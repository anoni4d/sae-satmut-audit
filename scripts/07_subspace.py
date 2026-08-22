#!/usr/bin/env python
"""Is the causal signal a stable SUBSPACE even though individual SAE features are not?

The deep dive (phase7) showed per-feature non-identifiability: across SAE retrains the *identity*
of causal features barely overlaps (Jaccard~0.05). The sharper question for a strong result is
whether the causal information nonetheless lives in a seed-stable low-dimensional subspace of the
residual stream — i.e. the directions are stable but the SAE's basis for them is not.

Per seed we take the decoder columns of the FDR-significant causal features as a set of unit
directions in R^d, build the subspace they span (SVD), and compare subspaces across seeds by
principal angles (mean squared canonical correlation). We compare that overlap against two nulls:
random subspaces of equal dimension, and the subspace of *non-causal* features. We also measure,
for each causal direction in the reference seed, its nearest cosine to (a) any direction and
(b) any causal direction in another seed — separating "direction absent" from "direction present
but causal label noisy".

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/07_subspace.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("subspace")
RNG = np.random.default_rng(0)


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p); ranked = p[order]
    q = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def subspace_basis(cols: np.ndarray, energy: float = 0.90, max_dim: int = 50):
    """cols: [d, n] unit columns. Return (U[:, :r], r) capturing `energy` of the spectrum."""
    if cols.shape[1] == 0:
        return np.zeros((cols.shape[0], 0)), 0
    U, s, _ = np.linalg.svd(cols, full_matrices=False)
    e = s ** 2
    r = int(np.searchsorted(np.cumsum(e) / e.sum(), energy) + 1)
    r = max(1, min(r, max_dim, U.shape[1]))
    return U[:, :r], r


def overlap(Ua: np.ndarray, Ub: np.ndarray) -> float:
    """Mean squared canonical correlation between two orthonormal bases = (1/r)||Ua^T Ub||_F^2,
    averaged over the smaller dimension. 1 = identical subspace, ~r_min/d for random."""
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return float("nan")
    M = Ua.T @ Ub
    r = min(Ua.shape[1], Ub.shape[1])
    return float((M ** 2).sum() / r)


def random_subspace(d, r):
    Q, _ = np.linalg.qr(RNG.standard_normal((d, r)))
    return Q[:, :r]


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    null_A = np.load(art / f"null_A_layer{L}.npy")
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))
    d = model.d_model

    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    log.info("subspace analysis: layer %d, d=%d, seeds=%s, thr=%.4f", L, d, seeds, thr)

    causal_cols, noncausal_cols, A_by_seed, W_by_seed, sets = {}, {}, {}, {}, {}
    for s in seeds:
        sae = S.load_sae(cfg, L, seed=s)
        W = sae.W_dec.detach().cpu().numpy()                       # [d, m], unit columns
        feats, _, _ = controls.run_causal_test(cfg, model, sae, satmut, L, with_ci=False)
        A = feats.set_index("feature_id")["A_f"]
        p = np.array([causal.empirical_pvalue(a, null_A) for a in feats.A_f])
        q = bh_fdr(p)
        feats = feats.assign(p_value=p, q_value=q)
        feats.to_csv(art / f"per_seed_feats_layer{L}_seed{s}.csv", index=False)
        causal_ids = feats.loc[feats.q_value < 0.05, "feature_id"].astype(int).to_numpy()
        noncausal_ids = feats.loc[feats.q_value > 0.5, "feature_id"].astype(int).to_numpy()
        causal_cols[s] = W[:, causal_ids]
        noncausal_cols[s] = W[:, RNG.choice(noncausal_ids, min(len(causal_ids), len(noncausal_ids)), replace=False)]
        A_by_seed[s], W_by_seed[s], sets[s] = A, W, set(causal_ids.tolist())
        log.info("seed %d: %d causal (q<0.05)", s, len(causal_ids))

    # per-seed causal subspaces
    bases, dims = {}, {}
    for s in seeds:
        bases[s], dims[s] = subspace_basis(causal_cols[s])
        log.info("seed %d: causal subspace dim (90%% energy) = %d  (from %d features)",
                 s, dims[s], causal_cols[s].shape[1])

    ref = seeds[0]
    pairs = [(ref, s) for s in seeds[1:]]
    res = {"layer": L, "d": d, "thr": float(thr), "seeds": seeds,
           "n_causal_per_seed": {int(s): int(causal_cols[s].shape[1]) for s in seeds},
           "causal_subspace_dim_per_seed": {int(s): int(dims[s]) for s in seeds},
           "pairs": []}

    for a, b in pairs:
        r = min(dims[a], dims[b])
        Ua, Ub = bases[a][:, :r], bases[b][:, :r]
        ov_causal = overlap(Ua, Ub)
        # null 1: random subspaces of dim r
        ov_rand = float(np.mean([overlap(random_subspace(d, r), random_subspace(d, r))
                                 for _ in range(50)]))
        # null 2: non-causal subspaces of seeds a,b
        Na, _ = subspace_basis(noncausal_cols[a]); Nb, _ = subspace_basis(noncausal_cols[b])
        rn = min(Na.shape[1], Nb.shape[1], r)
        ov_noncausal = overlap(Na[:, :rn], Nb[:, :rn])
        # nearest-cosine: each ref causal direction -> max cos to ANY / to CAUSAL col in seed b
        Ca, Cb = causal_cols[a], causal_cols[b]
        Wb_all = W_by_seed[b]
        cos_any = (Ca.T @ Wb_all)                          # [na, m]
        cos_causal = (Ca.T @ Cb)                           # [na, nb]
        nn_any = np.abs(cos_any).max(1) if cos_any.size else np.array([])
        nn_causal = np.abs(cos_causal).max(1) if cos_causal.size else np.array([])
        # feature-level Jaccard for reference
        jac = len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b]))
        res["pairs"].append({
            "seed_a": int(a), "seed_b": int(b), "r": int(r),
            "overlap_causal": ov_causal, "overlap_random_null": ov_rand,
            "overlap_noncausal_null": ov_noncausal,
            "nn_cos_any_median": float(np.median(nn_any)) if nn_any.size else float("nan"),
            "nn_cos_causal_median": float(np.median(nn_causal)) if nn_causal.size else float("nan"),
            "frac_causal_with_causal_partner_cos>0.5":
                float((nn_causal > 0.5).mean()) if nn_causal.size else float("nan"),
            "frac_causal_with_any_partner_cos>0.5":
                float((nn_any > 0.5).mean()) if nn_any.size else float("nan"),
            "feature_jaccard": jac,
        })
        log.info("seeds %d-%d: overlap causal=%.3f vs random=%.3f vs noncausal=%.3f | "
                 "nn_any=%.2f nn_causal=%.2f | jaccard=%.3f",
                 a, b, ov_causal, ov_rand, ov_noncausal,
                 res["pairs"][-1]["nn_cos_any_median"],
                 res["pairs"][-1]["nn_cos_causal_median"], jac)

    (art / f"subspace_layer{L}.json").write_text(json.dumps(res, indent=2))
    _write_phase8(cfg, L, res)
    mp = np.mean([p["overlap_causal"] for p in res["pairs"]])
    mr = np.mean([p["overlap_random_null"] for p in res["pairs"]])
    mj = np.mean([p["feature_jaccard"] for p in res["pairs"]])
    print(f"[phase8] causal-subspace overlap={mp:.3f} (random null={mr:.3f}); "
          f"feature jaccard={mj:.3f}; dims={list(res['causal_subspace_dim_per_seed'].values())}")


def _write_phase8(cfg, L, res):
    rows = []
    for p in res["pairs"]:
        rows.append([f"{p['seed_a']}-{p['seed_b']}", p["r"],
                     f"{p['overlap_causal']:.3f}", f"{p['overlap_random_null']:.3f}",
                     f"{p['overlap_noncausal_null']:.3f}",
                     f"{p['nn_cos_causal_median']:.2f}",
                     f"{p['frac_causal_with_causal_partner_cos>0.5']:.2f}",
                     f"{p['feature_jaccard']:.3f}"])
    dims = list(res["causal_subspace_dim_per_seed"].values())
    ncaus = list(res["n_causal_per_seed"].values())
    body = f"""# Phase 8 — Is the causal signal a stable subspace?

Layer L={L}, d={res['d']}. Per seed, the FDR-significant (q<0.05) causal features' decoder
columns span a low-dimensional subspace: **{ncaus} causal features → {dims}-dim subspaces**
(90% spectral energy). Across seeds we compare these subspaces by mean squared canonical
correlation (1 = identical, ~r/d for random).

{md_table(["seed pair", "r", "causal overlap", "random null", "non-causal null",
           "nn-cos to causal", "frac w/ causal partner", "feature Jaccard"], rows)}

**Reading.**
- If **causal overlap ≫ random null** while **feature Jaccard ≈ 0.05**, the causal information is a
  stable subspace even though the SAE's per-feature basis for it is not — i.e. subspace-level
  interpretation is identifiable where feature-level interpretation is not.
- `nn-cos to causal` / `frac w/ causal partner` separate *direction absent* (low) from *direction
  present but causal label noisy* (high): a high value means the causal directions recur across
  seeds but get re-bound to different SAE atoms.
- The small subspace dimension relative to the number of causal features quantifies feature
  splitting / redundancy.

**CHECKPOINT 8.** Does the subspace result warrant reframing the paper around
*subspace-level* causal identifiability + a seed-ensembling fix?
"""
    write_report(cfg, "phase8", body)


if __name__ == "__main__":
    main()
