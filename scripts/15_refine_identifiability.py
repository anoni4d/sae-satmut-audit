#!/usr/bin/env python
"""Refinement items 2, 3, 8 on the real layer-6 artifacts.

  (2) Threshold-crossing vs genuine basis arbitrariness:
      - continuous rank-stability (Spearman A_f, decoder-matched) is already the robust metric;
      - sweep the FDR threshold q and report cross-seed *matched* Jaccard(q) to show how much of the
        near-zero Jaccard is binarisation noise near the cutoff;
      - bootstrap a 95% CI on Jaccard at the operating point q=0.05.
  (3) Subspace claim with statistics:
      - bootstrap 95% CIs on the causal-subspace overlap and the non-causal-null overlap;
      - permutation p-value: is the causal subspace more cross-seed-stable than a random same-size
        feature subset?
  (8) FDR under the shared null:
      - re-run with Benjamini-Yekutieli (dependence-robust) and compare significant counts to BH,
        for both the R=1e3 and R=1e4 nulls.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/15_refine_identifiability.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dna_interp import causal, controls  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402

RNG = np.random.default_rng(0)


def chosen_layer(cfg, art):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = art / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 6


# ---------- shared helpers (mirrors 07_subspace.py) ----------
def bh_fdr(p):
    p = np.asarray(p, float); n = p.size
    order = np.argsort(p); ranked = p[order]
    q = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def by_fdr(p):
    """Benjamini-Yekutieli: BH inflated by the harmonic factor c(n)=sum 1/i (dependence-robust)."""
    p = np.asarray(p, float); n = p.size
    c = np.sum(1.0 / np.arange(1, n + 1))
    order = np.argsort(p); ranked = p[order]
    q = np.minimum.accumulate((ranked * n * c / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def subspace_basis(cols, energy=0.90, max_dim=50):
    if cols.shape[1] == 0:
        return np.zeros((cols.shape[0], 0)), 0
    U, s, _ = np.linalg.svd(cols, full_matrices=False)
    e = s ** 2
    r = int(np.searchsorted(np.cumsum(e) / e.sum(), energy) + 1)
    r = max(1, min(r, max_dim, U.shape[1]))
    return U[:, :r], r


def overlap(Ua, Ub):
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return float("nan")
    M = Ua.T @ Ub
    r = min(Ua.shape[1], Ub.shape[1])
    return float((M ** 2).sum() / r)


def load_decoder(art, L, seed):
    W = load_file(str(art / f"sae_layer{L}_seed{seed}.safetensors"))["W_dec"]  # [d, m]
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)


def main():
    cfg = load_config()
    art = Path(cfg.paths["artifacts"])
    L = chosen_layer(cfg, art)
    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    feats = {s: pd.read_csv(art / f"per_seed_feats_layer{L}_seed{s}.csv") for s in seeds}
    W = {s: load_decoder(art, L, s) for s in seeds}
    out = {"layer": L, "seeds": seeds}
    print(f"seeds={seeds}, n_features={len(feats[seeds[0]])}")

    # ---------------- ITEM 8: BY vs BH ----------------
    null1k = np.load(art / f"null_A_layer{L}.npy")
    null10k = np.load(art / f"null_A_layer{L}_R10000.npy")
    A0 = feats[seeds[0]]["A_f"].to_numpy()
    item8 = {}
    for tag, null in (("R1000", null1k), ("R10000", null10k)):
        p = np.array([causal.empirical_pvalue(a, null) for a in A0])
        qbh, qby = bh_fdr(p), by_fdr(p)
        item8[tag] = {
            "n_features": int(len(p)),
            "BH_q05": int((qbh < 0.05).sum()), "BH_q10": int((qbh < 0.10).sum()),
            "BY_q05": int((qby < 0.05).sum()), "BY_q10": int((qby < 0.10).sum()),
        }
        print(f"[item8/{tag}] BH q05={item8[tag]['BH_q05']} q10={item8[tag]['BH_q10']} | "
              f"BY q05={item8[tag]['BY_q05']} q10={item8[tag]['BY_q10']}")
    out["item8_fdr"] = item8

    # ---------------- ITEM 2: Jaccard vs threshold + bootstrap ----------------
    ref = seeds[0]
    qmap = {s: dict(zip(feats[s]["feature_id"].astype(int), feats[s]["q_value"])) for s in seeds}
    # matched feature pairs (Hungarian, cos>0.5) ref<->each other seed
    matches = {}
    for s in seeds[1:]:
        ra, rb, cos = controls.match_features(W[ref], W[s])
        keep = cos > 0.5
        matches[s] = list(zip(ra[keep].tolist(), rb[keep].tolist()))
    qgrid = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

    def matched_jaccard(s, qthr, pairs=None):
        pairs = matches[s] if pairs is None else pairs
        both = either = 0
        for fa, fb in pairs:
            ca = qmap[ref].get(fa, 1.0) < qthr
            cb = qmap[s].get(fb, 1.0) < qthr
            both += ca and cb
            either += ca or cb
        return both / either if either else float("nan")

    jac_curve = {}
    for q in qgrid:
        vals = [matched_jaccard(s, q) for s in seeds[1:]]
        jac_curve[q] = float(np.nanmean(vals))
    # bootstrap CI at q=0.05 (resample matched pairs)
    boot = []
    for _ in range(2000):
        vals = []
        for s in seeds[1:]:
            pr = matches[s]
            idx = RNG.integers(0, len(pr), len(pr))
            vals.append(matched_jaccard(s, 0.05, [pr[i] for i in idx]))
        boot.append(np.nanmean(vals))
    out["item2_jaccard_vs_q"] = jac_curve
    out["item2_jaccard_q05_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    out["item2_spearman_Af_pairs"] = [0.232, 0.256]  # from robustness_v2 (continuous, primary)
    print(f"[item2] Jaccard(q): " + ", ".join(f"{q}:{jac_curve[q]:.3f}" for q in qgrid))
    print(f"[item2] Jaccard@q05 95% CI = [{out['item2_jaccard_q05_ci'][0]:.3f}, "
          f"{out['item2_jaccard_q05_ci'][1]:.3f}]")

    # ---------------- ITEM 3: subspace CI + permutation ----------------
    causal_cols, noncausal_ids_by = {}, {}
    for s in seeds:
        f = feats[s]
        cids = f.loc[f.q_value < 0.05, "feature_id"].astype(int).to_numpy()
        nids = f.loc[f.q_value > 0.5, "feature_id"].astype(int).to_numpy()
        causal_cols[s] = W[s][:, cids]
        noncausal_ids_by[s] = nids
    pairs = [(ref, s) for s in seeds[1:]]

    def pair_overlap(colsA, colsB):
        Ua, _ = subspace_basis(colsA); Ub, _ = subspace_basis(colsB)
        r = min(Ua.shape[1], Ub.shape[1])
        return overlap(Ua[:, :r], Ub[:, :r])

    obs_causal = float(np.mean([pair_overlap(causal_cols[a], causal_cols[b]) for a, b in pairs]))
    # non-causal: random same-size subset
    nc_cols = {s: W[s][:, RNG.choice(noncausal_ids_by[s],
                                     min(causal_cols[s].shape[1], len(noncausal_ids_by[s])),
                                     replace=False)] for s in seeds}
    obs_noncausal = float(np.mean([pair_overlap(nc_cols[a], nc_cols[b]) for a, b in pairs]))

    # bootstrap CIs (resample columns with replacement)
    def boot_overlap(cols_by, B=600):
        vals = []
        for _ in range(B):
            res = {}
            for s in seeds:
                c = cols_by[s]
                if c.shape[1] == 0:
                    res[s] = c; continue
                idx = RNG.integers(0, c.shape[1], c.shape[1])
                res[s] = c[:, idx]
            vals.append(np.mean([pair_overlap(res[a], res[b]) for a, b in pairs]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    ci_causal = boot_overlap(causal_cols)
    ci_noncausal = boot_overlap(nc_cols)
    # permutation: relabel causal/noncausal pooled, same-size causal subset, overlap null
    perm = []
    for _ in range(1000):
        res = {}
        for s in seeds:
            pool = np.concatenate([feats[s].loc[feats[s].q_value < 0.05, "feature_id"].astype(int).to_numpy(),
                                   noncausal_ids_by[s]])
            sub = RNG.choice(pool, causal_cols[s].shape[1], replace=False)
            res[s] = W[s][:, sub]
        perm.append(np.mean([pair_overlap(res[a], res[b]) for a, b in pairs]))
    perm = np.array(perm)
    pval = float((perm >= obs_causal).mean())
    out["item3_subspace"] = {
        "obs_causal_overlap": obs_causal, "causal_overlap_95ci": ci_causal,
        "obs_noncausal_overlap": obs_noncausal, "noncausal_overlap_95ci": ci_noncausal,
        "perm_null_mean": float(perm.mean()), "perm_p_causal_gt_null": pval,
        "ci_overlap_between_causal_and_noncausal":
            bool(ci_causal[0] <= ci_noncausal[1] and ci_noncausal[0] <= ci_causal[1]),
    }
    print(f"[item3] causal overlap={obs_causal:.3f} CI{ci_causal} | "
          f"non-causal={obs_noncausal:.3f} CI{ci_noncausal} | "
          f"perm p(causal>null)={pval:.3f} | CIs overlap={out['item3_subspace']['ci_overlap_between_causal_and_noncausal']}")

    (art / "refine_items_2_3_8.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {art / 'refine_items_2_3_8.json'}")


if __name__ == "__main__":
    main()
