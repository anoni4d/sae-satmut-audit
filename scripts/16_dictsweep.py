#!/usr/bin/env python
"""Dictionary-size sweep: is the per-feature non-identifiability a property of SAE decomposition,
or just an artifact of a dictionary larger than the signal requires?

Two reviewers raised this. The concern is concrete: at 16x expansion (m=4096) over a d=256
residual stream, many latents may be competing to explain the same structure, and the resulting
feature splitting — not any deep non-identifiability — could be what destroys cross-seed
agreement. If cross-seed reproducibility rises sharply as the dictionary shrinks, the honest
conclusion is "this dictionary was too large". If it stays low at every width, the conclusion is
about the decomposition itself.

We sweep expansion in {4, 8, 16, 32} (m = 1024..8192) at fixed k=64 and layer L, three seeds each,
and report SAE quality (FVU, L0, dead fraction) alongside the causal-signal and identifiability
metrics. The 16x column reuses the existing headline SAEs — `sae._sae_path` only puts the width in
the filename for non-default widths, so nothing is retrained or overwritten there.

Aggregation is at LOCUS level (the 29 measurements are 21 independent loci; see
`data.locus_groups`), with the element-level number kept alongside for continuity.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/16_dictsweep.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S, viz
from dna_interp.activations import load_activations, load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, path, write_report

log = get_logger("dictsweep")

EXPANSIONS = [4, 8, 16, 32]
N_SEEDS = 3


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 6


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def ensure_trained(cfg, mm, layer, expansion, seed, d):
    """Train the (m, seed) SAE if its artifact is missing; otherwise load. Idempotent."""
    m = expansion * d
    try:
        sae = S.load_sae(cfg, layer, seed=seed, m=m, d=d)
        log.info("[m=%d seed=%d] loaded existing SAE", m, seed)
        rng = np.random.default_rng(seed)
        return sae, S.evaluate_sae(cfg, sae, mm, rng)
    except Exception:
        log.info("[m=%d seed=%d] training ...", m, seed)
        cfg.sae.expansion = expansion
        sae, metrics = S.train_sae(cfg, mm, layer, seed=seed)
        return sae, metrics


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    mm, _ = load_activations(cfg, L)
    d = mm.shape[1]
    base_expansion = int(cfg.sae.expansion)
    log.info("dictsweep: layer %d, d=%d, expansions %s x %d seeds, %d measurements / %d loci",
             L, d, EXPANSIONS, N_SEEDS, len(satmut), len(set(gmap.values())))

    results = {"layer": L, "d": d, "k": int(cfg.sae.k), "n_seeds": N_SEEDS,
               "n_measurements": len(satmut), "n_loci": len(set(gmap.values())),
               "by_expansion": {}}

    for exp in EXPANSIONS:
        m = exp * d
        cfg.sae.expansion = exp
        quality, A_by_seed, sets_by_seed, W_by_seed = [], {}, {}, {}

        # one null per width: b_dec differs slightly between dictionaries, so the null that
        # defines the threshold is recomputed rather than shared.
        sae0, q0 = ensure_trained(cfg, mm, L, exp, 0, d)
        null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                             sae0.b_dec.detach(), cfg.device, seed=cfg.seed)
        null_align = controls.cached_alignment(cfg, model, null_enc, satmut, L, f"null_m{m}")
        nullA = controls.aggregate_features(null_align, 1, groups=gmap).A_f.to_numpy()
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))

        for seed in range(N_SEEDS):
            sae, q = (sae0, q0) if seed == 0 else ensure_trained(cfg, mm, L, exp, seed, d)
            quality.append(q)
            align = controls.cached_alignment(cfg, model, sae, satmut, L, f"topk_m{m}_seed{seed}")
            feats = controls.aggregate_features(align, int(cfg.causal.min_active_elements),
                                                groups=gmap)
            A = feats.set_index("feature_id")["A_f"]
            p = np.array([causal.empirical_pvalue(a, nullA) for a in A.to_numpy()])
            q_val = bh_fdr(p)
            ids = A.index.to_numpy()
            A_by_seed[seed] = A
            sets_by_seed[seed] = set(ids[q_val < 0.05].tolist())
            W_by_seed[seed] = sae.W_dec.detach().cpu().numpy()
            log.info("[m=%d seed=%d] FVU=%.3f L0=%.1f dead=%.3f | frac=%.3f q<0.05=%d",
                     m, seed, q["FVU"], q["L0"], q["dead_frac"],
                     float((A.to_numpy() > thr).mean()), len(sets_by_seed[seed]))

        # cross-seed identifiability at this width
        raw_j, matched_j, spear = [], [], []
        for s in range(1, N_SEEDS):
            raw_j.append(_jaccard(sets_by_seed[0], sets_by_seed[s]))
            common = A_by_seed[0].index.intersection(A_by_seed[s].index)
            spear.append(causal.spearman_rho(A_by_seed[0].loc[common].to_numpy(),
                                             A_by_seed[s].loc[common].to_numpy()))
            ra, rb, cos = controls.match_features(W_by_seed[0], W_by_seed[s])
            keep = cos > float(cfg.causal.match_cos_threshold)
            both = either = 0
            for fa, fb in zip(ra[keep], rb[keep]):
                ca, cb = int(fa) in sets_by_seed[0], int(fb) in sets_by_seed[s]
                both += ca and cb
                either += ca or cb
            matched_j.append(both / either if either else float("nan"))

        A0 = A_by_seed[0].to_numpy()
        results["by_expansion"][exp] = {
            "m": m,
            "FVU": float(np.mean([q["FVU"] for q in quality])),
            "L0": float(np.mean([q["L0"] for q in quality])),
            "dead_frac": float(np.mean([q["dead_frac"] for q in quality])),
            "threshold": float(thr),
            "causal_fraction": float((A0 > thr).mean()),
            "n_features_evaluated": int(len(A0)),
            "n_fdr_q05": int(len(sets_by_seed[0])),
            "median_A_f": float(np.nanmedian(A0)),
            "cross_seed_jaccard_raw": float(np.nanmean(raw_j)),
            "cross_seed_jaccard_matched": float(np.nanmean(matched_j)),
            "cross_seed_spearman_A": float(np.nanmean(spear)),
        }
        log.info("[m=%d] SUMMARY frac=%.3f q05=%d | jaccard raw=%.3f matched=%.3f "
                 "spearman=%.3f | dead=%.3f",
                 m, results["by_expansion"][exp]["causal_fraction"],
                 results["by_expansion"][exp]["n_fdr_q05"],
                 results["by_expansion"][exp]["cross_seed_jaccard_raw"],
                 results["by_expansion"][exp]["cross_seed_jaccard_matched"],
                 results["by_expansion"][exp]["cross_seed_spearman_A"],
                 results["by_expansion"][exp]["dead_frac"])

    cfg.sae.expansion = base_expansion
    (art / f"dictsweep_layer{L}.json").write_text(json.dumps(results, indent=2))
    viz.plot_dictsweep(cfg, results)
    _write(cfg, L, results)

    js = [results["by_expansion"][e]["cross_seed_jaccard_matched"] for e in EXPANSIONS]
    print(f"[phase15] matched cross-seed Jaccard by expansion "
          f"{dict(zip(EXPANSIONS, [round(x, 3) for x in js]))}")


def _write(cfg, L, res):
    rows = []
    for e in EXPANSIONS:
        r = res["by_expansion"][e]
        rows.append([f"{e}x", r["m"], f"{r['FVU']:.3f}", f"{r['L0']:.0f}",
                     f"{r['dead_frac']:.3f}", f"{r['causal_fraction']:.3f}", r["n_fdr_q05"],
                     f"{r['cross_seed_jaccard_raw']:.3f}",
                     f"{r['cross_seed_jaccard_matched']:.3f}",
                     f"{r['cross_seed_spearman_A']:.3f}"])
    body = f"""# Phase 15 — Dictionary-size sweep (is non-identifiability just an over-large dictionary?)

Layer L={L}, d={res['d']}, k={res['k']}, {res['n_seeds']} seeds per width.
**{res['n_measurements']} measurements over {res['n_loci']} loci**; A_f aggregated at locus level.

{md_table(["expansion", "m", "FVU", "L0", "dead frac", "causal frac", "FDR q<0.05",
           "seed Jaccard (raw)", "seed Jaccard (matched)", "seed Spearman A_f"], rows)}

Figure: `reports/figures/dictsweep.png`

**Reading.** The aggregate causal signal (causal fraction, FDR count) is expected to persist at
every width — it is a population statement. The decisive column is cross-seed agreement:

- If **agreement rises sharply as m shrinks**, the instability is feature splitting in an
  over-parameterised dictionary, and the claim must narrow to "at this width".
- If **agreement stays low at every width**, dictionary size is not the explanation and the
  non-identifiability is a property of the unsupervised decomposition itself.
- The `dead frac` column is the companion diagnostic: the paper already flags that Caduceus's
  milder instability may track *effective* dictionary size (live atoms) rather than architecture.
  This sweep varies nominal size directly, so comparing the two columns separates them.

**CHECKPOINT 15.** Which reading do the numbers support, and does the claim need
narrowing to a width range?
"""
    write_report(cfg, "phase15", body)


if __name__ == "__main__":
    main()
