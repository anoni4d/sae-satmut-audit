#!/usr/bin/env python
"""Property-matched null (reviewer request: random directions are not comparable to SAE features).

The headline null draws random unit directions and passes them through the identical ISM pipeline.
Reviewers correctly note that such directions are not matched to real SAE features in sparsity,
activation frequency, threshold behaviour, or coverage across elements: a dense ReLU direction
fires on roughly half of all tokens where a TopK feature fires on ~1.6%, so the two are not
answering the same question.

We therefore add a second null whose directions are random but *sparsity-matched*: each gets a
per-direction threshold calibrated on the corpus so its firing density equals that of a randomly
drawn real feature (`controls.MatchedNullSAE`). We additionally report a density-STRATIFIED
threshold, comparing each real feature only against null directions in its own density decile.

Note on scale: the alignment metric is invariant to positive rescaling of the sensitivity vector,
so activation magnitude needs no matching and cannot affect the comparison — only the support and
sparsity of the sensitivity profile can, which is exactly what is matched here.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/20_matchednull.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import causal, controls, sae as S, viz
from dna_interp.activations import load_activations, load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("matchednull")


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 6


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    mm, _ = load_activations(cfg, L)
    R = int(cfg.causal.null_R)

    # real feature densities define the profile the null must match
    metrics = S.evaluate_sae(cfg, sae, mm, np.random.default_rng(0))
    dens_real_all = np.asarray(metrics["density"])
    log.info("real features: median density %.5f, live %d/%d",
             float(np.median(dens_real_all[dens_real_all > 0])),
             int((dens_real_all > 0).sum()), len(dens_real_all))

    align = controls.cached_alignment(cfg, model, sae, satmut, L, f"topk_seed{cfg.seed}")
    feats = controls.aggregate_features(align, int(cfg.causal.min_active_elements), groups=gmap)
    A_real = feats.set_index("feature_id")["A_f"]
    dens_real = dens_real_all[A_real.index.to_numpy()]

    # null 1: the original dense random-direction null
    dense_enc = controls.RandomFeatureSAE(model.d_model, R, sae.b_dec.detach(), cfg.device,
                                          seed=cfg.seed)
    dense_align = controls.cached_alignment(cfg, model, dense_enc, satmut, L, "null")
    A_dense = controls.aggregate_features(dense_align, 1, groups=gmap)

    # null 2: sparsity/density-matched directions
    matched_enc = controls.MatchedNullSAE.calibrated(
        mm, model.d_model, R, sae.b_dec.detach(), cfg.device, dens_real_all, seed=cfg.seed)
    matched_align = controls.cached_alignment(cfg, model, matched_enc, satmut, L, "null_matched")
    A_matched = controls.aggregate_features(matched_align, 1, groups=gmap)
    dens_null = np.asarray(matched_enc.target_density)[A_matched.feature_id.to_numpy()]

    out = {"layer": L, "null_R": R, "n_loci": len(set(gmap.values())),
           "real_density_median": float(np.median(dens_real[dens_real > 0])) if (dens_real > 0).any() else 0.0}
    rows = []
    A = A_real.to_numpy()
    for tag, nullA, cov in (("dense random directions", A_dense.A_f.to_numpy(),
                             len(A_dense) / R),
                            ("sparsity-matched directions", A_matched.A_f.to_numpy(),
                             len(A_matched) / R)):
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
        p = np.array([causal.empirical_pvalue(a, nullA) for a in A])
        q = bh_fdr(p)
        key = "dense" if "dense" in tag else "matched"
        out[key] = {"threshold": float(thr), "causal_fraction": float((A > thr).mean()),
                    "n_above": int((A > thr).sum()), "n_fdr_q05": int((q < 0.05).sum()),
                    "null_median_A": float(np.nanmedian(nullA)),
                    "frac_null_directions_active": float(cov)}
        rows.append([tag, f"{cov:.2f}", f"{np.nanmedian(nullA):+.4f}", f"{thr:.3f}",
                     f"{(A > thr).mean():.3f}", int((A > thr).sum()), int((q < 0.05).sum())])
        log.info("[%s] active %.2f | null median %.4f | thr %.3f | frac %.3f | q<0.05 %d",
                 tag, cov, float(np.nanmedian(nullA)), thr, float((A > thr).mean()),
                 int((q < 0.05).sum()))

    # density-stratified threshold against the matched null
    thr_strat = controls.stratified_threshold(A, dens_real, A_matched.A_f.to_numpy(), dens_null)
    out["stratified"] = {"causal_fraction": float((A > thr_strat).mean()),
                         "n_above": int((A > thr_strat).sum()),
                         "threshold_min": float(np.nanmin(thr_strat)),
                         "threshold_max": float(np.nanmax(thr_strat))}
    rows.append(["sparsity-matched, density-stratified", "-", "-",
                 f"{np.nanmin(thr_strat):.3f}-{np.nanmax(thr_strat):.3f}",
                 f"{(A > thr_strat).mean():.3f}", int((A > thr_strat).sum()), "-"])
    log.info("[stratified] thr range %.3f-%.3f | frac %.3f", float(np.nanmin(thr_strat)),
             float(np.nanmax(thr_strat)), float((A > thr_strat).mean()))

    (art / f"matchednull_layer{L}.json").write_text(json.dumps(out, indent=2))
    viz.plot_Af_distribution(cfg, A, A_dense.A_f.to_numpy(), out["dense"]["threshold"],
                             matched_null=A_matched.A_f.to_numpy())
    _write(cfg, L, out, rows)
    print(f"[phase18] dense-null frac={out['dense']['causal_fraction']:.3f} | "
          f"matched-null frac={out['matched']['causal_fraction']:.3f} | "
          f"density-stratified frac={out['stratified']['causal_fraction']:.3f}")


def _write(cfg, L, out, rows):
    body = f"""# Phase 18 — Property-matched null

Layer L={L}, R={out['null_R']} directions per null, locus-level aggregation over
{out['n_loci']} loci. Median firing density of the real evaluated features:
{out['real_density_median']:.5f}.

{md_table(["null", "frac directions active", "null median A", "threshold",
           "causal fraction", "n above thr", "FDR q<0.05"], rows)}

**Why a second null.** The dense random-direction null answers "could an arbitrary direction in
the residual stream score this well?". It is deliberately unconditioned, and as the
`frac directions active` column shows it is not comparable to a real feature in sparsity or in
coverage across elements. The sparsity-matched null answers the sharper question: "could an
arbitrary direction *with the same firing profile as a real feature* score this well?" Each of its
directions carries a per-direction threshold calibrated on the corpus to match the density of a
randomly drawn real feature.

Activation *scale* is deliberately not matched, because it cannot matter: the alignment metric
$a_f(e)=2\\sum_p I_f[p]\\,r(E_e[p])/\\sum_p I_f[p]-1$ is invariant to positive rescaling of $I_f$
(verified in `tests/`). Only the support and sparsity of the sensitivity profile can affect the
score, and those are what the matched null equalises.

The final row goes further and compares each real feature only against null directions in its own
density decile, so no conclusion rests on comparing a rare feature to mostly-dense directions.

**Reading.** If the causal fraction and FDR count survive under the matched and stratified nulls,
the aggregate signal is not an artifact of comparing sparse features to dense random directions.
If they collapse, the original null was too permissive and the headline must be restated against
the matched one.

**CHECKPOINT 18.** Confirm which null the paper should quote as primary.
"""
    write_report(cfg, "phase18", body)


if __name__ == "__main__":
    main()
