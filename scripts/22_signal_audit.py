#!/usr/bin/env python
"""Audit: is the aggregate 'causal signal' directional, or a dispersion artifact?

The headline test thresholds A_f at the 95th percentile of a null and reports the fraction of
features above it. That procedure is one-sided by construction but was never checked for
one-sidedness: if SAE features merely have a WIDER A_f distribution than the null (which sparser
units do, mechanically), then a large fraction clears the upper threshold, an equally large
fraction clears the lower one, and the enrichment is about spread rather than alignment.

This script runs the four checks that distinguish the two, all against the ORIGINAL null so none
of them depends on the sparsity-matched null:

  (1) SHIFT      Mann-Whitney of real vs null A_f, plus the common-language effect size
                 P(real > null). A directional signal must show a shift, not just a spread.
  (2) SYMMETRY   fraction of features above the null's 95th percentile vs below its 5th.
                 Equal fractions => spread, not alignment.
  (3) MIRROR-FDR the same Benjamini-Hochberg procedure applied to the upper and to the lower
                 tail. A directional signal yields many upper-tail discoveries and few lower.
  (4) RAW        fraction of per-(feature, element) alignments above zero, before any
                 aggregation or thresholding, tested across loci (the independent unit) with a
                 Wilcoxon signed-rank test so the clustering is respected.

Optionally (5) SCALING: the same fraction as a function of dictionary size, read from
`dictsweep_layer{L}.json`. A biological signal should not grow with how finely the representation
is chopped up; a dispersion artifact should.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/22_signal_audit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from _common import boot
from dna_interp import causal, controls, sae as S, viz
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("audit")


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
    ma = int(cfg.causal.min_active_elements)

    align = controls.cached_alignment(cfg, model, sae, satmut, L, f"topk_seed{cfg.seed}")
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         sae.b_dec.detach(), cfg.device, seed=cfg.seed)
    null_align = controls.cached_alignment(cfg, model, null_enc, satmut, L, "null")

    real = controls.aggregate_features(align, ma, groups=gmap).A_f.to_numpy()
    null = controls.aggregate_features(null_align, 1, groups=gmap).A_f.to_numpy()
    real, null = real[~np.isnan(real)], null[~np.isnan(null)]
    out = {"layer": L, "n_features": int(real.size), "n_null": int(null.size),
           "n_loci": len(set(gmap.values()))}

    # ---- (1) shift ----
    u, p_greater = mannwhitneyu(real, null, alternative="greater")
    out["shift"] = {
        "P_real_gt_null": float(u / (real.size * null.size)),
        "mannwhitney_p_greater": float(p_greater),
        "real_median": float(np.median(real)), "null_median": float(np.median(null)),
        "real_sd": float(real.std()), "null_sd": float(null.std()),
    }
    log.info("(1) SHIFT: P(real>null)=%.3f p=%.3g | medians %.4f vs %.4f | sd %.4f vs %.4f",
             out["shift"]["P_real_gt_null"], p_greater, out["shift"]["real_median"],
             out["shift"]["null_median"], out["shift"]["real_sd"], out["shift"]["null_sd"])

    # ---- (2) symmetry ----
    hi = float(np.percentile(null, float(cfg.causal.percentile)))
    lo = float(np.percentile(null, 100.0 - float(cfg.causal.percentile)))
    up, dn = float((real > hi).mean()), float((real < lo).mean())
    out["symmetry"] = {"thr_upper": hi, "thr_lower": lo, "frac_above": up, "frac_below": dn,
                       "one_sided_excess": up - dn}
    log.info("(2) SYMMETRY: above=%.3f below=%.3f -> one-sided excess %+.4f", up, dn, up - dn)

    # ---- (3) mirror FDR ----
    p_up = np.array([(1 + (null >= a).sum()) / (1 + null.size) for a in real])
    p_dn = np.array([(1 + (null <= a).sum()) / (1 + null.size) for a in real])
    q_up, q_dn = bh_fdr(p_up), bh_fdr(p_dn)
    out["mirror_fdr"] = {"causal_q05": int((q_up < 0.05).sum()),
                         "anti_q05": int((q_dn < 0.05).sum()),
                         "causal_q10": int((q_up < 0.10).sum()),
                         "anti_q10": int((q_dn < 0.10).sum())}
    log.info("(3) MIRROR-FDR: causal q<0.05=%d vs anti-aligned q<0.05=%d",
             out["mirror_fdr"]["causal_q05"], out["mirror_fdr"]["anti_q05"])

    # ---- (4) raw, per locus (the independent unit) ----
    raw = align.copy()
    raw["locus"] = raw.elem_id.map(lambda e: gmap.get(e, e))
    per_locus = raw.groupby("locus").a_f.apply(lambda s: float((s > 0).mean()))
    nraw = null_align.copy()
    nraw["locus"] = nraw.elem_id.map(lambda e: gmap.get(e, e))
    per_locus_null = nraw.groupby("locus").a_f.apply(lambda s: float((s > 0).mean()))
    stat, p_w = wilcoxon(per_locus.to_numpy() - 0.5, alternative="greater")
    common = per_locus.index.intersection(per_locus_null.index)
    stat2, p_w2 = wilcoxon(per_locus.loc[common].to_numpy() - per_locus_null.loc[common].to_numpy(),
                           alternative="greater")
    out["raw"] = {
        "frac_positive_overall": float((raw.a_f > 0).mean()),
        "frac_positive_null": float((nraw.a_f > 0).mean()),
        "per_locus_mean": float(per_locus.mean()),
        "wilcoxon_vs_half_p": float(p_w),
        "wilcoxon_vs_null_p": float(p_w2),
        "n_pairs": int(len(raw)),
    }
    log.info("(4) RAW: frac(a_f>0) real=%.4f null=%.4f | per-locus Wilcoxon vs 0.5 p=%.4g, "
             "vs null p=%.4g", out["raw"]["frac_positive_overall"],
             out["raw"]["frac_positive_null"], p_w, p_w2)

    # ---- (5) scaling with dictionary size ----
    ds_path = art / f"dictsweep_layer{L}.json"
    if ds_path.exists():
        ds = json.loads(ds_path.read_text())["by_expansion"]
        out["scaling"] = {str(k): {"m": v["m"], "causal_fraction": v["causal_fraction"],
                                   "n_fdr_q05": v["n_fdr_q05"], "dead_frac": v["dead_frac"]}
                          for k, v in ds.items()}
        log.info("(5) SCALING: %s", {v["m"]: round(v["causal_fraction"], 3) for v in ds.values()})

    (art / f"signal_audit_layer{L}.json").write_text(json.dumps(out, indent=2))
    viz.plot_signal_audit(cfg, real, null, hi, lo, out)
    _write(cfg, L, out)
    print(f"[phase20] one-sided excess={out['symmetry']['one_sided_excess']:+.4f} | "
          f"mirror-FDR causal={out['mirror_fdr']['causal_q05']} vs anti={out['mirror_fdr']['anti_q05']} | "
          f"P(real>null)={out['shift']['P_real_gt_null']:.3f}")


def _write(cfg, L, out):
    sh, sy, mf, rw = out["shift"], out["symmetry"], out["mirror_fdr"], out["raw"]
    scaling = ""
    if "scaling" in out:
        rows = [[v["m"], f"{v['causal_fraction']:.3f}", v["n_fdr_q05"], f"{v['dead_frac']:.3f}"]
                for v in sorted(out["scaling"].values(), key=lambda x: x["m"])]
        scaling = ("\n## (5) Does the signal scale with dictionary size?\n\n"
                   + md_table(["m", "causal fraction", "FDR q<0.05", "dead frac"], rows)
                   + "\n\nA biological signal should not depend on how finely the representation is\n"
                     "partitioned. A dispersion artifact should: more, sparser features means a wider\n"
                     "per-feature A_f distribution and so more units past a fixed one-sided threshold.\n")
    body = f"""# Phase 20 — Signal audit: directional effect or dispersion artifact?

Layer L={L}, {out['n_features']} features vs {out['n_null']} null directions, locus-level
aggregation over {out['n_loci']} loci. Everything below uses the ORIGINAL random-direction null.

## (1) Is the distribution shifted, or just wider?
| quantity | real features | null |
| --- | --- | --- |
| median A_f | {sh['real_median']:+.4f} | {sh['null_median']:+.4f} |
| sd A_f | {sh['real_sd']:.4f} | {sh['null_sd']:.4f} |

P(real > null) = **{sh['P_real_gt_null']:.3f}** (0.5 = chance), Mann-Whitney one-sided
p = **{sh['mannwhitney_p_greater']:.3g}**.

## (2) Is the excess one-sided?
| | fraction |
| --- | --- |
| above the null's upper threshold ({sy['thr_upper']:+.4f}) | **{sy['frac_above']:.3f}** |
| below the null's lower threshold ({sy['thr_lower']:+.4f}) | **{sy['frac_below']:.3f}** |
| one-sided excess | **{sy['one_sided_excess']:+.4f}** |

## (3) Mirror-FDR — the same procedure applied to each tail
| direction | q<0.05 | q<0.10 |
| --- | --- | --- |
| "causally aligned" (upper tail) | **{mf['causal_q05']}** | {mf['causal_q10']} |
| "anti-aligned" (lower tail) | **{mf['anti_q05']}** | {mf['anti_q10']} |

## (4) Before any aggregation or thresholding
Fraction of per-(feature, element) alignments above zero: real **{rw['frac_positive_overall']:.4f}**
vs null {rw['frac_positive_null']:.4f} over {rw['n_pairs']:,} pairs. Testing across loci (the
independent unit, Wilcoxon signed-rank): vs 0.5 p={rw['wilcoxon_vs_half_p']:.4g}; vs the matched
null p={rw['wilcoxon_vs_null_p']:.4g}.
{scaling}
**Reading.** A directional causal signal requires (1) a shift, (2) a one-sided excess, and (3) many
more upper- than lower-tail discoveries. If instead the excess is symmetric and the mirror counts
are comparable, the reported "causal fraction" measures how much MORE VARIABLE feature alignments
are than null-direction alignments — which sparser units are mechanically — and not whether they
are better aligned. In that case the correct headline is that no directional aggregate signal is
demonstrated, and any per-feature interpretation built on the threshold inherits the same problem.

**CHECKPOINT 20.** This audit is load-bearing for the paper's central claim; confirm the
reading before the writeup is finalised.
"""
    write_report(cfg, "phase20", body)


if __name__ == "__main__":
    main()
