#!/usr/bin/env python
"""Split-half noise ceiling + locus-level re-aggregation of the causal test.

Two jobs, one ISM pass per encoder.

(A) **Locus-level re-aggregation.** The 29 Kircher measurements are 21 loci (TERT x4, SORT1 x3,
    LDLR/PKLR/ZRS x2 share a byte-identical sequence). The original A_f weighted-median therefore
    counted TERT four times. We recompute the headline causal fraction / FDR with repeat
    measurements of a locus averaged first, and report both side by side.

(B) **Split-half noise ceiling** — the reviewers' question "how much of the cross-seed
    disagreement is statistical uncertainty rather than true non-identifiability?" Within ONE
    seed, split the loci into disjoint halves, compute A_f on each half, and measure their
    agreement. That is the reproducibility attainable with *zero* SAE variation: pure estimation
    noise at half the element budget. Cross-seed agreement at the same budget is then reported as
    a fraction of that ceiling. Cross-seed << ceiling => genuine non-identifiability; cross-seed
    ~ ceiling => the instability is mostly small-n noise.

Both reuse a cached per-(feature, measurement) alignment matrix, so re-running the analysis costs
no further ISM.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/17_noiseceiling.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, path, write_report

log = get_logger("noiseceiling")
N_SPLITS = 200


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


def cached_alignment(cfg, model, encoder, satmut, layer, tag):
    """compute_alignments, memoised to parquet — the expensive ISM pass happens once."""
    p = path(cfg, "artifacts", f"align_{tag}_layer{layer}.parquet")
    if p.exists():
        log.info("reusing cached alignment %s", p.name)
        return pd.read_parquet(p)
    log.info("running ISM for %s ...", tag)
    df = controls.compute_alignments(cfg, model, encoder, satmut, layer, restrict_active=True)
    df.to_parquet(p)
    return df


def A_from(align, groups=None, min_active=1):
    """feature_id -> A_f, at element level (groups=None) or locus level."""
    feats = controls.aggregate_features(align, min_active, with_ci=False, groups=groups)
    return feats.set_index("feature_id")["A_f"]


def _sets_and_rank(A, null_A, q_thr=0.05):
    ids = A.index.to_numpy()
    p = np.array([causal.empirical_pvalue(a, null_A) for a in A.to_numpy()])
    q = bh_fdr(p)
    return set(ids[q < q_thr].tolist()), A


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
    loci = sorted(set(gmap.values()))
    log.info("layer %d | %d measurements over %d loci", L, len(satmut), len(loci))

    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    log.info("seeds: %s", seeds)

    # ---- one ISM pass per encoder (SAEs + the null), cached ----
    aligns = {}
    for s in seeds:
        sae = S.load_sae(cfg, L, seed=s)
        aligns[s] = cached_alignment(cfg, model, sae, satmut, L, f"topk_seed{s}")
    sae0 = S.load_sae(cfg, L, seed=seeds[0])
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         sae0.b_dec.detach(), cfg.device, seed=cfg.seed)
    null_align = cached_alignment(cfg, model, null_enc, satmut, L, "null")

    out = {"layer": L, "seeds": seeds, "n_measurements": len(satmut), "n_loci": len(loci)}

    # ---- (A) element-level vs locus-level headline ----
    level_rows = []
    A_by_seed = {}
    for level, g in (("element", None), ("locus", gmap)):
        nullA = A_from(null_align, groups=g).to_numpy()
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
        per_seed = {}
        for s in seeds:
            A = A_from(aligns[s], groups=g)
            cset, _ = _sets_and_rank(A, nullA)
            per_seed[s] = {"A": A, "causal_set": cset}
            if s == seeds[0]:
                frac = float((A.to_numpy() > thr).mean())
                level_rows.append([level, f"{thr:.3f}", f"{frac:.3f}",
                                   int((A.to_numpy() > thr).sum()), len(A), len(cset)])
        A_by_seed[level] = per_seed
        out[f"{level}_level"] = {
            "threshold": float(thr),
            "causal_fraction": float((per_seed[seeds[0]]["A"].to_numpy() > thr).mean()),
            "n_features": int(len(per_seed[seeds[0]]["A"])),
            "n_fdr_q05": int(len(per_seed[seeds[0]]["causal_set"])),
            "median_A_f": float(np.nanmedian(per_seed[seeds[0]]["A"].to_numpy())),
        }
        log.info("[%s-level] thr=%.3f frac=%.3f FDR q<0.05=%d", level, thr,
                 out[f"{level}_level"]["causal_fraction"], out[f"{level}_level"]["n_fdr_q05"])

    # cross-seed agreement at locus level (the honest unit).
    # Feature INDICES are arbitrary across retrains, so any cross-seed comparison of A_f must go
    # through Hungarian decoder matching first; only the raw set-Jaccard is index-free.
    gm = gmap
    W = {s: S.load_sae(cfg, L, seed=s).W_dec.detach().cpu().numpy() for s in seeds}
    matches = {}
    for s in seeds[1:]:
        ra, rb, cos = controls.match_features(W[seeds[0]], W[s])
        keep = cos > float(cfg.causal.match_cos_threshold)
        matches[s] = (ra[keep], rb[keep])
        log.info("seed %d: %d/%d decoder columns matched above cos %.2f",
                 s, int(keep.sum()), len(cos), float(cfg.causal.match_cos_threshold))

    cross = []
    for s in seeds[1:]:
        a, b = A_by_seed["locus"][seeds[0]], A_by_seed["locus"][s]
        ra, rb = matches[s]
        pairs = [(int(x), int(y)) for x, y in zip(ra, rb)
                 if x in a["A"].index and y in b["A"].index]
        xa = np.array([a["A"].loc[x] for x, _ in pairs])
        xb = np.array([b["A"].loc[y] for _, y in pairs])
        both = sum(1 for x, y in pairs if x in a["causal_set"] and y in b["causal_set"])
        either = sum(1 for x, y in pairs if x in a["causal_set"] or y in b["causal_set"])
        cross.append({"pair": f"{seeds[0]}-{s}",
                      "jaccard_q05_raw": _jaccard(a["causal_set"], b["causal_set"]),
                      "jaccard_q05_matched": both / either if either else float("nan"),
                      "spearman_A_matched": causal.spearman_rho(xa, xb),
                      "n_matched": len(pairs)})
    out["cross_seed_locus_level"] = cross
    log.info("cross-seed (locus level): raw J=%.3f matched J=%.3f matched spearman=%.3f",
             np.mean([c["jaccard_q05_raw"] for c in cross]),
             np.mean([c["jaccard_q05_matched"] for c in cross]),
             np.mean([c["spearman_A_matched"] for c in cross]))

    # ---- (B) split-half noise ceiling ----
    # collapse repeat measurements to loci ONCE; inside the resampling loop the unit is the locus,
    # so every split is then a plain row filter rather than a repeated groupby.
    aligns_L = {s: controls.collapse_to_loci(aligns[s], gmap) for s in seeds}
    null_L = controls.collapse_to_loci(null_align, gmap)

    # A 2x2 design isolates the two sources of non-reproduction at a MATCHED element budget:
    #   same seed / different data  -> estimation noise alone (the ceiling)
    #   different seed / same data  -> SAE basis variation alone
    #   different seed / different data -> both
    # Without the second cell one cannot say how much of the near-zero cross-seed agreement is
    # the SAE and how much is simply having 21 loci.
    rng = np.random.default_rng(0)
    half = len(loci) // 2
    ceil_j, ceil_s, cross_j, cross_s = [], [], [], []
    seedonly_j, seedonly_s = [], []
    for _ in range(N_SPLITS):
        perm = rng.permutation(loci)
        LA, LB = set(perm[:half]), set(perm[half:2 * half])

        def sub(align, keep):
            return align[align.elem_id.isin(keep)]

        nullA_A = A_from(sub(null_L, LA)).to_numpy()
        nullA_B = A_from(sub(null_L, LB)).to_numpy()

        # ceiling: SAME seed, disjoint halves of the loci -> pure estimation noise
        a0 = A_from(sub(aligns_L[seeds[0]], LA))
        b0 = A_from(sub(aligns_L[seeds[0]], LB))
        sa, _ = _sets_and_rank(a0, nullA_A)
        sb, _ = _sets_and_rank(b0, nullA_B)
        common = a0.index.intersection(b0.index)
        ceil_j.append(_jaccard(sa, sb))
        ceil_s.append(causal.spearman_rho(a0.loc[common].to_numpy(), b0.loc[common].to_numpy()))

        # comparison: DIFFERENT seeds, same disjoint halves -> noise + SAE variation.
        # A_f is compared through the decoder matching (indices are arbitrary across seeds);
        # the set-Jaccard is the raw, index-free one, matching how the paper reports it.
        a1 = A_from(sub(aligns_L[seeds[0]], LA))
        b1 = A_from(sub(aligns_L[seeds[1]], LB))
        sa1, _ = _sets_and_rank(a1, nullA_A)
        sb1, _ = _sets_and_rank(b1, nullA_B)
        cross_j.append(_jaccard(sa1, sb1))
        ra, rb = matches[seeds[1]]
        pr = [(int(x), int(y)) for x, y in zip(ra, rb) if x in a1.index and y in b1.index]
        if len(pr) >= 3:
            cross_s.append(causal.spearman_rho(
                np.array([a1.loc[x] for x, _ in pr]), np.array([b1.loc[y] for _, y in pr])))
        else:
            cross_s.append(np.nan)

        # missing cell: DIFFERENT seeds on the SAME half -> SAE basis variation with the
        # element budget held fixed, directly comparable to the ceiling above
        a2 = a0                                             # seed 0 on half A (already computed)
        b2 = A_from(sub(aligns_L[seeds[1]], LA))
        sa2 = sa
        sb2, _ = _sets_and_rank(b2, nullA_A)
        seedonly_j.append(_jaccard(sa2, sb2))
        pr2 = [(int(x), int(y)) for x, y in zip(ra, rb) if x in a2.index and y in b2.index]
        if len(pr2) >= 3:
            seedonly_s.append(causal.spearman_rho(
                np.array([a2.loc[x] for x, _ in pr2]), np.array([b2.loc[y] for _, y in pr2])))
        else:
            seedonly_s.append(np.nan)

    def stat(v):
        v = np.array([x for x in v if not np.isnan(x)])
        return {"mean": float(v.mean()), "sd": float(v.std()),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}

    out["noise_ceiling"] = {
        "n_splits": N_SPLITS, "half_size_loci": half,
        "same_seed_diff_data_jaccard": stat(ceil_j),      # estimation noise alone (ceiling)
        "same_seed_diff_data_spearman": stat(ceil_s),
        "diff_seed_same_data_jaccard": stat(seedonly_j),  # SAE variation alone
        "diff_seed_same_data_spearman": stat(seedonly_s),
        "diff_seed_diff_data_jaccard": stat(cross_j),     # both
        "diff_seed_diff_data_spearman": stat(cross_s),
    }
    sj, oj, cj = np.nanmean(ceil_j), np.nanmean(seedonly_j), np.nanmean(cross_j)
    ss, os_, cs = np.nanmean(ceil_s), np.nanmean(seedonly_s), np.nanmean(cross_s)
    out["noise_ceiling"]["jaccard_frac_of_ceiling"] = float(cj / sj) if sj else float("nan")
    out["noise_ceiling"]["seedonly_frac_of_ceiling"] = float(oj / sj) if sj else float("nan")
    out["noise_ceiling"]["spearman_frac_of_ceiling"] = float(cs / ss) if ss else float("nan")
    log.info("ceiling (J): same-seed/diff-data=%.3f | diff-seed/same-data=%.3f | both=%.3f", sj, oj, cj)
    log.info("ceiling (rho): same-seed/diff-data=%.3f | diff-seed/same-data=%.3f | both=%.3f",
             ss, os_, cs)

    (art / f"noiseceiling_layer{L}.json").write_text(json.dumps(out, indent=2))
    _write(cfg, L, out, level_rows)
    print(f"[phase14] locus-level causal frac={out['locus_level']['causal_fraction']:.3f} "
          f"(element-level {out['element_level']['causal_fraction']:.3f}); "
          f"cross-seed agreement = {100 * out['noise_ceiling']['jaccard_frac_of_ceiling']:.0f}% "
          f"of the same-seed noise ceiling")


def _write(cfg, L, out, level_rows):
    nc = out["noise_ceiling"]
    cs = out["cross_seed_locus_level"]
    body = f"""# Phase 14 — Locus-level re-aggregation + the split-half noise ceiling

Layer L={L}. **{out['n_measurements']} satmut measurements over {out['n_loci']} independent loci.**

## (A) Does the headline change when repeat measurements are collapsed?

{md_table(["aggregation unit", "null 95th-pct thr", "causal fraction", "n above thr",
           "n features", "FDR q<0.05"], level_rows)}

Kircher assays several cell lines / timepoints per element over a byte-identical sequence, so the
element-level weighted median counts TERT 4x and SORT1 3x. The locus-level row treats each locus
once. If the two rows agree, the aggregate signal is not an artifact of repeated measurements.

## (B) How much cross-seed disagreement is just estimation noise?

A 2x2 design at a matched element budget ({nc['half_size_loci']} loci per side,
{nc['n_splits']} random splits) separates the two sources of non-reproduction. Holding the SAE
fixed and splitting the loci gives the agreement attainable from estimation noise alone — the
ceiling. Holding the data fixed and changing the seed gives the SAE's own contribution.

| what differs | Jaccard (q<0.05) | Spearman(A_f) |
| --- | --- | --- |
| data only — same seed, disjoint halves (**noise ceiling**) | {nc['same_seed_diff_data_jaccard']['mean']:.3f} [{nc['same_seed_diff_data_jaccard']['ci95'][0]:.3f}, {nc['same_seed_diff_data_jaccard']['ci95'][1]:.3f}] | {nc['same_seed_diff_data_spearman']['mean']:.3f} [{nc['same_seed_diff_data_spearman']['ci95'][0]:.3f}, {nc['same_seed_diff_data_spearman']['ci95'][1]:.3f}] |
| **seed only — different seeds, same half** | {nc['diff_seed_same_data_jaccard']['mean']:.3f} [{nc['diff_seed_same_data_jaccard']['ci95'][0]:.3f}, {nc['diff_seed_same_data_jaccard']['ci95'][1]:.3f}] | {nc['diff_seed_same_data_spearman']['mean']:.3f} [{nc['diff_seed_same_data_spearman']['ci95'][0]:.3f}, {nc['diff_seed_same_data_spearman']['ci95'][1]:.3f}] |
| both — different seeds, disjoint halves | {nc['diff_seed_diff_data_jaccard']['mean']:.3f} [{nc['diff_seed_diff_data_jaccard']['ci95'][0]:.3f}, {nc['diff_seed_diff_data_jaccard']['ci95'][1]:.3f}] | {nc['diff_seed_diff_data_spearman']['mean']:.3f} [{nc['diff_seed_diff_data_spearman']['ci95'][0]:.3f}, {nc['diff_seed_diff_data_spearman']['ci95'][1]:.3f}] |

Seed-only agreement as a fraction of the ceiling: **{100 * nc['seedonly_frac_of_ceiling']:.0f}%**.

Full-data cross-seed agreement at locus level (A_f compared through Hungarian decoder matching;
feature indices are arbitrary across retrains, so only the set-Jaccard is index-free):
{", ".join(f"{c['pair']}: raw J={c['jaccard_q05_raw']:.3f}, matched J={c['jaccard_q05_matched']:.3f}, matched rho={c['spearman_A_matched']:.3f} (n={c['n_matched']})" for c in cs)}.

**Reading.** The decisive comparison is the *seed-only* row against the ceiling. If changing the
seed costs much more agreement than resampling the data does, retraining genuinely destroys the
decomposition. If the ceiling is itself near zero — i.e. the same SAE scored on different loci
already disagrees this much — then per-feature causal calls are not reproducible *at this ground
truth budget* regardless of the SAE, and the non-identifiability claim must be stated as a joint
limit of the method and the available ground truth rather than a property of SAEs alone.

**CHECKPOINT 14.** Confirm which of the two readings the numbers support.
"""
    write_report(cfg, "phase14", body)


if __name__ == "__main__":
    main()
