#!/usr/bin/env python
"""Element-subsampling stability: is the headline a small-n (29-element) artifact, and would more
elements help? We capture the per-(feature, element) alignment matrix in ONE ISM pass, then
recompute everything on random element subsets of size n' — no further ISM:
  (1) causal fraction + FDR-significant count vs n'  -> does the headline converge / stay stable?
  (2) supervised probe Spearman vs n' (leave-one-element-out)  -> learning curve; does it plateau?

Stable curves defuse "it's just noise at n=29"; the non-identifiability is an SAE property, so it
cannot be a power artifact — this makes that visible.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/13_subsample.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import causal, controls, probe as P, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import BASE_TO_IDX, get_logger, md_table, write_report

log = get_logger("subsample")
RNG = np.random.default_rng(0)


def chosen_layer(cfg):
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed, family="topk")
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)

    # ---- one ISM pass (cached), collapsed to LOCI: the resampling unit must be the
    # independent locus, not the measurement, or subsets silently re-draw the same sequence ----
    log.info("loading per-(feature,measurement) alignment and collapsing to loci...")
    align = controls.collapse_to_loci(
        controls.cached_alignment(cfg, model, sae, satmut, L, f"topk_seed{cfg.seed}"), gmap)
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         sae.b_dec.detach(), cfg.device, seed=cfg.seed)
    null_A = controls.aggregate_features(
        controls.cached_alignment(cfg, model, null_enc, satmut, L, "null"), 1,
        groups=gmap).A_f.to_numpy()
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))

    elem_ids = sorted(align.elem_id.unique())
    n_elem = len(elem_ids)
    log.info("resampling unit = locus: %d loci from %d measurements", n_elem, len(satmut))
    # pivot to feature x element matrices of a_f and mass (NaN where feature inactive on element)
    feats = sorted(align.feature_id.unique())
    fidx = {f: i for i, f in enumerate(feats)}
    eidx = {e: j for j, e in enumerate(elem_ids)}
    A = np.full((len(feats), n_elem), np.nan)
    M = np.zeros((len(feats), n_elem))
    for r in align.itertuples():
        A[fidx[r.feature_id], eidx[r.elem_id]] = r.a_f
        M[fidx[r.feature_id], eidx[r.elem_id]] = r.mass
    log.info("alignment matrix: %d features x %d elements", len(feats), n_elem)

    def recompute(sel):
        """A_f, fraction, FDR count over an element subset `sel` (indices)."""
        out = np.full(len(feats), np.nan)
        for i in range(len(feats)):
            a = A[i, sel]; m = M[i, sel]
            ok = ~np.isnan(a)
            if ok.sum() >= int(cfg.causal.min_active_elements):
                out[i] = causal.weighted_median(a[ok], m[ok])
        valid = out[~np.isnan(out)]
        if valid.size == 0:
            return np.nan, np.nan, 0
        frac = float((valid > thr).mean())
        p = np.array([causal.empirical_pvalue(v, null_A) for v in valid])
        nfdr = int((bh_fdr(p) < 0.05).sum())
        return frac, float(np.median(valid)), nfdr

    # ---- (1) causal fraction + FDR vs n' ----
    grid = sorted({g for g in (4, 6, 8, 12, 16, n_elem) if g <= n_elem})
    B = 30
    curve = {}
    for nq in grid:
        fr, fd = [], []
        reps = 1 if nq == n_elem else B
        for _ in range(reps):
            sel = RNG.choice(n_elem, nq, replace=False) if nq < n_elem else np.arange(n_elem)
            f, _, nfdr = recompute(sel)
            fr.append(f); fd.append(nfdr)
        curve[nq] = {"frac_mean": float(np.nanmean(fr)), "frac_std": float(np.nanstd(fr)),
                     "fdr_mean": float(np.mean(fd)), "fdr_std": float(np.std(fd))}
        log.info("n'=%2d: causal frac %.3f±%.3f | FDR q<0.05 %.0f±%.0f",
                 nq, curve[nq]["frac_mean"], curve[nq]["frac_std"],
                 curve[nq]["fdr_mean"], curve[nq]["fdr_std"])

    # ---- (2) supervised probe Spearman vs training-set size (train and test on whole loci) ----
    by_locus = {}
    for e in satmut:
        h = model.feature_input(e.seq, L).detach().cpu().numpy().astype(np.float64)
        E = np.asarray(e.E_mean, float)
        n = min(len(h), len(E))
        if n >= 5:
            by_locus.setdefault(gmap[e.elem_id], []).append((h[:n], E[:n]))
    loci = sorted(by_locus)
    nb = len(loci)
    fit = P.make_ridge(10.0)
    probe_curve = {}
    for nq in [g for g in (4, 6, 8, 12, 16, nb - 1) if 0 < g < nb]:
        rhos = []
        for _ in range(B):
            perm = RNG.permutation(nb)
            tr = [loci[j] for j in perm[:nq]]
            te = loci[perm[nq]]
            Xtr = np.vstack([b[0] for l in tr for b in by_locus[l]])
            ytr = np.concatenate([b[1] for l in tr for b in by_locus[l]])
            Xte = np.vstack([b[0] for b in by_locus[te]])
            yte = np.concatenate([b[1] for b in by_locus[te]])
            pred, _ = fit(Xtr, ytr, Xte)
            rhos.append(causal.spearman_rho(pred, yte))
        probe_curve[nq] = {"rho_mean": float(np.nanmean(rhos)), "rho_std": float(np.nanstd(rhos))}
        log.info("probe train n'=%2d loci: rho %.3f±%.3f", nq,
                 probe_curve[nq]["rho_mean"], probe_curve[nq]["rho_std"])

    out = {"layer": L, "thr": float(thr), "n_loci": n_elem, "n_measurements": len(satmut),
           "frac_fdr_vs_n": curve, "probe_vs_n": probe_curve}
    (art / f"subsample_layer{L}.json").write_text(json.dumps(out, indent=2))
    _report(cfg, L, thr, out)
    full = curve[n_elem]
    print(f"[phase13] frac@{n_elem} loci: {full['frac_mean']:.3f} (FDR {full['fdr_mean']:.0f}); "
          f"frac@8 loci: {curve.get(8,{}).get('frac_mean',float('nan')):.3f}")


def _report(cfg, L, thr, out):
    crows = [[nq, f"{v['frac_mean']:.3f} ± {v['frac_std']:.3f}", f"{v['fdr_mean']:.0f} ± {v['fdr_std']:.0f}"]
             for nq, v in out["frac_fdr_vs_n"].items()]
    prows = [[nq, f"{v['rho_mean']:.3f} ± {v['rho_std']:.3f}"] for nq, v in out["probe_vs_n"].items()]
    body = f"""# Phase 13 — Locus-subsampling stability (the small-n question)

Layer L={L}, null threshold {thr:.3f}, **{out['n_measurements']} measurements over {out['n_loci']}
independent loci**. One ISM pass captured the per-(feature,measurement) alignment; everything below is
recomputed on random subsets of LOCI (no further ISM). 30 random subsets per size. The resampling unit
is the locus, not the measurement — subsampling measurements would re-draw the same sequence.

## Causal fraction + FDR-significant count vs number of loci n'
{md_table(["n'", "causal fraction", "FDR q<0.05 count"], crows)}

## Supervised probe Spearman vs training-set size (train and test on whole loci)
{md_table(["train n'", "probe rho"], prows)}

**Reading.** If the causal fraction is roughly flat / converging across n' and FDR scales smoothly,
the headline is not a small-n artifact. The probe learning curve shows whether more elements would
raise the (modest) decodability — if it is still rising at n=21, more ground truth is the lever; if
plateaued, that magnitude is a real ceiling. The seed/family non-identifiability is an SAE property and so
cannot be a power artifact by construction.

**CHECKPOINT 13.** Confirm the n-stability framing for the paper's limitations section.
"""
    write_report(cfg, "phase13", body)


if __name__ == "__main__":
    main()
