#!/usr/bin/env python
"""P5+P6 — ISM causal test + the four controls. Writes phase5.md (headline) and phase6.md."""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S, viz
from dna_interp.activations import load_model
from dna_interp.data import load_satmut
from dna_interp.utils import md_table, path, write_report


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    model = load_model(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    satmut = load_satmut(cfg)
    print(f"[phase4] causal test: layer {L}, {len(satmut)} satmut elements")

    feats, align, cache = controls.run_causal_test(cfg, model, sae, satmut, L)
    null_A = controls.random_feature_null(cfg, model, sae, satmut, L)
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))

    feats["label_causal"] = causal.classify_array(feats.A_f.to_numpy(), thr)
    feats["p_value"] = [causal.empirical_pvalue(a, null_A) for a in feats.A_f]
    frac = causal.causal_fraction(feats.A_f.to_numpy(), thr)
    feats.to_csv(path(cfg, "artifacts", f"causal_features_layer{L}.csv"), index=False)
    np.save(path(cfg, "artifacts", f"null_A_layer{L}.npy"), null_A)

    # merge biological-class labels from annotation if present
    ann_path = Path(cfg.paths["artifacts"]) / f"features_layer{L}.csv"
    if ann_path.exists():
        ann = pd.read_csv(ann_path)[["feature_id", "label", "top_motif", "element_class"]]
        feats = feats.merge(ann, on="feature_id", how="left")

    # ---- controls ----
    ll = controls.loglik_baseline(cfg, model, satmut)
    A_ll = ll["A_ll"]
    shuf = controls.shuffle_control(cfg, cache, seed=cfg.seed)
    shuf_frac = causal.causal_fraction(shuf, thr)

    # robustness across any extra trained seeds
    rob = robustness(cfg, model, satmut, L, thr)

    # ---- figures ----
    viz.plot_Af_distribution(cfg, feats.A_f.to_numpy(), null_A, thr, A_ll=A_ll, A_shuffle=shuf)
    if "label" in feats:
        viz.plot_class_breakdown(cfg, feats, thr)
    _example_figures(cfg, feats, cache, thr)

    # ---- reports ----
    _write_phase5(cfg, L, feats, null_A, thr, frac, A_ll)
    _write_phase6(cfg, L, feats, null_A, thr, frac, ll, shuf, shuf_frac, rob)
    print(f"[phase4] causal fraction={frac:.3f} (thr={thr:.3f}); "
          f"LL baseline A_ll={A_ll:.3f}; shuffle frac={shuf_frac:.3f}")


def robustness(cfg, model, satmut, L, thr):
    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(Path(cfg.paths["artifacts"]) / f"sae_layer{L}_seed*.safetensors")))
    if len(seeds) < 2:
        return {"n_seeds": len(seeds), "median_cv": float("nan"), "note": "single seed; rerun pipeline with more seeds for CV"}
    A_by_seed, W_decs = {}, {}
    for s in seeds:
        sae = S.load_sae(cfg, L, seed=s)
        feats, _, _ = controls.run_causal_test(cfg, model, sae, satmut, L, with_ci=False)
        A_by_seed[s] = feats
        W_decs[s] = sae.W_dec.detach().cpu().numpy()
    out = controls.robustness_cv(A_by_seed, W_decs,
                                 cos_threshold=float(cfg.causal.match_cos_threshold))
    out["n_seeds"] = len(seeds)
    return out


def _example_figures(cfg, feats, cache, thr, n=2):
    top = feats.dropna(subset=["A_f"]).sort_values("A_f", ascending=False).head(n)
    for _, row in top.iterrows():
        fid = int(row.feature_id)
        for eid, (E, Imap) in cache.items():
            if fid in Imap:
                viz.plot_ism_example(cfg, Imap[fid], E, fid, eid)
                break


def _write_phase5(cfg, L, feats, null_A, thr, frac, A_ll):
    valid = feats.dropna(subset=["A_f"])
    by_class = ""
    if "label" in feats:
        g = valid.groupby("label")["A_f"].agg(["count", "median",
                                               lambda x: (x > thr).mean()])
        g.columns = ["n", "median_A_f", "causal_frac"]
        by_class = "\n## By biological class\n\n" + md_table(
            ["class", "n", "median A_f", "causal fraction"],
            [[i, int(r.n), f"{r.median_A_f:.3f}", f"{r.causal_frac:.2f}"]
             for i, r in g.iterrows()])
    novel = valid[(valid.A_f > thr) & (valid.get("label", "") == "unannotated")] \
        if "label" in valid else valid.iloc[:0]
    robust_line = ""
    if "A_f_lo" in valid:
        n_robust = int((valid.A_f_lo > thr).sum())
        robust_line = (f"- CI-robust causal features (bootstrap A_f_lo > threshold): "
                       f"**{n_robust}** of {int((valid.A_f > thr).sum())} above-threshold\n")
    body = f"""# Phase 5 — Causal test (headline)

Layer L={L}. Threshold = 95th pct of random-feature null = **{thr:.3f}**.

- features evaluated (active on >= {cfg.causal.min_active_elements} element): **{len(valid)}**
- **causal-aligned fraction = {frac:.3f}** ({int((valid.A_f > thr).sum())}/{len(valid)})
- A_f: min={valid.A_f.min():.3f} median={valid.A_f.median():.3f} max={valid.A_f.max():.3f}
- model-intrinsic LL baseline A_ll = {A_ll:.3f}
- candidate novel elements (causal & unannotated): **{len(novel)}**
{robust_line}{by_class}

Figures: `Af_distribution.png`, `Af_by_class.png`, `ism_example_*.png`
Artifacts: `causal_features_layer{L}.csv`, `null_A_layer{L}.npy`

**CHECKPOINT 5** — the headline result. (Interpretation gated on the controls, Phase 6.)
"""
    write_report(cfg, "phase5", body)


def _rob_verdict(rob):
    if rob.get("n_seeds", 0) < 2:
        return rob.get("note", "single seed")
    cv = rob.get("median_cv", float("nan"))
    if np.isnan(cv) or rob.get("n_matched", 0) == 0:
        return "no cross-seed matches above cosine gate"
    return "OK" if cv < 0.5 else "unstable"


def _write_phase6(cfg, L, feats, null_A, thr, frac, ll, shuf, shuf_frac, rob):
    valid = feats.dropna(subset=["A_f"])
    causal_A = valid[valid.A_f > thr].A_f
    beats_ll = (causal_A > ll["A_ll"]).mean() if len(causal_A) else float("nan")
    collapse = shuf_frac <= (1 - cfg.causal.percentile / 100.0) * 2 + 0.02
    body = f"""# Phase 6 — Controls & robustness

| control | result | expectation | verdict |
| --- | --- | --- | --- |
| (1) random-feature null | thr(95pct)={thr:.3f}, null median={np.nanmedian(null_A):.3f} | features exceed null | {"OK" if frac > 0 else "—"} |
| (2) LL baseline | A_ll={ll['A_ll']:.3f}; {beats_ll:.0%} of causal features beat it | features add signal over reading the model | {"OK" if beats_ll > 0.5 else "WEAK"} |
| (3) shuffle (element<->E) | causal frac={shuf_frac:.3f} (vs real {frac:.3f}) | collapses to ~null | {"OK" if collapse else "CHECK"} |
| (4) robustness (seeds) | n_seeds={rob.get('n_seeds')}, matched={rob.get('n_matched','-')} (mean cos={rob.get('mean_matched_cos', float('nan')):.2f}), median CV(A_f)={rob.get('median_cv', float('nan')):.2f} | stable across seeds | {_rob_verdict(rob)} |

Real causal fraction = **{frac:.3f}**; shuffle = **{shuf_frac:.3f}**; null 95th pct = {thr:.3f}.

**Interpretation.** A high causal fraction is trustworthy only if (2)–(4) hold: features must
beat the model's own log-likelihood, the signal must vanish under shuffling, and A_f must be
seed-stable. A low fraction with airtight controls is itself a valid result (SPEC §1, §10).

**CHECKPOINT 6** — controls airtight.
"""
    write_report(cfg, "phase6", body)


if __name__ == "__main__":
    main()
