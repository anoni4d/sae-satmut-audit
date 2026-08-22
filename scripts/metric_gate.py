#!/usr/bin/env python
"""Validity gate for a candidate A_f metric (PLAN.md §3.2, step 4).

A metric is only adoptable if, using that metric end-to-end:
  (G1) threshold well-defined & features exceed the random-feature null  (real frac > 0.05);
  (G2) shuffle control collapses to ~null  (shuffle frac <= ~null level, and < real frac);
  (G3) features beat the model's own log-likelihood  (A_ll < threshold AND majority of
       causal features have A_f > A_ll).
Selection of the metric is by principle (monosemantic features -> precision); this script only
*validates*. Runs each metric through the identical pipeline and prints PASS/FAIL per gate.

Run:  python scripts/metric_gate.py smoke=true
"""
import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut
from dna_interp.utils import md_table, path, write_report


def chosen_layer(cfg):
    from pathlib import Path
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def gate_one(cfg, model, sae, satmut, L, metric):
    feats, _, cache = controls.run_causal_test(cfg, model, sae, satmut, L, metric=metric)
    null_A = controls.random_feature_null(cfg, model, sae, satmut, L, metric=metric)
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))
    frac = causal.causal_fraction(feats.A_f.to_numpy(), thr)
    shuf = controls.shuffle_control(cfg, cache, seed=cfg.seed, metric=metric)
    shuf_frac = causal.causal_fraction(shuf, thr)
    ll = controls.loglik_baseline(cfg, model, satmut, metric=metric)
    A_ll = ll["A_ll"]
    causal_feats = feats.A_f[feats.A_f > thr]
    beat_ll = float((causal_feats > A_ll).mean()) if len(causal_feats) else float("nan")

    null_level = 1.0 - float(cfg.causal.percentile) / 100.0     # by construction ~0.05
    g1 = (np.isfinite(thr)) and (frac > null_level)
    g2 = (shuf_frac <= null_level + 0.03) and (shuf_frac < frac)
    g3 = (A_ll < thr) and (beat_ll > 0.5)
    return {"metric": metric, "threshold": thr, "real_frac": frac, "shuffle_frac": shuf_frac,
            "A_ll": A_ll, "pct_beat_ll": beat_ll, "null_median": float(np.nanmedian(null_A)),
            "G1_null": g1, "G2_shuffle": g2, "G3_llbeat": g3,
            "PASS": bool(g1 and g2 and g3)}


def main():
    cfg = boot()
    model = load_model(cfg)
    L = chosen_layer(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    satmut = load_satmut(cfg)

    candidates = ["spearman_all", "precision_weighted"]
    rows = [gate_one(cfg, model, sae, satmut, L, m) for m in candidates]
    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "artifacts", "metric_gate.csv"), index=False)

    tbl = md_table(
        ["metric", "thr", "real frac", "shuffle frac", "A_ll", "% beat LL",
         "G1 null", "G2 shuffle", "G3 LL", "ADOPTABLE"],
        [[r["metric"], f"{r['threshold']:.3f}", f"{r['real_frac']:.3f}",
          f"{r['shuffle_frac']:.3f}", f"{r['A_ll']:.3f}", f"{r['pct_beat_ll']:.2f}",
          "PASS" if r["G1_null"] else "FAIL", "PASS" if r["G2_shuffle"] else "FAIL",
          "PASS" if r["G3_llbeat"] else "FAIL", "YES" if r["PASS"] else "NO"] for r in rows])

    body = f"""# Metric validity gate (PLAN.md §3.2, step 4)

Toy ground-truth control, layer L={L}, {len(satmut)} satmut elements, null R={cfg.causal.null_R}.
A metric is **adoptable** only if all three gates pass when used end-to-end.

{tbl}

- **G1 (null/threshold):** real causal fraction exceeds the ~{1 - cfg.causal.percentile/100:.2f}
  null level with a finite 95th-pct threshold.
- **G2 (shuffle collapse):** permuting element↔E drops the causal fraction to ~null and below
  the real fraction.
- **G3 (beats LL baseline):** the model's own log-likelihood readout scores below threshold and
  the majority of causal features beat it.

**Conclusion.** Selection is by principle (monosemantic SAE features ⇒ precision metric); this
gate confirms `precision_weighted` is *valid* (controls behave) before adopting it for the real
run. If it passes, set `causal.metric=precision_weighted` and pre-register before touching real data.
"""
    write_report(cfg, "metric_gate", body)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
