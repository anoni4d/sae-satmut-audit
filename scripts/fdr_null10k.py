#!/usr/bin/env python
"""Robustness of the FDR to the random-feature null: recompute the null with R=10^4 directions
(the per-feature A_f are unchanged; only the empirical p-values / threshold / BH counts change).
Pre-empts the "p floored at 1/1001, shared-draw correlation" critique."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def main():
    cfg = boot()
    L = int(Path(cfg.paths["artifacts"], "chosen_layer.txt").read_text().strip())
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    sae = S.load_sae(cfg, L, seed=0, family="topk")
    satmut = load_satmut(cfg)
    feats = pd.read_csv(art / f"causal_features_layer{L}_fdr.csv")   # has A_f
    A = feats.A_f.to_numpy()

    out = {"layer": L}
    for R in (1000, 10000):
        cfg.causal.null_R = R
        nullA = controls.random_feature_null(cfg, model, sae, satmut, L)
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
        p = np.array([causal.empirical_pvalue(a, nullA) for a in A])
        q = bh_fdr(p)
        out[f"R{R}"] = {"n_null": int(len(nullA)), "thr": float(thr),
                        "frac_above": float((A > thr).mean()),
                        "n_above": int((A > thr).sum()),
                        "n_fdr_q05": int((q < 0.05).sum()),
                        "n_fdr_q10": int((q < 0.10).sum()),
                        "min_p": float(np.nanmin(p))}
        print(f"[null R={R}] thr={thr:.4f} frac={out[f'R{R}']['frac_above']:.3f} "
              f"FDR q<0.05={out[f'R{R}']['n_fdr_q05']} q<0.10={out[f'R{R}']['n_fdr_q10']} min_p={out[f'R{R}']['min_p']:.2e}")
        if R == 10000:
            np.save(art / f"null_A_layer{L}_R10000.npy", nullA)
    (art / f"fdr_null10k_layer{L}.json").write_text(json.dumps(out, indent=2))
    print("[fdr10k] DONE", json.dumps(out[f"R10000"]))


if __name__ == "__main__":
    main()
