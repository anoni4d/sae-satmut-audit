#!/usr/bin/env python
"""Dictionary-size sweep summary, for Reviewers 2 and 4.

`scripts/16_dictsweep.py` ran the sweep and aggregated A_f over independent loci. The paper as
submitted aggregates over assayed measurements, so this recomputes the summary at BOTH levels from
the cached per-(feature, measurement) alignments — no further ISM — and prints them side by side.
Report the element-level column while the paper still uses that convention; switch to the locus
column if and when the locus correction is applied, so the two are never mixed.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/dictsweep_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dna_interp import causal, controls  # noqa: E402
from dna_interp.data import SatmutElement, locus_groups  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402

WIDTHS = [1024, 2048, 4096, 8192]
N_SEEDS = 3


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def main():
    cfg = load_config()
    art = Path(cfg.paths["artifacts"])
    L = int((art / "chosen_layer.txt").read_text().strip())
    d = np.load(art / "satmut.npz", allow_pickle=True)
    elems = [SatmutElement(str(i), str(s), np.asarray(m), np.asarray(sg), np.asarray(mx))
             for i, s, m, sg, mx in zip(d["ids"], d["seqs"], d["E_mean"], d["E_signed"], d["E_max"])]
    gmap = locus_groups(elems)
    sweep = json.loads((art / f"dictsweep_layer{L}.json").read_text())["by_expansion"]
    q_of = {int(v["m"]): v for v in sweep.values()}

    rows = {}
    for level, groups in (("element", None), ("locus", gmap)):
        rows[level] = []
        for m in WIDTHS:
            nullA = controls.aggregate_features(
                pd.read_parquet(art / f"align_null_m{m}_layer{L}.parquet"), 1,
                groups=groups).A_f.to_numpy()
            thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
            sets, A0 = {}, None
            for s in range(N_SEEDS):
                A = controls.aggregate_features(
                    pd.read_parquet(art / f"align_topk_m{m}_seed{s}_layer{L}.parquet"),
                    int(cfg.causal.min_active_elements), groups=groups).set_index("feature_id")["A_f"]
                q = bh_fdr(np.array([causal.empirical_pvalue(a, nullA) for a in A.to_numpy()]))
                sets[s] = set(A.index.to_numpy()[q < 0.05].astype(int).tolist())
                if s == 0:
                    A0 = A
            raw_j = float(np.mean([jaccard(sets[0], sets[s]) for s in range(1, N_SEEDS)]))
            rows[level].append({
                "m": m, "FVU": q_of[m]["FVU"], "dead": q_of[m]["dead_frac"],
                "frac": float((A0.to_numpy() > thr).mean()), "fdr": len(sets[0]),
                "seed_jaccard_raw": raw_j,
            })

    for level in ("element", "locus"):
        print(f"\n=== {level}-level aggregation ===")
        print(f"{'m':>6} {'FVU':>6} {'dead':>6} {'causal frac':>12} {'FDR q<.05':>10} {'seed J':>8}")
        for r in rows[level]:
            print(f"{r['m']:>6} {r['FVU']:>6.3f} {r['dead']:>6.3f} {r['frac']:>12.3f} "
                  f"{r['fdr']:>10d} {r['seed_jaccard_raw']:>8.3f}")
    (ROOT / "reports_real" / "dictsweep_table.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {ROOT / 'rebut' / 'dictsweep_table.json'}")


if __name__ == "__main__":
    main()
