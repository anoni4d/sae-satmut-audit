#!/usr/bin/env python
"""Every number the paper quotes, recomputed with loci as the unit of aggregation.

The submitted paper aggregates A_f over the 29 assayed measurements. Kircher reports several
conditions per element over byte-identical sequence, so those are 21 independent loci (see
`data.locus_groups`). This recomputes the paper's numbers on that unit, from the cached
per-(feature, measurement) alignments wherever possible so no further ISM is needed.

Emits rebut/locus_numbers.json and prints a submitted-vs-corrected table.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/locus_numbers.py
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dna_interp import causal, controls, sae as S  # noqa: E402
from dna_interp.data import SatmutElement, locus_groups  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def subspace_basis(cols, energy=0.90, max_dim=50):
    if cols.shape[1] == 0:
        return np.zeros((cols.shape[0], 0))
    U, s, _ = np.linalg.svd(cols, full_matrices=False)
    e = s ** 2
    r = int(np.searchsorted(np.cumsum(e) / e.sum(), energy) + 1)
    return U[:, :max(1, min(r, max_dim, U.shape[1]))]


def overlap(Ua, Ub):
    r = min(Ua.shape[1], Ub.shape[1])
    if r == 0:
        return float("nan")
    return float(((Ua[:, :r].T @ Ub[:, :r]) ** 2).sum() / r)


def main():
    cfg = load_config()
    art = Path(cfg.paths["artifacts"])
    L = int((art / "chosen_layer.txt").read_text().strip())
    rng = np.random.default_rng(0)

    d = np.load(art / "satmut.npz", allow_pickle=True)
    elems = [SatmutElement(str(i), str(s), np.asarray(m), np.asarray(sg), np.asarray(mx))
             for i, s, m, sg, mx in zip(d["ids"], d["seqs"], d["E_mean"], d["E_signed"], d["E_max"])]
    g = locus_groups(elems)
    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    A = lambda tag, grp: controls.aggregate_features(                      # noqa: E731
        pd.read_parquet(art / f"align_{tag}_layer{L}.parquet"),
        int(cfg.causal.min_active_elements), groups=grp)

    out = {}
    for level, grp in (("element", None), ("locus", g)):
        nullA = A("null", grp).A_f.to_numpy()
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
        per_seed, W = {}, {}
        for s in seeds:
            f = A(f"topk_seed{s}", grp).set_index("feature_id")["A_f"]
            q = bh(np.array([causal.empirical_pvalue(a, nullA) for a in f.to_numpy()]))
            ids = f.index.to_numpy().astype(int)
            per_seed[s] = {"A": f, "set": set(ids[q < 0.05].tolist()),
                           "above": set(ids[f.to_numpy() > thr].tolist())}
            W[s] = S.load_sae(cfg, L, seed=s).W_dec.detach().cpu().numpy()

        A0 = per_seed[seeds[0]]["A"].to_numpy()
        raw_j, mat_j, spear, spear_hi = [], [], [], []
        for s in seeds[1:]:
            a, b = per_seed[seeds[0]], per_seed[s]
            raw_j.append(jac(a["set"], b["set"]))
            ra, rb, cos = controls.match_features(W[seeds[0]], W[s])
            keep = cos > float(cfg.causal.match_cos_threshold)
            pr = [(int(x), int(y)) for x, y in zip(ra[keep], rb[keep])
                  if x in a["A"].index and y in b["A"].index]
            both = sum(1 for x, y in pr if x in a["set"] and y in b["set"])
            either = sum(1 for x, y in pr if x in a["set"] or y in b["set"])
            mat_j.append(both / either if either else np.nan)
            xa = np.array([a["A"].loc[x] for x, _ in pr]); xb = np.array([b["A"].loc[y] for _, y in pr])
            spear.append(causal.spearman_rho(xa, xb))
            hi = np.abs(xa) > thr
            spear_hi.append(causal.spearman_rho(xa[hi], xb[hi]) if hi.sum() >= 3 else np.nan)

        repro = np.mean([all(f in per_seed[s]["above"] for s in seeds[1:])
                         for f in per_seed[seeds[0]]["above"]]) if per_seed[seeds[0]]["above"] else np.nan

        # causal subspace overlap vs random and vs a same-size non-causal subset
        bases, nc_bases = {}, {}
        for s in seeds:
            f = per_seed[s]["A"]
            cid = np.array(sorted(per_seed[s]["set"]), dtype=int)
            nc_pool = f.index.to_numpy().astype(int)
            nc_pool = np.setdiff1d(nc_pool, cid)
            nc = rng.choice(nc_pool, min(len(cid), len(nc_pool)), replace=False)
            bases[s] = subspace_basis(W[s][:, cid])
            nc_bases[s] = subspace_basis(W[s][:, nc])
        ov = [overlap(bases[seeds[0]], bases[s]) for s in seeds[1:]]
        ov_nc = [overlap(nc_bases[seeds[0]], nc_bases[s]) for s in seeds[1:]]
        r = bases[seeds[0]].shape[1]
        ov_rand = float(np.mean([overlap(np.linalg.qr(rng.standard_normal((W[seeds[0]].shape[0], r)))[0],
                                         np.linalg.qr(rng.standard_normal((W[seeds[0]].shape[0], r)))[0])
                                 for _ in range(30)]))

        out[level] = {
            "threshold": float(thr), "n_features": int(len(A0)),
            "causal_fraction": float((A0 > thr).mean()), "n_above": int((A0 > thr).sum()),
            "n_fdr_q05": len(per_seed[seeds[0]]["set"]),
            "median_A_f": float(np.nanmedian(A0)),
            "jaccard_raw": float(np.nanmean(raw_j)), "jaccard_matched": float(np.nanmean(mat_j)),
            "spearman_Af": float(np.nanmean(spear)), "spearman_Af_high": float(np.nanmean(spear_hi)),
            "frac_reproduced_all_seeds": float(repro),
            "subspace_overlap_causal": float(np.mean(ov)),
            "subspace_overlap_noncausal": float(np.mean(ov_nc)),
            "subspace_overlap_random": ov_rand,
            "subspace_dim": int(r),
        }

    (ROOT / "reports_real" / "locus_numbers.json").write_text(json.dumps(out, indent=2))
    keys = [("causal_fraction", "causal fraction"), ("n_above", "n above thr"),
            ("n_fdr_q05", "FDR q<0.05"), ("threshold", "threshold"),
            ("jaccard_raw", "cross-seed Jaccard (raw)"), ("jaccard_matched", "  (matched)"),
            ("spearman_Af", "cross-seed Spearman(A_f)"), ("spearman_Af_high", "  (high-signal)"),
            ("frac_reproduced_all_seeds", "reproduced in every seed"),
            ("subspace_overlap_causal", "subspace overlap: causal"),
            ("subspace_overlap_noncausal", "  non-causal null"),
            ("subspace_overlap_random", "  random null")]
    print(f"\n{'quantity':<28}{'submitted (element)':>20}{'corrected (locus)':>20}")
    for k, lbl in keys:
        a, b = out["element"][k], out["locus"][k]
        fa = f"{a:.3f}" if isinstance(a, float) else str(a)
        fb = f"{b:.3f}" if isinstance(b, float) else str(b)
        print(f"{lbl:<28}{fa:>20}{fb:>20}")


if __name__ == "__main__":
    main()
