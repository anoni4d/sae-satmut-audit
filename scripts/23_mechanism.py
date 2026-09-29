#!/usr/bin/env python
"""The mechanism behind the audit: sparser units have wider A_f distributions. Shown two ways.

(A) EMPIRICAL. For real features, tail membership against the number of loci the feature is active
    on. If the one-sided threshold selects the sparsest units, the effect is visible directly, and
    it should be symmetric at every coverage level.

(B) ZERO-SIGNAL SIMULATION. Draw per-locus scores from a centred (signal-free by construction) pool
    of a_f values. Build "features" whose locus counts follow the real feature distribution, and
    "null directions" active on all loci, exactly as in the real pipeline. Nothing in this
    simulation contains any alignment. Two pools are used: the dense null's own scores (coverage
    variation alone) and the real features' scores, centred (coverage plus the greater per-locus
    noise that sparse units also carry).

Reads only cached artifacts (satmut.npz and the per-(feature, measurement) alignment parquets), so
no model or GPU is needed. Emits `mechanism_layer{L}.json` and `mechanism_sim_layer{L}.npz` in the
artifacts directory.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/23_mechanism.py
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

rng = np.random.default_rng(0)


def main():
    cfg = load_config(sys.argv[1:])
    art = Path(cfg.paths["artifacts"])
    L = int((art / "chosen_layer.txt").read_text().strip())

    d = np.load(art / "satmut.npz", allow_pickle=True)
    el = [SatmutElement(str(i), str(s), np.asarray(m), np.asarray(sg), np.asarray(mx))
          for i, s, m, sg, mx in zip(d["ids"], d["seqs"], d["E_mean"], d["E_signed"], d["E_max"])]
    g = locus_groups(el)
    real = controls.collapse_to_loci(
        pd.read_parquet(art / f"align_topk_seed{cfg.seed}_layer{L}.parquet"), g)
    null = controls.collapse_to_loci(pd.read_parquet(art / f"align_null_layer{L}.parquet"), g)

    ma = int(cfg.causal.min_active_elements)
    Ar = controls.aggregate_features(real, ma).set_index("feature_id")
    An = controls.aggregate_features(null, 1).A_f.to_numpy()
    hi, lo = np.percentile(An, 95), np.percentile(An, 5)
    n_loci = real.groupby("feature_id").elem_id.nunique()
    Ar = Ar.join(n_loci.rename("n_loci"))

    # ---------------- (A) empirical ----------------
    print("=== (A) tail membership vs. number of active loci (real features) ===")
    print(f"{'n_loci':>10} {'features':>9} {'|A_f|>thr':>10} {'above':>7} {'below':>7} {'sd(A_f)':>9}")
    bins = [(1, 8), (9, 12), (13, 16), (17, 19), (20, 21)]
    rows = []
    for lo_b, hi_b in bins:
        m = Ar[(Ar.n_loci >= lo_b) & (Ar.n_loci <= hi_b)]
        if len(m) < 20: continue
        up = float((m.A_f > hi).mean()); dn = float((m.A_f < lo).mean())
        rows.append((f"{lo_b}-{hi_b}", len(m), up + dn, up, dn, float(m.A_f.std())))
        print(f"{rows[-1][0]:>10} {len(m):>9} {up+dn:>10.3f} {up:>7.3f} {dn:>7.3f} {m.A_f.std():>9.4f}")
    r = causal.spearman_rho(Ar.n_loci.to_numpy(), np.abs(Ar.A_f.to_numpy()))
    print(f"  Spearman(n_loci, |A_f|) = {r:+.3f}   <- negative = sparser units score more extreme")

    # ---------------- (B) zero-signal simulation ----------------
    counts = n_loci.to_numpy()                       # real feature locus-coverage distribution
    n_all = int(real.elem_id.nunique())              # a dense null direction is active on every locus

    def sim(pool, n_units, n_draw):
        out = np.empty(n_units)
        for i in range(n_units):
            k = max(1, int(n_draw()))
            idx = rng.integers(0, len(pool), k)
            out[i] = causal.weighted_median(pool[idx, 0], pool[idx, 1])
        return out

    # Two pools of per-locus scores, each centred so no directional signal remains.
    # "null pool" isolates coverage alone; "feature pool" additionally carries the real
    # per-locus noise of sparse units, which is itself a consequence of sparsity.
    print("\n=== (B) zero-signal simulation (no alignment anywhere by construction) ===")
    pool_null = null[["a_f", "mass"]].to_numpy().copy()
    pool_null[:, 0] -= np.median(pool_null[:, 0])
    pool_feat = real[["a_f", "mass"]].to_numpy().copy()
    pool_feat[:, 0] -= np.median(pool_feat[:, 0])

    regimes = {}
    for key, lbl, pool in (("coverage_only", "coverage only (null-pool scores)", pool_null),
                           ("coverage_plus_noise",
                            "coverage + per-locus noise (feature-pool scores, centred)", pool_feat)):
        sf = sim(pool, len(Ar), lambda: rng.choice(counts))
        sn = sim(pool_null, 1000, lambda: n_all)
        h, l = np.percentile(sn, 95), np.percentile(sn, 5)
        regimes[key] = {"frac_above": float((sf > h).mean()), "frac_below": float((sf < l).mean()),
                        "sd_features": float(sf.std()), "sd_null": float(sn.std())}
        print(f"  [{lbl}]")
        print(f"     sd features {sf.std():.4f} vs null {sn.std():.4f} | "
              f"above {regimes[key]['frac_above']:.3f}  below {regimes[key]['frac_below']:.3f}")
        if pool is pool_feat:
            sim_feat, sim_null = sf, sn
    print(f"  -> observed in real data: {float((Ar.A_f > hi).mean()):.3f} above / "
          f"{float((Ar.A_f < lo).mean()):.3f} below")

    cp = regimes["coverage_plus_noise"]
    out = {"layer": L, "empirical_bins": rows, "spearman_nloci_absA": r,
           "simulation": regimes,
           # flat keys kept for the coverage+noise regime, the one Figure/§4 quote first
           "sim_frac_above": cp["frac_above"], "sim_frac_below": cp["frac_below"],
           "sim_sd_features": cp["sd_features"], "sim_sd_null": cp["sd_null"]}
    (art / f"mechanism_layer{L}.json").write_text(json.dumps(out, indent=2))
    np.savez(art / f"mechanism_sim_layer{L}.npz",
             sim_feat=sim_feat, sim_null=sim_null,
             n_loci=Ar.n_loci.to_numpy(), A_f=Ar.A_f.to_numpy(), hi=hi, lo=lo)
    print(f"wrote {art / f'mechanism_layer{L}.json'}")


if __name__ == "__main__":
    main()
