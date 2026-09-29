#!/usr/bin/env python
"""Render the paper's Figure 1 (signal audit) at the NeurIPS column width (5.5in).

The shared viz.plot_signal_audit renders at 9.5in for on-screen review; embedding that in a 5.5in
column shrinks every label by ~43% and the log-axis tick labels collide. This draws the same two
panels at final print size with fonts specified for that size, so \\includegraphics needs no scaling.

Left: A_f for SAE features against the random-direction null, both tails shaded.
Right: reported causal fraction and FDR count against dictionary size (reads dictsweep_layer{L}.json,
produced by scripts/16_dictsweep.py).

Writes <reports>/figures/signal_audit.png.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/fig_signal_audit.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dna_interp import controls  # noqa: E402
from dna_interp.data import SatmutElement, locus_groups  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402


def main():
    cfg = load_config(sys.argv[1:])
    art = Path(cfg.paths["artifacts"])
    L = int((art / "chosen_layer.txt").read_text().strip())
    out_path = ROOT / cfg.paths["reports"] / "figures" / "signal_audit.png"

    d = np.load(art / "satmut.npz", allow_pickle=True)
    elems = [SatmutElement(str(i), str(s), np.asarray(m), np.asarray(sg), np.asarray(mx))
             for i, s, m, sg, mx in zip(d["ids"], d["seqs"], d["E_mean"], d["E_signed"], d["E_max"])]
    g = locus_groups(elems)

    def locus_A(tag):
        df = pd.read_parquet(art / f"align_{tag}_layer{L}.parquet")
        return controls.aggregate_features(df, 1, groups=g).A_f.dropna().to_numpy()

    real, null = locus_A(f"topk_seed{cfg.seed}"), locus_A("null")
    hi, lo = np.percentile(null, 95), np.percentile(null, 5)
    up, dn = (real > hi).mean(), (real < lo).mean()
    sweep = json.loads((art / f"dictsweep_layer{L}.json").read_text())["by_expansion"]
    rows = sorted(sweep.values(), key=lambda v: v["m"])
    ms = [v["m"] for v in rows]
    frac = [v["causal_fraction"] for v in rows]
    fdr = [v["n_fdr_q05"] for v in rows]

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6})
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.15))

    # ---- left: both tails ----
    lo_x, hi_x = np.percentile(np.concatenate([real, null]), [0.3, 99.7])
    bins = np.linspace(lo_x, hi_x, 55)
    ax[0].hist(null, bins=bins, density=True, color="0.84", label="random-direction null")
    ax[0].hist(null, bins=bins, density=True, histtype="step", color="0.45", lw=0.9)
    ax[0].hist(real, bins=bins, density=True, histtype="step", color="tab:blue", lw=1.5,
               label="SAE features")
    for x in (hi, lo):
        ax[0].axvline(x, color="tab:red", ls="--", lw=1.0)
    ax[0].axvspan(hi, bins[-1], color="tab:red", alpha=0.10, lw=0)
    ax[0].axvspan(bins[0], lo, color="tab:red", alpha=0.10, lw=0)
    ax[0].annotate(f"{dn:.3f}", xy=(0.02, 0.60), xycoords="axes fraction", fontsize=6.5,
                   color="tab:red", ha="left")
    ax[0].annotate(f"{up:.3f}", xy=(0.98, 0.60), xycoords="axes fraction", fontsize=6.5,
                   color="tab:red", ha="right")
    ax[0].set(xlabel="$A_f$ (alignment with measured effect)", ylabel="density",
              title="Tail excess is symmetric")
    ax[0].legend(frameon=False, loc="upper left")

    # ---- right: scaling with dictionary size ----
    ax[1].plot(ms, frac, "o-", color="tab:blue", ms=3.5, lw=1.2, label="causal fraction")
    ax[1].axhline(0.05, color="tab:red", ls="--", lw=1.0, label="5% null rate")
    ax[1].set_xscale("log", base=2)
    ax[1].minorticks_off()                       # log minor ticks collide with the m labels
    ax[1].set_xticks(ms)
    ax[1].set_xticklabels([str(m) for m in ms])
    ax[1].set(xlabel="dictionary size $m$", ylabel="causal fraction",
              title="…and grows with dictionary size")
    ax[1].set_ylim(0, max(frac) * 1.25)
    axr = ax[1].twinx()
    axr.plot(ms, fdr, "s:", color="tab:orange", ms=3.5, lw=1.2, label="FDR $q<0.05$ count")
    axr.set_ylabel("FDR-significant features")
    axr.set_ylim(0, max(fdr) * 1.25)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    fig.tight_layout(pad=0.4, w_pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400)
    print(f"wrote {out_path}  ({up:.3f} above / {dn:.3f} below; "
          f"m={ms} frac={[round(f, 3) for f in frac]})")


if __name__ == "__main__":
    main()
