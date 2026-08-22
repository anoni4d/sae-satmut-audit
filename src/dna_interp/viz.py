"""Figures. Functions take artifact data and write PNGs into reports/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import path


def _fig(cfg, name):
    p = Path(cfg.paths["reports"]) / "figures" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_layer_stats(cfg, stats: list[dict]):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    layers = [s["layer"] for s in stats]
    ax[0].plot(layers, [s["mean_norm"] for s in stats], "o-")
    ax[0].set(xlabel="layer", ylabel="mean activation norm", title="Residual-stream norm")
    ax[1].plot(layers, [s["mean_var"] for s in stats], "o-", color="tab:orange")
    ax[1].set(xlabel="layer", ylabel="mean variance", title="Activation variance")
    fig.tight_layout()
    p = _fig(cfg, "layer_stats.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def plot_pca_scree(cfg, acts: np.ndarray, layer: int, k: int = 50):
    x = acts - acts.mean(0)
    cov = np.cov(x.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1][:k]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(ev) + 1), ev / ev.sum(), "o-")
    ax.set(xlabel="component", ylabel="explained variance ratio",
           title=f"PCA scree (layer {layer})")
    fig.tight_layout()
    p = _fig(cfg, f"pca_scree_layer{layer}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def plot_density_hist(cfg, density: np.ndarray, layer: int):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    d = density[density > 0]
    ax.hist(np.log10(d + 1e-9), bins=40)
    ax.set(xlabel="log10 feature density", ylabel="count",
           title=f"Feature density (layer {layer})")
    fig.tight_layout()
    p = _fig(cfg, f"density_layer{layer}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


def plot_Af_distribution(cfg, A_real: np.ndarray, A_null: np.ndarray, threshold: float,
                         A_ll: float | None = None, A_shuffle: np.ndarray | None = None,
                         matched_null: np.ndarray | None = None):
    """Left: full A_f distributions. Right: the upper tail, where the claim actually lives.

    Drawn as step outlines rather than alpha-blended fills — three translucent histograms over
    the same range are unreadable, especially in greyscale. Only the null is filled (it is the
    reference); everything else is a distinguishable line style.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    real = A_real[~np.isnan(A_real)]
    # scale x to where the mass actually is; A_f is bounded by [-1,1] but occupies a narrow band,
    # and plotting the full range leaves every distribution stacked in one unreadable spike
    pool = np.concatenate([v[~np.isnan(v)] for v in
                           (real, A_null, A_shuffle if A_shuffle is not None else np.array([]),
                            matched_null if matched_null is not None else np.array([]))
                           if v is not None and len(v)])
    lo, hi = np.percentile(pool, [0.2, 99.8])
    pad = 0.08 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 61)

    def draw(ax, bns):
        ax.hist(A_null, bins=bns, density=True, color="0.82", label="random-direction null")
        ax.hist(A_null, bins=bns, density=True, histtype="step", color="0.45", lw=1.2)
        if matched_null is not None and len(matched_null):
            ax.hist(matched_null[~np.isnan(matched_null)], bins=bns, density=True,
                    histtype="step", color="tab:green", lw=1.6, ls="-.",
                    label="sparsity-matched null")
        if A_shuffle is not None and len(A_shuffle):
            ax.hist(A_shuffle, bins=bns, density=True, histtype="step", color="tab:purple",
                    lw=1.6, ls="--", label="shuffle control")
        ax.hist(real, bins=bns, density=True, histtype="step", color="tab:blue", lw=2.2,
                label="SAE features")
        ax.axvline(threshold, color="tab:red", ls="--", lw=1.4,
                   label=f"95th pct null = {threshold:.3f}")
        if A_ll is not None:
            ax.axvline(A_ll, color="darkgreen", ls=":", lw=1.6,
                       label=f"log-likelihood baseline = {A_ll:.3f}")

    draw(axes[0], bins)
    axes[0].set(xlabel="$A_f$ (alignment with measured MPRA effect)", ylabel="density",
                title="Full distribution")
    axes[0].legend(fontsize=7, frameon=False)

    tail_hi = float(np.nanpercentile(real, 99.8)) if real.size else 1.0
    tail = np.linspace(threshold - 0.05, tail_hi, 40)
    draw(axes[1], tail)
    axes[1].set_xlim(threshold - 0.05, tail_hi)
    n_above = int((real > threshold).sum())
    axes[1].set(xlabel="$A_f$", ylabel="",
                title=f"Upper tail: {n_above} features above threshold")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = _fig(cfg, "Af_distribution.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_signal_audit(cfg, real, null, thr_hi, thr_lo, out: dict):
    """The diagnostic figure: is the tail excess one-sided, and does it scale with dictionary size?

    Left panel shades BOTH tails, because the whole question is whether the upper one is special.
    Right panel plots the reported signal against dictionary width, which separates a biological
    effect (flat) from a dispersion artifact (rising).
    """
    real = np.asarray(real)[~np.isnan(np.asarray(real))]
    null = np.asarray(null)[~np.isnan(np.asarray(null))]
    has_scaling = "scaling" in out
    fig, axes = plt.subplots(1, 2 if has_scaling else 1,
                             figsize=(9.5 if has_scaling else 5.6, 3.8))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    lo, hi = np.percentile(np.concatenate([real, null]), [0.2, 99.8])
    bins = np.linspace(lo, hi, 61)
    ax.hist(null, bins=bins, density=True, color="0.82", label="random-direction null")
    ax.hist(null, bins=bins, density=True, histtype="step", color="0.45", lw=1.2)
    ax.hist(real, bins=bins, density=True, histtype="step", color="tab:blue", lw=2.2,
            label="SAE features")
    ax.axvline(thr_hi, color="tab:red", ls="--", lw=1.3)
    ax.axvline(thr_lo, color="tab:red", ls="--", lw=1.3)
    ax.axvspan(thr_hi, bins[-1], color="tab:red", alpha=0.10)
    ax.axvspan(bins[0], thr_lo, color="tab:red", alpha=0.10)
    sy = out["symmetry"]
    ax.annotate(f"above: {sy['frac_above']:.3f}", xy=(0.97, 0.72), xycoords="axes fraction",
                ha="right", fontsize=8, color="tab:red")
    ax.annotate(f"below: {sy['frac_below']:.3f}", xy=(0.03, 0.72), xycoords="axes fraction",
                ha="left", fontsize=8, color="tab:red")
    ax.set(xlabel="$A_f$ (alignment with measured MPRA effect)", ylabel="density",
           title="Tail excess is symmetric")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    if has_scaling:
        s = sorted(out["scaling"].values(), key=lambda x: x["m"])
        ms = [v["m"] for v in s]
        ax2 = axes[1]
        ax2.plot(ms, [v["causal_fraction"] for v in s], "o-", color="tab:blue",
                 label="reported causal fraction")
        ax2.axhline(0.05, color="tab:red", ls="--", lw=1.2, label="5% null rate")
        ax2.set(xscale="log", xlabel="dictionary size $m$", ylabel="causal fraction",
                title="…and grows with dictionary size")
        ax2.minorticks_off()                      # log minor ticks collide with the m labels
        ax2.set_xticks(ms)
        ax2.set_xticklabels([str(m) for m in ms])
        axr = ax2.twinx()
        axr.plot(ms, [v["n_fdr_q05"] for v in s], "s:", color="tab:orange",
                 label="FDR $q<0.05$ count")
        axr.set_ylabel("FDR-significant features")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    p = _fig(cfg, "signal_audit.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def plot_dictsweep(cfg, res: dict):
    """Cross-seed identifiability vs dictionary width, with the SAE-quality companions."""
    exps = sorted(int(e) for e in res["by_expansion"])
    g = lambda k: [res["by_expansion"][e][k] if e in res["by_expansion"]
                   else res["by_expansion"][str(e)][k] for e in exps]  # noqa: E731
    ms = g("m")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax[0].plot(ms, g("cross_seed_jaccard_matched"), "o-", color="tab:blue",
               label="causal-set Jaccard (decoder-matched)")
    ax[0].plot(ms, g("cross_seed_jaccard_raw"), "s--", color="tab:cyan",
               label="causal-set Jaccard (raw)")
    ax[0].plot(ms, g("cross_seed_spearman_A"), "^-", color="tab:orange",
               label=r"Spearman($A_f$) across seeds")
    ax[0].set(xscale="log", xlabel="dictionary size $m$", ylabel="cross-seed agreement",
              title="Identifiability vs dictionary size")
    ax[0].set_xticks(ms); ax[0].set_xticklabels([str(m) for m in ms])
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].legend(fontsize=7, frameon=False)

    ax[1].plot(ms, g("causal_fraction"), "o-", color="tab:blue", label="causal fraction")
    ax[1].axhline(0.05, color="tab:red", ls="--", lw=1.2, label="5% null rate")
    ax[1].set(xscale="log", xlabel="dictionary size $m$", ylabel="causal fraction",
              title="Aggregate signal and dictionary quality")
    ax[1].set_xticks(ms); ax[1].set_xticklabels([str(m) for m in ms])
    ax2 = ax[1].twinx()
    ax2.plot(ms, g("dead_frac"), "v:", color="tab:grey", label="dead fraction")
    ax2.plot(ms, g("FVU"), "d:", color="tab:green", label="FVU")
    ax2.set_ylabel("dead fraction / FVU")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    p = _fig(cfg, "dictsweep.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_class_breakdown(cfg, feats_df, threshold: float):
    import pandas as pd  # noqa
    df = feats_df.dropna(subset=["A_f"])
    classes = sorted(df["label"].dropna().unique()) if "label" in df else ["all"]
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df[df.label == c]["A_f"].values if "label" in df else df["A_f"].values
            for c in classes]
    ax.boxplot(data, labels=classes, showmeans=True)
    ax.axhline(threshold, color="red", ls="--", label="null 95th pct")
    ax.set(ylabel="A_f", title="Causal alignment by feature class")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    p = _fig(cfg, "Af_by_class.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_ism_example(cfg, I_f: np.ndarray, E: np.ndarray, feat_id: int, elem_id: str,
                     rel_cut: float = 0.25):
    """Overlay of one feature's ISM sensitivity and the measured per-position effect.

    Drawn as a true overlay on twin axes (orange = sensitivity, blue = measured effect) so the
    colours match how the panel is described, and the positions carrying the feature's top
    sensitivity mass are shaded — that shading *is* the quantity the alignment metric scores:
    does the feature's sensitivity mass land where the measured effect is large? A monosemantic
    feature is sensitive at only a few positions, so it should not trace the whole effect profile.
    """
    I_f = np.asarray(I_f, float)
    E = np.asarray(E, float)
    n = min(len(I_f), len(E))
    I_f, E = I_f[:n], E[:n]
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax2 = ax.twinx()

    # Shade where the feature is *substantially* sensitive, relative to its own peak. A quantile
    # rule is wrong here: I_f is near-zero at most positions, so the top decile of its nonzero
    # values is mostly noise and shades the whole element.
    if np.any(I_f > 0):
        cut = rel_cut * float(I_f.max())
        for p_ in np.flatnonzero(I_f >= cut):
            ax.axvspan(p_ - 0.5, p_ + 0.5, color="tab:orange", alpha=0.25, lw=0)

    ax.plot(x, E, color="tab:blue", lw=1.4, label="measured effect $E_e$ (MPRA)")
    ax2.plot(x, I_f, color="tab:orange", lw=1.4, label=f"ISM sensitivity $I_f$ (feature {feat_id})")
    ax.set(xlabel="position in element (bp)", ylabel="measured effect $E_e$")
    ax2.set_ylabel(f"feature sensitivity $I_f$")
    ax.yaxis.label.set_color("tab:blue")
    ax2.yaxis.label.set_color("tab:orange")
    ax.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:orange")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{elem_id}: does the feature's sensitivity mass (shaded) sit on "
                 f"high-effect positions?", fontsize=9)
    fig.tight_layout()
    p = _fig(cfg, f"ism_example_{elem_id}_f{feat_id}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p
