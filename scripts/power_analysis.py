#!/usr/bin/env python
"""Power analysis for the satmut n-elements fork (PLAN.md §3.1 / Checkpoint 1).

Question: with n~20 satmut elements, do we have power to detect that causal (motif) features
align better than the random-feature null? We answer it parametrically on the toy ground-truth
control: ISM is run ONCE on a pool of elements (sensitivity I_f is independent of the effect),
then we (a) add varying noise to the effect vectors E (the real effect-size/SNR is unknown), and
(b) bootstrap-subsample n elements, re-aggregating A_f each time. Detection = a one-sided
Mann-Whitney test that true-causal features' A_f exceeds the null features' A_f (p<0.05).
Power(n) = fraction of bootstraps that detect.

CAVEAT: synthetic E with a chosen SNR — this gives the *machinery* and a sensitivity band, not
the literal Kircher answer. Re-run with real E once the Kircher tables are loaded.

Run:  python scripts/power_analysis.py smoke=true satmut.n_synthetic=60
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import MOTIF_NAMES, load_satmut
from dna_interp.utils import get_logger, md_table, path, write_report

log = get_logger("power")

N_GRID = [5, 8, 12, 16, 20, 28, 40]
SIGMAS = [1.0, 4.0, 8.0, 16.0]     # effect noise as multiples of per-element std(E)
B = 150                            # bootstraps per (n, sigma)


def ground_truth_labels(model, sae):
    R = model.R / (np.linalg.norm(model.R, axis=0, keepdims=True) + 1e-9)
    W = sae.W_dec.detach().cpu().numpy()
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
    cos = np.abs(W.T @ R)
    nm = len(MOTIF_NAMES)
    return (cos[:, :nm].max(1) > 0.3) & (cos[:, :nm].max(1) > cos[:, nm:].max(1))


def _feature_data(cfg, model, enc, satmut, L):
    """{feat: {elem_id: I_vec}} from one shared ISM pass."""
    _, cache = controls.compute_alignments(cfg, model, enc, satmut, L, cache_I=True)
    data = {}
    for eid, (E, Imap) in cache.items():
        for f, I in Imap.items():
            data.setdefault(int(f), {})[eid] = np.asarray(I)
    return data


def build_af_matrix(data, E_by_elem, metric_fn, elem_order):
    """Precompute a_f for every (feature, element) ONCE (bootstrap-invariant):
    returns M [F, n_elem] with NaN where the feature is inactive on that element."""
    eidx = {e: j for j, e in enumerate(elem_order)}
    feats = sorted(data)
    M = np.full((len(feats), len(elem_order)), np.nan)
    for i, f in enumerate(feats):
        for eid, I in data[f].items():
            M[i, eidx[eid]] = metric_fn(I, E_by_elem[eid])
    return M, np.array(feats)


def main():
    cfg = boot()
    model = load_model(cfg)
    if not hasattr(model, "R"):
        raise SystemExit("power_analysis requires the toy model (smoke=true).")
    L = int((path(cfg, "artifacts", "chosen_layer.txt")).read_text().strip())
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    satmut = load_satmut(cfg)
    metric_fn = causal.get_metric(cfg.causal.metric)
    rng = np.random.default_rng(cfg.seed)
    log.info("power analysis: pool=%d elements, metric=%s", len(satmut), cfg.causal.metric)

    is_causal = ground_truth_labels(model, sae)
    E_clean = {e.elem_id: np.asarray(e.E_mean) for e in satmut}
    elem_ids = list(E_clean)
    real_data = _feature_data(cfg, model, sae, satmut, L)
    enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                    sae.b_dec.detach(), cfg.device, seed=cfg.seed)
    null_data = _feature_data(cfg, model, enc, satmut, L)
    n_causal = int(np.sum([is_causal[f] for f in real_data]))
    log.info("ISM done; %d causal features; aggregation = unweighted median (power tool)", n_causal)

    rows = []
    for sigma in SIGMAS:
        E_noisy = {eid: E + sigma * (E.std() + 1e-9) * rng.standard_normal(len(E))
                   for eid, E in E_clean.items()}
        M_real, fids = build_af_matrix(real_data, E_noisy, metric_fn, elem_ids)
        M_null, _ = build_af_matrix(null_data, E_noisy, metric_fn, elem_ids)
        caus_mask = is_causal[fids]
        M_caus = M_real[caus_mask]
        for n in N_GRID:
            if n > len(elem_ids):
                continue
            detect = 0
            for _ in range(B):
                cols = rng.choice(len(elem_ids), size=n, replace=False)
                c = np.nanmedian(M_caus[:, cols], axis=1)
                u = np.nanmedian(M_null[:, cols], axis=1)
                c = c[~np.isnan(c)]; u = u[~np.isnan(u)]
                if c.size >= 3 and u.size >= 3:
                    try:
                        if mannwhitneyu(c, u, alternative="greater").pvalue < 0.05:
                            detect += 1
                    except ValueError:
                        pass
            rows.append({"sigma": sigma, "n": n, "power": detect / B})
            log.info("sigma=%.1f n=%2d power=%.2f", sigma, n, detect / B)

    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "artifacts", "power_analysis.csv"), index=False)
    _plot(cfg, df)
    _report(cfg, df, len(satmut), n_causal)
    print(df.pivot(index="n", columns="sigma", values="power").to_string())


def _plot(cfg, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for sigma, g in df.groupby("sigma"):
        ax.plot(g.n, g.power, "o-", label=f"noise σ={sigma}×std(E)")
    ax.axhline(0.8, color="red", ls="--", lw=1, label="80% power")
    ax.axvline(20, color="gray", ls=":", lw=1, label="n≈20 (Kircher)")
    ax.set(xlabel="number of satmut elements (n)", ylabel="detection power",
           title="Power to detect causal>null vs n (toy control)", ylim=(-0.02, 1.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path(cfg, "reports", "figures", "power_analysis.png"), dpi=130)
    plt.close(fig)


def _min_n(df, sigma, target=0.8):
    g = df[df.sigma == sigma].sort_values("n")
    ok = g[g.power >= target]
    return int(ok.n.iloc[0]) if len(ok) else None


def _report(cfg, df, n_pool, n_causal):
    piv = df.pivot(index="n", columns="sigma", values="power")
    headers = ["n"] + [f"σ={s}" for s in piv.columns]
    table = md_table(headers, [[int(n)] + [f"{piv.loc[n, s]:.2f}" for s in piv.columns]
                               for n in piv.index])
    at20 = df[df.n == 20].set_index("sigma")["power"].to_dict()
    minn = {s: _min_n(df, s) for s in df.sigma.unique()}
    body = f"""# Power analysis — the n≈20 fork (PLAN.md §3.1, Checkpoint 1)

Toy ground-truth control, metric `{cfg.causal.metric}`, pool of {n_pool} elements,
{n_causal} known-causal features, {B} bootstraps per cell. Detection = one-sided Mann-Whitney
(causal A_f > null A_f, p<0.05). Noise σ is added to E as a multiple of per-element std(E) — a
stand-in for the unknown real effect SNR.

Power(n) by noise level:

{table}

- **At n=20:** power = {", ".join(f"{p:.2f} (σ={s})" for s, p in sorted(at20.items()))}.
- **Smallest n for 80% power:** {", ".join(f"σ={s}: {('n='+str(v)) if v else '>'+str(max(N_GRID))}" for s, v in sorted(minn.items()))}.

Figure: `reports/figures/power_analysis.png`.

## Reading (decision support, not the decision)
- The fork is governed by the real effect **SNR**, not by n alone: read the n=20 row and the
  "smallest n for 80% power" line above against the noise level you believe matches Kircher.
- In the low/moderate-noise regime n≈20 is adequate; n≈20 only becomes the bottleneck in a
  high-noise regime, where the **eQTL/extra-satmut secondary (SPEC §11)** is needed or the
  result must carry explicit power limits.
- The real SNR is unknown until Kircher E is loaded; **re-run this with real E at Checkpoint 1**
  before committing. The decision quantity is power at the *real* noise level, read off the curve.
- Caveat: synthetic effect + unweighted-median aggregation; this is the *machinery* and a
  sensitivity band, calibrated to be informative — not the literal Kircher answer.

> **Decision at Checkpoint 1:** read the real-data power off this curve and choose:
> proceed at n≈20, add the eQTL/extra-satmut secondary, or report with stated power limits.
"""
    write_report(cfg, "power_analysis", body)


if __name__ == "__main__":
    main()
