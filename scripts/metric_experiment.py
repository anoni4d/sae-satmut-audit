#!/usr/bin/env python
"""Metric bake-off for the A_f alignment definition (addresses PLAN.md §3.2).

The current per-feature metric a_f(e) = Spearman(I_f, E_e) over ALL positions penalizes a
motif-specific feature when an element carries several motifs (E is high at OTHER motifs'
positions where I_f ~ 0). This experiment compares candidate metrics on the TOY ground-truth
control, where each SAE feature's true class is known from its decoder's alignment to the toy
model's motif vs GC vs random channels. The decisive comparison is threshold-free: how well
does A_f separate ground-truth-causal (motif) features from the rest (AUROC + median gap)?

Run:  python scripts/metric_experiment.py smoke=true
(Toy model required — it exposes the channel rotation R.)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import MOTIF_NAMES, load_satmut, seq_to_idx
from dna_interp.utils import BASE_TO_IDX, get_logger, md_table, path, write_report

log = get_logger("metric_exp")


# ---------------------------------------------------------------------------
# candidate per-element metrics  a_f(e) := metric(I_f, E_e, gc_e)  in [-1, 1]
# ---------------------------------------------------------------------------
def gc_profile(seq, w=11):
    idx = seq_to_idx(seq)
    gc = ((idx == BASE_TO_IDX["C"]) | (idx == BASE_TO_IDX["G"])).astype(float)
    ww = max(1, min(w, len(idx)))
    return np.convolve(gc, np.ones(ww) / ww, mode="same")


def _resid_on(a, b):
    a = a - a.mean(); b = b - b.mean()
    beta = (a * b).sum() / ((b * b).sum() + 1e-12)
    return a - beta * b


def m_spearman_all(I, E, gc):                       # current baseline
    return causal.spearman_rho(I, E)


def m_spearman_active(I, E, gc):                     # only where the feature is sensitive
    mask = I > 1e-9
    return causal.spearman_rho(I[mask], E[mask]) if mask.sum() >= 3 else 0.0


def m_precision_weighted(I, E, gc):                  # do high-I positions carry effect? (precision)
    if I.sum() <= 0:
        return 0.0
    er = (rankdata(E) - 1) / (len(E) - 1 + 1e-12)    # 0..1 percentile of E
    return float(2 * ((I * er).sum() / I.sum()) - 1)  # >0 iff I-mass sits on high-E positions


def m_partial_gc(I, E, gc):                          # remove the composition (GC) component
    if I.size < 3:
        return 0.0
    ri, re, rg = rankdata(I), rankdata(E), rankdata(gc)
    ai, ae = _resid_on(ri, rg), _resid_on(re, rg)
    den = np.sqrt((ai * ai).sum() * (ae * ae).sum()) + 1e-12
    return float((ai * ae).sum() / den)


def m_overlap_auroc(I, E, gc):                       # does I rank effectful positions high? (recall)
    pos = E > (np.median(E) + 1e-12)
    if pos.sum() == 0 or pos.all():
        return 0.0
    try:
        return float(2 * roc_auc_score(pos, I) - 1)
    except Exception:
        return 0.0


METRICS = {
    "spearman_all (baseline)": m_spearman_all,
    "spearman_active": m_spearman_active,
    "precision_weighted": m_precision_weighted,
    "partial_gc": m_partial_gc,
    "overlap_auroc": m_overlap_auroc,
}


# ---------------------------------------------------------------------------
# ground-truth feature labels from decoder vs known channel directions
# ---------------------------------------------------------------------------
def ground_truth_labels(model, sae):
    R = model.R / (np.linalg.norm(model.R, axis=0, keepdims=True) + 1e-9)   # [d, C]
    W = sae.W_dec.detach().cpu().numpy()
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)               # [d, m]
    cos = np.abs(W.T @ R)                                                    # [m, C]
    n_motif = len(MOTIF_NAMES)
    max_motif = cos[:, :n_motif].max(1)
    max_other = cos[:, n_motif:].max(1)
    is_causal = (max_motif > 0.3) & (max_motif > max_other)
    return is_causal, max_motif - max_other


def main():
    cfg = boot()
    model = load_model(cfg)
    if not hasattr(model, "R"):
        raise SystemExit("metric_experiment requires the toy model (run with smoke=true).")
    L = int((path(cfg, "artifacts", "chosen_layer.txt")).read_text().strip())
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    satmut = load_satmut(cfg)

    # one ISM pass -> cache I_f and E per (feature, element); plus activation mass for weighting
    _, cache = controls.compute_alignments(cfg, model, sae, satmut, L, cache_I=True)
    gcs = {e.elem_id: gc_profile(e.seq) for e in satmut}
    masses = {}
    from dna_interp import ism
    for e in satmut:
        _, mass, _ = ism.element_activation(model, sae, e.seq, L, cfg.device)
        masses[e.elem_id] = mass

    # per metric: aggregate A_f (weighted median over elements)
    is_causal, motifness = ground_truth_labels(model, sae)
    rows_long = {name: {} for name in METRICS}     # name -> {feat: [(a, w)]}
    for eid, (E, Imap) in cache.items():
        gc = gcs[eid]
        for f, I in Imap.items():
            w = float(masses[eid][f])
            for name, fn in METRICS.items():
                rows_long[name].setdefault(f, []).append((fn(I, E, gc), w))

    feat_ids = sorted({f for d in rows_long.values() for f in d})
    A = {name: np.full(len(feat_ids), np.nan) for name in METRICS}
    for name in METRICS:
        for i, f in enumerate(feat_ids):
            if f in rows_long[name]:
                a, w = zip(*rows_long[name][f])
                A[name][i] = causal.aggregate_A_f(list(a), list(w))
    lab = is_causal[np.array(feat_ids)]

    # evaluate separation
    results = []
    for name in METRICS:
        a = A[name]; ok = ~np.isnan(a)
        y, s = lab[ok], a[ok]
        auroc = roc_auc_score(y, s) if (y.sum() and (~y).sum()) else float("nan")
        gap = np.median(s[y]) - np.median(s[~y]) if y.sum() and (~y).sum() else float("nan")
        results.append({"metric": name, "AUROC": auroc, "gap": gap,
                        "median_causal": float(np.median(s[y])) if y.sum() else np.nan,
                        "median_other": float(np.median(s[~y])) if (~y).sum() else np.nan,
                        "n_causal": int(y.sum()), "n_other": int((~y).sum())})
    res = pd.DataFrame(results).sort_values("AUROC", ascending=False)
    res.to_csv(path(cfg, "artifacts", "metric_experiment.csv"), index=False)
    _plot(cfg, A, lab, res)

    best = res.iloc[0]["metric"]
    body = f"""# Metric bake-off — defining A_f (PLAN.md §3.2)

Toy ground-truth control, layer L={L}, {int(lab.sum())} causal (motif) features vs
{int((~lab).sum())} other, evaluated on {len(satmut)} satmut elements. Separation is
threshold-free: AUROC of A_f as a classifier of the *known* causal label, plus the median gap.

{md_table(["metric", "AUROC", "median(causal)", "median(other)", "gap"],
          [[r.metric, f"{r.AUROC:.3f}", f"{r.median_causal:.3f}",
            f"{r.median_other:.3f}", f"{r.gap:.3f}"] for r in res.itertuples()])}

**Best separator: `{best}`** (AUROC {res.iloc[0]['AUROC']:.3f}).

## Reading
- `spearman_all` is the current spec metric; if it is *not* the top row, the multi-motif
  penalty is real and material — a metric change is warranted before the real run.
- `precision_weighted` / `spearman_active` ask "where the feature is sensitive, is there
  effect?" (precision) — robust to a feature not covering *other* motifs in the element.
- `overlap_auroc` is recall-oriented (does the feature rank ALL effectful positions high?) and
  is *expected* to be penalized by multi-motif elements — useful as a contrast.
- `partial_gc` removes the composition (GC) confound.

Figure: `reports/figures/metric_comparison.png`. Artifact: `artifacts/metric_experiment.csv`.

> **Decision needed:** adopt `{best}` (or a precision-oriented variant) as the
> `a_f(e)` definition, or keep `spearman_all` with the caveat documented. This is load-bearing
> for the headline number on real, multi-TF regulatory elements.
"""
    write_report(cfg, "metric_experiment", body)
    print(res.to_string(index=False))
    print(f"\nBest separator: {best}")


def _plot(cfg, A, lab, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(A)
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 4), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), names):
        a = A[name]; ok = ~np.isnan(a)
        y = lab[ok]; s = a[ok]
        ax.boxplot([s[~y], s[y]], tick_labels=["other", "causal"], showmeans=True)
        au = res.set_index("metric").loc[name, "AUROC"]
        ax.set_title(f"{name}\nAUROC={au:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("A_f separation of ground-truth causal features, by metric", fontsize=10)
    fig.tight_layout()
    p = path(cfg, "reports", "figures", "metric_comparison.png")
    fig.savefig(p, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
