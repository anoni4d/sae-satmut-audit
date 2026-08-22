"""Causal-alignment math: Spearman alignment, weighted-median aggregation, null threshold,
classification. Pure functions (numpy/scipy) — this is the unit-tested core (§8.4)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho, robust to degenerate (constant / too-short) inputs -> 0.0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3 or a.size != b.size:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


# ---------------------------------------------------------------------------
# Per-element alignment metrics a_f(e) := metric(I_f, E_e) in [-1, 1].
# Selection rationale (see reports/metric_experiment.md, PLAN.md §3.2): SAE features are
# monosemantic, so a per-feature causal metric must be PRECISION-oriented (recall is a
# set-level property). `precision_weighted` is the adopted default; `spearman_all` is the
# original baseline, kept for comparison.
# ---------------------------------------------------------------------------
def m_spearman_all(I: np.ndarray, E: np.ndarray) -> float:
    """Rank correlation over all positions (original baseline; recall-contaminated)."""
    return spearman_rho(I, E)


def m_spearman_active(I: np.ndarray, E: np.ndarray) -> float:
    """Rank correlation over positions where the feature is sensitive (I>0)."""
    I = np.asarray(I, float); E = np.asarray(E, float)
    mask = I > 1e-9
    return spearman_rho(I[mask], E[mask]) if mask.sum() >= 3 else 0.0


def m_precision_weighted(I: np.ndarray, E: np.ndarray) -> float:
    """Sensitivity-weighted mean percentile-rank of E, centered to [-1,1]:
    >0 iff the feature's ISM sensitivity mass sits on high-effect positions (precision)."""
    I = np.asarray(I, float); E = np.asarray(E, float)
    if I.size < 3 or I.sum() <= 0:
        return 0.0
    er = (rankdata(E) - 1) / (len(E) - 1 + 1e-12)        # 0..1
    return float(2.0 * ((I * er).sum() / I.sum()) - 1.0)


def m_precision_weighted_alt(I_alt: np.ndarray, E_alt: np.ndarray,
                             alt_base: np.ndarray) -> float:
    """Substitution-level version of `m_precision_weighted`.

    Position-averaging asks "is the feature sensitive where mutations matter?". This asks the
    sharper question "is the feature sensitive to the *specific substitutions* that matter?" —
    a feature could be sensitive at a high-effect position but to the wrong allele. Scoring over
    (position, alt) pairs also triples the sample per element.

    I_alt   : [L, n_alt] un-averaged ISM sensitivity (from ism.feature_sensitivity).
    E_alt   : [L, 4] measured per-substitution effect, column = alt base index.
    alt_base: [L, n_alt] base index of each slot, -1 where unused.
    """
    I_alt = np.asarray(I_alt, float)
    E_alt = np.asarray(E_alt, float)
    alt_base = np.asarray(alt_base, int)
    n = min(I_alt.shape[0], E_alt.shape[0], alt_base.shape[0])
    if n < 3:
        return 0.0
    I_alt, E_alt, alt_base = I_alt[:n], E_alt[:n], alt_base[:n]

    valid = alt_base >= 0
    if not valid.any():
        return 0.0
    rows = np.repeat(np.arange(n)[:, None], alt_base.shape[1], axis=1)
    I_flat = I_alt[valid]
    E_flat = E_alt[rows[valid], alt_base[valid]]
    if I_flat.size < 3 or I_flat.sum() <= 0:
        return 0.0
    er = (rankdata(E_flat) - 1) / (len(E_flat) - 1 + 1e-12)
    return float(2.0 * ((I_flat * er).sum() / I_flat.sum()) - 1.0)


METRICS = {
    "spearman_all": m_spearman_all,
    "spearman_active": m_spearman_active,
    "precision_weighted": m_precision_weighted,
}


def get_metric(name: str):
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}; choices: {list(METRICS)}")
    return METRICS[name]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of `values` with non-negative `weights`."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if weights.sum() <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    cutoff = w.sum() / 2.0
    i = int(np.searchsorted(cum, cutoff))
    i = min(i, len(v) - 1)
    return float(v[i])


def aggregate_A_f(a_f_per_element: list[float], masses: list[float]) -> float:
    """A_f = weighted median over elements of a_f(e), weight = activation mass on e."""
    if not a_f_per_element:
        return float("nan")
    return weighted_median(np.array(a_f_per_element), np.array(masses))


def bootstrap_ci(values, weights=None, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the (weighted) median over elements — used to put error
    bars on A_f. Resamples elements with replacement. Returns (lo, hi); (nan,nan) if <2 points."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        return (float("nan"), float("nan"))
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = weighted_median(values[idx], weights[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def null_threshold(null_A: np.ndarray, percentile: float = 95.0) -> float:
    null_A = np.asarray(null_A, dtype=np.float64)
    null_A = null_A[~np.isnan(null_A)]
    if null_A.size == 0:
        return 0.0
    return float(np.percentile(null_A, percentile))


def classify(A_f: float, threshold: float) -> str:
    if np.isnan(A_f):
        return "inactive"
    return "causal" if A_f > threshold else "correlational"


def classify_array(A: np.ndarray, threshold: float) -> np.ndarray:
    out = np.where(A > threshold, "causal", "correlational").astype(object)
    out[np.isnan(A)] = "inactive"
    return out


def causal_fraction(A: np.ndarray, threshold: float) -> float:
    A = np.asarray(A, dtype=np.float64)
    valid = A[~np.isnan(A)]
    if valid.size == 0:
        return float("nan")
    return float((valid > threshold).mean())


def empirical_pvalue(A_f: float, null_A: np.ndarray) -> float:
    """One-sided p: fraction of null >= A_f (with +1 smoothing)."""
    null_A = np.asarray(null_A)
    null_A = null_A[~np.isnan(null_A)]
    if null_A.size == 0 or np.isnan(A_f):
        return float("nan")
    return float((1 + (null_A >= A_f).sum()) / (1 + null_A.size))
