"""Supervised probes of per-position effect, with locus-grouped cross-validation.

Every probe in this project asks the same question — can the measured per-position effect
`E_e[p]` be read linearly off a per-position representation? — and differs only in what the
representation is (activation value, ISM sensitivity, sequence composition) and which
estimator maps it to `E` (ridge, PLS at fixed rank, MLP). This module holds that shared
machinery so the four scripts that used to carry private copies (`08_probe`, `09_sensprobe`,
`11_probesweep`, `14_ranksweep`) agree by construction.

**Why grouped CV.** Folds must be held out by *locus*, not by assayed measurement. Kircher
reports several cell lines / timepoints per element, and those rows share a byte-identical
sequence (see `data.locus_groups`): holding out `LDLR.2` while `LDLR` — the same 318 bp with
an effect vector correlated at r=0.985 — stays in training is not held-out evaluation. Pass
`groups=` to `cross_val_probe` for locus-grouped folds; omit it to reproduce the older
leave-one-element-out numbers exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .causal import spearman_rho
from .utils import BASE_TO_IDX

# A block is one element's (X[n_pos, n_feat], y[n_pos]) pair.
Block = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# per-position feature builders
# ---------------------------------------------------------------------------
def seq_features(seq: str, gc_win: int = 11) -> np.ndarray:
    """Composition baseline: one-hot(base) + centered local GC fraction. [L, 5]."""
    L = len(seq)
    idx = np.array([BASE_TO_IDX.get(b, 0) for b in seq])
    oh = np.zeros((L, 4))
    oh[np.arange(L), idx] = 1.0
    gc = ((idx == BASE_TO_IDX["C"]) | (idx == BASE_TO_IDX["G"])).astype(float)
    k = max(1, min(gc_win, L))
    gcw = np.convolve(gc, np.ones(k) / k, mode="same") - 0.5
    return np.hstack([oh, gcw[:, None]])


def kmer_features(seq: str, k: int = 6) -> np.ndarray:
    """Stronger sequence baseline: one-hot of the k-mer centered on each position. [L, 4**k].

    A k-mer indicator is what gapped-k-mer / lsgkm-style models consume, so this is the
    honest "can plain local sequence do it?" control — strictly richer than one-hot+GC, and
    it bounds how much of the probe's signal is local composition rather than representation.
    Returned dense; at k=6 that is 4096 columns, fine for the ~13k positions here.
    """
    L = len(seq)
    idx = np.array([BASE_TO_IDX.get(b, 0) for b in seq], dtype=np.int64)
    half = k // 2
    pad = np.full(half, 0, dtype=np.int64)
    padded = np.concatenate([pad, idx, pad])
    # rolling base-4 code of the k-mer window centered at each position
    codes = np.zeros(L, dtype=np.int64)
    for j in range(k):
        codes = codes * 4 + padded[j : j + L]
    out = np.zeros((L, 4 ** k), dtype=np.float32)
    out[np.arange(L), codes] = 1.0
    return out


# ---------------------------------------------------------------------------
# estimators: fit(Xtr, ytr) -> predict(Xte); optionally expose a weight direction
# ---------------------------------------------------------------------------
def _standardize(Xtr: np.ndarray):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-8
    return mu, sd


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge on standardized X (bias folded via centering)."""
    mu, sd = _standardize(X)
    Xs = (X - mu) / sd
    ybar = y.mean()
    w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (y - ybar))
    return w, mu, sd, ybar


def ridge_pred(X: np.ndarray, w, mu, sd, ybar) -> np.ndarray:
    return ((X - mu) / sd) @ w + ybar


def make_ridge(alpha: float = 10.0):
    def fit_predict(Xtr, ytr, Xte):
        w, mu, sd, yb = ridge_fit(Xtr, ytr, alpha)
        return ridge_pred(Xte, w, mu, sd, yb), w

    return fit_predict


def make_pls(rank: int):
    """Rank-`rank` supervised linear probe (components ordered by covariance with the target)."""
    from sklearn.cross_decomposition import PLSRegression

    def fit_predict(Xtr, ytr, Xte):
        mu, sd = _standardize(Xtr)
        Xs, Xt = (Xtr - mu) / sd, (Xte - mu) / sd
        ncomp = max(1, min(rank, Xs.shape[1], Xs.shape[0] - 1))
        pls = PLSRegression(n_components=ncomp, scale=False)
        yb = ytr.mean()
        pls.fit(Xs, ytr - yb)
        return pls.predict(Xt).ravel() + yb, None

    return fit_predict


def make_mlp(hidden: int = 64, alpha: float = 1e-3, seed: int = 0, max_iter: int = 300):
    from sklearn.neural_network import MLPRegressor

    def fit_predict(Xtr, ytr, Xte):
        mu, sd = _standardize(Xtr)
        net = MLPRegressor(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=max_iter,
                           random_state=seed, early_stopping=False)
        yb = ytr.mean()
        net.fit((Xtr - mu) / sd, ytr - yb)
        return net.predict((Xte - mu) / sd) + yb, None

    return fit_predict


# ---------------------------------------------------------------------------
# grouped cross-validation
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    rhos: np.ndarray                     # held-out Spearman, one per FOLD
    fold_names: list[str] = field(default_factory=list)
    dir_stability_cos: float = float("nan")
    n_folds: int = 0
    n_blocks: int = 0

    @property
    def median_rho(self) -> float:
        return float(np.nanmedian(self.rhos)) if self.rhos.size else float("nan")

    @property
    def mean_rho(self) -> float:
        return float(np.nanmean(self.rhos)) if self.rhos.size else float("nan")

    @property
    def frac_pos(self) -> float:
        return float(np.mean(self.rhos > 0)) if self.rhos.size else float("nan")

    def as_dict(self) -> dict:
        return {"median_rho": self.median_rho, "mean_rho": self.mean_rho,
                "frac_pos": self.frac_pos, "dir_stability_cos": self.dir_stability_cos,
                "n_folds": self.n_folds, "n_blocks": self.n_blocks}


def cross_val_probe(blocks: list[Block], estimator=None, groups=None,
                    names: list[str] | None = None) -> ProbeResult:
    """Leave-one-GROUP-out CV over per-element blocks.

    blocks    : [(X_e, y_e)] one per assayed measurement.
    estimator : fit_predict(Xtr, ytr, Xte) -> (pred, weight_or_None); default ridge(alpha=10).
    groups    : group label per block (e.g. locus). None -> every block is its own group,
                which reproduces the original leave-one-element-out behaviour exactly.
    names     : optional label per block, used to name folds.

    Held-out Spearman is computed on the *concatenation* of the group's blocks, so a locus
    contributes one fold regardless of how many times it was assayed.
    """
    estimator = estimator or make_ridge(10.0)
    n = len(blocks)
    if n == 0:
        return ProbeResult(np.array([]), [], float("nan"), 0, 0)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    names = names or [str(i) for i in range(n)]

    rhos, ws, fold_names = [], [], []
    for g in sorted(set(groups.tolist())):
        te_idx = [i for i in range(n) if groups[i] == g]
        tr_idx = [i for i in range(n) if groups[i] != g]
        if not tr_idx or not te_idx:
            continue
        Xtr = np.vstack([blocks[j][0] for j in tr_idx])
        ytr = np.concatenate([blocks[j][1] for j in tr_idx])
        Xte = np.vstack([blocks[j][0] for j in te_idx])
        yte = np.concatenate([blocks[j][1] for j in te_idx])
        pred, w = estimator(Xtr, ytr, Xte)
        rhos.append(spearman_rho(pred, yte))
        if w is not None:
            ws.append(w / (np.linalg.norm(w) + 1e-9))
        fold_names.append("+".join(names[i] for i in te_idx))

    cos = float("nan")
    if len(ws) > 1:
        W = np.array(ws)
        iu = np.triu_indices(len(ws), 1)
        cos = float(np.mean(np.abs((W @ W.T)[iu])))
    return ProbeResult(np.array(rhos), fold_names, cos, len(rhos), n)


def paired_probe(blocks: list[Block], groups, names=None, estimator=None) -> dict:
    """Run the same probe ungrouped and locus-grouped, so the leakage delta is explicit."""
    ungrouped = cross_val_probe(blocks, estimator, groups=None, names=names)
    grouped = cross_val_probe(blocks, estimator, groups=groups, names=names)
    return {"element_level": ungrouped.as_dict(), "locus_level": grouped.as_dict(),
            "delta_median_rho": grouped.median_rho - ungrouped.median_rho}
