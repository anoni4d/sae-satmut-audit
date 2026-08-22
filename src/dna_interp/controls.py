"""Causal-test orchestration + the four controls (§8.5).

Orchestration lives here (not causal.py) because the controls reuse the *identical*
pipeline: the random-feature null, the shuffle control, and the real test all call
`compute_alignments`. ISM forward passes are shared across all features of an encoder, so
evaluating 1000 random directions costs essentially the same as evaluating one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import causal, ism
from .utils import get_logger

log = get_logger("controls")


# ---------------------------------------------------------------------------
# encoders with a common .encode(h)->[L, n] interface
# ---------------------------------------------------------------------------
class RandomFeatureSAE:
    """R random unit directions: z_r = ReLU(w_r . (h - b_dec)). The null encoder.

    Scope (reviewers asked): this null answers "could an arbitrary direction in the residual
    stream score this well?". It is deliberately *unconditioned* — dense ReLU directions fire on
    roughly half of all tokens, whereas a TopK feature fires on ~1.6%. For the sharper question
    "could an arbitrary direction with the same sparsity profile score this well?", use
    MatchedNullSAE below. We report both."""

    def __init__(self, d_model: int, R: int, b_dec: torch.Tensor, device: str, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(R, d_model, generator=g)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-9)
        self.W = W.to(device)
        self.b_dec = b_dec.to(device)
        self.device = device

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu((h - self.b_dec) @ self.W.t())


class MatchedNullSAE:
    """Random directions matched to the SAE's *sparsity* profile: z_r = ReLU(w_r.(h-b_dec) - θ_r).

    Each direction gets a per-direction threshold θ_r calibrated on the cached corpus activations
    so that its firing density equals that of a randomly drawn real SAE feature. This matches the
    properties reviewers flagged as unmatched in the dense null — activation frequency, sparsity,
    threshold behaviour, and hence coverage across elements (a rarely-firing direction is active
    on few elements, exactly like a real monosemantic feature).

    Activation *scale* needs no matching: the alignment metric
    `a_f(e) = 2*sum_p I_f[p] r(E[p]) / sum_p I_f[p] - 1` is invariant to positive rescaling of
    I_f, so only the support/shape of the sensitivity profile can affect the score.
    """

    def __init__(self, W: torch.Tensor, theta: torch.Tensor, b_dec: torch.Tensor, device: str,
                 target_density: np.ndarray | None = None):
        self.W = W.to(device)
        self.theta = theta.to(device)
        self.b_dec = b_dec.to(device)
        self.device = device
        self.target_density = target_density

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu((h - self.b_dec) @ self.W.t() - self.theta)

    @classmethod
    def calibrated(cls, mm, d_model: int, R: int, b_dec: torch.Tensor, device: str,
                   feature_density: np.ndarray, seed: int = 0, n_calib: int = 200_000):
        """Draw R directions and set each θ_r to hit a density sampled from `feature_density`.

        mm              : the cached activation memmap [n_tokens, d] (corpus statistics).
        feature_density : firing densities of the real SAE's live features, the profile to match.
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(R, d_model, generator=g)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-9)

        rng = np.random.default_rng(seed)
        live = np.asarray(feature_density)
        live = live[live > 0]
        target = rng.choice(live, size=R, replace=True) if live.size else np.full(R, 0.01)

        n = min(n_calib, mm.shape[0])
        sel = np.sort(rng.choice(mm.shape[0], size=n, replace=False))
        H = torch.tensor(np.asarray(mm[sel]), dtype=torch.float32)
        pre = (H - b_dec.detach().cpu()) @ W.t()                  # [n, R] on CPU (calibration only)
        # theta_r = the (1 - target_r) quantile, so exactly target_r of tokens exceed it
        q = torch.tensor(1.0 - target, dtype=torch.float32).clamp(0.0, 1.0)
        theta = torch.stack([torch.quantile(pre[:, r], q[r]) for r in range(R)])
        achieved = (pre > theta).float().mean(0).numpy()
        log.info("matched null: target density median %.4f -> achieved %.4f (R=%d)",
                 float(np.median(target)), float(np.median(achieved)), R)
        return cls(W, theta, b_dec.detach(), device, target_density=achieved)


def stratified_threshold(A_real: np.ndarray, dens_real: np.ndarray,
                         A_null: np.ndarray, dens_null: np.ndarray,
                         n_bins: int = 10, pct: float = 95.0) -> np.ndarray:
    """Per-feature threshold from the null directions in the same density bin.

    A single global percentile implicitly compares a rare feature against mostly-dense null
    directions. Binning on firing density compares like with like, which is what the
    'the null is not property-matched' critique asks for. Returns one threshold per real feature.
    """
    A_real = np.asarray(A_real, float)
    edges = np.quantile(dens_null[~np.isnan(dens_null)], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_null = np.digitize(dens_null, edges[1:-1])
    bin_real = np.digitize(dens_real, edges[1:-1])
    global_thr = causal.null_threshold(A_null, pct)
    out = np.full(A_real.shape, global_thr)
    for b in range(n_bins):
        vals = A_null[(bin_null == b) & ~np.isnan(A_null)]
        if vals.size >= 20:                       # enough null directions to estimate a quantile
            out[bin_real == b] = float(np.percentile(vals, pct))
    return out


# ---------------------------------------------------------------------------
# core: per-(feature, element) alignment
# ---------------------------------------------------------------------------
def _metric_name(cfg):
    return cfg.causal.get("metric", "spearman_all")


def compute_alignments(cfg, model, encoder, satmut, layer, restrict_active=True,
                       effect="E_mean", cache_I=False, metric=None):
    """Return a long DataFrame of (feature_id, elem_id, a_f, mass) and, if cache_I,
    a dict {elem_id: (E_vec, {feat: I_vec})} for reuse by the shuffle control."""
    device = cfg.device
    metric_fn = causal.get_metric(metric or _metric_name(cfg))
    rows = []
    cache = {}
    for e in satmut:
        summ, mass, active = ism.element_activation(model, encoder, e.seq, layer, device)
        feats = active if restrict_active else np.arange(len(summ))
        if feats.size == 0:
            continue
        I = ism.feature_sensitivity(model, encoder, e.seq, feats, layer, device,
                                    alts=cfg.ism.alts, summary=cfg.ism.summary,
                                    batch=int(cfg.ism.batch))
        E = getattr(e, effect)
        if cache_I:
            cache[e.elem_id] = (E, {})
        for f in feats:
            a = metric_fn(I[int(f)], E)
            rows.append({"feature_id": int(f), "elem_id": e.elem_id,
                         "a_f": a, "mass": float(mass[int(f)])})
            if cache_I:
                cache[e.elem_id][1][int(f)] = I[int(f)]
    df = pd.DataFrame(rows)
    return (df, cache) if cache_I else df


def collapse_to_loci(align_df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Average each feature's a_f over the repeat measurements of one locus.

    Kircher assays several cell lines / timepoints per element; those rows share an identical
    sequence (see `data.locus_groups`), so without this collapse TERT counts 4x and SORT1 3x
    in the weighted median. Masses are identical across a locus's rows (same sequence), so a
    plain mean is the mass-weighted mean.
    """
    if align_df.empty:
        return align_df
    df = align_df.copy()
    df["elem_id"] = df["elem_id"].map(lambda e: groups.get(e, e))
    return (df.groupby(["feature_id", "elem_id"], as_index=False)
              .agg(a_f=("a_f", "mean"), mass=("mass", "mean")))


def compute_alignments_subst(cfg, model, encoder, satmut, layer, restrict_active=True):
    """Per-(feature, measurement) alignment scored at SUBSTITUTION level.

    Mirrors `compute_alignments`, but keeps ISM sensitivity un-averaged over alternative bases
    and scores it against the assay's per-allele effects, so a feature is credited only when it
    responds to the substitution that actually changed expression. Returns a long DataFrame with
    both `a_f_subst` and the position-averaged `a_f` for direct comparison on one ISM pass.
    """
    device = cfg.device
    pos_metric = causal.get_metric(_metric_name(cfg))
    rows = []
    for e in satmut:
        if getattr(e, "E_alt", None) is None or np.asarray(e.E_alt).size == 0:
            log.warning("%s has no per-substitution effects; skipping", e.elem_id)
            continue
        summ, mass, active = ism.element_activation(model, encoder, e.seq, layer, device)
        feats = active if restrict_active else np.arange(len(summ))
        if feats.size == 0:
            continue
        I, per_alt, alt_base = ism.feature_sensitivity(
            model, encoder, e.seq, feats, layer, device, alts=cfg.ism.alts,
            summary=cfg.ism.summary, batch=int(cfg.ism.batch), return_per_alt=True)
        E_alt = np.asarray(e.E_alt, dtype=float)
        E_pos = getattr(e, "E_mean")
        for f in feats:
            f = int(f)
            rows.append({
                "feature_id": f, "elem_id": e.elem_id,
                "a_f": pos_metric(I[f], E_pos),
                "a_f_subst": causal.m_precision_weighted_alt(per_alt[f], E_alt, alt_base),
                "mass": float(mass[f]),
            })
    return pd.DataFrame(rows)


def cached_alignment(cfg, model, encoder, satmut, layer, tag: str) -> pd.DataFrame:
    """`compute_alignments` memoised to parquet, keyed by `tag`.

    ISM is the only expensive step in the whole analysis, and every downstream question
    (locus vs element aggregation, FDR variants, split-half resampling, dictionary sweeps)
    is a re-reduction of the same per-(feature, measurement) alignment matrix. Caching it
    means those cost seconds instead of another GPU hour."""
    from .utils import path

    p = path(cfg, "artifacts", f"align_{tag}_layer{layer}.parquet")
    if p.exists():
        log.info("reusing cached alignment %s", p.name)
        return pd.read_parquet(p)
    log.info("running ISM for %s ...", tag)
    df = compute_alignments(cfg, model, encoder, satmut, layer, restrict_active=True)
    df.to_parquet(p)
    return df


def aggregate_features(align_df: pd.DataFrame, min_active=1, with_ci=False,
                       n_boot=1000, groups: dict | None = None) -> pd.DataFrame:
    """Collapse per-(f,e) alignments into per-feature A_f (weighted median).
    with_ci adds bootstrap [A_f_lo, A_f_hi] over elements (opt-in: skip for the null/shuffle,
    which call this thousands of times).
    groups: optional elem_id -> locus_id map; when given, repeat measurements of one locus are
    averaged first so the median is taken over independent loci (default None = legacy path,
    kept so the pre-revision numbers stay exactly reproducible)."""
    if groups is not None:
        align_df = collapse_to_loci(align_df, groups)
    out = []
    for f, g in align_df.groupby("feature_id"):
        if len(g) < min_active:
            continue
        rec = {"feature_id": int(f),
               "A_f": causal.aggregate_A_f(list(g.a_f), list(g.mass)),
               "n_active_e": len(g), "mean_mass": float(g.mass.mean())}
        if with_ci:
            lo, hi = causal.bootstrap_ci(list(g.a_f), list(g.mass), n_boot=n_boot)
            rec["A_f_lo"], rec["A_f_hi"] = lo, hi
        out.append(rec)
    return pd.DataFrame(out).sort_values("A_f", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# the test + the four controls
# ---------------------------------------------------------------------------
def run_causal_test(cfg, model, sae, satmut, layer, effect="E_mean", metric=None, with_ci=True,
                    groups: dict | None = None):
    align, cache = compute_alignments(cfg, model, sae, satmut, layer, restrict_active=True,
                                      effect=effect, cache_I=True, metric=metric)
    feats = aggregate_features(align, int(cfg.causal.min_active_elements), with_ci=with_ci,
                               groups=groups)
    return feats, align, cache


def random_feature_null(cfg, model, sae, satmut, layer, effect="E_mean", metric=None,
                        groups: dict | None = None, encoder=None) -> np.ndarray:
    """Control 1: null distribution of A over R random directions (identical pipeline).

    `encoder` overrides the default dense random-direction null — pass a MatchedNullSAE to get
    the sparsity/density-matched null instead (the two answer different questions; see its
    docstring)."""
    R = int(cfg.causal.null_R)
    enc = encoder or RandomFeatureSAE(model.d_model, R, sae.b_dec.detach(), cfg.device,
                                      seed=cfg.seed)
    align = compute_alignments(cfg, model, enc, satmut, layer,
                               restrict_active=True, effect=effect, metric=metric)
    feats = aggregate_features(align, int(cfg.causal.min_active_elements), groups=groups)
    log.info("null: %d/%d random features active", len(feats), R)
    return feats.A_f.to_numpy()


def loglik_baseline(cfg, model, satmut, effect="E_mean", metric=None) -> dict:
    """Control 2: per-position model log-likelihood ISM, scored with the SAME metric as
    features so the comparison is apples-to-apples."""
    metric_fn = causal.get_metric(metric or _metric_name(cfg))
    vals, masses = [], []
    for e in satmut:
        I_ll = ism.loglik_ism(model, e.seq, alts=cfg.ism.alts, batch=int(cfg.ism.batch))
        vals.append(metric_fn(I_ll, getattr(e, effect)))
        masses.append(1.0)
    A_ll = causal.aggregate_A_f(vals, masses)
    return {"A_ll": A_ll, "per_element": vals}


def shuffle_control(cfg, cache, seed=0, metric=None, groups: dict | None = None) -> np.ndarray:
    """Control 3: permute the element<->E pairing using cached I vectors; A must collapse.

    With `groups`, the permutation is derangement-like *at locus level*: an element never
    receives the effect vector of its own locus. That matters here — TERT is assayed four
    times over an identical sequence with effect vectors correlated at r>0.87, so a naive
    element permutation can hand an element a near-copy of its own target and weaken the
    control."""
    metric_fn = causal.get_metric(metric or _metric_name(cfg))
    rng = np.random.default_rng(seed)
    elem_ids = list(cache.keys())
    n_e = len(elem_ids)
    loci = [groups.get(e, e) for e in elem_ids] if groups else list(elem_ids)

    # sample a permutation that never pairs an element with an effect vector from its own locus
    perm = None
    for _ in range(1000):
        cand = rng.permutation(n_e)
        if all(loci[i] != loci[cand[i]] for i in range(n_e)):
            perm = cand
            break
    if perm is None:                      # degenerate (e.g. a single locus): fall back
        perm = rng.permutation(n_e)
        log.warning("shuffle_control: no locus-disjoint permutation found; using a plain one")

    rows = []
    for i, eid in enumerate(elem_ids):
        E_other, _ = cache[elem_ids[perm[i]]]
        _, I_map = cache[eid]
        for f, I in I_map.items():
            n = min(len(I), len(E_other))
            rows.append({"feature_id": f, "elem_id": eid,
                         "a_f": metric_fn(I[:n], E_other[:n]), "mass": 1.0})
    feats = aggregate_features(pd.DataFrame(rows), 1, groups=groups)
    return feats.A_f.to_numpy()


def match_features(W_dec_a: np.ndarray, W_dec_b: np.ndarray):
    """Hungarian match decoder columns across two SAEs by cosine similarity.
    W_dec_*: [d, m]. Returns (idx_a, idx_b, cos) for matched pairs."""
    from scipy.optimize import linear_sum_assignment

    A = W_dec_a / (np.linalg.norm(W_dec_a, axis=0, keepdims=True) + 1e-9)
    B = W_dec_b / (np.linalg.norm(W_dec_b, axis=0, keepdims=True) + 1e-9)
    S = A.T @ B                                   # [m_a, m_b] cosine
    ra, rb = linear_sum_assignment(-S)
    return ra, rb, S[ra, rb]


def robustness_cv(A_by_seed: dict[int, pd.DataFrame], W_decs: dict[int, np.ndarray],
                  cos_threshold: float = 0.5):
    """Match features across seeds to a reference seed (Hungarian on decoder cosine); report
    CV of A_f for features matched above `cos_threshold` in every seed, plus the achieved
    matched-cosine so the stability of the matching itself is visible."""
    seeds = sorted(A_by_seed)
    ref = seeds[0]
    ref_A = A_by_seed[ref].set_index("feature_id")["A_f"]
    series = {ref: ref_A.to_dict()}
    matched_cos: list[float] = []
    for s in seeds[1:]:
        ra, rb, cos = match_features(W_decs[ref], W_decs[s])
        a_s = A_by_seed[s].set_index("feature_id")["A_f"]
        m = {}
        for fa, fb, c in zip(ra, rb, cos):
            if c > cos_threshold and fa in ref_A.index and fb in a_s.index:
                m[int(fa)] = a_s.get(fb, np.nan)
                matched_cos.append(float(c))
        series[s] = m
    common = set(series[ref])
    for s in seeds[1:]:
        common &= set(series[s])
    common = [f for f in common if not np.isnan(series[ref][f])]
    cvs = []
    for f in common:
        vals = np.array([series[s].get(f, np.nan) for s in seeds])
        vals = vals[~np.isnan(vals)]
        if vals.size >= 2 and abs(vals.mean()) > 1e-6:
            cvs.append(abs(vals.std() / vals.mean()))
    return {"n_matched": len(common),
            "median_cv": float(np.median(cvs)) if cvs else float("nan"),
            "mean_cv": float(np.mean(cvs)) if cvs else float("nan"),
            "mean_matched_cos": float(np.mean(matched_cos)) if matched_cos else float("nan"),
            "cos_threshold": cos_threshold}
