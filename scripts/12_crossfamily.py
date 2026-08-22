#!/usr/bin/env python
"""Is the non-identifiability TopK-specific? Train Gated + JumpReLU SAEs on the same layer-6
activations and run the identical causal pipeline, then ask:
  (i)  within each family: does the aggregate causal signal appear and is the per-feature causal set
       seed-stable? (Jaccard of q<0.05 set across seeds, Hungarian decoder matching)
  (ii) across families: do TopK / Gated / JumpReLU find the SAME causal features? (cross-family
       Jaccard + Spearman of A_f). Low overlap ⇒ the causal decomposition is method-arbitrary,
       subsuming the seed result.

Coefficients for Gated (sae.l1_coeff) / JumpReLU (sae.l0_coeff, sae.theta_init) are passed as
standard config overrides after calibration (scripts/calib_families.py). Reuses the shared
random-feature null threshold and the existing TopK seeds. Idempotent: skips training / causal
passes whose artifacts already exist.

Run e.g.:
  DNA_INTERP_CONFIG=config/real.yaml python scripts/12_crossfamily.py \
      sae.l1_coeff=0.03 sae.l0_coeff=0.002 sae.theta_init=0.05
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_activations, load_model
from dna_interp.data import load_satmut
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("crossfamily")
FAMILIES = [f for f in os.environ.get("CF_FAMILIES", "topk,gated,jumprelu").split(",") if f]
N_SEEDS = 3


def chosen_layer(cfg):
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else int(cfg.activations.get("chosen_layer") or 1)


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def set_family_cfg(cfg, fam):
    # coefficients (sae.l1_coeff / sae.l0_coeff / sae.theta_init) come from CLI config overrides
    cfg.sae.family = fam


def ensure_trained(cfg, mm, L, fam):
    """Train N_SEEDS for `fam` if missing; return per-seed eval metrics."""
    set_family_cfg(cfg, fam)
    metrics = {}
    for s in range(N_SEEDS):
        p = S._sae_path(cfg, L, s, fam)
        if p.exists():
            sae = S.load_sae(cfg, L, s, family=fam)
            metrics[s] = S.evaluate_sae(cfg, sae, mm, np.random.default_rng(s))
            log.info("[%s] seed %d exists: FVU=%.3f L0=%.1f", fam, s, metrics[s]["FVU"], metrics[s]["L0"])
        else:
            cfg.seed = s
            _, m = S.train_sae(cfg, mm, L, seed=s)
            metrics[s] = m
            log.info("[%s] seed %d TRAINED: FVU=%.3f L0=%.1f dead=%.3f", fam, s, m["FVU"], m["L0"], m["dead_frac"])
    return metrics


def causal_for_seed(cfg, model, satmut, L, fam, s, null_A, thr):
    """Return feats df (A_f, q, causal) for one (family, seed); cache to csv."""
    art = Path(cfg.paths["artifacts"])
    cache = art / f"cf_feats_{fam}_seed{s}.csv"
    sae = S.load_sae(cfg, L, s, family=fam)
    W = sae.W_dec.detach().cpu().numpy()
    if cache.exists():
        return pd.read_csv(cache), W
    feats, _, _ = controls.run_causal_test(cfg, model, sae, satmut, L, with_ci=False)
    feats["p_value"] = [causal.empirical_pvalue(a, null_A) for a in feats.A_f]
    feats["q_value"] = bh_fdr(feats.p_value.to_numpy())
    feats["causal"] = feats.q_value < 0.05
    feats.to_csv(cache, index=False)
    return feats, W


def within_family_jaccard(feats_by_seed, W_by_seed, cos_thr):
    """Jaccard of q<0.05 causal sets across seeds (ref=seed0), decoder-matched."""
    seeds = sorted(feats_by_seed)
    ref = seeds[0]
    ref_causal = set(feats_by_seed[ref].loc[feats_by_seed[ref].causal, "feature_id"].astype(int))
    jacc, spear = [], []
    refA = feats_by_seed[ref].set_index("feature_id")["A_f"]
    for s in seeds[1:]:
        ra, rb, cos = controls.match_features(W_by_seed[ref], W_by_seed[s])
        a_s = feats_by_seed[s].set_index("feature_id")["A_f"]
        s_causal = set(feats_by_seed[s].loc[feats_by_seed[s].causal, "feature_id"].astype(int))
        matched_ref_causal, xa, xb = set(), [], []
        for fa, fb, c in zip(ra, rb, cos):
            if c <= cos_thr:
                continue
            if fa in refA.index and fb in a_s.index:
                xa.append(refA.loc[fa]); xb.append(a_s.loc[fb])
                if fa in ref_causal and fb in s_causal:
                    matched_ref_causal.add(int(fa))
        union = len(ref_causal | s_causal)
        jacc.append(len(matched_ref_causal) / union if union else float("nan"))
        spear.append(causal.spearman_rho(np.array(xa), np.array(xb)))
    return float(np.nanmean(jacc)), float(np.nanmean(spear)), len(ref_causal)


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    mm, _ = load_activations(cfg, L)
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    null_A = np.load(art / f"null_A_layer{L}.npy")
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))
    cos_thr = float(cfg.causal.match_cos_threshold)
    coeffs = {"l1_coeff": float(cfg.sae.get("l1_coeff", 0)),
              "l0_coeff": float(cfg.sae.get("l0_coeff", 0)),
              "theta_init": float(cfg.sae.get("theta_init", 0)),
              "epochs": int(cfg.sae.epochs)}
    log.info("crossfamily: L=%d thr=%.4f, coeffs=%s", L, thr, coeffs)

    # ---- train (idempotent) ----
    fam_metrics = {}
    for fam in FAMILIES:
        if fam == "topk":
            # reuse existing topk seeds
            fam_metrics[fam] = {s: S.evaluate_sae(cfg, S.load_sae(cfg, L, s, family="topk"), mm,
                                                  np.random.default_rng(s)) for s in range(N_SEEDS)}
        else:
            fam_metrics[fam] = ensure_trained(cfg, mm, L, fam)

    # ---- causal test per (family, seed) ----
    feats_by, W_by = {f: {} for f in FAMILIES}, {f: {} for f in FAMILIES}
    for fam in FAMILIES:
        set_family_cfg(cfg, fam)
        for s in range(N_SEEDS):
            feats_by[fam][s], W_by[fam][s] = causal_for_seed(cfg, model, satmut, L, fam, s, null_A, thr)
            log.info("[%s seed %d] causal frac=%.3f q<0.05=%d FVU=%.3f L0=%.1f", fam, s,
                     float((feats_by[fam][s].A_f > thr).mean()),
                     int(feats_by[fam][s].causal.sum()),
                     fam_metrics[fam][s]["FVU"], fam_metrics[fam][s]["L0"])

    # ---- within-family seed identifiability ----
    within = {}
    for fam in FAMILIES:
        j, sp, ncaus = within_family_jaccard(feats_by[fam], W_by[fam], cos_thr)
        within[fam] = {"jaccard_seed": j, "spearman_seed": sp, "n_causal_q05_seed0": ncaus,
                       "FVU": float(np.mean([fam_metrics[fam][s]["FVU"] for s in range(N_SEEDS)])),
                       "L0": float(np.mean([fam_metrics[fam][s]["L0"] for s in range(N_SEEDS)])),
                       "causal_frac_seed0": float((feats_by[fam][0].A_f > thr).mean())}

    # ---- cross-family (seed 0) ----
    cross = {}
    for i, fa in enumerate(FAMILIES):
        for fb in FAMILIES[i + 1:]:
            ra, rb, cos = controls.match_features(W_by[fa][0], W_by[fb][0])
            Aa = feats_by[fa][0].set_index("feature_id")["A_f"]
            Ab = feats_by[fb][0].set_index("feature_id")["A_f"]
            ca = set(feats_by[fa][0].loc[feats_by[fa][0].causal, "feature_id"].astype(int))
            cb = set(feats_by[fb][0].loc[feats_by[fb][0].causal, "feature_id"].astype(int))
            matched_causal, xa, xb = set(), [], []
            for x, y, c in zip(ra, rb, cos):
                if c <= cos_thr:
                    continue
                if x in Aa.index and y in Ab.index:
                    xa.append(Aa.loc[x]); xb.append(Ab.loc[y])
                    if x in ca and y in cb:
                        matched_causal.add(int(x))
            union = len(ca | cb)
            cross[f"{fa}-{fb}"] = {
                "jaccard_causal": len(matched_causal) / union if union else float("nan"),
                "spearman_Af": causal.spearman_rho(np.array(xa), np.array(xb)),
                "median_matched_cos": float(np.median(cos[cos > cos_thr])) if (cos > cos_thr).any() else float("nan"),
            }
            log.info("cross %s-%s: jaccard_causal=%.3f spearman_Af=%.3f",
                     fa, fb, cross[f"{fa}-{fb}"]["jaccard_causal"], cross[f"{fa}-{fb}"]["spearman_Af"])

    out = {"layer": L, "thr": float(thr), "coeffs": coeffs, "within": within, "cross": cross}
    (art / f"crossfamily_layer{L}.json").write_text(json.dumps(out, indent=2))
    _report(cfg, L, thr, within, cross)
    print(f"[phase12] within-family seed Jaccard: " +
          ", ".join(f"{f}={within[f]['jaccard_seed']:.3f}" for f in FAMILIES) +
          " | cross-family causal Jaccard: " +
          ", ".join(f"{k}={v['jaccard_causal']:.3f}" for k, v in cross.items()))


def _report(cfg, L, thr, within, cross):
    wrows = [[f, f"{within[f]['FVU']:.3f}", f"{within[f]['L0']:.0f}",
              f"{within[f]['causal_frac_seed0']:.3f}", within[f]['n_causal_q05_seed0'],
              f"{within[f]['jaccard_seed']:.3f}", f"{within[f]['spearman_seed']:.3f}"]
             for f in FAMILIES]
    crows = [[k, f"{v['jaccard_causal']:.3f}", f"{v['spearman_Af']:.3f}", f"{v['median_matched_cos']:.2f}"]
             for k, v in cross.items()]
    body = f"""# Phase 12 — Cross-SAE-family identifiability

Layer L={L}, null threshold {thr:.3f}. Three SAE families on the same activations; same causal pipeline.

## Per family (within-family seed identifiability)
{md_table(["family", "FVU", "L0", "causal frac", "q<0.05", "seed Jaccard", "seed Spearman A_f"], wrows)}

## Across families (seed 0, decoder-matched)
{md_table(["pair", "causal-set Jaccard", "Spearman A_f", "median matched cos"], crows)}

**Reading.** If every family shows an aggregate causal signal but low within-family seed Jaccard,
non-identifiability is not TopK-specific. If the cross-family causal Jaccard is also low, different
SAE families recover *different* causal features from the same activations — the causal decomposition
is method-arbitrary, which subsumes the seed-instability result.

**CHECKPOINT 12.** Confirm the cross-family verdict for the paper's generality claim.
"""
    write_report(cfg, "phase12", body)


if __name__ == "__main__":
    main()
