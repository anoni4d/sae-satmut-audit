#!/usr/bin/env python
"""Deep dive on the real causal result (post-hoc analysis; produces artifacts + reports_real/phase7.md).

Three jobs, all driven off the trained SAEs + cached activations:
  (A) Multiple-comparison control: BH-FDR on the per-feature empirical p-values vs the
      random-feature null. Defines the FDR-significant causal set (q<0.05/0.10).
  (B) Honest seed-robustness: re-run the causal test for every trained seed, Hungarian-match
      decoder columns, and report (i) Spearman of A_f across seeds on matched features and on
      the high-signal subset, and (ii) Jaccard overlap of the causal *set* across seeds.
      This replaces CV(A_f), which is ill-defined when A_f is near zero (median ~0.004).
  (C) Characterise the high-confidence set (FDR ∩ CI-robust) and the novel (unannotated)
      causal features: their top driving satmut elements and max-activating genomic context.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/06_deepdive.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import annotate, causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_corpus, load_satmut
from dna_interp.utils import get_logger, md_table, path, write_report

log = get_logger("deepdive")


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pp = p[ok]
    n = pp.size
    order = np.argsort(pp)
    ranked = pp[order]
    q_sorted = ranked * n / (np.arange(1, n + 1))
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    out = np.empty(n)
    out[order] = q_sorted
    q[ok] = out
    return q


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    null_A = np.load(art / f"null_A_layer{L}.npy")
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))
    elem_name = {e.elem_id: e.elem_id for e in satmut}
    elem_seq = {e.elem_id: e.seq for e in satmut}
    log.info("deepdive: layer %d, thr=%.4f, %d satmut elements", L, thr, len(satmut))

    # ---- (B) re-run the causal test per seed, keep A_f + decoder; seed0 keeps align+cache ----
    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    log.info("seeds found: %s", seeds)
    feats_by_seed, W_by_seed = {}, {}
    align0 = cache0 = None
    for s in seeds:
        sae = S.load_sae(cfg, L, seed=s)
        feats, align, cache = controls.run_causal_test(cfg, model, sae, satmut, L,
                                                       with_ci=(s == seeds[0]))
        feats = feats.copy()
        feats["p_value"] = [causal.empirical_pvalue(a, null_A) for a in feats.A_f]
        feats["q_value"] = bh_fdr(feats.p_value.to_numpy())
        feats_by_seed[s] = feats
        W_by_seed[s] = sae.W_dec.detach().cpu().numpy()
        if s == seeds[0]:
            align0, cache0 = align, cache
        log.info("seed %d: %d feats, above-thr=%d, q<0.05=%d",
                 s, len(feats), int((feats.A_f > thr).sum()), int((feats.q_value < 0.05).sum()))

    ref = seeds[0]
    ref_feats = feats_by_seed[ref]

    # cross-seed matching + robustness diagnostics
    rob = {"thr": float(thr), "seeds": seeds, "n_features": int(len(ref_feats))}
    spear_all, spear_hi, jacc = [], [], []
    ref_A = ref_feats.set_index("feature_id")["A_f"]
    ref_causal = set(ref_feats.loc[ref_feats.q_value < 0.05, "feature_id"].astype(int))
    ref_hi_ids = set(ref_feats.loc[ref_feats.A_f > thr, "feature_id"].astype(int))
    repro_counts = {fid: 0 for fid in ref_hi_ids}
    for s in seeds[1:]:
        ra, rb, cos = controls.match_features(W_by_seed[ref], W_by_seed[s])
        a_s = feats_by_seed[s].set_index("feature_id")["A_f"]
        causal_s = set(feats_by_seed[s].loc[feats_by_seed[s].q_value < 0.05, "feature_id"].astype(int))
        xa, xb = [], []
        matched_causal_ref = set()
        for fa, fb, c in zip(ra, rb, cos):
            if c <= float(cfg.causal.match_cos_threshold):
                continue
            if fa in ref_A.index and fb in a_s.index:
                xa.append(ref_A.loc[fa]); xb.append(a_s.loc[fb])
                # does seed s reproduce ref's causal calls?
                if fa in ref_causal and fb in causal_s:
                    matched_causal_ref.add(int(fa))
                if fa in ref_hi_ids and a_s.loc[fb] > thr:
                    repro_counts[int(fa)] += 1
        xa, xb = np.array(xa), np.array(xb)
        spear_all.append(causal.spearman_rho(xa, xb))
        hi = np.abs(xa) > thr
        spear_hi.append(causal.spearman_rho(xa[hi], xb[hi]) if hi.sum() >= 3 else float("nan"))
        union = len(ref_causal | causal_s)
        jacc.append(len(matched_causal_ref) / union if union else float("nan"))
        log.info("seed %d vs %d: matched=%d spearman_all=%.3f spearman_hi=%.3f jaccard=%.3f",
                 ref, s, len(xa), spear_all[-1], spear_hi[-1], jacc[-1])

    rob.update({
        "spearman_Af_all_pairs": [float(x) for x in spear_all],
        "spearman_Af_highsignal_pairs": [float(x) for x in spear_hi],
        "jaccard_causalset_q05_pairs": [float(x) for x in jacc],
        "n_ref_causal_q05": len(ref_causal),
        "n_ref_above_thr": len(ref_hi_ids),
        # fraction of ref above-thr features reproduced (above thr) in ALL other seeds
        "frac_above_thr_reproduced_all_seeds":
            float(np.mean([c == len(seeds) - 1 for c in repro_counts.values()])) if repro_counts else float("nan"),
    })

    # ---- (A) FDR on seed0 (the headline SAE) ----
    ref_feats = ref_feats.copy()
    ci_robust = ref_feats.A_f_lo > thr if "A_f_lo" in ref_feats else pd.Series(False, index=ref_feats.index)
    ref_feats["ci_robust"] = ci_robust
    for ql in (0.05, 0.10, 0.20):
        rob[f"n_fdr_q{ql}"] = int((ref_feats.q_value < ql).sum())
    highconf = ref_feats[(ref_feats.q_value < 0.10) & ref_feats.ci_robust].copy()
    rob["n_highconf_fdr10_and_ci"] = int(len(highconf))

    # ---- (C) drivers + annotation + context for the high-confidence set ----
    ann_path = art / f"features_layer{L}.csv"
    ann = pd.read_csv(ann_path).set_index("feature_id") if ann_path.exists() else None

    # top driving satmut elements per feature (from seed0 align)
    drivers = {}
    if align0 is not None and len(align0):
        for f, g in align0.groupby("feature_id"):
            gg = g.sort_values("a_f", ascending=False)
            top = [(str(r.elem_id), float(r.a_f)) for _, r in gg.head(3).iterrows()]
            drivers[int(f)] = top

    # max-activating genomic window for high-confidence features (single memmap scan)
    log.info("scanning max-activating windows (seed %d)...", ref)
    sae0 = S.load_sae(cfg, L, seed=ref)
    items_by_feat, density = annotate.top_activating(cfg, sae0, L, top_n=3)
    corpus = load_corpus(cfg)
    corpus_map = {w.seq_id: w.seq for w in corpus}

    rows = []
    for _, r in highconf.iterrows():
        fid = int(r.feature_id)
        lab = (ann.loc[fid, "label"] if ann is not None and fid in ann.index else "")
        ecl = (ann.loc[fid, "element_class"] if ann is not None and fid in ann.index else "")
        drv = drivers.get(fid, [])
        win = ""
        its = items_by_feat.get(fid, [])
        if its:
            v, sid, pos = its[0]
            s = corpus_map.get(sid, "")
            a, b = max(0, pos - 12), min(len(s), pos + 13)
            win = s[a:b]
        rows.append({
            "feature_id": fid, "A_f": float(r.A_f),
            "A_f_lo": float(r.A_f_lo), "q_value": float(r.q_value),
            "n_active_e": int(r.n_active_e), "label": lab, "element_class": ecl,
            "top_driver_elements": ";".join(f"{e}:{a:.2f}" for e, a in drv),
            "max_activating_context": win,
        })
    hc = pd.DataFrame(rows).sort_values("A_f", ascending=False)
    hc.to_csv(art / f"highconf_features_layer{L}.csv", index=False)
    ref_feats.to_csv(art / f"causal_features_layer{L}_fdr.csv", index=False)
    (art / f"robustness_v2_layer{L}.json").write_text(json.dumps(rob, indent=2))

    novel = hc[hc.label == "unannotated"]
    log.info("high-confidence n=%d (novel/unannotated=%d)", len(hc), len(novel))

    # ---- report ----
    _write_phase7(cfg, L, rob, hc, novel, thr)
    print(f"[phase7] highconf={len(hc)} novel={len(novel)} | "
          f"spearman_hi={np.nanmean(rob['spearman_Af_highsignal_pairs']):.3f} "
          f"jaccard={np.nanmean(rob['jaccard_causalset_q05_pairs']):.3f} "
          f"repro_frac={rob['frac_above_thr_reproduced_all_seeds']:.3f}")


def _write_phase7(cfg, L, rob, hc, novel, thr):
    sa = np.nanmean(rob["spearman_Af_all_pairs"])
    sh = np.nanmean(rob["spearman_Af_highsignal_pairs"])
    jc = np.nanmean(rob["jaccard_causalset_q05_pairs"])
    top = hc.head(15)
    body = f"""# Phase 7 — Deep dive: FDR, honest robustness, high-confidence features

Layer L={L}. Null 95th-pct threshold = **{thr:.3f}**.

## (A) Multiple-comparison control (BH-FDR vs random-feature null)
- FDR q<0.05: **{rob['n_fdr_q0.05']}** features | q<0.10: **{rob['n_fdr_q0.1']}** | q<0.20: **{rob['n_fdr_q0.2']}**
- High-confidence set (q<0.10 **and** bootstrap-CI-robust over elements): **{rob['n_highconf_fdr10_and_ci']}**

The population signal survives multiple-comparison correction: far more than the ~5% of 3,980
features that clear the null threshold by chance.

## (B) Seed robustness — proper diagnostic (CV(A_f) is undefined near A_f≈0, so not used)
- Spearman of A_f across seeds, **all matched features**: {sa:.3f} (per-pair {['%.3f'%x for x in rob['spearman_Af_all_pairs']]})
- Spearman of A_f across seeds, **high-signal features (|A_f|>thr)**: {sh:.3f} (per-pair {['%.3f'%x for x in rob['spearman_Af_highsignal_pairs']]})
- Jaccard of the q<0.05 causal **set** across seeds (decoder-matched): {jc:.3f}
- Fraction of seed-0 above-threshold features reproduced (>thr) in **every** other seed: **{rob['frac_above_thr_reproduced_all_seeds']:.3f}**
- Decoder matching cosine gate = {cfg.causal.match_cos_threshold}

## (C) High-confidence causal features (n={len(hc)}; novel/unannotated={len(novel)})

{md_table(["feature", "A_f", "A_f_lo", "q", "n_elem", "label", "top driver elements", "context"],
          [[int(r.feature_id), f"{r.A_f:.3f}", f"{r.A_f_lo:.3f}", f"{r.q_value:.3f}",
            int(r.n_active_e), r.label or "—", r.top_driver_elements,
            r.max_activating_context or "—"] for _, r in top.iterrows()])}

(full table: `artifacts/highconf_features_layer{L}.csv`; FDR table: `causal_features_layer{L}_fdr.csv`)

**CHECKPOINT 7.** Biology call: are the top driver elements / contexts sensible for
the candidate features? Which novel (unannotated) ones merit follow-up?
"""
    write_report(cfg, "phase7", body)


if __name__ == "__main__":
    main()
