#!/usr/bin/env python
"""Repeat-masking control: does the aggregate causal signal survive removing Alu/low-complexity
positions? 61% of high-confidence novel features are repeat detectors, so we must check whether the
headline causal fraction is genuinely regulatory or repeat-driven.

For each satmut element we build a per-position repeat mask (Alu-consensus k-mer hits + homopolymer
runs), then score every feature's alignment on the KEPT (non-repeat) positions only, recomputing the
random-feature null and FDR with the same masking. We report masked vs unmasked side by side on the
same ISM pass.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/10_repeatmask.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, ism, sae as S
from dna_interp.activations import load_model
from dna_interp.annotate import repeat_mask          # shared with the subspace analysis
from dna_interp.controls import RandomFeatureSAE
from dna_interp.data import load_satmut
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("repeatmask")
pw = causal.m_precision_weighted


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def bh_fdr(p):
    p = np.asarray(p, float); n = p.size; o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def score_encoder(model, enc, satmut, masks, L, device, cfg, feats_per_elem):
    """Return long DataFrames (feature_id, a_masked, a_full, mass) over given features per element."""
    rk, rf = [], []
    for e in satmut:
        keep = ~masks[e.elem_id]
        feats = feats_per_elem[e.elem_id]
        if feats is None or len(feats) == 0:
            continue
        I = ism.feature_sensitivity(model, enc, e.seq, feats, L, device,
                                    alts=cfg.ism.alts, summary=cfg.ism.summary, batch=int(cfg.ism.batch))
        E = np.asarray(e.E_mean, float)
        n = min(len(E), len(keep))
        k = keep[:n]
        if k.sum() < 10:
            continue
        for f in feats:
            If = np.asarray(I[int(f)], float)[:n]
            rk.append((int(f), e.elem_id, pw(If[k], E[:n][k]), float(If[k].sum() + 1e-9)))
            rf.append((int(f), e.elem_id, pw(If, E[:n]), float(If.sum() + 1e-9)))
    cols = ["feature_id", "elem_id", "a", "mass"]
    return pd.DataFrame(rk, columns=cols), pd.DataFrame(rf, columns=cols)


def agg(df):
    out = []
    for f, g in df.groupby("feature_id"):
        out.append((int(f), causal.aggregate_A_f(list(g.a), list(g.mass)), len(g)))
    return pd.DataFrame(out, columns=["feature_id", "A_f", "n_active_e"])


def main():
    cfg = boot()
    L = chosen_layer(cfg); device = cfg.device
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    satmut = load_satmut(cfg)
    masks = {e.elem_id: repeat_mask(e.seq) for e in satmut}
    frac_masked_pos = float(np.mean([masks[e.elem_id][:len(e.E_mean)].mean() for e in satmut]))
    log.info("repeat-masked positions: mean %.1f%% per element", 100 * frac_masked_pos)

    # active features per element (shared between real test and the null's restriction set)
    active = {}
    for e in satmut:
        _, _, a = ism.element_activation(model, sae, e.seq, L, device)
        active[e.elem_id] = a

    log.info("scoring real SAE features (masked + full)...")
    rk, rf = score_encoder(model, sae, satmut, masks, L, device, cfg, active)
    A_masked = agg(rk); A_full = agg(rf)

    R = int(cfg.causal.null_R)
    enc = RandomFeatureSAE(model.d_model, R, sae.b_dec.detach(), device, seed=cfg.seed)
    null_feats = {e.elem_id: np.arange(R) for e in satmut}
    log.info("scoring %d random-feature null (masked + full)...", R)
    nk, nf = score_encoder(model, enc, satmut, masks, L, device, cfg, null_feats)
    nullA_masked = agg(nk).A_f.to_numpy(); nullA_full = agg(nf).A_f.to_numpy()

    res = {"layer": L, "frac_masked_positions": frac_masked_pos, "null_R": R}
    for tag, Adf, nullA in (("masked", A_masked, nullA_masked), ("full", A_full, nullA_full)):
        thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
        Av = Adf.A_f.to_numpy()
        p = np.array([causal.empirical_pvalue(a, nullA) for a in Av])
        q = bh_fdr(p)
        res[tag] = {"n_features": int(len(Av)), "thr": float(thr),
                    "fraction_above": float(np.mean(Av > thr)),
                    "n_above": int((Av > thr).sum()),
                    "median_A_f": float(np.median(Av)),
                    "n_fdr_q05": int((q < 0.05).sum()),
                    "n_fdr_q10": int((q < 0.10).sum())}
        log.info("%s: thr=%.3f fraction=%.3f (n_above=%d) FDR q05=%d median_A=%.3f",
                 tag, thr, res[tag]["fraction_above"], res[tag]["n_above"],
                 res[tag]["n_fdr_q05"], res[tag]["median_A_f"])
    (art / f"repeatmask_layer{L}.json").write_text(json.dumps(res, indent=2))

    m, fu = res["masked"], res["full"]
    body = f"""# Phase 10 — Repeat-masking control

Layer L={L}. Per-element Alu/low-complexity positions masked (mean **{100*frac_masked_pos:.1f}%** of
positions removed). Same ISM pass scored on kept positions only; null + FDR recomputed under masking.

| | causal fraction | n above thr | thr | FDR q<0.05 | median A_f |
| --- | --- | --- | --- | --- | --- |
| full (unmasked) | {fu['fraction_above']:.3f} | {fu['n_above']} | {fu['thr']:.3f} | {fu['n_fdr_q05']} | {fu['median_A_f']:.3f} |
| **repeat-masked** | **{m['fraction_above']:.3f}** | {m['n_above']} | {m['thr']:.3f} | {m['n_fdr_q05']} | {m['median_A_f']:.3f} |

**Reading.** If the masked fraction stays well above the 5% chance rate and FDR keeps many features,
the aggregate causal signal is genuinely regulatory, not a repeat artifact (headline holds/strengthens).
If it collapses toward chance, the honest headline becomes "the aggregate signal is largely
repeat-driven" — a different but valid finding.

**CHECKPOINT 10.** Decide which way it landed and how it reframes the headline.
"""
    write_report(cfg, "phase10", body)
    print(f"[phase10] full fraction={fu['fraction_above']:.3f} (FDR {fu['n_fdr_q05']}) -> "
          f"masked fraction={m['fraction_above']:.3f} (FDR {m['n_fdr_q05']}); "
          f"masked {100*frac_masked_pos:.0f}% of positions")


if __name__ == "__main__":
    main()
