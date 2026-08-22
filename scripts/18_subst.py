#!/usr/bin/env python
"""Substitution-specific causal test (reviewer request: not position-averaged).

The headline test averages ISM sensitivity over the three alternative bases at each position and
compares it to the position-averaged measured effect. That asks "is the feature sensitive where
mutations matter?" but not "is it sensitive to the substitutions that matter?" — a feature can be
sensitive at a high-effect position and yet respond to the wrong allele, and position-averaging
scores those identically (demonstrated in `tests/test_causal.py`).

Here we keep both the ISM sensitivity and the assay effects resolved per (position, alternative
base) and score over those pairs, which also triples the sample per element. One ISM pass yields
both scorings, so the comparison is exactly paired — same features, same elements, same forward
passes. Aggregation is at locus level (21 loci; see `data.locus_groups`).

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/18_subst.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, path, write_report

log = get_logger("subst")


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 6


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _cached_subst(cfg, model, encoder, satmut, layer, tag):
    p = path(cfg, "artifacts", f"subst_{tag}_layer{layer}.parquet")
    if p.exists():
        log.info("reusing cached %s", p.name)
        return pd.read_parquet(p)
    log.info("running per-alt ISM for %s ...", tag)
    df = controls.compute_alignments_subst(cfg, model, encoder, satmut, layer)
    df.to_parquet(p)
    return df


def _agg(align, col, gmap, min_active):
    """Aggregate one alignment column to per-feature A_f at locus level.

    Select the wanted column BEFORE renaming: the frame carries both `a_f` and `a_f_subst`, so
    renaming first would produce two columns called `a_f` and the downstream groupby would receive
    a DataFrame where it expects a Series.
    """
    d = align[["feature_id", "elem_id", col, "mass"]].rename(columns={col: "a_f"})
    feats = controls.aggregate_features(d, min_active, groups=gmap)
    return feats.set_index("feature_id")["A_f"]


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    ma = int(cfg.causal.min_active_elements)
    log.info("subst: layer %d, %d measurements over %d loci", L, len(satmut), len(set(gmap.values())))

    align = _cached_subst(cfg, model, sae, satmut, L, "topk_seed0")
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         sae.b_dec.detach(), cfg.device, seed=cfg.seed)
    null_align = _cached_subst(cfg, model, null_enc, satmut, L, "null")

    out = {"layer": L, "n_measurements": len(satmut), "n_loci": len(set(gmap.values()))}
    A = {}
    for level, col in (("position", "a_f"), ("substitution", "a_f_subst")):
        A_real = _agg(align, col, gmap, ma)
        A_null = _agg(null_align, col, gmap, 1).to_numpy()
        thr = causal.null_threshold(A_null, float(cfg.causal.percentile))
        p = np.array([causal.empirical_pvalue(a, A_null) for a in A_real.to_numpy()])
        q = bh_fdr(p)
        A[level] = A_real
        out[level] = {
            "threshold": float(thr),
            "causal_fraction": float((A_real.to_numpy() > thr).mean()),
            "n_above_thr": int((A_real.to_numpy() > thr).sum()),
            "n_features": int(len(A_real)),
            "n_fdr_q05": int((q < 0.05).sum()),
            "median_A_f": float(np.nanmedian(A_real.to_numpy())),
        }
        log.info("[%s-level] thr=%.3f frac=%.3f q<0.05=%d median=%.4f", level, thr,
                 out[level]["causal_fraction"], out[level]["n_fdr_q05"],
                 out[level]["median_A_f"])

    common = A["position"].index.intersection(A["substitution"].index)
    out["spearman_position_vs_substitution"] = causal.spearman_rho(
        A["position"].loc[common].to_numpy(), A["substitution"].loc[common].to_numpy())
    log.info("agreement between the two scorings: Spearman=%.3f",
             out["spearman_position_vs_substitution"])

    (art / f"subst_layer{L}.json").write_text(json.dumps(out, indent=2))
    _write(cfg, L, out)
    print(f"[phase16] substitution-level frac={out['substitution']['causal_fraction']:.3f} "
          f"(position-level {out['position']['causal_fraction']:.3f}); "
          f"Spearman between scorings={out['spearman_position_vs_substitution']:.3f}")


def _write(cfg, L, out):
    rows = [[lvl, f"{out[lvl]['threshold']:.3f}", f"{out[lvl]['causal_fraction']:.3f}",
             out[lvl]["n_above_thr"], out[lvl]["n_features"], out[lvl]["n_fdr_q05"],
             f"{out[lvl]['median_A_f']:.4f}"] for lvl in ("position", "substitution")]
    body = f"""# Phase 16 — Substitution-specific causal test

Layer L={L}. **{out['n_measurements']} measurements over {out['n_loci']} loci**, locus-level
aggregation, same SAE / same ISM pass for both rows.

{md_table(["scoring", "null thr", "causal fraction", "n above thr", "n features", "FDR q<0.05",
           "median A_f"], rows)}

Spearman between the two per-feature scorings: **{out['spearman_position_vs_substitution']:.3f}**.

**What the two rows mean.** *Position* averages ISM sensitivity over the three alternative bases
and compares it to the position-averaged measured effect — "is the feature sensitive where
mutations matter?". *Substitution* keeps both resolved per (position, alt base) — "is the feature
sensitive to the substitutions that matter?". The second is strictly the harder question: a
feature sensitive at a high-effect position but responding to the wrong allele scores well on the
first and poorly on the second (see `tests/test_causal.py::test_precision_weighted_alt_...`).

**Reading.** If the substitution-level fraction holds up, the aggregate signal is allele-specific,
not merely positional, and the headline strengthens. If it drops toward the null, the model tracks
*which positions* are mutable but not *which change* matters — a real and reportable limit on how
mechanistically the features can be read.

**CHECKPOINT 16.** Confirm which reading the numbers support.
"""
    write_report(cfg, "phase16", body)


if __name__ == "__main__":
    main()
