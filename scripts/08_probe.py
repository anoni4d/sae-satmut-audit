#!/usr/bin/env python
"""Probe-vs-SAE contrast: is the per-position satmut effect LINEARLY decodable and STABLE from the
residual stream, even though unsupervised SAE features are not identifiable?

For each satmut measurement we take per-position activations h[p] in R^d (layer L) as features and
the per-position effect E_e[p] as target, then cross-validate by holding out a whole LOCUS:
  - predictive Spearman(pred, E) on each held-out locus -> is the effect decodable & generalizing?
  - stability of the fitted weight direction across folds (mean pairwise cosine) -> is there a
    stable causal direction (unlike the seed-unstable SAE features)?

Folds are grouped by locus because Kircher assays several cell lines / timepoints of the same
element over a byte-identical sequence (TERT x4, SORT1 x3, LDLR/PKLR/ZRS x2; see
`data.locus_groups`). Leave-one-*element*-out would train on the held-out sequence itself. We
report both levels so the size of that leak is explicit.

Baselines: sequence composition (one-hot + local GC) and a stronger local k-mer indicator probe,
each given its own best alpha so the margin over them is conservative.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/08_probe.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import probe
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("probe")

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
KMER_K = 5


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def _sweep_alpha(blocks, groups, names):
    """Pick alpha on the locus-grouped (honest) fold structure; return both levels at that alpha."""
    best_alpha, best = None, -np.inf
    for a in ALPHAS:
        r = probe.cross_val_probe(blocks, probe.make_ridge(a), groups=groups, names=names)
        log.info("  alpha=%-6.0f locus-level median rho=%.3f (frac>0 %.2f, stab %.3f)",
                 a, r.median_rho, r.frac_pos, r.dir_stability_cos)
        if r.median_rho > best:
            best_alpha, best = a, r.median_rho
    paired = probe.paired_probe(blocks, groups=groups, names=names,
                               estimator=probe.make_ridge(best_alpha))
    paired["best_alpha"] = best_alpha
    return paired


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    n_loci = len(set(gmap.values()))
    log.info("probe: layer %d, d=%d, %d measurements over %d loci",
             L, model.d_model, len(satmut), n_loci)

    act_blocks, seq_blocks, kmer_blocks, names, groups = [], [], [], [], []
    for e in satmut:
        h = model.feature_input(e.seq, L).detach().cpu().numpy().astype(np.float64)  # [L, d]
        E = np.asarray(e.E_mean, dtype=np.float64)
        n = min(len(h), len(E))
        if n < 5:
            continue
        act_blocks.append((h[:n], E[:n]))
        seq_blocks.append((probe.seq_features(e.seq[:n]), E[:n]))
        kmer_blocks.append((probe.kmer_features(e.seq[:n], k=KMER_K).astype(np.float64), E[:n]))
        names.append(e.elem_id)
        groups.append(gmap[e.elem_id])
    log.info("usable measurements: %d over %d loci", len(act_blocks), len(set(groups)))

    out = {"layer": L, "d": int(model.d_model), "model": str(cfg.model.name),
           "n_measurements": len(act_blocks), "n_loci": len(set(groups)),
           "kmer_k": KMER_K, "alphas_swept": list(ALPHAS)}

    log.info("activation probe h[p]:")
    out["activation_probe"] = _sweep_alpha(act_blocks, groups, names)
    log.info("composition baseline (one-hot + GC):")
    out["composition_probe"] = _sweep_alpha(seq_blocks, groups, names)
    log.info("k-mer baseline (k=%d):", KMER_K)
    out["kmer_probe"] = _sweep_alpha(kmer_blocks, groups, names)

    # per-locus detail for the honest fold structure
    ap = probe.cross_val_probe(act_blocks, probe.make_ridge(out["activation_probe"]["best_alpha"]),
                               groups=groups, names=names)
    out["activation_probe"]["per_locus"] = {n: float(r) for n, r in zip(ap.fold_names, ap.rhos)}

    (art / f"probe_layer{L}.json").write_text(json.dumps(out, indent=2))
    _write_phase9(cfg, L, out)
    a_el, a_lo = out["activation_probe"]["element_level"], out["activation_probe"]["locus_level"]
    print(f"[phase9] activation probe: element-level rho={a_el['median_rho']:.3f} -> "
          f"locus-level rho={a_lo['median_rho']:.3f} (delta {out['activation_probe']['delta_median_rho']:+.3f}); "
          f"composition={out['composition_probe']['locus_level']['median_rho']:.3f}; "
          f"kmer={out['kmer_probe']['locus_level']['median_rho']:.3f}")


def _row(label, d):
    return [label, f"{d['median_rho']:.3f}", f"{d['mean_rho']:.3f}",
            f"{d['frac_pos']:.2f}", f"{d['dir_stability_cos']:.3f}"]


def _write_phase9(cfg, L, out):
    ap, cp, kp = out["activation_probe"], out["composition_probe"], out["kmer_probe"]
    hdr = ["probe", "median Spearman(pred,E)", "mean", "frac folds >0", "weight-dir stability (cos)"]
    locus_tbl = md_table(hdr, [
        _row(f"**{out['model']} activations** (h[p]∈R^{out['d']})", ap["locus_level"]),
        _row(f"k-mer baseline (k={out['kmer_k']}, local)", kp["locus_level"]),
        _row("sequence composition (one-hot+GC)", cp["locus_level"]),
    ])
    elem_tbl = md_table(hdr, [
        _row(f"{out['model']} activations", ap["element_level"]),
        _row(f"k-mer baseline (k={out['kmer_k']})", kp["element_level"]),
        _row("sequence composition", cp["element_level"]),
    ])
    body = f"""# Phase 9 — Supervised probe vs. SAE (is the causal effect linearly decodable & stable?)

Layer L={L}, d={out['d']}, model `{out['model']}`.
**{out['n_measurements']} satmut measurements over {out['n_loci']} independent loci**; ridge with
alpha swept over {out['alphas_swept']} and chosen on the locus-grouped folds
(activation alpha={ap['best_alpha']:.0f}, composition alpha={cp['best_alpha']:.0f},
k-mer alpha={kp['best_alpha']:.0f}).

## Locus-grouped CV (primary — folds hold out a whole locus)

{locus_tbl}

## Element-level CV (legacy — leaks duplicated loci; shown for comparison only)

{elem_tbl}

**Leakage delta** (locus-level minus element-level median rho): activations
**{ap['delta_median_rho']:+.3f}**, k-mer {kp['delta_median_rho']:+.3f},
composition {cp['delta_median_rho']:+.3f}.

Kircher reports several cell lines / timepoints per element over a byte-identical sequence
(TERT x4, SORT1 x3, LDLR/PKLR/ZRS x2). Leave-one-*element*-out therefore trains on the held-out
sequence — with effect vectors correlated up to r=0.985 — so the element-level row is optimistic.
The locus-grouped row is the number to quote.

**Reading.** The supervised probe lives on the *fixed* model (no SAE seed), so a high
weight-direction stability across folds means a stable causal direction exists in the
representation. Contrast with the SAE result (phase7/8): individual causal features are
seed-unstable (Jaccard 0.046) and their subspace only modestly beats a non-causal null.
- If the activation probe **generalizes (median rho>0, stable) and beats both baselines**, the
  causal information is linearly present and stable, and the failure is the *unsupervised
  decomposition*, not the model.
- If it is **≈0 / ≈baselines**, the layer does not linearly encode per-bp effect, so the SAE is
  not uniquely to blame — a weaker, model-level statement.

The k-mer probe is the stronger sequence control (local {out['kmer_k']}-mer indicator, the feature
class gapped-k-mer models consume): it bounds how much of the probe's signal is plain local
composition rather than learned representation.

**CHECKPOINT 9.** Which way did it land, and does it warrant the probe-vs-SAE framing?
"""
    write_report(cfg, "phase9", body)


if __name__ == "__main__":
    main()
