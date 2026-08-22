#!/usr/bin/env python
"""Probe ceiling: is the modest rho~0.12 a layer-6 / linearity artifact, or a true ceiling?

(1) Linear ridge probe of per-position effect E from activation value h[p], swept across ALL layers
    (leave-one-element-out). Finds the best layer L*.
(2) Nonlinear (small-MLP) probe at L* as a capacity ceiling, vs the linear probe and vs sequence
    composition. If some layer or the MLP reaches rho ~0.3-0.5, the positive half of the result goes
    from "modest" to "strong".

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/11_probesweep.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import probe
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("probesweep")


def ridge_loo(blocks, alpha, groups=None):
    r = probe.cross_val_probe(blocks, probe.make_ridge(alpha), groups=groups)
    return r.median_rho, r.frac_pos


def mlp_loo(blocks, hidden=64, alpha=1e-3, seed=0, groups=None):
    r = probe.cross_val_probe(blocks, probe.make_mlp(hidden, alpha, seed), groups=groups)
    return r.median_rho, r.frac_pos


def main():
    cfg = boot()
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    N = model.n_layers
    log.info("probesweep: %d layers, %d measurements over %d loci",
             N, len(satmut), len(set(gmap.values())))

    # cache per-(element, layer) activations + targets
    E_by = {e.elem_id: np.asarray(e.E_mean, float) for e in satmut}
    layer_blocks = {L: [] for L in range(1, N + 1)}
    seq_blocks, groups = [], []
    for e in satmut:
        E = E_by[e.elem_id]
        for L in range(1, N + 1):
            h = model.feature_input(e.seq, L).detach().cpu().numpy().astype(np.float64)
            n = min(len(h), len(E))
            if n >= 5:
                layer_blocks[L].append((h[:n], E[:n]))
        seq_blocks.append((probe.seq_features(e.seq[:len(E)]), E[:len(E)]))
        groups.append(gmap[e.elem_id])
    log.info("cached activations for all layers")

    out = {"n_layers": N, "n_measurements": len(satmut), "n_loci": len(set(groups)),
           "alpha": 10.0, "by_layer": {}}
    for L in range(1, N + 1):
        rho, fpos = ridge_loo(layer_blocks[L], alpha=10.0, groups=groups)
        rho_el, _ = ridge_loo(layer_blocks[L], alpha=10.0)          # legacy, for the delta
        out["by_layer"][L] = {"linear_rho": rho, "frac_pos": fpos, "linear_rho_element": rho_el}
        log.info("layer %d: locus-level rho=%.3f (frac>0 %.2f) | element-level rho=%.3f",
                 L, rho, fpos, rho_el)

    best_L = max(out["by_layer"], key=lambda L: out["by_layer"][L]["linear_rho"])
    out["best_layer"] = int(best_L)
    log.info("best layer = %d (rho=%.3f); fitting MLP there...",
             best_L, out["by_layer"][best_L]["linear_rho"])
    mlp_rho, mlp_fpos = mlp_loo(layer_blocks[best_L], groups=groups)
    comp_rho, comp_fpos = ridge_loo(seq_blocks, alpha=10.0, groups=groups)
    out["mlp_best_layer"] = {"rho": mlp_rho, "frac_pos": mlp_fpos}
    out["composition"] = {"rho": comp_rho, "frac_pos": comp_fpos}

    art = Path(cfg.paths["artifacts"])
    (art / "probesweep.json").write_text(json.dumps(out, indent=2))

    rows = [[L, f"{out['by_layer'][L]['linear_rho']:.3f}", f"{out['by_layer'][L]['frac_pos']:.2f}",
             f"{out['by_layer'][L]['linear_rho_element']:.3f}"] for L in range(1, N + 1)]
    body = f"""# Phase 11 — Probe layer sweep + nonlinear ceiling

Linear ridge probe of per-position effect E from activation value h[p], swept across all {N}
layers; MLP probe at the best layer as a capacity ceiling.
**{out['n_measurements']} measurements over {out['n_loci']} independent loci**; folds hold out a
whole locus (leave-one-*element*-out is shown alongside — it trains on the held-out sequence for
the {out['n_measurements'] - out['n_loci']} duplicated measurements, so it is optimistic).

{md_table(["layer", "locus-level median rho", "frac loci >0", "element-level rho (leaky)"], rows)}

- **best layer L\\*={best_L}** (locus-level linear rho={out['by_layer'][best_L]['linear_rho']:.3f};
  element-level {out['by_layer'][best_L]['linear_rho_element']:.3f})
- **MLP at L\\***: rho={mlp_rho:.3f} (frac>0 {mlp_fpos:.2f}) — nonlinear ceiling
- composition baseline: rho={comp_rho:.3f}

**Reading.** If a layer or the MLP reaches rho ~0.3-0.5, the causal effect is richly decodable and
the modest L6 linear number was an artifact of layer/linearity → the positive half strengthens. If
all layers and the MLP stay low, that is a genuine ceiling → report the signal as real but weak,
and the contrast with SAE non-identifiability stands on stability, not magnitude.

**CHECKPOINT 11.** Does the ceiling change the framing/venue ambition?
"""
    write_report(cfg, "phase11", body)
    print(f"[phase11] best layer L*={best_L} linear rho={out['by_layer'][best_L]['linear_rho']:.3f}; "
          f"MLP rho={mlp_rho:.3f}; composition={comp_rho:.3f}")


if __name__ == "__main__":
    main()
