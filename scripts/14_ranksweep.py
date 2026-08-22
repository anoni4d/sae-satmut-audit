#!/usr/bin/env python
"""Item 1 (refinement): rank sweep of the supervised probe.

The paper's hinge claim is that the causal information is "concentrated in one recoverable
direction". The original evidence was a single full-rank ridge probe (rho~0.19 at L*=5) plus an MLP
that does not beat it -- which rules out *nonlinearity*, not *dimensionality*. This script tests
dimensionality directly.

For each satmut element we take per-position residual-stream activations h[p] in R^d at layer L*
and the per-position effect E[p]. A rank-r linear probe = PLS regression with r components
(supervised reduced-rank: components ordered by covariance with the target), fit leave-one-element-out.
We report held-out median Spearman(pred, E) as a function of r in {1,2,3,5,10}, full-rank ridge as the
ceiling reference (reproduces the paper's number), and the effective dimensionality (smallest r
reaching 90% of the achievable rho).

Decision logic (per work order):
  - rho plateaus at r=1  -> "one direction" claim stands; cite this sweep.
  - rho keeps climbing   -> signal is distributed across >=k dims; weaken the claim.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/14_ranksweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dna_interp import probe  # noqa: E402
from dna_interp.activations import load_model  # noqa: E402
from dna_interp.data import load_satmut, locus_groups  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402

# L* is the probe's best layer (a finding, not the SAE's chosen layer); override via env if needed.
LSTAR = int(__import__("os").environ.get("RANKSWEEP_LAYER", "5"))
RANKS = [1, 2, 3, 5, 10]


def ridge_loo(blocks, alpha=10.0, groups=None):
    r = probe.cross_val_probe(blocks, probe.make_ridge(alpha), groups=groups)
    return r.median_rho, r.frac_pos, list(r.rhos)


def pls_loo(blocks, r, groups=None):
    res = probe.cross_val_probe(blocks, probe.make_pls(r), groups=groups)
    return res.median_rho, res.frac_pos, list(res.rhos)


def main():
    cfg = load_config()
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    print(f"[ranksweep] L*={LSTAR}, d={model.d_model}, {len(satmut)} measurements "
          f"over {len(set(gmap.values()))} loci")

    blocks, groups = [], []
    for e in satmut:
        h = model.feature_input(e.seq, LSTAR).detach().cpu().numpy().astype(np.float64)
        E = np.asarray(e.E_mean, dtype=np.float64)
        n = min(len(h), len(E))
        if n >= 5:
            blocks.append((h[:n], E[:n]))
            groups.append(gmap[e.elem_id])
    print(f"[ranksweep] usable: {len(blocks)} measurements over {len(set(groups))} loci "
          f"(folds hold out a whole locus)")

    out = {"layer": LSTAR, "d": int(model.d_model), "n_measurements": len(blocks),
           "n_loci": len(set(groups)), "ranks": {}}
    for r in RANKS:
        med, fpos, _ = pls_loo(blocks, r, groups=groups)
        med_el, _, _ = pls_loo(blocks, r)
        out["ranks"][r] = {"median_rho": med, "frac_pos": fpos, "median_rho_element": med_el}
        print(f"[ranksweep] PLS rank r={r:2d}: locus-level rho={med:.4f} (frac>0 {fpos:.2f}) "
              f"| element-level {med_el:.4f}")

    rmed, rfpos, _ = ridge_loo(blocks, alpha=10.0, groups=groups)
    out["full_ridge"] = {"median_rho": rmed, "frac_pos": rfpos, "alpha": 10.0}
    print(f"[ranksweep] full-rank ridge (alpha=10): median rho={rmed:.4f} (frac>0 {rfpos:.2f})")

    rho1 = out["ranks"][1]["median_rho"]
    rho_max = max([out["ranks"][r]["median_rho"] for r in RANKS] + [rmed])
    out["rho_r1"] = rho1
    out["rho_max"] = rho_max
    out["gain_r1_to_max"] = rho_max - rho1
    out["r1_frac_of_max"] = rho1 / rho_max if rho_max else float("nan")
    # effective dim: smallest swept r reaching 90% of rho_max
    eff = None
    for r in RANKS:
        if out["ranks"][r]["median_rho"] >= 0.90 * rho_max:
            eff = r
            break
    out["eff_dim_90pct"] = eff
    print(f"[ranksweep] rho(r=1)={rho1:.4f}  rho_max={rho_max:.4f}  "
          f"r1/max={out['r1_frac_of_max']:.2f}  eff_dim(90%)={eff}")

    dest = Path(cfg.paths["artifacts"]) / f"ranksweep_layer{LSTAR}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[ranksweep] wrote {dest}")


if __name__ == "__main__":
    main()
