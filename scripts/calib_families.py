#!/usr/bin/env python
"""Calibrate Gated l1_coeff and JumpReLU l0_coeff on the real layer-6 activations so their average
L0 lands near TopK's k=64 (fair cross-family comparison). Short runs; prints FVU + L0 per setting."""
from pathlib import Path
import numpy as np
from _common import boot
from dna_interp import sae as S
from dna_interp.activations import load_activations
from dna_interp.utils import get_logger

log = get_logger("calib")


def main():
    cfg = boot()
    L = int(Path(cfg.paths["artifacts"], "chosen_layer.txt").read_text().strip())
    mm, _ = load_activations(cfg, L)
    cfg.sae.epochs = 6                     # full-length: converged FVU differs hugely from 2-epoch
    cfg.sae.ram_max_tokens = 4_000_000
    log.info("CALIB on layer %d, %dx%d (target iso-FVU~0.10 to match TopK)", L, mm.shape[0], mm.shape[1])

    runs = (
        [("gated", "l1_coeff", v, {}) for v in (0.3, 0.8, 2.0)] +
        [("jumprelu", "l0_coeff", v, {"theta_init": 0.05}) for v in (0.01, 0.03, 0.08)]
    )
    for fam, key, val, extra in runs:
        cfg.sae.family = fam
        for k, v in extra.items():
            cfg.sae[k] = v
        cfg.sae[key] = val
        sae, m = S.train_sae(cfg, mm, L, seed=0)
        log.info("CALIB-RESULT [%s] %s=%s theta_init=%s -> FVU=%.3f  L0=%.1f  dead=%.3f",
                 fam, key, val, extra.get("theta_init", "-"), m["FVU"], m["L0"], m["dead_frac"])


if __name__ == "__main__":
    main()
