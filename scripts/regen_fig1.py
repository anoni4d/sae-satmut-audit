#!/usr/bin/env python
"""Regenerate Figure 1 for the rebuttal, addressing Reviewer 4.

Two defects were real. (a) The colours were mislabelled: the code drew ISM sensitivity in blue and
the measured effect in red, while the caption said orange and blue, and described an overlay where
the figure was two stacked panels. (b) The left panel layered three alpha-blended histograms over a
range where they all pile up, so they could not be told apart.

The numbers are unchanged from the submitted version (element-level A_f, the original null and
threshold); only the rendering is corrected. Output goes to rebut/figures/, which the rebuttal
paper's graphicspath searches before the original figure directory.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/regen_fig1.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from dna_interp import causal, ism, sae as S, viz  # noqa: E402
from dna_interp.activations import load_model  # noqa: E402
from dna_interp.data import load_satmut  # noqa: E402
from dna_interp.utils import load_config  # noqa: E402

ELEM, FEAT = "FOXE1", 2795          # the example used in the submitted Figure 1


def main():
    cfg = load_config()
    L = int((Path(cfg.paths["artifacts"]) / "chosen_layer.txt").read_text().strip())
    art = Path(cfg.paths["artifacts"])
    out = SimpleNamespace(paths={"reports": str(ROOT / "reports_real")})

    # ---- left panel: A_f distribution, same numbers, readable rendering ----
    feats = __import__("pandas").read_csv(art / f"causal_features_layer{L}.csv")
    null_A = np.load(art / f"null_A_layer{L}.npy")
    thr = causal.null_threshold(null_A, float(cfg.causal.percentile))
    A = feats.A_f.to_numpy()
    print(f"A_f: n={len(A)}, threshold={thr:.3f}, above={int((A>thr).sum())}")
    p = viz.plot_Af_distribution(out, A, null_A, thr, A_ll=-0.004)
    print("wrote", p)

    # ---- right panel: the ISM example, corrected colours and a true overlay ----
    model = load_model(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    e = next(x for x in load_satmut(cfg) if x.elem_id == ELEM)
    I = ism.feature_sensitivity(model, sae, e.seq, np.array([FEAT]), L, cfg.device,
                                alts=cfg.ism.alts, summary=cfg.ism.summary,
                                batch=int(cfg.ism.batch))
    p = viz.plot_ism_example(out, I[FEAT], np.asarray(e.E_mean, float), FEAT, ELEM)
    print("wrote", p)


if __name__ == "__main__":
    main()
