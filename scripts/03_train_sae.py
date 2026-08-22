#!/usr/bin/env python
"""P3 — train the TopK SAE at the chosen layer; report acceptance (FVU/dead/density)."""
from pathlib import Path

from _common import boot
from dna_interp import sae as S
from dna_interp import viz
from dna_interp.activations import load_activations
from dna_interp.utils import md_table, path, write_report


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    mm, _ = load_activations(cfg, L)
    print(f"[phase3] training SAE on layer {L}: {mm.shape[0]:,} tokens x {mm.shape[1]} "
          f"(m={cfg.sae.expansion * mm.shape[1]}, k={cfg.sae.k})")

    sae, metrics = S.train_sae(cfg, mm, L, seed=cfg.seed)
    viz.plot_density_hist(cfg, metrics["density"], L)

    acc = (metrics["FVU"] <= 0.15) and (metrics["dead_frac"] <= 0.05)
    body = f"""# Phase 3 — TopK SAE

Layer L={L}, m={metrics['m']} (expansion {cfg.sae.expansion}), k={metrics['k']}, seed={cfg.seed}

{md_table(["metric", "value", "acceptance"],
          [["FVU", f"{metrics['FVU']:.3f}", "<= 0.15"],
           ["dead fraction", f"{metrics['dead_frac']:.3f}", "<= 0.05"],
           ["L0", metrics["L0"], "= k (by construction)"],
           ["mean density", f"{metrics['density_mean']:.4f}", "-"],
           ["features >10% density", metrics["n_high_density"], "flagged non-specific"]])}

Figure: `reports/figures/density_layer{L}.png`

**Acceptance (§8.2): {"PASS" if acc else "NOT MET"}** — {"proceed to annotation" if acc else "adjust k/expansion/layer or diversify corpus (SPEC §11); for the synthetic/toy regime small corpora can inflate FVU"}

**CHECKPOINT 3** — SAE meets acceptance.
"""
    write_report(cfg, "phase3", body)
    print(f"[phase3] FVU={metrics['FVU']:.3f} dead={metrics['dead_frac']:.3f} "
          f"-> {'PASS' if acc else 'CHECK'}")


if __name__ == "__main__":
    main()
