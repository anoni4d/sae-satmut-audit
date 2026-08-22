#!/usr/bin/env python
"""P2 — extract residual-stream activations across a layer sweep; pick L. Writes phase2.md."""
import numpy as np

from _common import boot
from dna_interp import viz
from dna_interp.activations import extract_activations, load_model, sweep_layers
from dna_interp.data import load_corpus
from dna_interp.utils import md_table, path, write_report


def main():
    cfg = boot()
    model = load_model(cfg)
    corpus = load_corpus(cfg)
    layers = sweep_layers(cfg, model)
    print(f"[phase2] model={type(model).__name__} d_model={model.d_model} "
          f"layers={model.n_layers}; sweeping {layers}")

    all_stats = []
    for L in layers:
        _, _, stats = extract_activations(cfg, model, corpus, L)
        all_stats.append(stats)
        mm = np.memmap(path(cfg, "artifacts", f"acts_layer{L}.memmap"),
                       dtype=np.float32, mode="r", shape=(stats["n_tokens"], stats["d_model"]))
        sample = np.asarray(mm[np.random.default_rng(0).choice(
            stats["n_tokens"], size=min(4000, stats["n_tokens"]), replace=False)])
        viz.plot_pca_scree(cfg, sample, L)

    viz.plot_layer_stats(cfg, all_stats)

    # heuristic pick: highest mean variance (richest residual stream) among swept layers
    chosen = max(all_stats, key=lambda s: s["mean_var"])["layer"]
    (path(cfg, "artifacts", "chosen_layer.txt")).write_text(str(chosen))

    body = f"""# Phase 2 — Activations

Model: **{type(model).__name__}** (d_model={model.d_model}, usable layers={model.n_layers})
Swept layers: {layers}

{md_table(["layer", "n_tokens", "mean_norm", "mean_var", "max_var"],
          [[s["layer"], f"{s['n_tokens']:,}", f"{s['mean_norm']:.3f}",
            f"{s['mean_var']:.4f}", f"{s['max_var']:.4f}"] for s in all_stats])}

Figures: `reports/figures/layer_stats.png`, `reports/figures/pca_scree_layer*.png`

**Chosen layer L = {chosen}** (max mean variance among swept layers; revisit at Checkpoint 3
if SAE quality differs). Set `activations.chosen_layer={chosen}` for downstream steps.

**CHECKPOINT 2** — confirm layer choice.
"""
    write_report(cfg, "phase2", body)
    print(f"[phase2] chosen layer L={chosen}")


if __name__ == "__main__":
    main()
