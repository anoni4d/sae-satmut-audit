#!/usr/bin/env python
"""P4 — feature annotation (max-activating examples, motif enrichment, class overlap)."""
from pathlib import Path

from _common import boot
from dna_interp import annotate, sae as S
from dna_interp.data import load_corpus
from dna_interp.utils import md_table, write_report


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    sae = S.load_sae(cfg, L, seed=cfg.seed)
    corpus = load_corpus(cfg)
    df = annotate.annotate_features(cfg, sae, L, corpus)

    counts = df.label.value_counts().to_dict()
    body = f"""# Phase 4 — Annotation

Layer L={L}, {len(df)} features.

{md_table(["label", "count"], [[k, v] for k, v in counts.items()])}

- features with significant motif (p<1e-4): **{int((df.motif_enrichment_p < 1e-4).sum())}**
- top motifs: {", ".join(sorted(set(df[df.top_motif != ''].top_motif))[:10])}

Artifact: `artifacts/features_layer{L}.csv`

**CHECKPOINT 4** — spot-check biological plausibility of top features.
"""
    write_report(cfg, "phase4", body)
    print(f"[phase4] annotated {len(df)} features -> {counts}")


if __name__ == "__main__":
    main()
