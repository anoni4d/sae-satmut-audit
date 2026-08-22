#!/usr/bin/env python
"""P1 — build SAE corpus + satmut validation set; write reports/phase1.md (incl. power check)."""
import numpy as np
import pandas as pd

from _common import boot
from dna_interp import data as D
from dna_interp.utils import md_table, write_report


def main():
    cfg = boot()
    corpus = D.build_corpus(cfg)
    satmut = D.load_satmut(cfg)

    # corpus stats
    md = pd.DataFrame([{"cls": w.cls, "len": len(w.seq),
                        "n_motif": len(w.planted)} for w in corpus])
    by_cls = md.groupby("cls").agg(n=("len", "size"), mean_len=("len", "mean"),
                                   mean_motifs=("n_motif", "mean")).reset_index()
    total_tokens = int(md.len.sum())

    # satmut stats + power check
    n_e = len(satmut)
    pos_total = sum(e.L for e in satmut)
    nz = sum(int((e.E_mean > 0).sum()) for e in satmut)
    e_all = np.concatenate([e.E_mean for e in satmut])
    power_ok = (n_e >= 20) and (nz >= 1000)

    body = f"""# Phase 1 — Data

## SAE corpus
- model context / window length: **{cfg.corpus.context_len} bp**
- total windows: **{len(corpus)}**, total tokens: **{total_tokens:,}**
- source: {"synthetic (planted motifs)" if not cfg.corpus.genome_fasta else "GRCh38 real"}

{md_table(["class", "n_windows", "mean_len", "mean_motifs/window"],
          [[r.cls, int(r.n), f"{r.mean_len:.0f}", f"{r.mean_motifs:.2f}"] for r in by_cls.itertuples()])}

> Full-run target is >=~1e8 cached activation vectors (scale `corpus.n_windows`); the value
> above is the current (smoke/dev) corpus.

## satmut validation ({cfg.satmut.source})
- elements: **{n_e}**, total positions: **{pos_total:,}**, positions with nonzero E: **{nz:,}**
- E_mean: min={e_all.min():.3f} median={np.median(e_all):.3f} max={e_all.max():.3f}

## Power check
- n_elements = {n_e} (Kircher 2019 has ~20 disease-associated elements)
- nonzero-effect positions = {nz}
- **Verdict: {"adequate for a first pass" if power_ok else "UNDERPOWERED"}** — {"proceed; revisit with eQTL secondary only if A_f CIs are wide" if power_ok else "add GTEx fine-mapped eQTLs or more published satmut elements (see SPEC §11), or accept explicit power limits"}

Artifacts: `artifacts/corpus.fasta`, `artifacts/corpus_meta.csv`, `artifacts/satmut.npz`

**CHECKPOINT 1** — confirm sizes / E distribution / power decision before extraction.
"""
    p = write_report(cfg, "phase1", body)
    print(f"[phase1] corpus={len(corpus)} windows, satmut={n_e} elements -> {p}")


if __name__ == "__main__":
    main()
