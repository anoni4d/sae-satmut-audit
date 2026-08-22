# Sparsity, Not Alignment

**A saturation-mutagenesis audit of sparse-autoencoder features in a genomic foundation model.**

This repository trains sparse autoencoders (SAEs) on the residual stream of genomic foundation
models (HyenaDNA, Caduceus) and tests, per feature, whether its in-silico-mutagenesis sensitivity
aligns with measured per-base-pair effects from saturation-mutagenesis MPRA (Kircher et al. 2019),
against a random-direction null with false-discovery control.

**Result.** Applied directly, the test reports a discovery: 18.2% of features clear the null's 95th
percentile against a 5% chance rate, 291 survive Benjamini–Hochberg correction, and the effect
persists under label shuffling, repeat masking, a log-likelihood baseline, locus subsampling, three
seeds, two SAE families and two architectures.

**That result is an artifact.** The procedure thresholds one tail but never checks that the excess is
one-sided, and it is not: 18.6% of features fall beyond the opposite threshold, the identical FDR
procedure applied to the lower tail returns *more* discoveries (324 vs 291), the two distributions
differ in scale rather than location (P(real > null) = 0.51), and the reported effect grows
monotonically with dictionary size (0.135 → 0.219 as m goes 1024 → 8192). What the threshold selects
is features whose alignment score is more *variable* than a dense random direction's — which sparser
units are mechanically. A split-half noise ceiling (Jaccard 0.044) further shows the accompanying
cross-seed and cross-family irreproducibility is indistinguishable from what estimation noise alone
produces at this ground-truth budget, so it is not attributable to the decomposition either.

We do **not** claim that no alignment exists: with 21 independent loci the direct test is
underpowered (Wilcoxon p = 0.55). "Not established" is the defensible statement.

The four diagnostics that expose this — symmetry, mirror-FDR, shift-vs-spread, and scaling with
dictionary size — are in `scripts/22_signal_audit.py`, and each is a few lines on top of an existing
pipeline.

---

## Repository layout

```
src/dna_interp/      the library
  data.py            corpus + Kircher satmut loaders, tokenization, genome FASTA
  activations.py     model backends (HyenaDNA, Caduceus, deterministic toy) + residual-stream hook
  sae.py             SAE families (TopK, Gated, JumpReLU) behind one interface; train loop, metrics
  ism.py             single-nt in-silico mutagenesis -> per-position feature sensitivity I_f
  causal.py          alignment metrics (precision-weighted), weighted-median A_f, null threshold
  controls.py        random-feature + sparsity-matched nulls, LL baseline, shuffle, seed robustness
  probe.py           shared supervised-probe machinery + locus-grouped cross-validation
  annotate.py        max-activating examples, motif enrichment, repeat mask, element-class overlap
  viz.py             figures
scripts/             one CLI per phase (01..22) + run_pipeline.py and analysis utilities
config/              default.yaml (smoke), real.yaml (HyenaDNA+GRCh38+Kircher), caduceus.yaml
tests/               causal math, ISM == brute force, data loaders
notebooks/           analysis.ipynb (annotation review, figures)
reports_real/        figures for the HyenaDNA run
reports_caduceus/    figures for the Caduceus run
```

### Analyses added in review

| script | what it does |
|---|---|
| `16_dictsweep.py` | dictionary-size sweep, expansion 4–32 x 3 seeds, at fixed k |
| `17_noiseceiling.py` | split-half noise ceiling: reproducibility attainable with the dictionary held fixed |
| `18_subst.py` | scores alignment per (position, alternative base) rather than position-averaged |
| `19_subspace_bio.py` | what the selected features' subspace responds to, vs a matched non-causal one |
| `20_matchednull.py` | sparsity-matched null: random directions thresholded to a real feature's firing density |
| `21_crossfamily_locus.py` | TopK vs Gated agreement, recomputed on independent loci |
| `22_signal_audit.py` | one-sidedness checks: tail symmetry, mirror-FDR, shift-vs-spread, scaling with m |
| `locus_numbers.py` | every reported number at both element and locus aggregation |
| `dictsweep_table.py` | sweep summary at both aggregation levels |
| `regen_fig1.py` | regenerates Figure 1 |

**Unit of analysis.** The Kircher panel reports several conditions per element over byte-identical
sequence, so its 29 rows are 21 independent loci. `data.locus_groups` groups by assayed sequence;
pass `groups=` to `controls.aggregate_features` and `probe.cross_val_probe` to aggregate and
cross-validate on that unit. Omitting it reproduces the earlier measurement-level numbers.

## Install
```bash
pip install -e ".[dev]"
pytest -q                       # causal math, ISM==brute force, loaders
```
Python 3.11+, PyTorch (CUDA). HyenaDNA loads via `transformers` (`trust_remote_code`).
**Caduceus** additionally needs `mamba-ssm` + `causal-conv1d` (CUDA kernels) — use a torch build with
prebuilt wheels (e.g. torch 2.4/cu121) in a separate env.

## Reproduce
```bash
# tiny end-to-end smoke on a deterministic toy model (no downloads, ~1 min)
python scripts/run_pipeline.py smoke=true

# the real HyenaDNA study (needs data; see below)
DNA_INTERP_CONFIG=config/real.yaml python scripts/run_pipeline.py n_seeds=3   # P1..P6
DNA_INTERP_CONFIG=config/real.yaml python scripts/06_deepdive.py              # FDR + seed robustness
DNA_INTERP_CONFIG=config/real.yaml python scripts/07_subspace.py             # subspace stability
DNA_INTERP_CONFIG=config/real.yaml python scripts/08_probe.py                # supervised probe
DNA_INTERP_CONFIG=config/real.yaml python scripts/09_sensprobe.py            # sensitivity-matched probe
DNA_INTERP_CONFIG=config/real.yaml python scripts/10_repeatmask.py           # repeat-masking control
DNA_INTERP_CONFIG=config/real.yaml python scripts/11_probesweep.py           # probe layer sweep
DNA_INTERP_CONFIG=config/real.yaml CF_FAMILIES="topk,gated" \
    python scripts/12_crossfamily.py sae.l1_coeff=0.3 sae.epochs=6           # cross-SAE-family
DNA_INTERP_CONFIG=config/real.yaml python scripts/13_subsample.py            # n-subsampling stability

# the cross-model (Caduceus) replication
DNA_INTERP_CONFIG=config/caduceus.yaml python scripts/run_pipeline.py n_seeds=3
```
Everything is config-driven with fixed seeds; activations/SAEs/ISM are cached to disk. Set
`DNA_INTERP_CONFIG` to pick the config; the paths in `config/*.yaml` are repo-relative placeholders —
point `paths.*` and `corpus.genome_fasta/ccre_bed/gencode_gtf` at your data.

## Data
- **Genome:** GRCh38 / hg38 FASTA (UCSC).
- **Corpus annotations:** ENCODE SCREEN cCRE BED, GENCODE basic genes BED.
- **Ground truth:** Kircher et al. 2019 saturation-mutagenesis MPRA tables (kircherlab satMutMPRA
  portal / GEO GSE126550), placed in `<data_raw>/kircher2019/`.

## Notes
- Single question, single model family: SAEs on a genomic foundation model, tested against
  saturation-mutagenesis ground truth. Out of scope: model zoo, genome-wide atlas, steering/design.
- **Disclosure:** implementation and analysis were carried out with assistance from an LLM coding
  assistant; all scientific decisions, controls and interpretation are the authors'.
- **License:** MIT (see `LICENSE`).

## Anonymity

This repository accompanies a double-blind submission. Author names, affiliations, prior-version
citations and venue references have been removed. It will be replaced by a non-anonymous version
after review.
