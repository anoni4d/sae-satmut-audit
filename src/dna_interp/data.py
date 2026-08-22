"""Data layer: planted-motif definitions, SAE corpus, satmut validation set, genome FASTA.

Two regimes, one code path:
  * synthetic (default / smoke / method-validation): a deterministic genome with planted
    motifs; satmut elements whose per-bp effect vector E is *defined* from those motifs.
    This gives ground-truth causal structure so the whole pipeline can be validated.
  * real: GRCh38 via pyfaidx + ENCODE cCRE / GENCODE BEDs for the corpus, and the
    Kircher 2019 saturation-mutagenesis tables for validation (loaders below; activated
    by setting the corresponding paths in config).

The planted MOTIFS are the shared ground truth: the corpus plants them, the satmut effect
vectors are built from them, and the toy model (activations.py) builds detectors for them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import BASE_TO_IDX, BASES, get_logger, path

log = get_logger("data")

# ---------------------------------------------------------------------------
# Ground-truth motif catalogue (consensus + regulatory "effect weight").
# Loosely modelled on real TF motifs; weights are arbitrary positive effect sizes.
# ---------------------------------------------------------------------------
MOTIFS: dict[str, dict] = {
    "TATA": {"consensus": "TATAAA", "weight": 1.0},
    "GATA": {"consensus": "GATAAG", "weight": 0.8},
    "EBOX": {"consensus": "CACGTG", "weight": 0.9},
    "ETS":  {"consensus": "GGAAGT", "weight": 0.7},
    "SP1":  {"consensus": "GGGGCGGGG", "weight": 0.6},
}
MOTIF_NAMES = list(MOTIFS)


def pwm_from_consensus(consensus: str, p_match: float = 0.85) -> np.ndarray:
    """Return a [L,4] probability matrix peaked on the consensus base."""
    L = len(consensus)
    pwm = np.full((L, 4), (1.0 - p_match) / 3.0, dtype=np.float64)
    for j, ch in enumerate(consensus):
        pwm[j, BASE_TO_IDX[ch]] = p_match
    return pwm


def pwm_logodds(pwm: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    bg = bg if bg is not None else np.full(4, 0.25)
    return np.log(pwm + 1e-9) - np.log(bg + 1e-9)


# precompute log-odds matrices once
_MOTIF_LO = {name: pwm_logodds(pwm_from_consensus(m["consensus"])) for name, m in MOTIFS.items()}


def seq_to_idx(seq: str) -> np.ndarray:
    """ACGT string -> int8 indices; non-ACGT mapped to 0 (A)."""
    return np.array([BASE_TO_IDX.get(c, 0) for c in seq.upper()], dtype=np.int8)


def idx_to_seq(idx: np.ndarray) -> str:
    return "".join(BASES[i] for i in idx)


def motif_scan(seq_idx: np.ndarray, name: str) -> np.ndarray:
    """Per-start-position log-odds score of `name` along seq. length = len(seq)-L+1."""
    lo = _MOTIF_LO[name]
    L = lo.shape[0]
    n = len(seq_idx) - L + 1
    if n <= 0:
        return np.zeros(0)
    out = np.empty(n, dtype=np.float64)
    for s in range(n):
        out[s] = lo[np.arange(L), seq_idx[s:s + L]].sum()
    return out


# ---------------------------------------------------------------------------
# Synthetic genome with planted motifs
# ---------------------------------------------------------------------------
def _random_seq(n: int, gc: float, rng: np.random.Generator) -> np.ndarray:
    # base order A,C,G,T -> probs from gc
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return rng.choice(4, size=n, p=p).astype(np.int8)


def _plant(seq_idx: np.ndarray, name: str, start: int) -> list[tuple[int, int, str]]:
    cons = MOTIFS[name]["consensus"]
    m = seq_to_idx(cons)
    seq_idx[start:start + len(m)] = m
    return [(start, start + len(m), name)]


@dataclass
class CorpusWindow:
    seq_id: str
    seq: str
    cls: str           # ccre | genic | background
    planted: list      # list of [start, end, motif]


def build_corpus(cfg) -> list[CorpusWindow]:
    """Build (or load real) SAE-training windows. Returns list and caches to disk."""
    rng = np.random.default_rng(cfg.seed)
    ctx = int(cfg.corpus.context_len)
    n = int(cfg.corpus.n_windows)
    mix = cfg.corpus.class_mix

    if cfg.corpus.genome_fasta:
        return _build_corpus_real(cfg)

    counts = {c: int(round(n * float(w))) for c, w in mix.items()}
    counts["background"] += n - sum(counts.values())  # fix rounding

    windows: list[CorpusWindow] = []
    wid = 0
    gc = float(cfg.corpus.synthetic_gc)
    for cls, cnt in counts.items():
        for _ in range(cnt):
            s = _random_seq(ctx, gc, rng)
            planted: list = []
            if cls in ("ccre", "genic"):
                # plant 1-4 motifs; ccre richer in regulatory motifs
                k = rng.integers(2, 5) if cls == "ccre" else rng.integers(1, 3)
                for _m in range(int(k)):
                    name = MOTIF_NAMES[rng.integers(len(MOTIF_NAMES))]
                    L = len(MOTIFS[name]["consensus"])
                    pos = int(rng.integers(0, ctx - L))
                    planted += _plant(s, name, pos)
            windows.append(CorpusWindow(f"w{wid:06d}", idx_to_seq(s), cls, planted))
            wid += 1
    rng.shuffle(windows)
    _save_corpus(cfg, windows)
    log.info("built synthetic corpus: %d windows x %d bp (%s)", len(windows), ctx, counts)
    return windows


def _build_corpus_real(cfg) -> list[CorpusWindow]:
    """Real GRCh38 windows from cCRE/GENCODE BEDs + GC-matched background."""
    import pyfaidx  # noqa

    fa = pyfaidx.Fasta(cfg.corpus.genome_fasta, sequence_always_upper=True)
    ctx = int(cfg.corpus.context_len)
    n = int(cfg.corpus.n_windows)
    rng = np.random.default_rng(cfg.seed)
    windows: list[CorpusWindow] = []

    def grab(chrom, start):
        s = str(fa[chrom][start:start + ctx])
        return s if len(s) == ctx and s.count("N") < ctx * 0.05 else None

    beds = {"ccre": cfg.corpus.ccre_bed, "genic": cfg.corpus.gencode_gtf}
    per = {c: int(round(n * float(w))) for c, w in cfg.corpus.class_mix.items()}
    wid = 0
    for cls, bed in beds.items():
        if not bed:
            continue
        regions = pd.read_csv(bed, sep="\t", header=None, comment="#",
                              usecols=[0, 1, 2], names=["chrom", "start", "end"])
        regions = regions.sample(min(len(regions), per.get(cls, 0)), random_state=cfg.seed)
        for _, r in regions.iterrows():
            mid = (int(r.start) + int(r.end)) // 2
            s = grab(r.chrom, max(0, mid - ctx // 2))
            if s:
                windows.append(CorpusWindow(f"w{wid:06d}", s, cls, []))
                wid += 1
    # GC-matched random background
    chroms = [c for c in fa.keys() if "_" not in c and c not in ("chrM", "MT")]
    while len([w for w in windows if w.cls == "background"]) < per.get("background", 0):
        chrom = chroms[rng.integers(len(chroms))]
        clen = len(fa[chrom])
        if clen <= ctx:
            continue
        s = grab(chrom, int(rng.integers(0, clen - ctx)))
        if s:
            windows.append(CorpusWindow(f"w{wid:06d}", s, "background", []))
            wid += 1
    rng.shuffle(windows)
    _save_corpus(cfg, windows)
    log.info("built real corpus: %d windows", len(windows))
    return windows


def _save_corpus(cfg, windows: list[CorpusWindow]) -> None:
    fa = path(cfg, "artifacts", "corpus.fasta")
    meta = []
    with open(fa, "w") as fh:
        for w in windows:
            fh.write(f">{w.seq_id} {w.cls}\n{w.seq}\n")
            meta.append({"seq_id": w.seq_id, "cls": w.cls, "len": len(w.seq),
                         "planted": json.dumps(w.planted)})
    pd.DataFrame(meta).to_csv(path(cfg, "artifacts", "corpus_meta.csv"), index=False)


def load_corpus(cfg) -> list[CorpusWindow]:
    fa = path(cfg, "artifacts", "corpus.fasta")
    meta = path(cfg, "artifacts", "corpus_meta.csv")
    if not fa.exists():
        return build_corpus(cfg)
    md = pd.read_csv(meta).set_index("seq_id")
    out = []
    sid = seq = None
    with open(fa) as fh:
        for line in fh:
            if line.startswith(">"):
                if sid is not None:
                    out.append(CorpusWindow(sid, seq, md.loc[sid, "cls"],
                                            json.loads(md.loc[sid, "planted"])))
                sid = line[1:].split()[0]
                seq = ""
            else:
                seq += line.strip()
        if sid is not None:
            out.append(CorpusWindow(sid, seq, md.loc[sid, "cls"],
                                    json.loads(md.loc[sid, "planted"])))
    return out


# ---------------------------------------------------------------------------
# Saturation-mutagenesis validation elements
# ---------------------------------------------------------------------------
@dataclass
class SatmutElement:
    elem_id: str
    seq: str
    E_mean: np.ndarray         # E[p] = mean over 3 alts of |effect|  (default GT)
    E_signed: np.ndarray       # mean over 3 alts of signed effect
    E_max: np.ndarray          # max over 3 alts of |effect|
    planted: list = field(default_factory=list)
    # [L, 4] per-substitution effect magnitude, column = alt base index (BASE_TO_IDX order).
    # The reference allele's own column is 0. Retained (rather than only its position-average)
    # so the causal test can be scored per SUBSTITUTION, not just per position — averaging over
    # the three alts discards which specific change at a position actually matters.
    E_alt: np.ndarray | None = None

    @property
    def L(self) -> int:
        return len(self.seq)


def _effect_from_motifs(seq_idx: np.ndarray, planted: list) -> np.ndarray:
    """Per-bp, per-alt effect tensor [L,4]: how much disrupting base p to alt b changes
    regulatory activity. Effect at p is driven by the planted motif covering p, scaled by
    how far the alt moves away from the consensus log-odds. Outside motifs ~ 0."""
    L = len(seq_idx)
    eff = np.zeros((L, 4), dtype=np.float64)
    for start, end, name in planted:
        lo = _MOTIF_LO[name]
        w = MOTIFS[name]["weight"]
        ref_score = lo[np.arange(end - start), seq_idx[start:end]].sum()
        for off, p in enumerate(range(start, end)):
            ref_b = seq_idx[p]
            for b in range(4):
                # score change if base p were b (single-position swap within the motif)
                delta = lo[off, b] - lo[off, ref_b]
                # disruptive (negative delta) reduces activity -> magnitude of effect
                eff[p, b] += w * abs(delta)
    return eff


def build_synthetic_satmut(cfg) -> list[SatmutElement]:
    rng = np.random.default_rng(cfg.seed + 777)
    n = int(cfg.satmut.n_synthetic)
    L = int(cfg.satmut.element_len)
    gc = float(cfg.corpus.synthetic_gc)
    out = []
    for i in range(n):
        s = _random_seq(L, gc, rng)
        planted: list = []
        k = int(rng.integers(2, 5))
        for _ in range(k):
            name = MOTIF_NAMES[rng.integers(len(MOTIF_NAMES))]
            mlen = len(MOTIFS[name]["consensus"])
            pos = int(rng.integers(0, L - mlen))
            planted += _plant(s, name, pos)
        eff = _effect_from_motifs(s, planted)              # [L,4]
        # mask the reference allele (no effect for "mutating" to itself)
        for p in range(L):
            eff[p, s[p]] = 0.0
        alts = eff.sum(1) / 3.0                              # mean over 3 alts of |effect|
        e_mean = alts
        e_signed = -eff.sum(1) / 3.0                         # disruption = negative
        e_max = eff.max(1)
        out.append(SatmutElement(f"e{i:03d}", idx_to_seq(s), e_mean, e_signed, e_max, planted,
                                 E_alt=eff))
    _save_satmut(cfg, out)
    log.info("built %d synthetic satmut elements (len %d)", n, L)
    return out


def load_kircher(cfg) -> list[SatmutElement]:
    """Parse Kircher 2019 satmut tables from cfg.satmut.kircher_dir.

    Supports the kircherlab portal / GitHub combined table `elements.tsv[.gz]`
    (cols: Chrom,Pos,Ref,Alt,Barcodes,DNA,RNA,Coefficient,pValue,Element,Release) as well as
    per-element tsv/csv files. Keeps SNV substitutions (drops 1-bp deletions `Alt='-'`),
    filters to the requested genome release, and aggregates the 3 alt alleles per position into
    E_mean/E_signed/E_max. Effect column = Coefficient (log2 expression effect)."""
    d = Path(cfg.satmut.kircher_dir)
    release = cfg.satmut.get("release", "GRCh38")
    combined = (list(d.glob("elements.tsv.gz")) + list(d.glob("elements.tsv"))
                + list(d.glob("*elements*.tsv*")))
    out: list[SatmutElement] = []
    if combined:
        df = pd.read_csv(combined[0], sep="\t", low_memory=False)
        df = _filter_kircher(df, release)
        out.extend(_parse_kircher_frame(df, combined[0].stem))
    else:
        files = sorted(d.glob("*.tsv")) + sorted(d.glob("*.txt")) + sorted(d.glob("*.csv"))
        if not files:
            raise FileNotFoundError(
                f"No Kircher tables in {d}. Download elements.tsv.gz from "
                "github.com/kircherlab/MPRA_SaturationMutagenesis/tree/master/data "
                "(or the satMutMPRA portal), or use satmut.source=synthetic.")
        for f in files:
            sep = "," if f.suffix == ".csv" else "\t"
            df = _filter_kircher(pd.read_csv(f, sep=sep), release)
            out.extend(_parse_kircher_frame(df, f.stem))
    out = [e for e in out if e.L >= 10]
    _save_satmut(cfg, out)
    log.info("loaded %d Kircher satmut elements (release %s)", len(out), release)
    return out


def _filter_kircher(df: pd.DataFrame, release: str) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "release" in cols:
        df = df[df[cols["release"]].astype(str).str.upper() == release.upper()]
    if "alt" in cols:
        df = df[df[cols["alt"]].astype(str).str.upper().isin(list("ACGT"))]   # drop deletions
    return df


def _parse_kircher_frame(df: pd.DataFrame, default_id: str) -> list[SatmutElement]:
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    pos_c = pick("position", "pos", "start")
    ref_c = pick("ref", "reference", "ref_allele")
    alt_c = pick("alt", "alt_allele", "variant")
    eff_c = pick("coefficient", "value", "log2foldchange", "log2fc", "effect",
                 "expression_log2", "score")
    elem_c = pick("element", "region", "name")
    seq_c = pick("ref_base", "refbase")  # not a full sequence; reconstructed below
    if None in (pos_c, ref_c, alt_c, eff_c):
        raise ValueError(f"{default_id}: cannot find position/ref/alt/effect columns in {list(df.columns)}")

    groups = df.groupby(elem_c) if elem_c else [(default_id, df)]
    elements = []
    for ename, g in groups:
        g = g.copy()
        g["_p"] = g[pos_c].astype(int)
        p0 = g["_p"].min()
        L = int(g["_p"].max() - p0 + 1)
        ref_base = {int(r[pos_c]) - p0: str(r[ref_c]).upper()[0] for _, r in g.iterrows()}
        seq = "".join(ref_base.get(i, "A") for i in range(L))
        eff = np.zeros((L, 4))
        for _, r in g.iterrows():
            p = int(r[pos_c]) - p0
            b = BASE_TO_IDX.get(str(r[alt_c]).upper()[0])
            if b is not None:
                eff[p, b] = abs(float(r[eff_c]))
        signed = np.zeros((L, 4))
        for _, r in g.iterrows():
            p = int(r[pos_c]) - p0
            b = BASE_TO_IDX.get(str(r[alt_c]).upper()[0])
            if b is not None:
                signed[p, b] = float(r[eff_c])
        nz = (eff > 0).sum(1).clip(min=1)
        elements.append(SatmutElement(
            elem_id=str(ename), seq=seq,
            E_mean=eff.sum(1) / nz, E_signed=signed.sum(1) / nz, E_max=eff.max(1),
            planted=[], E_alt=eff))
    return elements


def load_satmut(cfg) -> list[SatmutElement]:
    if cfg.satmut.source == "kircher":
        return load_kircher(cfg)
    return build_synthetic_satmut(cfg)


# ---------------------------------------------------------------------------
# Locus grouping — the independent unit of the validation set.
#
# Kircher assays several conditions / cell lines of the SAME element and reports them as
# separate rows: TERT x4 cell lines, SORT1 x3, LDLR / PKLR / ZRS x2. Those share a
# byte-identical assayed sequence and near-identical effect vectors (r up to 0.985), so the
# 29 measurements are only 21 independent loci. Treating them as independent (a) leaks the
# held-out sequence into training under leave-one-*element*-out CV, and (b) over-weights
# those loci 2-4x when aggregating A_f. Group by sequence hash rather than by parsing names:
# no curation, and directly checkable against the data.
# ---------------------------------------------------------------------------
def _locus_name(ids: list[str]) -> str:
    """Name a group by the longest common prefix of its member ids (TERT-GAa/GBM/... -> TERT)."""
    ids = sorted(ids)
    if len(ids) == 1:
        return ids[0]
    import os

    p = os.path.commonprefix(ids).rstrip("-._ ")
    return p or ids[0]


def locus_groups(elems: list[SatmutElement]) -> dict[str, str]:
    """Map elem_id -> locus_id, grouping measurements that share an identical assayed sequence."""
    import hashlib

    by_seq: dict[str, list[str]] = {}
    for e in elems:
        h = hashlib.sha1(e.seq.upper().encode()).hexdigest()
        by_seq.setdefault(h, []).append(e.elem_id)
    out: dict[str, str] = {}
    for members in by_seq.values():
        name = _locus_name(members)
        for m in members:
            out[m] = name
    return out


def locus_index(elems: list[SatmutElement]) -> np.ndarray:
    """Integer group label per element, aligned to `elems` — for group-aware CV splitters."""
    g = locus_groups(elems)
    order = {name: i for i, name in enumerate(sorted(set(g.values())))}
    return np.array([order[g[e.elem_id]] for e in elems], dtype=int)


def _save_satmut(cfg, elems: list[SatmutElement]) -> None:
    np.savez(
        path(cfg, "artifacts", "satmut.npz"),
        ids=np.array([e.elem_id for e in elems]),
        seqs=np.array([e.seq for e in elems]),
        E_mean=np.array([e.E_mean for e in elems], dtype=object),
        E_signed=np.array([e.E_signed for e in elems], dtype=object),
        E_max=np.array([e.E_max for e in elems], dtype=object),
        E_alt=np.array([e.E_alt if e.E_alt is not None else np.zeros((0, 4)) for e in elems],
                       dtype=object),
        planted=np.array([json.dumps(e.planted) for e in elems]),
        allow_pickle=True,
    )
