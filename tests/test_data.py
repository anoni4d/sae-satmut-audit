import numpy as np
import pandas as pd

from dna_interp import data as D
from dna_interp.utils import load_config


def _smoke_cfg(tmp_path):
    cfg = load_config(["smoke=true"])
    for k in cfg.paths:
        cfg.paths[k] = str(tmp_path / k)
    return cfg


def test_motif_scan_recovers_planted():
    seq = "AAAA" + "TATAAA" + "AAAA"
    sc = D.motif_scan(D.seq_to_idx(seq), "TATA")
    assert int(np.argmax(sc)) == 4                 # motif starts at index 4
    assert sc.max() > 4.0


def test_seq_idx_roundtrip():
    s = "ACGTACGTTTGGCCA"
    assert D.idx_to_seq(D.seq_to_idx(s)) == s


def test_build_corpus_balance_and_planting(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    corpus = D.build_corpus(cfg)
    assert len(corpus) == cfg.corpus.n_windows
    assert all(len(w.seq) == cfg.corpus.context_len for w in corpus)
    classes = {w.cls for w in corpus}
    assert {"ccre", "genic", "background"} <= classes
    # planted motifs actually present in ccre windows
    ccre = [w for w in corpus if w.cls == "ccre" and w.planted]
    w = ccre[0]
    s, e, name = w.planted[0]
    assert w.seq[s:e] == D.MOTIFS[name]["consensus"]
    # determinism
    corpus2 = D.build_corpus(cfg)
    assert [w.seq for w in corpus] == [w.seq for w in corpus2]


def test_synthetic_satmut_structure(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    elems = D.build_synthetic_satmut(cfg)
    assert len(elems) == cfg.satmut.n_synthetic
    e = elems[0]
    assert e.E_mean.shape == (e.L,)
    assert (e.E_mean >= 0).all()
    # effect concentrated at planted motif positions
    in_motif = np.zeros(e.L, bool)
    for s, en, _ in e.planted:
        in_motif[s:en] = True
    assert e.E_mean[in_motif].mean() > e.E_mean[~in_motif].mean()


def test_kircher_realdata_if_present(tmp_path):
    """Validate the parser against the actual kircherlab elements.tsv.gz when it is present."""
    import pytest
    from pathlib import Path
    real = Path("data/raw/kircher2019/elements.tsv.gz")
    if not real.exists():
        pytest.skip("real Kircher data not downloaded")
    cfg = load_config(["satmut.source=kircher", "satmut.kircher_dir=data/raw/kircher2019"])
    elems = D.load_kircher(cfg)
    assert len(elems) >= 20                       # ~20 loci (+ time/haplotype variants)
    for e in elems[:5]:
        assert set(e.seq) <= set("ACGT")          # reconstructed reference is clean DNA
        assert e.E_mean.shape == (e.L,)
        assert (e.E_mean >= 0).all()
    # known high-signal element present
    assert any(x.elem_id == "SORT1" for x in elems)


def test_kircher_parser(tmp_path):
    df = pd.DataFrame({
        "element": ["X"] * 9,
        "position": [10, 10, 10, 11, 11, 11, 12, 12, 12],
        "ref": ["A"] * 9,
        "alt": ["C", "G", "T"] * 3,
        "log2FC": [-1.0, -0.5, -0.2, 0.0, 0.1, -0.1, -2.0, -1.5, -1.0],
    })
    elems = D._parse_kircher_frame(df, "X")
    assert len(elems) == 1
    e = elems[0]
    assert e.L == 3
    assert e.E_mean[2] > e.E_mean[1]               # position 12 most disruptive
