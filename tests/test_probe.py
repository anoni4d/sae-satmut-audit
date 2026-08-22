"""Locus grouping + grouped cross-validation.

The canary test below is the point of the whole module: with duplicated measurements in the
validation set, ungrouped CV scores high by memorising the held-out sequence from its twin,
while grouped CV correctly reports no signal.
"""
import numpy as np

from dna_interp import probe
from dna_interp.data import SatmutElement, locus_groups, locus_index


def _elem(eid, seq):
    L = len(seq)
    z = np.zeros(L)
    return SatmutElement(eid, seq, z, z, z)


def test_locus_groups_by_sequence_identity():
    seq_a, seq_b = "ACGTACGTAC", "TTTTGGGGCC"
    elems = [_elem("TERT-GBM", seq_a), _elem("TERT-HEK", seq_a),
             _elem("TERT-GSc", seq_a), _elem("BCL11A", seq_b)]
    g = locus_groups(elems)
    # identical sequences collapse to one locus, named by the common prefix
    assert g["TERT-GBM"] == g["TERT-HEK"] == g["TERT-GSc"] == "TERT"
    # a distinct sequence stays its own locus and keeps its own name
    assert g["BCL11A"] == "BCL11A"
    assert len(set(g.values())) == 2


def test_locus_groups_distinguishes_same_prefix_different_sequence():
    """MYCrs11986220 / MYCrs6983267 share a prefix but are different sequences -> 2 loci."""
    elems = [_elem("MYCrs11986220", "ACGTACGTAC"), _elem("MYCrs6983267", "ACGTACGTAA")]
    g = locus_groups(elems)
    assert g["MYCrs11986220"] != g["MYCrs6983267"]


def test_locus_index_alignment():
    seq_a, seq_b = "ACGTACGTAC", "TTTTGGGGCC"
    elems = [_elem("A.1", seq_a), _elem("B", seq_b), _elem("A.2", seq_a)]
    idx = locus_index(elems)
    assert idx.shape == (3,)
    assert idx[0] == idx[2] and idx[0] != idx[1]


def _blocks_with_a_duplicated_locus(seed=0):
    """Blocks 0,1 are the SAME element assayed twice; 2..5 are unrelated noise."""
    rng = np.random.default_rng(seed)
    d, n = 8, 60
    X_dup = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y_dup = X_dup @ w_true                      # perfectly linear in its own features
    dup = [(X_dup, y_dup), (X_dup, y_dup.copy())]
    other = [(rng.standard_normal((n, d)), rng.standard_normal(n)) for _ in range(4)]
    return dup + other


def test_grouped_cv_blocks_the_duplicate_leak():
    """Ungrouped CV memorises the twin; grouped CV does not. This is the LDLR/LDLR.2 case."""
    blocks = _blocks_with_a_duplicated_locus()
    groups = [0, 0, 1, 2, 3, 4]               # blocks 0 and 1 are one locus
    est = probe.make_ridge(1.0)

    ungrouped = probe.cross_val_probe(blocks, est, groups=None)
    grouped = probe.cross_val_probe(blocks, est, groups=groups)

    # fold 0 held out while its identical twin stays in training -> near-perfect recovery
    assert ungrouped.rhos[0] > 0.9
    # held out together, nothing in training explains it
    assert abs(grouped.rhos[0]) < 0.5
    assert grouped.median_rho < ungrouped.median_rho
    assert grouped.n_folds == 5 and ungrouped.n_folds == 6


def test_ungrouped_matches_legacy_leave_one_element_out():
    """groups=None must reproduce the pre-revision leave-one-element-out numbers exactly."""
    from dna_interp.causal import spearman_rho

    blocks = _blocks_with_a_duplicated_locus(seed=3)
    alpha = 10.0

    legacy = []
    for i in range(len(blocks)):
        Xtr = np.vstack([blocks[j][0] for j in range(len(blocks)) if j != i])
        ytr = np.concatenate([blocks[j][1] for j in range(len(blocks)) if j != i])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xs = (Xtr - mu) / sd
        yb = ytr.mean()
        w = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (ytr - yb))
        Xte, yte = blocks[i]
        legacy.append(spearman_rho(((Xte - mu) / sd) @ w + yb, yte))

    got = probe.cross_val_probe(blocks, probe.make_ridge(alpha), groups=None)
    assert np.allclose(got.rhos, np.array(legacy), atol=1e-12)


def test_paired_probe_reports_both_levels():
    blocks = _blocks_with_a_duplicated_locus()
    out = probe.paired_probe(blocks, groups=[0, 0, 1, 2, 3, 4])
    assert set(out) == {"element_level", "locus_level", "delta_median_rho"}
    assert out["locus_level"]["n_folds"] == 5
    assert out["element_level"]["n_folds"] == 6


def test_seq_and_kmer_features_shapes():
    seq = "ACGTACGTACGT"
    assert probe.seq_features(seq).shape == (len(seq), 5)
    km = probe.kmer_features(seq, k=3)
    assert km.shape == (len(seq), 64)
    assert np.allclose(km.sum(1), 1.0)         # exactly one k-mer indicator per position
