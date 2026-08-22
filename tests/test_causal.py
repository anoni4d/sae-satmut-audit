import numpy as np

from dna_interp import causal


def test_spearman_monotone_and_random():
    x = np.arange(50, dtype=float)
    assert causal.spearman_rho(x, x ** 2) > 0.99          # monotone -> ~1
    assert causal.spearman_rho(x, -x) < -0.99
    rng = np.random.default_rng(0)
    rhos = [causal.spearman_rho(rng.standard_normal(50), rng.standard_normal(50))
            for _ in range(200)]
    assert abs(np.mean(rhos)) < 0.1                        # random -> ~0


def test_spearman_degenerate():
    assert causal.spearman_rho(np.ones(10), np.arange(10.0)) == 0.0
    assert causal.spearman_rho([1, 2], [1, 2]) == 0.0      # too short


def test_synthetic_feature_known_sensitivity():
    """§8.4: a feature whose sensitivity IS the effect must align ~1; a random feature ~0."""
    rng = np.random.default_rng(1)
    L = 200
    E = np.abs(rng.standard_normal(L))
    I_causal = E + 0.05 * rng.standard_normal(L)           # sensitivity tracks effect
    I_random = rng.standard_normal(L)
    assert causal.spearman_rho(I_causal, E) > 0.9
    assert abs(causal.spearman_rho(I_random, E)) < 0.3


def test_weighted_median():
    assert causal.weighted_median(np.array([1.0, 2, 3]), np.array([1.0, 1, 1])) == 2.0
    # heavy weight pulls the median
    assert causal.weighted_median(np.array([1.0, 2, 3]), np.array([1.0, 1, 10])) == 3.0


def test_null_threshold_and_classify():
    null = np.linspace(-1, 1, 1001)
    thr = causal.null_threshold(null, 95)
    assert abs(thr - 0.9) < 0.05
    assert causal.classify(0.95, thr) == "causal"
    assert causal.classify(0.0, thr) == "correlational"
    assert causal.classify(np.nan, thr) == "inactive"


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 40)
    lo, hi = causal.bootstrap_ci(vals, n_boot=1000)
    assert lo < np.median(vals) < hi          # CI brackets the point estimate
    assert hi - lo < 0.2                       # reasonably tight for n=40
    assert all(np.isnan(x) for x in causal.bootstrap_ci([1.0]))


def test_metric_registry():
    rng = np.random.default_rng(0)
    L = 100
    E = np.zeros(L); E[20:26] = rng.uniform(1, 2, 6)     # effect on one motif block
    I = np.zeros(L); I[20:26] = rng.uniform(1, 2, 6)     # feature sensitive there
    # precision metric: feature sensitivity sits on high-effect positions -> strongly positive
    assert causal.get_metric("precision_weighted")(I, E) > 0.5
    # a feature sensitive only on zero-effect positions -> negative/low
    I2 = np.zeros(L); I2[60:66] = 1.0
    assert causal.get_metric("precision_weighted")(I2, E) < causal.get_metric("precision_weighted")(I, E)


def test_causal_fraction_and_pvalue():
    null = np.random.default_rng(0).standard_normal(1000) * 0.2
    A = np.array([0.9, 0.8, 0.0, -0.5, np.nan])
    thr = causal.null_threshold(null, 95)
    assert 0.0 <= causal.causal_fraction(A, thr) <= 1.0
    assert causal.empirical_pvalue(0.9, null) < 0.05
    assert causal.empirical_pvalue(0.0, null) > 0.1


def test_precision_weighted_alt_distinguishes_the_specific_substitution():
    """The point of substitution-level scoring: position-averaging cannot tell a feature that
    responds to the allele that matters from one that responds to the wrong allele."""
    L, n_alt = 60, 3
    E_alt = np.zeros((L, 4))
    # at position 20, only a change to base index 2 (G) has a large measured effect
    E_alt[20, 2] = 3.0
    # alt slots at position 20 carry bases 1, 2, 3
    alt_base = np.tile(np.array([1, 2, 3]), (L, 1))

    right = np.zeros((L, n_alt)); right[20, 1] = 1.0     # sensitive to the G substitution
    wrong = np.zeros((L, n_alt)); wrong[20, 0] = 1.0     # sensitive at the same POSITION, wrong allele

    a_right = causal.m_precision_weighted_alt(right, E_alt, alt_base)
    a_wrong = causal.m_precision_weighted_alt(wrong, E_alt, alt_base)
    assert a_right > 0.9
    assert a_right > a_wrong

    # position-averaged scoring collapses both to the same vector -> cannot separate them
    E_pos = E_alt.sum(1) / 3.0
    assert causal.m_precision_weighted(right.mean(1), E_pos) == \
           causal.m_precision_weighted(wrong.mean(1), E_pos)


def test_precision_weighted_alt_degenerate_inputs():
    alt_base = np.tile(np.array([1, 2, 3]), (10, 1))
    assert causal.m_precision_weighted_alt(np.zeros((10, 3)), np.zeros((10, 4)), alt_base) == 0.0
    assert causal.m_precision_weighted_alt(np.zeros((2, 3)), np.zeros((2, 4)),
                                           alt_base[:2]) == 0.0
