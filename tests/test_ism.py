import numpy as np
import torch

from dna_interp import ism
from dna_interp.activations import ToyGenomicModel
from dna_interp.sae import SAEConfig, TopKSAE


def _tiny_setup(seed=0):
    model = ToyGenomicModel(d_model=32, n_layers=2, device="cpu", max_length=64, seed=seed)
    rng = np.random.default_rng(seed)
    seqs = ["".join("ACGT"[i] for i in rng.integers(0, 4, 20)) for _ in range(40)]
    H = np.concatenate([model.feature_input(s, 1).numpy() for s in seqs], 0)
    sae = TopKSAE(SAEConfig(d_model=32, m=64, k=8, k_aux=16))
    sae.init_from_sample(H)
    sae.normalize_decoder()
    return model, sae, seqs


def test_ism_matches_bruteforce():
    """§8.4: batched feature_sensitivity must equal an explicit brute-force loop."""
    model, sae, seqs = _tiny_setup()
    seq = seqs[0]
    _, _, active = ism.element_activation(model, sae, seq, 1, "cpu")
    assert active.size > 0
    feats = active[:5]
    I = ism.feature_sensitivity(model, sae, seq, feats, 1, "cpu", alts="all", summary="max")
    for f in feats:
        ref = ism.ism_bruteforce(model, sae, seq, int(f), 1, "cpu", summary="max")
        assert np.allclose(I[int(f)], ref, atol=1e-4), f"feature {f} mismatch"


def test_ism_token_summary_matches_bruteforce():
    model, sae, seqs = _tiny_setup(seed=2)
    seq = seqs[1]
    _, _, active = ism.element_activation(model, sae, seq, 1, "cpu")
    feats = active[:3]
    I = ism.feature_sensitivity(model, sae, seq, feats, 1, "cpu", alts="all", summary="token")
    for f in feats:
        ref = ism.ism_bruteforce(model, sae, seq, int(f), 1, "cpu", summary="token")
        assert np.allclose(I[int(f)], ref, atol=1e-4)


def test_loglik_ism_shape_and_nonneg():
    model, _, seqs = _tiny_setup()
    I = ism.loglik_ism(model, seqs[0], alts="all")
    assert I.shape == (20,)
    assert (I >= 0).all()


def test_per_alt_matches_the_averaged_sensitivity():
    """return_per_alt exposes the tensor that is averaged away; the mean must reproduce I_f."""
    from dna_interp.utils import BASE_TO_IDX, BASES
    model, sae, seqs = _tiny_setup(seed=1)
    seq = seqs[0]
    _, _, active = ism.element_activation(model, sae, seq, 1, "cpu")
    feats = active[:4]
    I, per_alt, alt_base = ism.feature_sensitivity(model, sae, seq, feats, 1, "cpu",
                                                   alts="all", summary="max",
                                                   return_per_alt=True)
    for f in feats:
        assert np.allclose(per_alt[int(f)].mean(1), I[int(f)], atol=1e-9)

    # every slot names a real alternative base, never the reference
    assert alt_base.shape == (len(seq), 3)
    assert (alt_base >= 0).all()
    for p, ref in enumerate(seq):
        assert BASE_TO_IDX[ref] not in set(alt_base[p].tolist())
        assert sorted(alt_base[p].tolist()) == sorted(
            BASE_TO_IDX[b] for b in BASES if b != ref)
