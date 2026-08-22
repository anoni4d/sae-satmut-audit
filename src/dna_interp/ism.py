"""Single-nucleotide in-silico mutagenesis (ISM).

Per (feature f, satmut element e) we build a per-position sensitivity vector I_f in R^{L_e}:
  for each position p, each alt base b' != x[p]: mutate, forward, read f's activation
  (sequence summary = max over token positions, or the token at p);  Δ = |z_f(x) - z_f(x')|.
  I_f[p] = mean_{b'} Δ.

ISM is run only for features active on the element (caller passes the active set), and base
activations are cached. A brute-force reference (`ism_bruteforce`) backs the unit tests.
"""
from __future__ import annotations

import numpy as np
import torch

from .utils import BASE_TO_IDX, BASES, get_logger

log = get_logger("ism")

_ALT = {b: [c for c in BASES if c != b] for b in BASES}


def _summary(z: torch.Tensor, summary: str, pos: int | None = None) -> torch.Tensor:
    """z: [L, m] post-TopK activations -> [m] sequence summary."""
    if summary == "token" and pos is not None:
        return z[pos]
    return z.max(0).values


@torch.no_grad()
def encode_seq(model, sae, seq: str, layer: int, device: str) -> torch.Tensor:
    h = model.feature_input(seq, layer).to(device)
    return sae.encode(h)                       # [L, m]


@torch.no_grad()
def element_activation(model, sae, seq: str, layer: int, device: str):
    """Return (summary[m], mass[m], active_idx) for the base sequence."""
    z = encode_seq(model, sae, seq, layer, device)
    summ = z.max(0).values
    mass = z.sum(0)
    active = torch.nonzero(summ > 0, as_tuple=False).flatten().cpu().numpy()
    return summ.cpu().numpy(), mass.cpu().numpy(), active


def _variants(seq: str, alts: str):
    """Yield (p, alt_base) for all single-nt substitutions."""
    for p, ref in enumerate(seq):
        choices = _ALT[ref] if alts == "all" else _ALT[ref][:1]
        for b in choices:
            yield p, b


@torch.no_grad()
def feature_sensitivity(model, sae, seq: str, features: np.ndarray, layer: int,
                        device: str, alts: str = "all", summary: str = "max",
                        batch: int = 64, return_per_alt: bool = False):
    """Return {feature_id: I_f[L]} for the requested features via batched ISM.

    With `return_per_alt`, additionally return the un-averaged sensitivity and the identity of
    each alt slot: ({feat: I[L]}, {feat: I_alt[L, n_alt]}, alt_base[L, n_alt]), where
    `alt_base[p, a]` is the BASE_TO_IDX code of the alternative base in slot `a` at position `p`
    (-1 where unused). The per-alt tensor is computed anyway — averaging it away discards which
    specific substitution the feature responds to, which is what a substitution-level analysis
    needs to line up with the assay's per-allele measurements.
    """
    L = len(seq)
    feats = np.asarray(features, dtype=np.int64)
    if feats.size == 0:
        return ({}, {}, np.zeros((L, 0), dtype=np.int64)) if return_per_alt else {}

    base_z = encode_seq(model, sae, seq, layer, device)
    base_summ = _summary(base_z, summary).index_select(0, torch.tensor(feats, device=device))

    var_list = list(_variants(seq, alts))
    n_alt = 3 if alts == "all" else 1
    deltas = np.zeros((L, n_alt, feats.size), dtype=np.float64)
    counts = np.zeros((L, n_alt), dtype=np.int64)
    alt_base = np.full((L, n_alt), -1, dtype=np.int64)

    for i in range(0, len(var_list), batch):
        chunk = var_list[i : i + batch]
        seqs = []
        for p, b in chunk:
            seqs.append(seq[:p] + b + seq[p + 1 :])
        h, lens = model.hidden(seqs, layer)
        h = h.to(device)
        for j, (p, b) in enumerate(chunk):
            z = sae.encode(h[j, : lens[j]])
            summ = _summary(z, summary, pos=p).index_select(0, torch.tensor(feats, device=device))
            d = (base_summ - summ).abs().cpu().numpy()
            a = counts[p].sum()       # alt index within this position
            deltas[p, a] = d
            counts[p, a] = 1
            alt_base[p, a] = BASE_TO_IDX[b]
    I = deltas.sum(1) / np.clip(counts.sum(1, keepdims=True), 1, None)   # [L, F] mean over alts
    out = {int(f): I[:, k] for k, f in enumerate(feats)}
    if not return_per_alt:
        return out
    per_alt = {int(f): deltas[:, :, k] for k, f in enumerate(feats)}     # [L, n_alt]
    return out, per_alt, alt_base


@torch.no_grad()
def loglik_ism(model, seq: str, alts: str = "all", batch: int = 64) -> np.ndarray:
    """Per-position model-intrinsic effect: mean_{b'} |LL(x) - LL(x')|  (the LL baseline)."""
    L = len(seq)
    base = model.sequence_loglik(seq)
    out = np.zeros(L)
    cnt = np.zeros(L)
    for p, b in _variants(seq, alts):
        ll = model.sequence_loglik(seq[:p] + b + seq[p + 1 :])
        out[p] += abs(base - ll)
        cnt[p] += 1
    return out / np.clip(cnt, 1, None)


# ---------------------------------------------------------------------------
# Brute-force reference (tests only): no batching, explicit loops.
# ---------------------------------------------------------------------------
@torch.no_grad()
def ism_bruteforce(model, sae, seq: str, feature: int, layer: int, device: str,
                   summary: str = "max") -> np.ndarray:
    L = len(seq)
    base_z = encode_seq(model, sae, seq, layer, device)
    base = _summary(base_z, summary)[feature].item()
    I = np.zeros(L)
    for p in range(L):
        ref = seq[p]
        diffs = []
        for b in BASES:
            if b == ref:
                continue
            mut = seq[:p] + b + seq[p + 1 :]
            z = encode_seq(model, sae, mut, layer, device)
            val = _summary(z, summary, pos=p)[feature].item()
            diffs.append(abs(base - val))
        I[p] = float(np.mean(diffs))
    return I
