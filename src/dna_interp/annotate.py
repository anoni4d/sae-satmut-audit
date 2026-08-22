"""Feature annotation: max-activating examples, motif enrichment, element-class overlap.

Motif scanning uses MOODS/`fimo` + JASPAR when available; otherwise it falls back to the
internal planted-motif catalogue (data.MOTIFS) — which is exactly the right reference for
the synthetic/toy ground-truth regime and recovers planted motifs by hypergeometric test.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

from . import data as D
from .activations import load_activations
from .utils import get_logger, path

log = get_logger("annotate")

# ---------------------------------------------------------------------------
# Repeat / low-complexity annotation. Used by the repeat-masking control and by the
# subspace-characterisation analysis, so it lives here rather than in one script.
# ---------------------------------------------------------------------------
ALUY = ("GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTCAGGAGATCGAGACC"
        "ATCCCGGCTAAAACGGTGAAACCCCGTCTCTACTAAAAATACAAAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAG"
        "CTACTCGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCA"
        "GCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAAAA")


def _rc(s: str) -> str:
    c = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(c.get(x, "N") for x in reversed(s))


def _kmers(s: str, k: int = 11) -> set[str]:
    return {s[i:i + k] for i in range(len(s) - k + 1)}


ALU_K = _kmers(ALUY) | _kmers(_rc(ALUY))


def repeat_mask(seq: str, k: int = 11) -> np.ndarray:
    """Boolean per-position mask of Alu-consensus k-mer hits and homopolymer runs (>=6)."""
    s = seq.upper()
    L = len(s)
    m = np.zeros(L, bool)
    for i in range(L - k + 1):
        if s[i:i + k] in ALU_K:
            m[i:i + k] = True
    for mt in re.finditer(r"(.)\1{5,}", s):
        m[mt.start():mt.end()] = True
    return m


@torch.no_grad()
def top_activating(cfg, sae, layer: int, top_n: int = 20):
    """Scan cached activations; return per-feature top-N (value, seq_id, position) and density."""
    mm, idx_df = load_activations(cfg, layer)
    device = cfg.device
    m = sae.c.m
    n = mm.shape[0]
    # running top-N via simple lists (m can be large but top_n small)
    best_val = np.full((m, top_n), -np.inf)
    best_tok = np.full((m, top_n), -1, dtype=np.int64)
    fire = np.zeros(m, dtype=np.int64)
    bs = 8192
    for i in range(0, n, bs):
        h = torch.tensor(np.asarray(mm[i : i + bs]), device=device, dtype=torch.float32)
        z = sae.encode(h).cpu().numpy()                 # [b, m]
        fire += (z > 0).sum(0)
        # for each feature, merge this block's max candidates
        blk_max = z.max(0)                              # [m]
        blk_arg = z.argmax(0) + i
        replace = blk_max > best_val[:, -1]
        if replace.any():
            for f in np.nonzero(replace)[0]:
                vals = np.append(best_val[f], blk_max[f])
                toks = np.append(best_tok[f], blk_arg[f])
                order = np.argsort(-vals)[:top_n]
                best_val[f] = vals[order]
                best_tok[f] = toks[order]
    density = fire / n
    tok2 = idx_df.set_index("token_idx")
    out = {}
    for f in range(m):
        items = []
        for v, t in zip(best_val[f], best_tok[f]):
            if t >= 0 and np.isfinite(v):
                r = tok2.loc[t]
                items.append((float(v), str(r.seq_id), int(r.position)))
        out[f] = items
    return out, density


def _windows_for_feature(corpus_map, items, flank=15):
    seqs = []
    for _v, sid, pos in items:
        s = corpus_map.get(sid)
        if s is None:
            continue
        a, b = max(0, pos - flank), min(len(s), pos + flank + 1)
        seqs.append(s[a:b])
    return seqs


def motif_enrichment(windows: list[str], bg_freq: dict[str, float], n_bg: int):
    """Hypergeometric enrichment of each catalogue motif in `windows` vs background freq."""
    res = []
    K = len(windows)
    if K == 0:
        return res
    for name in D.MOTIF_NAMES:
        hits = sum(1 for w in windows if _has_motif(w, name))
        p_bg = bg_freq.get(name, 1e-3)
        # P(>= hits) under background hit rate
        exp_bg = max(1, int(round(p_bg * n_bg)))
        pval = hypergeom.sf(hits - 1, n_bg + K, exp_bg + hits, K)
        res.append((name, hits / K, float(pval)))
    res.sort(key=lambda r: r[2])
    return res


def _has_motif(window: str, name: str, thresh: float = 4.0) -> bool:
    idx = D.seq_to_idx(window)
    sc = D.motif_scan(idx, name)
    return bool(len(sc) and sc.max() > thresh)


def _bg_freq(corpus, n_sample=500):
    rng = np.random.default_rng(0)
    bg = [w.seq for w in corpus if w.cls == "background"]
    if not bg:
        bg = [w.seq for w in corpus]
    sample = rng.choice(len(bg), size=min(n_sample, len(bg)), replace=False)
    freq = {name: 0 for name in D.MOTIF_NAMES}
    for j in sample:
        s = bg[j]
        for name in D.MOTIF_NAMES:
            if _has_motif(s, name):
                freq[name] += 1
    n = len(sample)
    return {k: v / n for k, v in freq.items()}, n


def annotate_features(cfg, sae, layer: int, corpus, top_n: int = 20) -> pd.DataFrame:
    items_by_feat, density = top_activating(cfg, sae, layer, top_n)
    corpus_map = {w.seq_id: w.seq for w in corpus}
    cls_map = {w.seq_id: w.cls for w in corpus}
    bg_freq, n_bg = _bg_freq(corpus)

    rows = []
    for f, items in items_by_feat.items():
        if not items:
            rows.append({"feature_id": f, "density": float(density[f]), "top_motif": "",
                         "motif_enrichment_p": np.nan, "element_class": "",
                         "label": "dead"})
            continue
        wins = _windows_for_feature(corpus_map, items)
        enr = motif_enrichment(wins, bg_freq, n_bg)
        top_motif, _, p = enr[0] if enr else ("", 0, np.nan)
        sig = (not np.isnan(p)) and p < 1e-4
        # dominant element class among top-activating windows
        classes = [cls_map.get(sid, "") for _v, sid, _p in items]
        elem_class = max(set(classes), key=classes.count) if classes else ""
        label = "motif" if sig else ("element" if elem_class == "ccre" else "unannotated")
        rows.append({"feature_id": f, "density": float(density[f]),
                     "top_motif": top_motif if sig else "",
                     "motif_enrichment_p": p, "element_class": elem_class, "label": label})
    df = pd.DataFrame(rows).sort_values("feature_id").reset_index(drop=True)
    df.to_csv(path(cfg, "artifacts", f"features_layer{layer}.csv"), index=False)
    return df
