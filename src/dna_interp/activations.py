"""Genomic model wrappers + residual-stream activation extraction.

One interface, two backends:
  * HyenaDNAModel  — the real single-nt foundation model (HuggingFace, trust_remote_code).
  * ToyGenomicModel — a deterministic single-nt stand-in whose residual stream is a fixed
    random rotation of interpretable channels (motif detectors + GC + random distractors).
    Used for tests, smoke, and as the ground-truth positive control: motif-detector
    features are causal-by-construction, GC features are correlational-by-construction.

Both expose, at single-nucleotide resolution:
  - hidden(seqs, layer) -> [B, Lmax, d]  (residual stream = output of block `layer`)
  - feature_input(seq, layer) -> [L, d]  (special tokens stripped; token i <-> base i)
  - sequence_loglik(seq) -> float        (for the model-intrinsic ISM baseline)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from . import data as D
from .utils import BASE_TO_IDX, BASES, get_logger, path

log = get_logger("activations")


class GenomicModel(ABC):
    n_layers: int
    d_model: int
    max_length: int
    device: str

    @abstractmethod
    def hidden(self, seqs: list[str], layer: int) -> tuple[torch.Tensor, list[int]]:
        """Residual stream at block `layer`: padded [B, Lmax, d] and true lengths."""

    def feature_input(self, seq: str, layer: int) -> torch.Tensor:
        """[L, d] residual stream for one sequence, special tokens stripped."""
        h, lens = self.hidden([seq], layer)
        return h[0, : lens[0]]

    @abstractmethod
    def sequence_loglik(self, seq: str) -> float:
        """Total log-likelihood of `seq` under the model (for the LL baseline)."""

    def valid_layers(self) -> list[int]:
        return list(range(1, self.n_layers + 1))


# ---------------------------------------------------------------------------
# HyenaDNA
# ---------------------------------------------------------------------------
class HyenaDNAModel(GenomicModel):
    def __init__(self, name: str, device: str, max_length: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
        self.model.eval().to(device)
        self.device = device
        self.max_length = int(max_length)
        cfg = self.model.config
        d = next((getattr(cfg, a) for a in ("d_model", "hidden_size", "n_embd")
                  if getattr(cfg, a, None) is not None), None)
        if d is None:
            raise AttributeError("could not determine d_model from model config")
        self.d_model = int(d)
        # number of usable residual-stream layers = (#hidden_states - 1); probe once
        with torch.no_grad():
            ids = self.tok("ACGT", return_tensors="pt")["input_ids"].to(device)
            hs = self.model(ids, output_hidden_states=True).hidden_states
        self.n_layers = len(hs) - 1
        # how many special tokens get appended (e.g. trailing [SEP]); detect via length
        self._suffix = ids.shape[1] - 4
        log.info("HyenaDNA %s: d_model=%d, layers(usable)=%d, suffix_tok=%d",
                 name, self.d_model, self.n_layers, self._suffix)

    @torch.no_grad()
    def hidden(self, seqs, layer):
        enc = self.tok(list(seqs), return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(self.device)
        mask = enc.get("attention_mask")
        out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[layer]                       # [B, T, d]
        if mask is not None:
            lens = [int(m.sum()) - self._suffix for m in mask]
        else:
            lens = [len(s) for s in seqs]
        lens = [max(0, min(l, h.shape[1])) for l in lens]
        return h.float().cpu(), lens

    @torch.no_grad()
    def sequence_loglik(self, seq: str) -> float:
        ids = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)
        out = self.model(ids)
        logits = out.logits[:, :-1].log_softmax(-1)
        tgt = ids[:, 1:]
        ll = logits.gather(-1, tgt.unsqueeze(-1)).sum().item()
        return float(ll)


# ---------------------------------------------------------------------------
# Caduceus (BiMamda, single-nt) — second architecture for the cross-model generality test.
# NOTE: requires mamba-ssm + causal-conv1d (custom CUDA kernels). Validated on cloud GPU only;
# the dev laptop lacks nvcc / a matching wheel (see SETUP_CADUCEUS.md). Mirrors HyenaDNAModel.
# Caduceus is a masked-LM, so sequence_loglik is a pseudo-log-likelihood (sum of per-position
# log-probs of the true base from a single unmasked forward) — fine for the relative LL baseline.
# ---------------------------------------------------------------------------
class CaduceusModel(GenomicModel):
    def __init__(self, name: str, device: str, max_length: int):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
        self.model.eval().to(device)
        self.device = device
        self.max_length = int(max_length)
        cfg = self.model.config
        d = next((getattr(cfg, a) for a in ("d_model", "hidden_size", "n_embd")
                  if getattr(cfg, a, None) is not None), None)
        if d is None:
            raise AttributeError("could not determine d_model from Caduceus config")
        self.d_model = int(d)
        with torch.no_grad():
            ids = self.tok("ACGT", return_tensors="pt")["input_ids"].to(device)
            hs = self.model(ids, output_hidden_states=True).hidden_states
        self.n_layers = len(hs) - 1
        self._suffix = ids.shape[1] - 4          # trailing special token(s), detected like HyenaDNA
        log.info("Caduceus %s: d_model=%d, layers(usable)=%d, suffix_tok=%d",
                 name, self.d_model, self.n_layers, self._suffix)

    @torch.no_grad()
    def hidden(self, seqs, layer):
        enc = self.tok(list(seqs), return_tensors="pt", padding=True)
        ids = enc["input_ids"].to(self.device)
        mask = enc.get("attention_mask")
        h = self.model(ids, output_hidden_states=True).hidden_states[layer]   # [B, T, d]
        if mask is not None:
            lens = [int(m.sum()) - self._suffix for m in mask]
        else:
            lens = [len(s) for s in seqs]
        lens = [max(0, min(l, h.shape[1])) for l in lens]
        return h.float().cpu(), lens

    @torch.no_grad()
    def sequence_loglik(self, seq: str) -> float:
        ids = self.tok(seq, return_tensors="pt")["input_ids"].to(self.device)
        logits = self.model(ids).logits.log_softmax(-1)        # [1, T, V], bidirectional MLM
        # pseudo-LL: sum log P(true base | full context) over the nucleotide positions
        n = min(len(seq), ids.shape[1])
        ll = logits[0, torch.arange(n), ids[0, :n]].sum().item()
        return float(ll)


# ---------------------------------------------------------------------------
# Toy model: ground-truth positive control
# ---------------------------------------------------------------------------
class ToyGenomicModel(GenomicModel):
    """Residual stream = R @ g(seq), with g = [motif detectors | GC | random distractors].

    Channel semantics (the ground truth the causal test must recover):
      - motif channels  : causal — sensitive only at the bases of a present motif.
      - GC channels      : correlational — depend on local composition, ~uniform ISM.
      - random channels  : distractors — structured but unrelated to satmut effect E.
    """

    def __init__(self, d_model: int, n_layers: int, device: str, max_length: int, seed: int = 0):
        self.device = device
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.max_length = int(max_length)
        self.motif_names = D.MOTIF_NAMES
        n_motif = len(self.motif_names)
        self.n_gc = 2
        self.gc_windows = [11, 31]
        n_random = max(4, self.d_model // 8)
        self.C = n_motif + self.n_gc + n_random
        self.n_random = n_random

        rng = np.random.default_rng(seed + 12345)
        # fixed random rotation channels->d_model (sparse-ish, unit-norm columns)
        R = rng.standard_normal((self.d_model, self.C))
        R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-9
        self.R = R
        # random distractor projection over a k-mer window of one-hot bases
        self.rand_w = 7
        self.rand_proj = rng.standard_normal((n_random, self.rand_w * 4))
        # bigram background log-prob table for the LL baseline (composition-based)
        bg = rng.dirichlet(np.ones(4) * 2.0, size=4)        # P(next | prev)
        self.bigram_logp = np.log(bg + 1e-9)
        self.unigram_logp = np.log(rng.dirichlet(np.ones(4) * 2.0) + 1e-9)
        log.info("ToyGenomicModel: d_model=%d layers=%d channels=%d (motif=%d gc=%d rand=%d)",
                 self.d_model, self.n_layers, self.C, n_motif, self.n_gc, n_random)

    # ---- channel features g(seq) -> [L, C] ----
    def _channels(self, seq: str) -> np.ndarray:
        idx = D.seq_to_idx(seq)
        L = len(idx)
        g = np.zeros((L, self.C), dtype=np.float64)
        col = 0
        # motif detectors: assign start-position log-odds score to position i
        for name in self.motif_names:
            sc = D.motif_scan(idx, name)
            if len(sc):
                g[: len(sc), col] = np.maximum(sc, 0.0)
            col += 1
        # GC channels: centered fraction in sliding window
        gc = (idx == BASE_TO_IDX["C"]) | (idx == BASE_TO_IDX["G"])
        gc = gc.astype(np.float64)
        for w in self.gc_windows:
            ww = max(1, min(w, L))
            k = np.ones(ww) / ww
            sm = np.convolve(gc, k, mode="same") - 0.5
            g[:, col] = sm
            col += 1
        # random distractors: fixed projection of one-hot k-mer window
        oh = np.zeros((L, 4))
        oh[np.arange(L), idx] = 1.0
        w = self.rand_w
        padded = np.vstack([np.zeros((w // 2, 4)), oh, np.zeros((w // 2, 4))])
        for i in range(L):
            window = padded[i : i + w].reshape(-1)
            g[i, col : col + self.n_random] = self.rand_proj @ window
        return g

    def _layerify(self, g: np.ndarray, layer: int) -> np.ndarray:
        """Deeper layers mix more context (progressive smoothing) -> layer choice matters."""
        if layer <= 1:
            return g
        k = np.ones(3) / 3.0
        out = g.copy()
        for _ in range(layer - 1):
            out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, out)
        return out

    def _hidden_one(self, seq: str, layer: int) -> np.ndarray:
        g = self._layerify(self._channels(seq), layer)      # [L, C]
        return g @ self.R.T                                  # [L, d]

    def hidden(self, seqs, layer):
        mats = [self._hidden_one(s, layer) for s in seqs]
        lens = [m.shape[0] for m in mats]
        Lmax = max(lens)
        out = np.zeros((len(mats), Lmax, self.d_model), dtype=np.float32)
        for i, m in enumerate(mats):
            out[i, : m.shape[0]] = m
        return torch.from_numpy(out), lens

    def sequence_loglik(self, seq: str) -> float:
        idx = D.seq_to_idx(seq)
        ll = float(self.unigram_logp[idx[0]])
        for i in range(1, len(idx)):
            ll += float(self.bigram_logp[idx[i - 1], idx[i]])
        return ll


# ---------------------------------------------------------------------------
# factory + extraction
# ---------------------------------------------------------------------------
def load_model(cfg) -> GenomicModel:
    name = cfg.model.name
    if name == "toy":
        return ToyGenomicModel(cfg.model.toy_d_model, cfg.model.toy_n_layers,
                               cfg.device, cfg.model.max_length, seed=cfg.seed)
    backend = str(cfg.model.get("backend", "")).lower()
    is_caduceus = backend == "caduceus" or "caduceus" in name.lower()
    try:
        if is_caduceus:
            return CaduceusModel(name, cfg.device, cfg.model.max_length)
        return HyenaDNAModel(name, cfg.device, cfg.model.max_length)
    except Exception as e:  # noqa
        if cfg.model.get("fallback_to_toy", True):
            log.warning("HyenaDNA load failed (%s); falling back to toy model.", e)
            return ToyGenomicModel(cfg.model.toy_d_model, cfg.model.toy_n_layers,
                                   cfg.device, cfg.model.max_length, seed=cfg.seed)
        raise


def sweep_layers(cfg, model: GenomicModel) -> list[int]:
    if cfg.activations.layers:
        return list(cfg.activations.layers)
    N = model.n_layers
    cand = sorted({max(1, N // 4), max(1, N // 2), max(1, (3 * N) // 4)})
    return cand


def extract_activations(cfg, model: GenomicModel, windows, layer: int):
    """Run corpus through the model; write per-token residual-stream rows to a memmap and
    an index table (token_idx -> seq_id, position). Returns (memmap_path, index_df, stats)."""
    import pandas as pd

    d = model.d_model
    n_tokens = sum(min(len(w.seq), model.max_length) for w in windows)
    mm_path = path(cfg, "artifacts", f"acts_layer{layer}.memmap")
    mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_tokens, d))

    bs = int(cfg.activations.batch_size)
    win_ids: list[str] = []
    win_lens: list[int] = []
    cursor = 0
    sums = np.zeros(d, dtype=np.float64)
    sqs = np.zeros(d, dtype=np.float64)
    for i in range(0, len(windows), bs):
        batch = windows[i : i + bs]
        seqs = [w.seq[: model.max_length] for w in batch]
        h, lens = model.hidden(seqs, layer)
        h = h.numpy()
        for j, w in enumerate(batch):
            L = lens[j]
            block = h[j, :L]
            mm[cursor : cursor + L] = block
            win_ids.append(w.seq_id)
            win_lens.append(L)
            sums += block.sum(0)
            sqs += (block.astype(np.float64) ** 2).sum(0)
            cursor += L
        if (i // bs) % 20 == 0:
            log.info("layer %d: %d/%d windows", layer, i + len(batch), len(windows))
    mm.flush()
    # build the token->(seq_id, position) index vectorized (avoids per-token appends)
    win_lens_arr = np.array(win_lens, dtype=np.int64)
    positions = np.concatenate([np.arange(L, dtype=np.int32) for L in win_lens]) if win_lens \
        else np.zeros(0, np.int32)
    seq_ids = np.repeat(np.array(win_ids, dtype=object), win_lens_arr)
    idx_df = pd.DataFrame({"token_idx": np.arange(cursor, dtype=np.int64),
                           "seq_id": pd.Categorical(seq_ids), "position": positions})
    idx_df.to_parquet(path(cfg, "artifacts", f"acts_layer{layer}_index.parquet"))

    mean = sums / cursor
    var = sqs / cursor - mean ** 2
    stats = {"layer": layer, "n_tokens": int(cursor), "d_model": d,
             "mean_norm": float(np.sqrt((mean ** 2).sum())),
             "mean_var": float(var.mean()), "max_var": float(var.max())}
    np.save(path(cfg, "artifacts", f"acts_layer{layer}_meanstd.npy"),
            np.stack([mean, np.sqrt(np.clip(var, 0, None))]))
    return mm_path, idx_df, stats


def load_activations(cfg, layer: int) -> tuple[np.memmap, "pd.DataFrame"]:
    import pandas as pd

    idx_df = pd.read_parquet(path(cfg, "artifacts", f"acts_layer{layer}_index.parquet"))
    meta = np.load(path(cfg, "artifacts", f"acts_layer{layer}_meanstd.npy"))
    d = meta.shape[1]
    n = len(idx_df)
    mm = np.memmap(path(cfg, "artifacts", f"acts_layer{layer}.memmap"),
                   dtype=np.float32, mode="r", shape=(n, d))
    return mm, idx_df
