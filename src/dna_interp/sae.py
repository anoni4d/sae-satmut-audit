"""Sparse autoencoders, three families behind one interface.

All families expose the same downstream contract used by ism/controls/annotate:
  encode(h) -> z   (post-sparsity feature activations, [B, m], z>=0 on active features)
  decode(z) -> ĥ ;  W_dec [d, m] (unit-norm columns), b_dec [d] ;  c.m, c.k
so the entire causal pipeline runs unchanged regardless of family.

  - TopK (Gao et al. 2024): z = TopK_k(ReLU(W_enc(h-b_dec)+b_enc)); L0=k by construction; AuxK
    revives dead latents.
  - Gated (Rajamanoharan et al. 2024): separate gate (Heaviside) and magnitude (ReLU) paths sharing
    W_gate; L1 on the gate pre-activations + a frozen-decoder auxiliary reconstruction.
  - JumpReLU (Rajamanoharan et al. 2024): z = π·H(π-θ) with a learned per-feature threshold θ and an
    L0 penalty trained through a rectangular straight-through estimator.

Gated/JumpReLU sparsity is set by a coefficient (the knob TopK avoids); we report the achieved L0 so
cross-family comparisons are transparent. This module is the substrate for the "is non-identifiability
TopK-specific?" experiment (see scripts/12_crossfamily.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .utils import get_logger, path

log = get_logger("sae")


def geometric_median(x: np.ndarray, n_iter: int = 100, eps: float = 1e-7) -> np.ndarray:
    y = x.mean(0)
    for _ in range(n_iter):
        d = np.linalg.norm(x - y, axis=1) + eps
        w = 1.0 / d
        y_new = (w[:, None] * x).sum(0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


@dataclass
class SAEConfig:
    d_model: int
    m: int
    k: int
    k_aux: int = 256
    aux_alpha: float = 1.0 / 32
    family: str = "topk"
    # gated / jumprelu sparsity knobs (ignored by topk)
    l1_coeff: float = 1e-2          # gated: weight on ||ReLU(pi_gate)||_1
    l0_coeff: float = 1e-2          # jumprelu: weight on E[L0]
    bandwidth: float = 0.1          # jumprelu: STE kernel bandwidth eps
    theta_init: float = 0.1         # jumprelu: initial threshold


# ---------------------------------------------------------------------------
# shared base: init, decoder normalization, the downstream encode/decode contract
# ---------------------------------------------------------------------------
class _BaseSAE(nn.Module):
    def __init__(self, c: SAEConfig):
        super().__init__()
        self.c = c
        self.W_dec = nn.Parameter(torch.zeros(c.d_model, c.m))
        self.b_dec = nn.Parameter(torch.zeros(c.d_model))
        self.register_buffer("steps_since_fire", torch.zeros(c.m, dtype=torch.long))

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-9)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    @torch.no_grad()
    def _init_decoder(self, sample: np.ndarray):
        med = geometric_median(sample.astype(np.float64))
        self.b_dec.copy_(torch.tensor(med, dtype=torch.float32))
        W = torch.randn(self.c.d_model, self.c.m)
        W /= W.norm(dim=0, keepdim=True) + 1e-9
        self.W_dec.copy_(W)
        return W

    def auxk_loss(self, h, recon, preacts, dead_mask):
        """AuxK: reconstruct the residual with the top-k_aux *dead* latents, reviving them. Shared
        across families so L1/L0 SAEs (which otherwise let latents die) get TopK's revival fix."""
        if dead_mask.sum() == 0:
            return h.new_zeros(())
        resid = h - recon.detach()
        da = torch.relu(preacts).clone()
        da[:, ~dead_mask] = 0.0
        kk = min(self.c.k_aux, int(dead_mask.sum()))
        vals, idx = da.topk(kk, dim=-1)
        z_aux = torch.zeros_like(da).scatter_(-1, idx, vals)
        resid_hat = z_aux @ self.W_dec.t()
        return self.c.aux_alpha * ((resid - resid_hat) ** 2).sum(-1).mean()

    # families implement: init_from_sample, encode, forward, extra_loss


# ---------------------------------------------------------------------------
# TopK (unchanged behaviour)
# ---------------------------------------------------------------------------
class TopKSAE(_BaseSAE):
    def __init__(self, c: SAEConfig):
        super().__init__(c)
        self.W_enc = nn.Parameter(torch.zeros(c.m, c.d_model))
        self.b_enc = nn.Parameter(torch.zeros(c.m))

    @torch.no_grad()
    def init_from_sample(self, sample):
        W = self._init_decoder(sample)
        self.W_enc.copy_(W.t().contiguous())
        self.b_enc.zero_()

    def preacts(self, h):
        return torch.relu((h - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def _topk(self, acts, k):
        if k >= acts.shape[-1]:
            return acts
        vals, idx = acts.topk(k, dim=-1)
        z = torch.zeros_like(acts)
        z.scatter_(-1, idx, vals)
        return z

    def encode(self, h):
        return self._topk(self.preacts(h), self.c.k)

    def forward(self, h):
        acts = self.preacts(h)
        z = self._topk(acts, self.c.k)
        return self.decode(z), z, acts

    def extra_loss(self, h, recon, z, acts, dead_mask):
        return self.auxk_loss(h, recon, acts, dead_mask)


# ---------------------------------------------------------------------------
# Gated SAE
# ---------------------------------------------------------------------------
class GatedSAE(_BaseSAE):
    def __init__(self, c: SAEConfig):
        super().__init__(c)
        self.W_gate = nn.Parameter(torch.zeros(c.m, c.d_model))
        self.b_gate = nn.Parameter(torch.zeros(c.m))
        self.r_mag = nn.Parameter(torch.zeros(c.m))      # log-scale magnitude rescaling
        self.b_mag = nn.Parameter(torch.zeros(c.m))

    @torch.no_grad()
    def init_from_sample(self, sample):
        W = self._init_decoder(sample)
        self.W_gate.copy_(W.t().contiguous())
        self.b_gate.zero_(); self.b_mag.zero_(); self.r_mag.zero_()

    def _pis(self, h):
        c = h - self.b_dec
        pi_gate = c @ self.W_gate.t() + self.b_gate
        W_mag = self.W_gate * torch.exp(self.r_mag)[:, None]
        pi_mag = c @ W_mag.t() + self.b_mag
        return pi_gate, pi_mag

    def encode(self, h):
        pi_gate, pi_mag = self._pis(h)
        return (pi_gate > 0).float() * torch.relu(pi_mag)

    def forward(self, h):
        pi_gate, pi_mag = self._pis(h)
        z = (pi_gate > 0).float() * torch.relu(pi_mag)
        return self.decode(z), z, pi_gate

    def extra_loss(self, h, recon, z, pi_gate, dead_mask):
        gate_act = torch.relu(pi_gate)
        l1 = self.c.l1_coeff * gate_act.sum(-1).mean()
        # frozen-decoder auxiliary reconstruction ties the gate to real directions
        recon_gate = gate_act @ self.W_dec.detach().t() + self.b_dec.detach()
        l_aux = ((h - recon_gate) ** 2).sum(-1).mean()
        return l1 + l_aux


# ---------------------------------------------------------------------------
# JumpReLU SAE  (rectangular straight-through estimator for the step)
# ---------------------------------------------------------------------------
class _HeavisideSTE(torch.autograd.Function):
    """Forward H(x); backward (1/eps)*rect(x/eps) — the JumpReLU pseudo-derivative."""
    @staticmethod
    def forward(ctx, x, bandwidth):
        ctx.save_for_backward(x)
        ctx.eps = float(bandwidth)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        eps = ctx.eps
        pseudo = (x.abs() < (eps / 2.0)).float() / eps
        return g * pseudo, None


class JumpReLUSAE(_BaseSAE):
    def __init__(self, c: SAEConfig):
        super().__init__(c)
        self.W_enc = nn.Parameter(torch.zeros(c.m, c.d_model))
        self.b_enc = nn.Parameter(torch.zeros(c.m))
        self.log_theta = nn.Parameter(torch.full((c.m,), float(np.log(c.theta_init))))

    @torch.no_grad()
    def init_from_sample(self, sample):
        W = self._init_decoder(sample)
        self.W_enc.copy_(W.t().contiguous())
        self.b_enc.zero_()

    def _pre(self, h):
        return (h - self.b_dec) @ self.W_enc.t() + self.b_enc

    def _theta(self):
        return torch.exp(self.log_theta)

    def encode(self, h):
        pi = self._pre(h)
        theta = self._theta()
        # inference: hard step (no STE needed); z = pi where pi>theta, else 0
        return pi * (pi > theta).float()

    def forward(self, h):
        pi = self._pre(h)
        theta = self._theta()
        gate = _HeavisideSTE.apply(pi - theta, self.c.bandwidth)
        z = pi * gate
        return self.decode(z), z, pi

    def extra_loss(self, h, recon, z, pi, dead_mask):
        theta = self._theta()
        l0 = _HeavisideSTE.apply(pi - theta, self.c.bandwidth).sum(-1).mean()
        return self.c.l0_coeff * l0 + self.auxk_loss(h, recon, pi, dead_mask)


FAMILIES = {"topk": TopKSAE, "gated": GatedSAE, "jumprelu": JumpReLUSAE}


def build_sae(c: SAEConfig) -> _BaseSAE:
    if c.family not in FAMILIES:
        raise ValueError(f"unknown SAE family {c.family!r}; choices {list(FAMILIES)}")
    return FAMILIES[c.family](c)


def _config_from_cfg(cfg, d: int) -> SAEConfig:
    s = cfg.sae
    return SAEConfig(
        d_model=d, m=int(s.expansion) * d, k=int(s.k),
        k_aux=int(s.k_aux), aux_alpha=float(s.aux_alpha),
        family=str(s.get("family", "topk")),
        l1_coeff=float(s.get("l1_coeff", 1e-2)),
        l0_coeff=float(s.get("l0_coeff", 1e-2)),
        bandwidth=float(s.get("bandwidth", 0.1)),
        theta_init=float(s.get("theta_init", 0.1)),
    )


# ---------------------------------------------------------------------------
# data streaming (RAM buffer; see note) + family-agnostic training
# ---------------------------------------------------------------------------
def _load_ram_buffer(mm: np.memmap, n_use: int, rng: np.random.Generator) -> np.ndarray:
    """Pull a random subset of `n_use` rows into a contiguous float32 RAM array (sorted gather =
    near-sequential read), so training is GPU-bound rather than disk-IO-bound on the multi-GB memmap."""
    n = mm.shape[0]
    if n_use >= n:
        return np.asarray(mm, dtype=np.float32)
    sel = np.sort(rng.choice(n, size=n_use, replace=False))
    return np.asarray(mm[sel], dtype=np.float32)


def _iter_batches(buf: np.ndarray, batch: int, rng: np.random.Generator, epochs: int):
    n = buf.shape[0]
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            yield buf[order[i : i + batch]]


def train_sae(cfg, mm: np.memmap, layer: int, seed: int | None = None) -> tuple[_BaseSAE, dict]:
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = cfg.device
    d = mm.shape[1]
    sc = _config_from_cfg(cfg, d)
    sae = build_sae(sc).to(device)

    ram_cap = int(cfg.sae.get("ram_max_tokens", 5_000_000))
    buf = _load_ram_buffer(mm, ram_cap, rng)
    log.info("[%s] loaded %d/%d tokens into RAM (%.2f GB) for SAE training",
             sc.family, buf.shape[0], mm.shape[0], buf.nbytes / 1e9)

    sample = buf[rng.choice(buf.shape[0], size=min(8192, buf.shape[0]), replace=False)]
    sae.init_from_sample(sample)
    sae.normalize_decoder()

    opt = torch.optim.Adam(sae.parameters(), lr=float(cfg.sae.lr))
    batch = int(cfg.sae.batch)
    dead_steps = max(1, int(cfg.sae.dead_token_window) // batch)
    data_mean = torch.tensor(sample.mean(0), device=device)
    total_var = ((torch.tensor(sample, device=device) - data_mean) ** 2).sum(-1).mean().item()

    step = 0
    for hb in _iter_batches(buf, batch, rng, int(cfg.sae.epochs)):
        h = torch.tensor(hb, device=device, dtype=torch.float32)
        recon, z, acts = sae(h)
        recon_loss = ((h - recon) ** 2).sum(-1).mean()

        fired = (z > 0).any(0)
        sae.steps_since_fire += 1
        sae.steps_since_fire[fired] = 0
        dead_mask = sae.steps_since_fire > dead_steps
        loss = recon_loss + sae.extra_loss(h, recon, z, acts, dead_mask)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()

        if step % 50 == 0:
            fvu = recon_loss.item() / (total_var + 1e-9)
            l0 = float((z > 0).float().sum(-1).mean().item())
            log.info("[%s] layer %d step %d | recon %.4f | FVU %.3f | L0 %.1f | dead %d/%d",
                     sc.family, layer, step, recon_loss.item(), fvu, l0,
                     int(dead_mask.sum()), sc.m)
        step += 1

    metrics = evaluate_sae(cfg, sae, mm, rng)
    metrics.update({"layer": layer, "seed": seed, "m": sc.m, "k": sc.k,
                    "family": sc.family, "steps": step})
    save_sae(cfg, sae, layer, seed)
    return sae, metrics


@torch.no_grad()
def evaluate_sae(cfg, sae: _BaseSAE, mm: np.memmap, rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    device = cfg.device
    n_eval = min(20000, mm.shape[0])
    sel = rng.choice(mm.shape[0], size=n_eval, replace=False)
    fire_counts = torch.zeros(sae.c.m, device=device)
    tot_resid = tot_var = l0_sum = 0.0
    seen = 0
    mean = torch.tensor(np.asarray(mm[sel]).mean(0), device=device)
    for i in range(0, n_eval, 4096):
        hb = np.asarray(mm[sel[i : i + 4096]])
        h = torch.tensor(hb, device=device, dtype=torch.float32)
        recon, z, _ = sae(h)
        tot_resid += ((h - recon) ** 2).sum().item()
        tot_var += ((h - mean) ** 2).sum().item()
        fire_counts += (z > 0).float().sum(0)
        l0_sum += (z > 0).float().sum().item()
        seen += h.shape[0]
    density = (fire_counts / seen).cpu().numpy()
    return {
        "FVU": float(tot_resid / (tot_var + 1e-9)),
        "dead_frac": float((density == 0).mean()),
        "L0": float(l0_sum / seen),
        "n_high_density": int((density > float(cfg.sae.density_warn)).sum()),
        "density_mean": float(density.mean()),
        "density_median": float(np.median(density)),
        "density": density,
    }


DEFAULT_EXPANSION = 16


def _sae_path(cfg, layer: int, seed: int, family: str, m: int | None = None, d: int | None = None):
    """Artifact path for one SAE.

    The dictionary width is encoded in the stem only when it differs from the default expansion,
    so the width sweep (scripts/16_dictsweep.py) cannot silently overwrite the headline SAEs
    trained at 16x — those keep their original `sae_layer{L}_seed{s}` names.
    """
    stem = f"sae_layer{layer}_seed{seed}" if family == "topk" \
        else f"sae_{family}_layer{layer}_seed{seed}"
    if m is not None and d and m != DEFAULT_EXPANSION * d:
        stem = stem.replace(f"layer{layer}", f"layer{layer}_m{m}")
    return path(cfg, "artifacts", f"{stem}.safetensors")


def save_sae(cfg, sae: _BaseSAE, layer: int, seed: int) -> str:
    p = _sae_path(cfg, layer, seed, sae.c.family, m=sae.c.m, d=sae.c.d_model)
    save_file({k: v.detach().cpu() for k, v in sae.state_dict().items()}, str(p))
    (p.with_suffix(".json")).write_text(json.dumps(asdict(sae.c)))
    return str(p)


def load_sae(cfg, layer: int, seed: int = 0, device: str | None = None,
             family: str = "topk", m: int | None = None, d: int | None = None) -> _BaseSAE:
    p = _sae_path(cfg, layer, seed, family, m=m, d=d)
    sc = SAEConfig(**json.loads(p.with_suffix(".json").read_text()))
    sae = build_sae(sc)
    sae.load_state_dict(load_file(str(p)))
    sae.to(device or cfg.device).eval()
    return sae
