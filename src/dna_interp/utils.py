"""Shared utilities: config loading, seeding, paths, logging, small helpers."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

# DNA alphabet used everywhere (single-nucleotide resolution).
BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def get_logger(name: str = "dna_interp") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _deep_merge(base: DictConfig, override: DictConfig) -> DictConfig:
    return OmegaConf.merge(base, override)


def load_config(overrides: list[str] | None = None,
                config_path: str | os.PathLike | None = None) -> DictConfig:
    """Load default.yaml, apply smoke_overrides if smoke=true, then dotlist CLI overrides.

    Order: default <- (smoke_overrides if smoke) <- CLI dotlist overrides.
    The CLI may itself set smoke=true; we resolve that before applying smoke_overrides.
    """
    base = config_path or os.environ.get("DNA_INTERP_CONFIG") or DEFAULT_CONFIG
    cfg = OmegaConf.load(base)
    overrides = list(overrides or [])

    # First pass: detect smoke flag from CLI so smoke_overrides apply correctly.
    cli = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    smoke = bool(OmegaConf.select(cli, "smoke", default=cfg.get("smoke", False)))

    if smoke:
        smoke_ov = cfg.get("smoke_overrides", {})
        cfg = _deep_merge(cfg, OmegaConf.create({"smoke": True}))
        cfg = _deep_merge(cfg, smoke_ov)

    if overrides:
        cfg = _deep_merge(cfg, cli)

    # Resolve device early.
    cfg.device = _resolve_device(cfg.get("device", "cuda"))
    return cfg


def _resolve_device(requested: str) -> str:
    try:
        import torch

        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        return "cpu"
    except Exception:
        return "cpu"


def ensure_dirs(cfg: DictConfig) -> None:
    for key in ("data_raw", "data_cache", "artifacts", "reports"):
        Path(cfg.paths[key]).mkdir(parents=True, exist_ok=True)
    (Path(cfg.paths["reports"]) / "figures").mkdir(parents=True, exist_ok=True)


def path(cfg: DictConfig, key: str, *parts: str) -> Path:
    p = Path(cfg.paths[key]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_config_snapshot(cfg: DictConfig, dest: str | os.PathLike) -> None:
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, dest)


def write_report(cfg: DictConfig, phase: str, body: str) -> Path:
    """Write reports/<phase>.md and return the path."""
    p = Path(cfg.paths["reports"]) / f"{phase}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
