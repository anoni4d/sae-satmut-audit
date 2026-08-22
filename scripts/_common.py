"""Shared script bootstrap: makes src/ importable and parses CLI dotlist overrides."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dna_interp.utils import load_config, ensure_dirs, set_seed, save_config_snapshot  # noqa


def boot():
    cfg = load_config(sys.argv[1:])
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    save_config_snapshot(cfg, Path(cfg.paths["artifacts"]) / "config_snapshot.yaml")
    return cfg
