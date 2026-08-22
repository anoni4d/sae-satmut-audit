#!/usr/bin/env python
"""One-command end-to-end driver (P1->P6). Runs the documented per-phase scripts in order,
training multiple SAE seeds so the robustness control has data.

    python scripts/run_pipeline.py smoke=true
    python scripts/run_pipeline.py            # uses config/default.yaml as-is
    python scripts/run_pipeline.py n_seeds=3 sae.k=32
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script, overrides):
    cmd = [sys.executable, str(ROOT / script), *overrides]
    print(f"\n{'=' * 70}\n>>> {' '.join(cmd)}\n{'=' * 70}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    overrides = [a for a in sys.argv[1:] if not a.startswith("n_seeds=")]
    n_seeds = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("n_seeds=")), 3)

    run("01_build_corpus.py", overrides)
    run("02_extract_activations.py", overrides)
    for s in range(n_seeds):                       # multiple seeds -> robustness control
        run("03_train_sae.py", overrides + [f"seed={s}"])
    run("05_annotate.py", overrides)
    run("04_run_ism.py", overrides)
    print("\nPipeline complete. See reports/phase1..6.md and reports/figures/.")


if __name__ == "__main__":
    main()
