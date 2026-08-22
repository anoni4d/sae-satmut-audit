#!/usr/bin/env python
"""What does the stable causal subspace actually encode? (reviewer: its biological
interpretation remains limited)

The subspace analysis showed the causal directions span a cross-seed-stable subspace even though
individual features do not reproduce. That is a statement about geometry, not biology. Here we ask
what the subspace *responds to*: we project each element's per-position residual stream onto the
causal subspace and test whether the positions with large projection are enriched for

  - Alu / low-complexity repeat sequence (the leading confound: the "novel" candidate features
    are largely repeat detectors),
  - GC content (plain composition),
  - large measured MPRA effect (the quantity we actually care about),

each against the matched non-causal subspace of the same dimension, so the comparison isolates
what is specific to the *causal* directions rather than true of any decoder subspace.

Reuses the cached alignment matrices (no ISM) plus one forward pass per element.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/19_subspace_bio.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.annotate import repeat_mask
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import BASE_TO_IDX, get_logger, md_table, write_report

log = get_logger("subspacebio")
RNG = np.random.default_rng(0)


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 6


def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def subspace_basis(cols, energy=0.90, max_dim=50):
    if cols.shape[1] == 0:
        return np.zeros((cols.shape[0], 0))
    U, s, _ = np.linalg.svd(cols, full_matrices=False)
    e = s ** 2
    r = int(np.searchsorted(np.cumsum(e) / e.sum(), energy) + 1)
    return U[:, :max(1, min(r, max_dim, U.shape[1]))]


def auroc(scores, labels):
    """Rank-based AUROC; labels boolean. 0.5 = no association."""
    labels = np.asarray(labels, bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gc_track(seq, win=11):
    idx = np.array([BASE_TO_IDX.get(b, 0) for b in seq])
    gc = ((idx == BASE_TO_IDX["C"]) | (idx == BASE_TO_IDX["G"])).astype(float)
    k = max(1, min(win, len(seq)))
    return np.convolve(gc, np.ones(k) / k, mode="same")


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    seeds = sorted(int(re.search(r"seed(\d+)", f).group(1))
                   for f in glob.glob(str(art / f"sae_layer{L}_seed*.safetensors")))
    log.info("subspace-bio: layer %d, seeds %s", L, seeds)

    # causal / non-causal feature sets at locus level, from the cached alignments
    saes = {s: S.load_sae(cfg, L, seed=s) for s in seeds}
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         saes[seeds[0]].b_dec.detach(), cfg.device, seed=cfg.seed)
    null_align = controls.cached_alignment(cfg, model, null_enc, satmut, L, "null")
    nullA = controls.aggregate_features(null_align, 1, groups=gmap).A_f.to_numpy()

    bases = {}
    for s in seeds:
        sae = saes[s]
        W = sae.W_dec.detach().cpu().numpy()
        align = controls.cached_alignment(cfg, model, sae, satmut, L, f"topk_seed{s}")
        feats = controls.aggregate_features(align, int(cfg.causal.min_active_elements),
                                            groups=gmap)
        p = np.array([causal.empirical_pvalue(a, nullA) for a in feats.A_f])
        q = bh_fdr(p)
        ids = feats.feature_id.to_numpy().astype(int)
        causal_ids = ids[q < 0.05]
        noncausal_pool = ids[q > 0.5]
        nc = RNG.choice(noncausal_pool, min(len(causal_ids), len(noncausal_pool)), replace=False)
        bases[s] = {"causal": subspace_basis(W[:, causal_ids]),
                    "noncausal": subspace_basis(W[:, nc]),
                    "n_causal": int(len(causal_ids))}
        log.info("seed %d: %d causal features -> %d-dim subspace",
                 s, len(causal_ids), bases[s]["causal"].shape[1])

    # project each locus's positions onto each subspace and test what high projection tracks
    rows = {"causal": {"repeat": [], "gc": [], "effect": []},
            "noncausal": {"repeat": [], "gc": [], "effect": []}}
    seen = set()
    for e in satmut:
        if gmap[e.elem_id] in seen:       # one representative per locus (sequences are identical)
            continue
        seen.add(gmap[e.elem_id])
        h = model.feature_input(e.seq, L).detach().cpu().numpy().astype(np.float64)
        E = np.asarray(e.E_mean, float)
        n = min(len(h), len(E))
        h, E = h[:n], E[:n]
        rmask = repeat_mask(e.seq)[:n]
        gc = gc_track(e.seq)[:n]
        for kind in ("causal", "noncausal"):
            U = bases[seeds[0]][kind]
            if U.shape[1] == 0:
                continue
            proj = np.linalg.norm(h @ U, axis=1)          # per-position projection magnitude
            rows[kind]["repeat"].append(auroc(proj, rmask) if rmask.any() else np.nan)
            rows[kind]["gc"].append(causal.spearman_rho(proj, gc))
            rows[kind]["effect"].append(causal.spearman_rho(proj, E))

    out = {"layer": L, "n_loci": len(seen),
           "subspace_dim": {k: int(bases[seeds[0]][k].shape[1]) for k in ("causal", "noncausal")},
           "n_causal_features": bases[seeds[0]]["n_causal"]}
    for kind in ("causal", "noncausal"):
        out[kind] = {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                         "frac_pos": float(np.nanmean(np.array(v) > (0.5 if k == "repeat" else 0)))}
                     for k, v in rows[kind].items()}
        log.info("[%s] repeat AUROC=%.3f | GC rho=%.3f | effect rho=%.3f", kind,
                 out[kind]["repeat"]["mean"], out[kind]["gc"]["mean"], out[kind]["effect"]["mean"])

    (art / f"subspace_bio_layer{L}.json").write_text(json.dumps(out, indent=2))
    _write(cfg, L, out)
    print(f"[phase17] causal subspace: repeat AUROC={out['causal']['repeat']['mean']:.3f}, "
          f"GC rho={out['causal']['gc']['mean']:.3f}, effect rho={out['causal']['effect']['mean']:.3f} "
          f"(non-causal: {out['noncausal']['repeat']['mean']:.3f} / "
          f"{out['noncausal']['gc']['mean']:.3f} / {out['noncausal']['effect']['mean']:.3f})")


def _write(cfg, L, out):
    rows = []
    for kind, label in (("causal", "causal subspace"), ("noncausal", "non-causal subspace (matched dim)")):
        rows.append([label, out["subspace_dim"][kind],
                     f"{out[kind]['repeat']['mean']:.3f}", f"{out[kind]['gc']['mean']:.3f}",
                     f"{out[kind]['effect']['mean']:.3f}",
                     f"{out[kind]['effect']['frac_pos']:.2f}"])
    body = f"""# Phase 17 — What does the causal subspace encode?

Layer L={L}, {out['n_loci']} loci (one representative per locus), {out['n_causal_features']} causal
features spanning a {out['subspace_dim']['causal']}-dimensional subspace. For each position we take
the norm of its residual-stream projection onto the subspace and ask what that tracks.

{md_table(["subspace", "dim", "repeat AUROC", "GC rho", "measured-effect rho", "frac loci rho>0"], rows)}

**Reading.** Values are means over loci. `repeat AUROC` = 0.5 means the projection is unrelated to
Alu/low-complexity positions; > 0.5 means the subspace is partly a repeat detector. `GC rho`
measures plain composition. `measured-effect rho` is the quantity of interest. The non-causal row
is the control: any decoder subspace of the same dimension will track *something*, so only the gap
between the rows is attributable to the causal selection.

This is a characterisation, not a mechanism: it says what the subspace covaries with in these 21
elements, which is the honest limit of what a per-position projection can show.

**CHECKPOINT 17.** Biology call on what the subspace is tracking.
"""
    write_report(cfg, "phase17", body)


if __name__ == "__main__":
    main()
