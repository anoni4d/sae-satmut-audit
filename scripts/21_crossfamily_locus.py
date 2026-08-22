#!/usr/bin/env python
"""Cross-family identifiability, recomputed at LOCUS level so it is comparable to the noise ceiling.

The cross-family causal-set Jaccard is now a headline number: it is the one identifiability
comparison that falls clearly below the split-half noise floor, and therefore the one effect
attributable to the decomposition rather than to the size of the validation set. That argument only
holds if both quantities are computed on the same unit, so this recomputes the TopK-vs-Gated
comparison with repeat measurements collapsed to their 21 independent loci (the original ran at
measurement level).

Reuses cached TopK alignments; runs and caches the Gated ISM passes once.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/21_crossfamily_locus.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import causal, controls, sae as S
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import get_logger, md_table, write_report

log = get_logger("crossfamily-locus")
N_SEEDS = 3


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


def _jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    ma = int(cfg.causal.min_active_elements)

    sae0 = S.load_sae(cfg, L, seed=0)
    null_enc = controls.RandomFeatureSAE(model.d_model, int(cfg.causal.null_R),
                                         sae0.b_dec.detach(), cfg.device, seed=cfg.seed)
    nullA = controls.aggregate_features(
        controls.cached_alignment(cfg, model, null_enc, satmut, L, "null"), 1,
        groups=gmap).A_f.to_numpy()
    thr = causal.null_threshold(nullA, float(cfg.causal.percentile))
    log.info("layer %d, %d loci, locus-level threshold %.4f", L, len(set(gmap.values())), thr)

    fam = {}
    for family in ("topk", "gated"):
        fam[family] = {}
        for seed in range(N_SEEDS):
            sae = S.load_sae(cfg, L, seed=seed, family=family)
            tag = f"{family}_seed{seed}"
            align = controls.cached_alignment(cfg, model, sae, satmut, L, tag)
            feats = controls.aggregate_features(align, ma, groups=gmap)
            A = feats.set_index("feature_id")["A_f"]
            q = bh_fdr(np.array([causal.empirical_pvalue(a, nullA) for a in A.to_numpy()]))
            ids = A.index.to_numpy().astype(int)
            fam[family][seed] = {
                "A": A, "set": set(ids[q < 0.05].tolist()),
                "W": sae.W_dec.detach().cpu().numpy(),
            }
            log.info("[%s seed %d] %d features, frac=%.3f, q<0.05=%d", family, seed, len(A),
                     float((A.to_numpy() > thr).mean()), int((q < 0.05).sum()))

    def compare(a, b):
        """raw + decoder-matched Jaccard and matched Spearman between two runs."""
        ra, rb, cos = controls.match_features(a["W"], b["W"])
        keep = cos > float(cfg.causal.match_cos_threshold)
        pairs = [(int(x), int(y)) for x, y in zip(ra[keep], rb[keep])
                 if x in a["A"].index and y in b["A"].index]
        both = sum(1 for x, y in pairs if x in a["set"] and y in b["set"])
        either = sum(1 for x, y in pairs if x in a["set"] or y in b["set"])
        xa = np.array([a["A"].loc[x] for x, _ in pairs])
        xb = np.array([b["A"].loc[y] for _, y in pairs])
        return {"jaccard_raw": _jaccard(a["set"], b["set"]),
                "jaccard_matched": both / either if either else float("nan"),
                "spearman_matched": causal.spearman_rho(xa, xb),
                "n_matched": len(pairs), "median_cos": float(np.median(cos[keep]))
                if keep.any() else float("nan")}

    out = {"layer": L, "threshold": float(thr), "n_loci": len(set(gmap.values())),
           "within": {}, "cross": {}}
    for family in ("topk", "gated"):
        vals = [compare(fam[family][0], fam[family][s]) for s in range(1, N_SEEDS)]
        out["within"][family] = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
        log.info("within %s: raw J=%.3f matched J=%.3f spearman=%.3f", family,
                 out["within"][family]["jaccard_raw"],
                 out["within"][family]["jaccard_matched"],
                 out["within"][family]["spearman_matched"])

    cross = [compare(fam["topk"][s], fam["gated"][s]) for s in range(N_SEEDS)]
    out["cross"]["topk_gated"] = {k: float(np.nanmean([c[k] for c in cross])) for k in cross[0]}
    log.info("cross topk-gated: raw J=%.3f matched J=%.3f spearman=%.3f (median cos %.2f)",
             out["cross"]["topk_gated"]["jaccard_raw"],
             out["cross"]["topk_gated"]["jaccard_matched"],
             out["cross"]["topk_gated"]["spearman_matched"],
             out["cross"]["topk_gated"]["median_cos"])

    # contextualise against the split-half noise ceiling if it has been computed
    nc_path = art / f"noiseceiling_layer{L}.json"
    if nc_path.exists():
        nc = json.loads(nc_path.read_text())["noise_ceiling"]
        out["noise_ceiling_jaccard"] = nc["same_seed_diff_data_jaccard"]["mean"]
        log.info("noise ceiling (same SAE, split loci) J=%.3f", out["noise_ceiling_jaccard"])

    (art / f"crossfamily_locus_layer{L}.json").write_text(json.dumps(out, indent=2))
    _write(cfg, L, out)
    print(f"[phase19] locus-level cross-family raw J="
          f"{out['cross']['topk_gated']['jaccard_raw']:.3f} "
          f"(within-topk {out['within']['topk']['jaccard_raw']:.3f}, "
          f"within-gated {out['within']['gated']['jaccard_raw']:.3f}, "
          f"ceiling {out.get('noise_ceiling_jaccard', float('nan')):.3f})")


def _write(cfg, L, out):
    rows = []
    for label, d in (("within TopK (seeds)", out["within"]["topk"]),
                     ("within Gated (seeds)", out["within"]["gated"]),
                     ("**across families** (TopK vs Gated)", out["cross"]["topk_gated"])):
        rows.append([label, f"{d['jaccard_raw']:.3f}", f"{d['jaccard_matched']:.3f}",
                     f"{d['spearman_matched']:.3f}", f"{d['median_cos']:.2f}"])
    ceil = out.get("noise_ceiling_jaccard")
    ceil_line = (f"\nSplit-half noise ceiling (same SAE, disjoint halves of the loci): "
                 f"**{ceil:.3f}**.\n" if ceil is not None else "")
    body = f"""# Phase 19 — Cross-family identifiability at locus level

Layer L={L}, {out['n_loci']} independent loci, locus-level threshold {out['threshold']:.3f}.
Recomputed on the same unit as the noise ceiling so the two are directly comparable.

{md_table(["comparison", "causal-set Jaccard (raw)", "Jaccard (decoder-matched)",
           "Spearman A_f (matched)", "median matched cos"], rows)}
{ceil_line}
**Reading.** Within-family seed agreement should be read against the noise ceiling: if it sits at
the ceiling, re-seeding costs nothing beyond estimation noise. The across-family row is the
decisive one — if it falls clearly *below* the ceiling, two dictionaries disagree by more than the
validation set's resolution can explain, and that disagreement is attributable to the
decomposition rather than to sample size.

**CHECKPOINT 19.** Confirm the cross-family contrast for the paper's central claim.
"""
    write_report(cfg, "phase19", body)


if __name__ == "__main__":
    main()
