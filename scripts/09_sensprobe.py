#!/usr/bin/env python
"""Sensitivity-matched probe: is the per-position effect decodable from the model's per-position
ISM SENSITIVITY (not its activation value), via a single stable direction?

The phase-9 probe used activation values h[p]; the satmut effect and the SAE test are about
*sensitivity* to mutation. This closes that gap: for each position p we form the model-space
sensitivity vector s[p] = mean over alt bases of |h(x)[p]-h(x'_p)[p]| in R^d (how mutating p
perturbs the layer-L residual stream), then run leave-one-element-out ridge s[p] -> E_e[p].
A high, stable (cross-fold cosine) predictive direction means the causal *sensitivity* signal is
linearly present and identifiable in the model, even though unsupervised SAE features are not.

Run:  DNA_INTERP_CONFIG=config/real.yaml python scripts/09_sensprobe.py
"""
import json
from pathlib import Path

import numpy as np

from _common import boot
from dna_interp import probe
from dna_interp.activations import load_model
from dna_interp.data import load_satmut, locus_groups
from dna_interp.utils import BASES, get_logger, write_report

log = get_logger("sensprobe")
_ALT = {b: [c for c in BASES if c != b] for b in BASES}


def chosen_layer(cfg):
    if cfg.activations.chosen_layer is not None:
        return int(cfg.activations.chosen_layer)
    f = Path(cfg.paths["artifacts"]) / "chosen_layer.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def leave_element_out(blocks, alpha, groups=None):
    """Locus-grouped CV when `groups` is given; identical to the old element-level CV otherwise."""
    r = probe.cross_val_probe(blocks, probe.make_ridge(alpha), groups=groups)
    return r.rhos, r.dir_stability_cos


def sensitivity_block(model, seq, L, batch=64):
    """s[p] in R^d = mean over alts of |h(x)[p]-h(x'_p)[p]|."""
    h0, lens0 = model.hidden([seq], L)
    h0 = h0[0, : lens0[0]].numpy().astype(np.float64)
    Lp = h0.shape[0]
    var = [(p, b) for p in range(min(len(seq), Lp)) for b in _ALT[seq[p]]]
    S = np.zeros((Lp, h0.shape[1])); cnt = np.zeros(Lp)
    for i in range(0, len(var), batch):
        chunk = var[i : i + batch]
        seqs = [seq[:p] + b + seq[p + 1 :] for p, b in chunk]
        h, lens = model.hidden(seqs, L); h = h.numpy()
        for j, (p, b) in enumerate(chunk):
            if p < lens[j]:
                S[p] += np.abs(h[j, p] - h0[p]); cnt[p] += 1
    return S / np.clip(cnt, 1, None)[:, None]


def main():
    cfg = boot()
    L = chosen_layer(cfg)
    art = Path(cfg.paths["artifacts"])
    model = load_model(cfg)
    satmut = load_satmut(cfg)
    gmap = locus_groups(satmut)
    log.info("sensprobe: layer %d, d=%d, %d measurements over %d loci",
             L, model.d_model, len(satmut), len(set(gmap.values())))

    sens_blocks, seq_blocks, names, groups = [], [], [], []
    for k, e in enumerate(satmut):
        S = sensitivity_block(model, e.seq, L, batch=int(cfg.ism.batch))
        E = np.asarray(e.E_mean, dtype=np.float64)
        n = min(len(S), len(E))
        if n < 5:
            continue
        sens_blocks.append((S[:n], E[:n]))
        seq_blocks.append((probe.seq_features(e.seq[:n]), E[:n]))
        names.append(e.elem_id)
        groups.append(gmap[e.elem_id])
        if k % 5 == 0:
            log.info("  %d/%d measurements done", k + 1, len(satmut))

    out = {"layer": L, "d": int(model.d_model), "n_measurements": len(sens_blocks),
           "n_loci": len(set(groups)), "alphas": {}}
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        rho, cos = leave_element_out(sens_blocks, alpha, groups=groups)
        out["alphas"][alpha] = {"median_rho": float(np.nanmedian(rho)),
                                "mean_rho": float(np.nanmean(rho)),
                                "frac_pos": float(np.mean(rho > 0)), "dir_stability_cos": cos}
        log.info("alpha=%.0f: sensitivity probe median rho=%.3f (mean %.3f, frac>0 %.2f) stab=%.3f",
                 alpha, np.nanmedian(rho), np.nanmean(rho), np.mean(rho > 0), cos)
        if best is None or np.nanmedian(rho) > best[1]:
            best = (alpha, float(np.nanmedian(rho)), rho, cos)
    alpha, _, rho_s, cos_s = best
    rho_seq, _ = leave_element_out(seq_blocks, alpha, groups=groups)
    res_locus = probe.cross_val_probe(sens_blocks, probe.make_ridge(alpha),
                                      groups=groups, names=names)
    rho_elem, cos_elem = leave_element_out(sens_blocks, alpha)      # legacy, for the delta

    out.update({"best_alpha": alpha,
                "sensitivity_probe": {"median_rho": float(np.nanmedian(rho_s)),
                                      "mean_rho": float(np.nanmean(rho_s)),
                                      "frac_pos": float(np.mean(rho_s > 0)),
                                      "dir_stability_cos": cos_s,
                                      "per_locus": {n: float(r) for n, r in
                                                    zip(res_locus.fold_names, res_locus.rhos)}},
                "sensitivity_probe_element_level": {
                    "median_rho": float(np.nanmedian(rho_elem)),
                    "frac_pos": float(np.mean(rho_elem > 0)),
                    "dir_stability_cos": cos_elem},
                "composition_probe": {"median_rho": float(np.nanmedian(rho_seq)),
                                      "frac_pos": float(np.mean(rho_seq > 0))}})
    (art / f"sensprobe_layer{L}.json").write_text(json.dumps(out, indent=2))

    sp = out["sensitivity_probe"]; cp = out["composition_probe"]
    se = out["sensitivity_probe_element_level"]
    body = f"""# Phase 9b — Sensitivity-matched probe (apples-to-apples with the SAE test)

Layer L={L}, d={out['d']}, **{out['n_measurements']} measurements over {out['n_loci']} independent
loci**, leave-one-*locus*-out ridge (alpha={alpha:.0f}).
Features = per-position model ISM sensitivity s[p]=mean_alt|h(x)[p]-h(x')[p]| in R^{out['d']}
(the same *sensitivity* quantity the per-feature causal test uses, not activation value).

| probe | median rho(pred,E) | mean | frac folds >0 | weight-dir stability (cos) |
| --- | --- | --- | --- | --- |
| **model ISM sensitivity** s[p] | **{sp['median_rho']:.3f}** | {sp['mean_rho']:.3f} | {sp['frac_pos']:.2f} | **{sp['dir_stability_cos']:.3f}** |
| sequence composition | {cp['median_rho']:.3f} | - | {cp['frac_pos']:.2f} | - |
| _(same probe, element-level CV — leaky)_ | _{se['median_rho']:.3f}_ | - | _{se['frac_pos']:.2f}_ | _{se['dir_stability_cos']:.3f}_ |

Folds hold out a whole locus: Kircher assays several cell lines / timepoints of the same element
over a byte-identical sequence, so leave-one-*element*-out trains on the held-out sequence itself
(last row, shown only for comparison).

This is sensitivity-matched to both the satmut assay and the SAE feature test. A stable direction
means the causal *sensitivity* signal is linearly identifiable in the model; contrast with the
unsupervised SAE result (phase7/8) where no stable causal feature set exists. Together with phase9
(activation-value probe) it makes the probe-vs-SAE contrast apples-to-apples.

**CHECKPOINT 9b.** Confirm the sensitivity-matched contrast for the writeup.
"""
    write_report(cfg, "phase9b", body)
    print(f"[phase9b] sensitivity-probe median rho={sp['median_rho']:.3f} "
          f"(dir-stability cos={sp['dir_stability_cos']:.3f}); composition={cp['median_rho']:.3f}; alpha={alpha:.0f}")


if __name__ == "__main__":
    main()
