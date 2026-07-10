"""Detection-significance math: bin the model per instrument, combine with the
per-bin depth uncertainty, and score the science goal.

Significance of "molecule X is present" is the nested-model chi-square distance
between the full spectrum and the spectrum with X's opacity removed, evaluated
on the instrument's bins:

    sigma_detect = sqrt( sum_bins ((d_full - d_without_X) / sigma_bin)^2 )

which is the standard likelihood-ratio proxy used in proposal planning
(e.g. PandExo-based feature-SNR estimates).
"""
from __future__ import annotations

import numpy as np

from . import instruments as ins
from . import noise as noise_mod


def bin_model(wl_model: np.ndarray, depth: np.ndarray, edges: np.ndarray):
    """Mean model depth per bin; bins with no model points get NaN."""
    idx = np.digitize(wl_model, edges) - 1
    nb = len(edges) - 1
    out = np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if sel.any():
            out[b] = float(depth[sel].mean())
    return out


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol: str,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_ppm: float) -> dict:
    """One instrument mode -> binned model, sigmas, and detection significance.

    Bins cover the intersection of the mode's science band, the model's coverage,
    and the pixels pandeia actually returned.
    """
    m = ins.MODES[mode_key]
    wl_model = model["wl_um"]
    order = np.argsort(wl_model)
    wl_model = wl_model[order]
    depth = model["depth"][order]
    mols = [str(x) for x in model["mols"]]
    depth_wo = model["depth_wo"][mols.index(target_mol)][order]

    wl_pix = np.asarray(mode_result["wl"])
    lo = max(m["wl_min"], float(wl_model.min()), float(wl_pix.min()))
    hi = min(m["wl_max"], float(wl_model.max()), float(wl_pix.max()))
    if hi <= lo:
        raise ValueError(f"{mode_key}: no overlap between instrument band and model")

    edges = noise_mod.make_bins(lo, hi, R_bin)
    nz = noise_mod.depth_error_bins(mode_result, edges, t_in_s, t_out_s,
                                    n_transits, floor_ppm)
    d_full = bin_model(wl_model, depth, edges)
    d_wo = bin_model(wl_model, depth_wo, edges)

    # keep bins that have noise pixels AND model coverage
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep_noise = np.isin(np.round(centers, 12), np.round(nz["wl_center"], 12))
    keep = keep_noise & np.isfinite(d_full) & np.isfinite(d_wo)
    sig_map = dict(zip(np.round(nz["wl_center"], 12), nz["sigma"]))
    sigma = np.array([sig_map[c] for c in np.round(centers[keep], 12)])

    wl_c = centers[keep]
    d_full_b, d_wo_b = d_full[keep], d_wo[keep]
    sigma_detect = float(np.sqrt(np.sum(((d_full_b - d_wo_b) / sigma) ** 2)))

    # Fisher Jacobian, binned on the same bins (rows: free params ..., lnR0)
    jac_bins = None
    if "jac" in model:
        jac_bins = np.stack([bin_model(wl_model, row[order], edges)[keep]
                             for row in model["jac"]])

    return dict(
        jac_bins=jac_bins,
        mode_key=mode_key, label=m["label"],
        wl=wl_c, depth=d_full_b, depth_wo=d_wo_b, sigma=sigma,
        sigma_detect=sigma_detect,
        median_sigma_ppm=float(np.median(sigma) * 1e6),
        n_bins=int(keep.sum()),
        ngroup=int(mode_result["ngroup"]),
        sat_frac=float(mode_result["sat_frac"]),
        saturated=bool(mode_result.get("saturated", False)),
        t_cycle_s=float(mode_result["t_cycle_s"]),
        warnings=mode_result.get("warnings", {}),
    )
