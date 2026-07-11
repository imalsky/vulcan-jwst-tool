"""Detection-significance math: bin the model per instrument THROUGH THE SAME
count-space measurement operator as the noise, combine with the per-bin depth
uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted): the binned model is the
expectation value of the same estimator whose variance the noise module
reports. Pixels Pandeia flags as fully saturated are excluded from the
operator; partially saturated pixels are kept but counted per bin
(n_pix_partial_sat) so affected channels are visible, not silent.

Significance of "molecule X is present" is the nested-model chi-square distance
between the full spectrum and the spectrum with X's opacity removed, evaluated
on the instrument's bins -- WITH a free constant depth offset profiled out:

    chi2 = sum_b (s_b/sigma_b)^2 - (sum_b s_b/sigma_b^2)^2 / sum_b sigma_b^-2
    sigma_detect = sqrt(chi2),   s_b = d_full - d_without_X

The offset marginalization removes the common-mode (absolute-depth) part of the
molecule's contribution -- the part a real fit reabsorbs into the continuum /
reference radius -- so removing a molecule's flat continuum no longer counts as
signal. This matches how the Fisher combined forecast treats per-mode offsets.
It is a FIXED-MODEL DISTINGUISHABILITY, not a retrieval detection significance:
temperature, clouds, and the other abundances are not re-fit, so it upper-bounds
what a full retrieval would report.

Multi-transit extrapolation uses the noise-model components (photon term scales
as 1/N, the R-anchored floor does not), so "transits to target" saturates
honestly instead of promising 1/sqrt(N) forever.
"""
from __future__ import annotations

import numpy as np

from . import binning
from . import instruments as ins
from . import noise as noise_mod

# hard cap for the transits-to-target search: beyond this the answer is
# "effectively unreachable" for any real proposal anyway
N_TRANSITS_CAP = 500


def detection_significance(signal: np.ndarray, sigma: np.ndarray,
                           marginalize_offset: bool = True) -> float:
    """sqrt(Delta chi^2) of a binned signal against noise, offset-profiled.

    ``marginalize_offset=True`` (default) projects out a constant depth offset
    (see module docstring); False reproduces the raw nested-model quadrature sum.
    """
    signal = np.asarray(signal, float)
    sigma = np.asarray(sigma, float)
    chi2 = float(np.sum((signal / sigma) ** 2))
    if marginalize_offset and signal.size > 1:
        w = 1.0 / sigma ** 2
        chi2 -= float(np.sum(signal * w) ** 2 / np.sum(w))
    return float(np.sqrt(max(chi2, 0.0)))


def sigma_at_transits(result: dict, n_transits: int) -> np.ndarray:
    """Per-bin depth sigma of an evaluated mode re-scaled to ``n_transits``.

    Photon/detector variance scales as 1/N from the evaluated count; the
    R-anchored floor is N-independent.
    """
    n0 = int(result["n_transits_eval"])
    scale = n0 / float(max(1, int(n_transits)))
    return np.sqrt(np.asarray(result["var_phot"]) * scale
                   + np.asarray(result["floor"]) ** 2)


def transits_to_target(result: dict, target_sig: float) -> dict:
    """Smallest transit count reaching ``target_sig`` for the detect goal.

    Returns dict(n=int|None, reachable=bool, sig_inf=float): ``sig_inf`` is the
    floor-limited ceiling (infinite transits); ``n`` is None when the target
    exceeds it (no number of transits reaches the target -- say so instead of
    quoting an optimistic 1/sqrt(N) number).
    """
    if result.get("depth_wo") is None:
        return dict(n=None, reachable=False, sig_inf=float("nan"))
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    floor = np.asarray(result["floor"])
    sig_inf = detection_significance(signal, np.maximum(floor, 1e-30))
    if target_sig > sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, N_TRANSITS_CAP + 1):
        if detection_significance(signal, sigma_at_transits(result, n)) >= target_sig:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_ppm: float) -> dict:
    """One instrument mode -> binned model, sigmas, and detection significance.

    Bins cover the intersection of the mode's science band, the model's coverage,
    and the pixels pandeia actually returned. Model, Jacobians, and noise are all
    binned through ONE count-space operator (module docstring). ``target_mol=None``
    (the parameter-constraint science goal) skips the molecule-removed comparison:
    ``sigma_detect`` comes back NaN and ``depth_wo`` None.
    """
    m = ins.MODES[mode_key]
    wl_model = model["wl_um"]
    order = np.argsort(wl_model)
    wl_model = wl_model[order]
    depth = model["depth"][order]
    mols = [str(x) for x in model["mols"]]
    if target_mol is not None and target_mol not in mols:
        raise ValueError(
            f"target molecule {target_mol!r} is not in the cached model's RT set "
            f"{mols} -- re-run the forward model with it enabled (extra_mols)")
    depth_wo = (model["depth_wo"][mols.index(target_mol)][order]
                if target_mol is not None else None)

    wl_pix = np.asarray(mode_result["wl"])
    flux_pix = np.asarray(mode_result["flux"])
    # channel-level saturation (worker_version >= 3): fully saturated pixels are
    # excluded from the estimator; partially saturated ones kept but counted.
    n_full_sat = np.asarray(mode_result.get("n_full_sat", np.zeros(wl_pix.size)))
    n_part_sat = np.asarray(mode_result.get("n_part_sat", np.zeros(wl_pix.size)))
    # degenerate-wavelength pixels (pandeia grid artifacts, e.g. the G395H
    # red-edge pileup) claim spectral information that does not exist -- drop
    # them and report the count (binning.DEGENERATE_WL_FRAC rationale).
    degen = binning.degenerate_wl_mask(wl_pix)
    usable = (n_full_sat == 0) & ~degen

    lo = max(m["wl_min"], float(wl_model.min()), float(wl_pix[usable].min()))
    hi = min(m["wl_max"], float(wl_model.max()), float(wl_pix[usable].max()))
    if hi <= lo:
        raise ValueError(f"{mode_key}: no overlap between instrument band and model")

    edges = noise_mod.make_bins(lo, hi, R_bin)
    op = binning.build_operator(wl_pix, flux_pix, edges,
                                wl_lo=float(wl_model.min()),
                                wl_hi=float(wl_model.max()), valid=usable)
    nz = noise_mod.depth_error_bins(mode_result, edges, t_in_s, t_out_s,
                                    n_transits, floor_ppm, op=op)
    d_full_b = binning.bin_model(op, wl_model, depth)
    if depth_wo is not None:
        d_wo_b = binning.bin_model(op, wl_model, depth_wo)
        sigma_detect = detection_significance(d_full_b - d_wo_b, nz["sigma"])
    else:
        d_wo_b, sigma_detect = None, float("nan")

    # Fisher Jacobian through the SAME operator (rows: free params ..., lnR0)
    jac_bins = None
    if "jac" in model:
        jac_bins = np.stack([binning.bin_model(op, wl_model, row[order])
                             for row in model["jac"]])

    return dict(
        jac_bins=jac_bins,
        mode_key=mode_key, label=m["label"],
        wl=nz["wl_center"], depth=d_full_b, depth_wo=d_wo_b, sigma=nz["sigma"],
        var_phot=nz["var_phot"], floor=nz["floor"],
        n_transits_eval=int(nz["n_transits"]),
        sigma_detect=sigma_detect,
        median_sigma_ppm=float(np.median(nz["sigma"]) * 1e6),
        n_bins=int(nz["wl_center"].size),
        n_pix_partial_sat=binning.bin_counts(op, n_part_sat > 0).astype(int),
        n_pix_full_sat_dropped=int(np.sum(n_full_sat > 0)),
        n_pix_degenerate_dropped=int(degen.sum()),
        ngroup=int(mode_result["ngroup"]),
        sat_frac=float(mode_result["sat_frac"]),
        sat_ngroups=mode_result.get("sat_ngroups"),
        saturated=bool(mode_result.get("saturated", False)),
        t_cycle_s=float(mode_result["t_cycle_s"]),
        warnings=mode_result.get("warnings", {}),
    )
