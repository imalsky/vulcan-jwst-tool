"""Detection-significance math: bin the model per instrument THROUGH THE SAME
count-space measurement operator as the noise, combine with the per-bin depth
uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted): the binned model is the
expectation value of the same estimator whose variance the noise module
reports. Pixels Pandeia flags as fully saturated are excluded from the
operator; partially saturated pixels are kept but counted per bin
(n_pix_partial_sat) so affected channels are visible, not silent.

For modes whose final bins approach the NATIVE resolving power (MIRI LRS
R~40-160, NIRSpec PRISM R~30-300, blue SOSS) the model is first blurred to
the instrument's R(lambda) exported by the pandeia worker
(binning.smooth_to_native_r); for high-R gratings this is automatically a
no-op, matching the sub-ppm edge-effect estimate at R_bin=100.

sigma_detect is a CONDITIONAL MATCHED-TEMPLATE S/N at the specified
atmospheric state: the nested-model chi-square distance between the full
spectrum and the spectrum with one molecule's opacity removed, with the
calibration nuisances profiled out --

    chi2 = s^T W s - b^T A^{-1} b,   W = diag(1/sigma^2),
    b = U W s,  A = U W U^T,         s_b = d_full - d_without_X

where the rows of U are one constant depth offset PLUS one step per extra
detector segment (NRS1|NRS2 for the two-detector NIRSpec gratings: real
G395H fits universally float such offsets, at the tens-of-ppm level --
Moran+2023, Madhusudhan+2023). The offset/step profiling removes the part of
the molecule's signature a real fit would reabsorb into the continuum or
per-detector calibration. It is NOT a retrieval detection significance:
temperature, clouds, and the other abundances are not re-fit, so it
upper-bounds what a full retrieval would report. When a Fisher Jacobian is
available, ``sigma_detect_proj`` additionally projects the template against
the T-P and lnR0 derivative directions (still conditional -- chemistry and
clouds stay fixed) and is the number to prefer for narrow margins.

Multi-transit extrapolation uses the noise-model components (photon term
scales as 1/N, the R-anchored floor does not), so "transits to target"
saturates honestly instead of promising 1/sqrt(N) forever.
"""
from __future__ import annotations

import numpy as np

from . import binning
from . import instruments as ins
from . import noise as noise_mod

# hard cap for the transits-to-target search: beyond this the answer is
# "effectively unreachable" for any real proposal anyway
N_TRANSITS_CAP = 500

# Jacobian rows treated as NUISANCE directions for sigma_detect_proj:
# temperature-structure parameters and the reference radius. Chemistry rows
# (lnZ, dlnCO, lnKzz) are the science axes -- projecting them out would eat
# the very signal being scored. Must track forward.TP_PARAM_NAMES.
_NUISANCE_JAC = frozenset(
    {"dT", "T_iso", "Tirr", "Tint", "log_kappa", "log_gamma", "lnR0"})


def _segment_rows(seg: np.ndarray) -> list[np.ndarray]:
    """Indicator rows (one per detector segment beyond the first) for the
    per-segment calibration-offset nuisances. Together with the constant
    offset they span exactly the per-segment offset space."""
    seg = np.asarray(seg, int)
    return [(seg == s).astype(float) for s in range(1, int(seg.max()) + 1)]


def detection_significance(signal: np.ndarray, sigma: np.ndarray,
                           nuisance: list[np.ndarray] | None = None,
                           marginalize_offset: bool = True) -> float:
    """sqrt(Delta chi^2) of a binned signal against noise, with linear
    nuisance directions profiled out (rank-aware).

    ``marginalize_offset=True`` (default) includes a constant depth offset;
    ``nuisance`` adds arbitrary extra rows (detector-segment steps, binned
    T-P/lnR0 Jacobian rows). Directions that are numerically null in the
    weighted normal matrix are dropped rather than inverted.
    """
    signal = np.asarray(signal, float)
    sigma = np.asarray(sigma, float)
    w = 1.0 / sigma ** 2
    chi2 = float(np.sum(w * signal ** 2))
    rows = ([np.ones_like(signal)] if marginalize_offset and signal.size > 1
            else [])
    rows += [np.asarray(r, float) for r in (nuisance or [])]
    if rows:
        U = np.stack(rows)
        A = (U * w) @ U.T
        b = (U * w) @ signal
        ew, ev = np.linalg.eigh(0.5 * (A + A.T))
        good = ew > 1e-12 * max(float(ew[-1]), 1e-300)
        if good.any():
            proj = ev[:, good].T @ b
            chi2 -= float(np.sum(proj ** 2 / ew[good]))
    return float(np.sqrt(max(chi2, 0.0)))


def sigma_at_transits(result: dict, n_transits: int) -> np.ndarray:
    """Per-bin depth sigma of an evaluated mode re-scaled to ``n_transits``.

    Photon/detector variance (inflation included) scales as 1/N from the
    evaluated count; the R-anchored floor is N-independent.
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
    quoting an optimistic 1/sqrt(N) number). Segment offsets stay profiled.
    """
    if result.get("depth_wo") is None:
        return dict(n=None, reachable=False, sig_inf=float("nan"))
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    steps = _segment_rows(result["seg"]) if "seg" in result else []
    floor = np.asarray(result["floor"])
    sig_inf = detection_significance(signal, np.maximum(floor, 1e-30),
                                     nuisance=steps)
    if target_sig > sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, N_TRANSITS_CAP + 1):
        if detection_significance(signal, sigma_at_transits(result, n),
                                  nuisance=steps) >= target_sig:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_ppm: float, noise_inflation: float = 1.0) -> dict:
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

    # blur the model to the instrument's native R(lambda) where the worker
    # exported it (worker_version >= 4); a no-op for high-R modes. Pixel cells
    # extend at most one native pixel past the bin span, hence the margin.
    r_native = mode_result.get("r_native")
    lsf_applied = False
    jac_rows = None
    if "jac" in model:
        jac_rows = [np.asarray(row)[order] for row in model["jac"]]
    if r_native is not None:
        r_nat = np.asarray(r_native, float)
        b_lo = max(float(wl_model.min()), lo * 0.97)
        b_hi = min(float(wl_model.max()), hi * 1.03)
        depth_sm = binning.smooth_to_native_r(wl_model, depth, wl_pix, r_nat,
                                              b_lo, b_hi)
        lsf_applied = bool(np.any(depth_sm != depth))
        depth = depth_sm
        if depth_wo is not None:
            depth_wo = binning.smooth_to_native_r(wl_model, depth_wo, wl_pix,
                                                  r_nat, b_lo, b_hi)
        if jac_rows is not None and lsf_applied:
            jac_rows = [binning.smooth_to_native_r(wl_model, row, wl_pix,
                                                   r_nat, b_lo, b_hi)
                        for row in jac_rows]

    edges = noise_mod.make_bins(lo, hi, R_bin)
    op = binning.build_operator(wl_pix, flux_pix, edges,
                                wl_lo=float(wl_model.min()),
                                wl_hi=float(wl_model.max()), valid=usable)
    nz = noise_mod.depth_error_bins(mode_result, edges, t_in_s, t_out_s,
                                    n_transits, floor_ppm, op=op,
                                    noise_inflation=noise_inflation)

    # detector segments (NRS1|NRS2 for the two-detector gratings) -> one
    # calibration-offset nuisance per segment in every score/forecast
    seg_full = np.zeros(wl_pix.size, int)
    seg_full[usable] = binning.segment_ids(wl_pix[usable])
    seg = binning.bin_segments(op, seg_full)
    steps = _segment_rows(seg)

    d_full_b = binning.bin_model(op, wl_model, depth)
    jac_bins = None
    jac_names = ([str(x) for x in model["jac_names"]]
                 if "jac_names" in model else [])
    if jac_rows is not None:
        jac_bins = np.stack([binning.bin_model(op, wl_model, row)
                             for row in jac_rows])
    if depth_wo is not None:
        d_wo_b = binning.bin_model(op, wl_model, depth_wo)
        s_b = d_full_b - d_wo_b
        sigma_detect = detection_significance(s_b, nz["sigma"], nuisance=steps)
        # nuisance-projected variant: also profile the T-P + lnR0 Jacobian
        # directions (chemistry/clouds stay fixed -- still conditional)
        sigma_detect_proj = float("nan")
        if jac_bins is not None and jac_names:
            nuis = steps + [jac_bins[i] for i, n in enumerate(jac_names)
                            if n in _NUISANCE_JAC]
            sigma_detect_proj = detection_significance(s_b, nz["sigma"],
                                                       nuisance=nuis)
    else:
        d_wo_b, sigma_detect, sigma_detect_proj = None, float("nan"), float("nan")

    keep = op["keep"]
    return dict(
        jac_bins=jac_bins,
        mode_key=mode_key, label=m["label"],
        wl=nz["wl_center"],
        wl_eff=binning.bin_values(op, wl_pix),
        bin_lo=edges[:-1][keep], bin_hi=edges[1:][keep],
        seg=seg, n_segments=int(seg.max()) + 1 if seg.size else 1,
        depth=d_full_b, depth_wo=d_wo_b, sigma=nz["sigma"],
        var_phot=nz["var_phot"], floor=nz["floor"],
        noise_infl=float(noise_inflation), lsf_applied=lsf_applied,
        n_transits_eval=int(nz["n_transits"]),
        sigma_detect=sigma_detect, sigma_detect_proj=sigma_detect_proj,
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
