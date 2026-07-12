"""Fisher-information parameter forecast from the autodiff spectrum Jacobian.

Given each instrument mode's binned Jacobian J (n_par, n_bins) and per-bin depth
sigma, the Fisher matrix is

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisance handling (mirrors the zco_information campaign):
  * per mode: the free parameters + lnR0 (reference-radius) + one constant
    depth OFFSET per detector SEGMENT are jointly fit; the nuisances are
    marginalized out of the report. A single-detector mode has one segment
    (its offset is degenerate with lnR0 and drops out via the rank-aware
    inversion); the two-detector NIRSpec gratings (G395H, G235H) get a
    separate NRS1 and NRS2 offset, because a detector-to-detector step of
    tens of ppm is universal in real fits (Moran+2023, Madhusudhan+2023) and
    can otherwise masquerade as atmospheric structure.
  * combined (all selected modes): one SHARED lnR0 plus one constant depth
    OFFSET per SEGMENT across all modes (absolute-calibration nuisance between
    detectors/visits), all marginalized. Offsets are what make
    multi-instrument combinations honest -- within a single band an offset and
    lnR0 are nearly degenerate.

A parameter with (numerically) no spectral response, or with weight in a
numerically null Fisher direction (a degeneracy), comes back as inf, shown as
"unconstrained" by the GUI rather than a fake number. The inversion is ALWAYS
rank-aware (eigendecomposition + relative threshold): np.linalg.inv on an
ill-conditioned Fisher matrix returns misleading finite numbers without
raising, so it is never used. Forecast sigmas are local Cramer-Rao lower
bounds under the quoted noise model -- best cases, not posterior widths.
"""
from __future__ import annotations

import numpy as np

_LN10 = np.log(10.0)

# report-unit conversion: sigma in ln-units -> display units
_TO_DISPLAY = {"lnZ": 1.0 / _LN10, "lnKzz": 1.0 / _LN10}

# Relative eigenvalue threshold: Fisher directions below REL_EIG_TOL x the
# largest eigenvalue are treated as numerically unconstrained (null space).
# eigh's noise floor is ~1e-16 x wmax; 1e-10 keeps 6 decades of margin while
# still spanning any physically meaningful constraint ratio.
REL_EIG_TOL = 1e-10
# A reported parameter whose loading on any null eigenvector exceeds this is
# flagged inf (it lives partly in an unconstrained direction).
NULL_LOAD_TOL = 1e-6


def _marg_sigmas(F: np.ndarray, n_report: int,
                 diag: dict | None = None) -> np.ndarray:
    """Rank-aware marginalized sigmas for the first n_report parameters of F.

    Unconditional eigendecomposition (F is symmetric PSD by construction) with
    an explicit relative threshold; parameters loaded on null directions come
    back inf. Pass a dict as ``diag`` to receive rank / dimension / condition
    number / eigenvalues (the audit-required numerical-health fields).
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    wmax = float(w[-1]) if w.size else 0.0
    good = w > REL_EIG_TOL * max(wmax, 1e-300)
    if diag is not None:
        diag.update(
            fisher_dimension=int(F.shape[0]),
            fisher_rank=int(good.sum()),
            condition_number=(wmax / float(w[good].min()) if good.any()
                              else float("inf")),
            eigenvalues=w.copy(),
            rel_eig_tol=REL_EIG_TOL,
        )
    if not good.any():
        return np.full(n_report, np.inf)
    cov_diag = ((V[:, good] ** 2) / w[good]).sum(axis=1)
    out = np.sqrt(cov_diag[:n_report])
    if (~good).any():
        load = np.abs(V[:, ~good]).max(axis=1)[:n_report]
        out[load > NULL_LOAD_TOL] = np.inf
    return out


def display_sigma(name: str, sigma: float) -> float:
    return sigma * _TO_DISPLAY.get(name, 1.0)


def _segment_offset_rows(result: dict) -> np.ndarray:
    """One indicator row per detector segment beyond the first, from
    result["seg"] (per-bin segment id from detect.evaluate_mode). Empty
    (0, n_bins) for a single-segment mode. Row s is 1 on the bins of segment
    s+1, 0 elsewhere -- the constant offset (or shared lnR0) already spans the
    first segment, so only the STEPS relative to it are added here."""
    nb = np.asarray(result["sigma"]).size
    seg = np.asarray(result.get("seg", np.zeros(nb, int)), int)
    n_seg = int(seg.max()) + 1 if seg.size else 1
    return np.stack([(seg == s).astype(float) for s in range(1, n_seg)]) \
        if n_seg > 1 else np.zeros((0, nb))


def _slope_nuisance_rows(result: dict) -> np.ndarray:
    """Per-segment slope rows the mode's noise scenario profiles (stored by
    detect.evaluate_mode; empty (0, n_bins) when the scenario floats none)."""
    nb = np.asarray(result["sigma"]).size
    return np.asarray(result.get("slope_rows", np.zeros((0, nb))), float)


def _fisher(Jn: np.ndarray, result: dict) -> np.ndarray:
    """J C^-1 J^T under the mode's noise model: the stored scenario covariance
    when present (correlated floor), else the exact diagonal fast path."""
    cov = result.get("cov")
    if cov is None:
        s = np.asarray(result["sigma"])
        return (Jn / s[None, :] ** 2) @ Jn.T
    return Jn @ np.linalg.solve(np.asarray(cov, float), Jn.T)


def mode_forecast(result: dict, free_names: list[str],
                  diag: dict | None = None) -> dict:
    """Per-mode marginalized sigmas. result needs jac_bins (n_par, n_bins) whose rows
    are [free..., lnR0], sigma (n_bins,), and (optionally) seg (n_bins,) for the
    per-segment offset nuisances, cov (correlated-scenario covariance) and
    slope_rows (per-segment slope nuisances). ``diag``: see _marg_sigmas."""
    J = np.asarray(result["jac_bins"])
    steps = _segment_offset_rows(result)          # (n_steps, n_bins)
    slope = _slope_nuisance_rows(result)          # (n_slopes, n_bins)
    Jn = np.vstack([J] + [x for x in (steps, slope) if x.size])
    F = _fisher(Jn, result)
    sig = _marg_sigmas(F, len(free_names), diag=diag)
    return dict(zip(free_names, sig))


def combined_forecast(results: list[dict], free_names: list[str],
                      diag: dict | None = None) -> dict:
    """All modes jointly: shared free params + shared lnR0 + one depth offset
    per detector SEGMENT (NRS1/NRS2 counted separately for the two-detector
    gratings) + each mode's scenario slope nuisances, all marginalized. Noise
    is block-diagonal across modes (each block the mode's scenario covariance
    or diagonal sigma; no cross-mode noise correlation is modeled)."""
    n_f = len(free_names)
    # count nuisance columns: one offset per segment + this mode's slope rows
    seg_counts, slope_counts = [], []
    for r in results:
        nb = np.asarray(r["sigma"]).size
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        seg_counts.append(int(seg.max()) + 1 if seg.size else 1)
        slope_counts.append(_slope_nuisance_rows(r).shape[0])
    n_nui = int(sum(seg_counts)) + int(sum(slope_counts))
    n_tot = n_f + 1 + n_nui                        # free + lnR0 + nuisances
    F = np.zeros((n_tot, n_tot))
    col = n_f + 1
    for r, n_seg, n_sl in zip(results, seg_counts, slope_counts):
        J = np.asarray(r["jac_bins"])             # rows: free..., lnR0
        nb = J.shape[1]
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        Jg = np.zeros((n_tot, nb))
        Jg[:n_f] = J[:n_f]
        Jg[n_f] = J[n_f]                          # shared lnR0
        for s_id in range(n_seg):                 # this mode's per-segment offsets
            Jg[col + s_id] = (seg == s_id).astype(float)
        col += n_seg
        if n_sl:                                  # this mode's slope nuisances
            Jg[col:col + n_sl] = _slope_nuisance_rows(r)
            col += n_sl
        F += _fisher(Jg, r)
    sig = _marg_sigmas(F, n_f, diag=diag)
    return dict(zip(free_names, sig))


def transits_to_target(result: dict, free_names: list[str], gp: str,
                       target_display: float, sigma_at_transits) -> dict:
    """Smallest transit count at which the marginalized (display-unit) forecast on
    ``gp`` reaches ``target_display`` -- with the systematic floor respected.

    ``sigma_at_transits(result, n) -> per-bin sigma`` comes from detect.py (photon
    variance scales 1/N, R-anchored floor does not). Returns
    dict(n=int|None, reachable=bool, sig_inf=float): ``sig_inf`` is the
    floor-limited best case (display units); ``n`` is None when the target beats
    it -- no transit count reaches the target, which the old 1/sqrt(N)
    extrapolation could never say.
    """
    from . import detect as _detect  # local import: fisher stays numpy-only otherwise

    def _sig_with(sigma, cov):
        r2 = dict(result)
        r2["sigma"] = sigma
        r2["cov"] = cov
        return display_sigma(gp, mode_forecast(r2, free_names)[gp])

    sig_inf = _sig_with(np.maximum(np.asarray(result["floor"]), 1e-30),
                        _detect.cov_at_transits(result, 1, floor_only=True))
    if not np.isfinite(sig_inf) or target_display < sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        if _sig_with(sigma_at_transits(result, n),
                     _detect.cov_at_transits(result, n)) <= target_display:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)
