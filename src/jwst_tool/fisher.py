"""Fisher-information parameter forecast from the autodiff spectrum Jacobian.

Given each instrument mode's binned Jacobian J (n_par, n_bins) and per-bin depth
sigma, the Fisher matrix is

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisance handling (mirrors the zco_information campaign):
  * per mode: the free parameters + lnR0 (reference-radius) are jointly fit;
    lnR0 is marginalized out of the report.
  * combined (all selected modes): one SHARED lnR0 plus one constant depth
    OFFSET per mode (absolute-calibration nuisance between visits), all
    marginalized. Offsets are what make multi-instrument combinations honest --
    within a single band an offset and lnR0 are nearly degenerate.

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


def mode_forecast(result: dict, free_names: list[str],
                  diag: dict | None = None) -> dict:
    """Per-mode marginalized sigmas. result needs jac_bins (n_par, n_bins) whose rows
    are [free..., lnR0] and sigma (n_bins,). ``diag``: see _marg_sigmas."""
    J = np.asarray(result["jac_bins"])
    s = np.asarray(result["sigma"])
    F = (J / s[None, :] ** 2) @ J.T
    sig = _marg_sigmas(F, len(free_names), diag=diag)
    return dict(zip(free_names, sig))


def combined_forecast(results: list[dict], free_names: list[str],
                      diag: dict | None = None) -> dict:
    """All modes jointly: shared free params + shared lnR0 + one offset per mode."""
    n_f = len(free_names)
    n_modes = len(results)
    n_tot = n_f + 1 + n_modes                     # free + lnR0 + offsets
    F = np.zeros((n_tot, n_tot))
    for m, r in enumerate(results):
        J = np.asarray(r["jac_bins"])             # rows: free..., lnR0
        s = np.asarray(r["sigma"])
        nb = J.shape[1]
        Jg = np.zeros((n_tot, nb))
        Jg[:n_f] = J[:n_f]
        Jg[n_f] = J[n_f]                          # shared lnR0
        Jg[n_f + 1 + m] = 1.0                     # this mode's depth offset
        F += (Jg / s[None, :] ** 2) @ Jg.T
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

    def _sig_with(sigma):
        r2 = dict(result); r2["sigma"] = sigma
        return display_sigma(gp, mode_forecast(r2, free_names)[gp])

    sig_inf = _sig_with(np.maximum(np.asarray(result["floor"]), 1e-30))
    if not np.isfinite(sig_inf) or target_display < sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        if _sig_with(sigma_at_transits(result, n)) <= target_display:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)
