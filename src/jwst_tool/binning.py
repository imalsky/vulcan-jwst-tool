"""One measurement operator per instrument mode: count-space binning shared by
the noise, the model, and every Jacobian row.

Why this module exists (2026-07-11 external audit, P0.1): the tool used to
report inverse-variance combined bin errors (noise.py) while binning the model
and Jacobians with local wavelength/trapezoid weights (detect.py). Those are
two DIFFERENT estimators, so the forecast model point was not the expectation
value of the statistic whose variance was being quoted. Everything now goes
through one operator built from the Pandeia extracted-pixel grid:

    pixels -> bin     d_b   = sum_i(w_i d_i) / sum_i(w_i),   w_i = F_i
    variance          Var_b = sum_i(w_i^2 Var_i) / (sum_i w_i)^2
    model -> pixel    d_i   = average of the piecewise-linear model over the
                              pixel's wavelength cell (midpoint edges)

Count-space weights (w_i = extracted stellar count rate F_i) are the estimator
real reductions implement: the binned light curve is the SUM of extracted
counts across the bin's pixels, so the fitted bin depth is the flux-weighted
mean of the per-pixel depths. In the photon-dominated limit (Var_i ~ 1/F_i)
this coincides with inverse-variance weighting; where background or read noise
contribute, count-space is slightly wider -- the honest variance of the
estimator actually used. The same weights bin the model, the removed-molecule
model, and the Jacobians, so signal, derivative, and noise all describe the
same statistic.

The model is averaged over each pixel's wavelength cell (exact integral of the
piecewise-linear model), never point-sampled, so a model grid finer than the
pixel grid (e.g. MIRI LRS) cannot alias. No extra LSF blur is applied on top:
Pandeia's extraction already carries the PSF/throughput into F_i and sigma_i,
and for a depth spectrum smooth on pixel scales the residual intra-pixel LSF
correction is second order. (A full Pandeia response matrix -- monochromatic
impulses through the 3D engine -- would refine this; documented limitation.)
"""
from __future__ import annotations

import numpy as np

# A pixel's cell half-width toward a neighbor is half that gap, capped at
# GAP_CAP x the smaller of its two adjacent gaps. The cap only matters at
# detector gaps (NRS1/NRS2) and band ends, where a raw midpoint cell would
# smear the model across wavelengths the pixel never sees.
GAP_CAP = 1.5

# Pixels whose local wavelength spacing is below this fraction of the mode's
# median spacing sit on a DEGENERATE wavelength solution (e.g. pandeia_data
# 3.0rc3 G395H piles ~700 samples within <1e-4 um at the NRS2 red edge, spacing
# down to 3.7e-6 um vs 6.6e-4 median). Counting them as independent spectral
# samples overstates the information in that bin (~sqrt(n) too-small sigma)
# and mislocates their flux in wavelength, so they are excluded -- loudly, via
# the n_pix_degenerate count surfaced by detect.evaluate_mode. Real dispersion
# gradients (PRISM, MIRI LRS) vary smoothly by factors of a few, far above
# this cut.
DEGENERATE_WL_FRAC = 0.02


def degenerate_wl_mask(wl_pix: np.ndarray) -> np.ndarray:
    """True for pixels on a degenerate wavelength solution (see above).
    Returned in the INPUT pixel order."""
    wl_pix = np.asarray(wl_pix, float)
    order = np.argsort(wl_pix)
    wl_s = wl_pix[order]
    if wl_s.size < 3:
        return np.zeros(wl_pix.size, bool)
    gaps = np.diff(wl_s)
    local = np.minimum(np.concatenate([[gaps[0]], gaps]),
                       np.concatenate([gaps, [gaps[-1]]]))
    bad_sorted = local < DEGENERATE_WL_FRAC * np.median(gaps)
    bad = np.zeros(wl_pix.size, bool)
    bad[order] = bad_sorted
    return bad


def _pixel_cells(wl_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cell_lo, cell_hi) per sorted pixel: midpoint edges, gap-capped."""
    n = wl_sorted.size
    if n == 1:
        return wl_sorted.copy(), wl_sorted.copy()
    gaps = np.diff(wl_sorted)
    g_lo = np.concatenate([[gaps[0]], gaps])       # gap toward the previous pixel
    g_hi = np.concatenate([gaps, [gaps[-1]]])      # gap toward the next pixel
    g_min = np.minimum(g_lo, g_hi)
    lo = wl_sorted - np.minimum(0.5 * g_lo, GAP_CAP * g_min)
    hi = wl_sorted + np.minimum(0.5 * g_hi, GAP_CAP * g_min)
    return lo, hi


def build_operator(wl_pix: np.ndarray, w_pix: np.ndarray, edges: np.ndarray,
                   wl_lo: float = -np.inf, wl_hi: float = np.inf,
                   valid: np.ndarray | None = None) -> dict:
    """Build the count-space measurement operator for one mode.

    wl_pix, w_pix : Pandeia pixel wavelengths (any order) and weights
                    (extracted stellar count rates); w_pix must be > 0.
    edges         : final bin edges (ascending).
    wl_lo, wl_hi  : model wavelength span; pixel cells are clipped to it and
                    pixels whose cell falls outside are dropped (their model
                    expectation is undefined) -- from BOTH model and noise,
                    so the two stay the same estimator.
    valid         : optional per-pixel bool mask (e.g. saturation exclusion).

    Returns dict:
      keep     (n_bins,)  bins with >=1 usable pixel
      wl_center(n_keep,)  edge midpoints of kept bins
      n_pix    (n_keep,)  usable pixels per kept bin
      pix_idx  (n_use,)   indices into the INPUT pixel arrays (callers pass
                          full-length per-pixel arrays; the operator selects)
      pix_w, cell_lo, cell_hi (n_use,), pix_bin (n_use,) index into kept bins
    """
    wl_pix = np.asarray(wl_pix, float)
    w_pix = np.asarray(w_pix, float)
    order = np.argsort(wl_pix)
    wl_s = wl_pix[order]
    lo_s, hi_s = _pixel_cells(wl_s)
    lo_s = np.maximum(lo_s, wl_lo)
    hi_s = np.minimum(hi_s, wl_hi)

    ok = (hi_s > lo_s) & (w_pix[order] > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)[order]
    bin_raw = np.digitize(wl_s, edges) - 1
    nb = len(edges) - 1
    ok &= (bin_raw >= 0) & (bin_raw < nb)

    idx = order[ok]
    bins = bin_raw[ok]
    keep = np.zeros(nb, dtype=bool)
    keep[bins] = True
    remap = np.cumsum(keep) - 1                     # bin id -> kept-bin id
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_pix = np.bincount(remap[bins], minlength=int(keep.sum()))
    return dict(keep=keep, wl_center=centers[keep],
                n_pix=n_pix.astype(int),
                pix_idx=idx, pix_w=w_pix[idx],
                cell_lo=lo_s[ok], cell_hi=hi_s[ok],
                pix_bin=remap[bins])


def _wsum(op: dict, values_per_pixel: np.ndarray) -> np.ndarray:
    out = np.zeros(op["wl_center"].size)
    np.add.at(out, op["pix_bin"], values_per_pixel)
    return out


def bin_values(op: dict, v_pix: np.ndarray) -> np.ndarray:
    """Count-weighted mean of a per-pixel quantity, per kept bin.
    ``v_pix`` is full-length (aligned with the arrays given to build_operator)."""
    v = np.asarray(v_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] * v) / _wsum(op, op["pix_w"])


def bin_variance(op: dict, var_pix: np.ndarray) -> np.ndarray:
    """Variance of the count-weighted bin estimator: sum(w^2 Var)/(sum w)^2."""
    v = np.asarray(var_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] ** 2 * v) / _wsum(op, op["pix_w"]) ** 2


def bin_counts(op: dict, flag_pix: np.ndarray) -> np.ndarray:
    """Plain per-kept-bin sum of a per-pixel count/flag (e.g. saturation)."""
    v = np.asarray(flag_pix, float)[op["pix_idx"]]
    return _wsum(op, v)


def bin_model(op: dict, wl_model: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    """Bin a native model through the operator: exact cell average of the
    piecewise-linear model per pixel, then the count-weighted bin mean.

    ``wl_model`` must be ascending and span every pixel cell (build_operator's
    wl_lo/wl_hi clipping guarantees this). Linear in y_model, so the binned
    Jacobian is the operator applied to each Jacobian row. Exact for a
    constant model (constant-depth conservation)."""
    wl = np.asarray(wl_model, float)
    y = np.asarray(y_model, float)
    # cumulative trapezoid integral at the model nodes; linear interpolation of
    # it is exact for constant y and O(h^2) otherwise -- same order as the model
    icum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(wl))])
    ia = np.interp(op["cell_lo"], wl, icum)
    ib = np.interp(op["cell_hi"], wl, icum)
    d_pix = (ib - ia) / (op["cell_hi"] - op["cell_lo"])
    return _wsum(op, op["pix_w"] * d_pix) / _wsum(op, op["pix_w"])
