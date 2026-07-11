"""Operator tests for the count-space measurement operator (2026-07 audit 8.1):
constant-depth conservation, estimator/variance consistency (Monte Carlo),
model-vs-noise same-estimator identity, Jacobian linearity, gap handling."""
import numpy as np
import pytest

from jwst_tool import binning, noise as noise_mod


def _fake_pixels(rng, n=800, wl0=1.0, wl1=5.0):
    wl = np.sort(rng.uniform(wl0, wl1, n))
    flux = 1e3 * (1.2 + np.sin(3.0 * wl)) * rng.uniform(0.8, 1.2, n)
    return wl, flux


def test_constant_depth_conservation():
    rng = np.random.default_rng(0)
    wl, flux = _fake_pixels(rng)
    edges = noise_mod.make_bins(1.1, 4.9, 80.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=0.9, wl_hi=5.1)
    wl_model = np.linspace(0.9, 5.1, 3000)
    d0 = 0.0123
    binned = binning.bin_model(op, wl_model, np.full(wl_model.size, d0))
    # exact up to float64 cumsum roundoff (sequential sum over the model grid)
    assert np.allclose(binned, d0, rtol=0, atol=1e-12)


def test_model_binning_is_linear():
    """Binned Jacobian == Jacobian of the binned model (operator linearity)."""
    rng = np.random.default_rng(1)
    wl, flux = _fake_pixels(rng)
    edges = noise_mod.make_bins(1.1, 4.9, 60.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=0.9, wl_hi=5.1)
    wl_model = np.linspace(0.9, 5.1, 2000)
    a = 1e-2 * (1 + np.sin(8 * wl_model))
    b = 1e-3 * np.cos(5 * wl_model)
    lhs = binning.bin_model(op, wl_model, 2.0 * a + 3.0 * b)
    rhs = 2.0 * binning.bin_model(op, wl_model, a) + 3.0 * binning.bin_model(op, wl_model, b)
    assert np.allclose(lhs, rhs, rtol=1e-12)


def test_estimator_mc_mean_and_variance():
    """The reported (bin value, bin variance) must be the mean and variance of
    the actual count-weighted estimator -- the audit's central consistency
    requirement, checked by Monte Carlo."""
    rng = np.random.default_rng(2)
    wl, flux = _fake_pixels(rng, n=400, wl0=2.0, wl1=3.0)
    edges = np.array([2.0, 2.3, 2.6, 3.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=3.1)
    wl_model = np.linspace(1.9, 3.1, 1500)
    depth = 0.01 * (1.0 + 0.3 * np.sin(6 * wl_model))
    sig_pix = 5e-4 * rng.uniform(0.5, 2.0, wl.size)

    d_true_bin = binning.bin_model(op, wl_model, depth)
    var_bin = binning.bin_variance(op, sig_pix ** 2)

    # simulate the estimator: per-pixel depth draws, count-weighted bin means
    d_pix_expect = np.interp(wl, wl_model, depth)
    n_mc = 4000
    draws = d_pix_expect[None, :] + sig_pix[None, :] * rng.standard_normal((n_mc, wl.size))
    est = np.stack([binning.bin_values(op, d) for d in draws])
    mc_mean, mc_var = est.mean(axis=0), est.var(axis=0)

    # mean matches the binned model within 5 MC standard errors (+ the O(h^2)
    # cell-average vs point-sample difference, < 1e-6 on this smooth model)
    se = np.sqrt(var_bin / n_mc)
    assert np.all(np.abs(mc_mean - d_true_bin) < 5.0 * se + 1e-6)
    assert np.allclose(mc_var, var_bin, rtol=0.15)


def test_noise_and_model_share_bins_and_estimator():
    """depth_error_bins with the same operator returns bins aligned 1:1 with the
    binned model, and count-space variance (not inverse-variance)."""
    rng = np.random.default_rng(3)
    wl, flux = _fake_pixels(rng, n=300, wl0=2.0, wl1=4.0)
    noise = np.sqrt(flux) * rng.uniform(0.9, 1.1, wl.size)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise.tolist(), t_cycle_s=20.0)
    edges = noise_mod.make_bins(2.05, 3.95, 50.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=4.1)
    nz = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0, op=op)
    assert nz["wl_center"].shape == op["wl_center"].shape
    var_pix = noise_mod.pixel_depth_variance(mode_result, 3600.0, 3600.0, 1)
    assert np.allclose(nz["var_phot"], binning.bin_variance(op, var_pix))
    # count-space variance >= inverse-variance combination (equality iff
    # weights are exactly ivar-optimal)
    inv = np.zeros(op["wl_center"].size)
    np.add.at(inv, op["pix_bin"], 1.0 / var_pix[op["pix_idx"]])
    assert np.all(nz["var_phot"] >= 1.0 / inv - 1e-18)


def test_detector_gap_does_not_leak_model():
    """Pixels at a detector-gap edge must not average the model across the gap
    (cell half-widths are capped)."""
    wl = np.concatenate([np.linspace(2.0, 2.5, 100), np.linspace(3.5, 4.0, 100)])
    flux = np.full(wl.size, 1e3)
    edges = np.array([2.0, 2.6, 3.4, 4.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=4.1)
    # a model that is 0 on the blue side, 1 in the gap, 0 on the red side:
    wl_model = np.linspace(1.9, 4.1, 5000)
    d = ((wl_model > 2.6) & (wl_model < 3.4)).astype(float)
    binned = binning.bin_model(op, wl_model, d)
    # kept bins are the two detector sides; gap-only bin has no pixels
    assert op["keep"].tolist() == [True, False, True]
    assert np.all(binned < 0.02)


def test_full_sat_pixels_excluded_via_valid_mask():
    rng = np.random.default_rng(4)
    wl, flux = _fake_pixels(rng, n=200, wl0=2.0, wl1=3.0)
    valid = np.ones(wl.size, bool)
    valid[:50] = False
    edges = np.array([2.0, 2.5, 3.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=3.1, valid=valid)
    assert not np.isin(op["pix_idx"], np.where(~valid)[0]).any()


def test_rebinning_nested_vs_direct():
    """Audit 8.1 test 5 analogue: binning pixels directly to coarse bins equals
    count-weighted recombination of fine bins built from the same pixels."""
    rng = np.random.default_rng(5)
    wl, flux = _fake_pixels(rng, n=1000, wl0=2.0, wl1=4.0)
    coarse = np.geomspace(2.0, 4.0, 9)
    fine = np.geomspace(2.0, 4.0, 33)          # 4 fine bins per coarse bin
    wl_model = np.linspace(1.9, 4.1, 4000)
    d = 0.01 * (1 + 0.2 * np.sin(7 * wl_model))
    op_c = binning.build_operator(wl, flux, coarse, wl_lo=1.9, wl_hi=4.1)
    op_f = binning.build_operator(wl, flux, fine, wl_lo=1.9, wl_hi=4.1)
    d_c = binning.bin_model(op_c, wl_model, d)
    d_f = binning.bin_model(op_f, wl_model, d)
    wsum_f = np.zeros(op_f["wl_center"].size)
    np.add.at(wsum_f, op_f["pix_bin"], op_f["pix_w"])
    # recombine fine bins into coarse with their pixel-count weights
    fmap = np.digitize(op_f["wl_center"], coarse) - 1
    num = np.zeros(len(coarse) - 1)
    den = np.zeros(len(coarse) - 1)
    np.add.at(num, fmap, wsum_f * d_f)
    np.add.at(den, fmap, wsum_f)
    assert np.allclose(num[den > 0] / den[den > 0], d_c, rtol=1e-12)


def test_degenerate_wavelength_pixels_flagged():
    """A pileup of near-duplicate wavelengths (the pandeia_data-3.0rc3 G395H
    red-edge artifact) must be flagged; smooth dispersion gradients must not."""
    base = np.linspace(2.0, 4.0, 500)
    pile = 3.0 + np.arange(300) * 4e-6            # 300 samples within 1.2e-3 um
    wl = np.concatenate([base, pile])
    bad = binning.degenerate_wl_mask(wl)
    assert bad[500:].all()
    assert not bad[:499].any()
    # a smooth 5x dispersion change across the band is NOT degenerate
    t = np.linspace(0, 1, 800)
    wl_smooth = 5.0 + np.cumsum(0.001 * (1 + 4 * t))
    assert not binning.degenerate_wl_mask(wl_smooth).any()
