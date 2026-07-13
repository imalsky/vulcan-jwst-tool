"""Detection-score + noise-math tests (2026-07 revised audit): offset/segment
profiling in the matched-template score, and the loud sub-cycle-window error."""
import numpy as np
import pytest

from jwst_tool import detect, noise as noise_mod


def test_constant_offset_profiled_out():
    """A pure constant depth signal carries no distinguishing information once
    the offset is profiled (it is the offset)."""
    sig = np.full(20, 3e-5)
    err = np.full(20, 1e-5)
    assert detect.detection_significance(sig, err, marginalize_offset=True) == \
        pytest.approx(0.0, abs=1e-9)
    # without profiling it is just the quadrature sum
    raw = detect.detection_significance(sig, err, marginalize_offset=False)
    assert raw == pytest.approx(np.sqrt(np.sum((sig / err) ** 2)))


def test_segment_step_profiled_out():
    """A signal that is a per-detector STEP (one level on NRS1, another on
    NRS2) must profile to ~0 once the segment offsets are supplied -- otherwise
    a calibration step reads as a molecular detection."""
    seg = np.array([0] * 15 + [1] * 15)
    signal = np.where(seg == 0, 2e-5, 9e-5)        # different level per detector
    err = np.full(seg.size, 1e-5)
    steps = detect._segment_rows(seg)
    # offset alone cannot remove a two-level step
    only_off = detect.detection_significance(signal, err, marginalize_offset=True)
    assert only_off > 5.0
    # offset + segment step removes it entirely
    with_seg = detect.detection_significance(signal, err, nuisance=steps,
                                             marginalize_offset=True)
    assert with_seg == pytest.approx(0.0, abs=1e-6)


def test_real_feature_survives_offset_and_step():
    """A localized band (not flat, not a step) keeps most of its S/N under
    offset+step profiling."""
    seg = np.array([0] * 20 + [1] * 20)
    wl = np.linspace(3.0, 5.0, 40)
    signal = 8e-5 * np.exp(-0.5 * ((wl - 4.05) / 0.05) ** 2)   # narrow SO2-like
    err = np.full(wl.size, 1e-5)
    steps = detect._segment_rows(seg)
    raw = detect.detection_significance(signal, err, marginalize_offset=False)
    prof = detect.detection_significance(signal, err, nuisance=steps,
                                         marginalize_offset=True)
    assert prof > 0.8 * raw       # a real feature is barely touched


def test_pixel_variance_raises_on_subcycle_window():
    """A transit window shorter than one integration cycle must raise, not
    silently pretend one integration fits (the retired max(1, ...))."""
    mode_result = dict(wl=[3.0, 3.1], flux=[1e3, 1e3],
                       noise_1int=[30.0, 30.0], t_cycle_s=100.0)
    with pytest.raises(ValueError, match="shorter than one integration"):
        noise_mod.pixel_depth_variance(mode_result, t_in_s=50.0, t_out_s=3600.0,
                                       n_transits=1)
    with pytest.raises(ValueError, match="n_transits"):
        noise_mod.pixel_depth_variance(mode_result, t_in_s=3600.0,
                                       t_out_s=3600.0, n_transits=0)


def test_noise_inflation_scales_variance():
    """noise_inflation multiplies sigma (variance by its square) and averages
    down with transits like the photon term."""
    rng = np.random.default_rng(0)
    wl = np.sort(rng.uniform(3.0, 5.0, 300))
    flux = np.full(wl.size, 1e3)
    noise = np.full(wl.size, 30.0)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise.tolist(), t_cycle_s=20.0)
    edges = noise_mod.make_bins(3.05, 4.95, 60.0)
    a = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0)
    b = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0,
                                   noise_inflation=1.2)
    assert np.allclose(b["var_phot"], a["var_phot"] * 1.2 ** 2)


def test_one_bin_offset_profiles_to_zero():
    """2026-07-12 recheck P2-D: with a free constant offset, a single bin has
    no shape information -- the score must be 0, not |s|/sigma (the old
    size>1 guard returned a false 3-sigma 'detection')."""
    s = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                      marginalize_offset=True)
    assert s == pytest.approx(0.0, abs=1e-9)
    # consistency: two identical bins were already 0; one bin now matches
    s2 = detect.detection_significance(np.array([3e-4, 3e-4]),
                                       np.array([1e-4, 1e-4]),
                                       marginalize_offset=True)
    assert s2 == pytest.approx(0.0, abs=1e-9)
    # with the offset explicitly disabled the one-bin score is |s|/sigma
    s3 = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                       marginalize_offset=False)
    assert s3 == pytest.approx(3.0, rel=1e-12)


def _lsf_mode_inputs(depth_baseline):
    """Minimal evaluate_mode inputs for a low-R mode where the native-R blur
    is active: PRISM-like R_native=100, narrow Jacobian feature."""
    wl_pix = np.linspace(1.0, 2.0, 600)
    flux = np.full(wl_pix.size, 1e6)
    mode_result = dict(
        wl=wl_pix.tolist(), flux=flux.tolist(),
        noise_1int=np.full(wl_pix.size, 1e3).tolist(),
        t_cycle_s=10.0, r_native=np.full(wl_pix.size, 100.0).tolist(),
        n_full_sat=np.zeros(wl_pix.size).tolist(),
        n_part_sat=np.zeros(wl_pix.size).tolist(),
        ngroup=10, sat_frac=0.5, saturated=False)
    wl_model = np.linspace(0.95, 2.05, 4000)
    jac_row = 1e-3 * np.exp(-0.5 * ((wl_model - 1.5) / 0.002) ** 2)
    model = dict(wl_um=wl_model, depth=depth_baseline(wl_model),
                 mols=["H2O"], jac=[jac_row], jac_names=["p0"])
    return mode_result, model


def test_jacobian_lsf_does_not_depend_on_baseline_shape():
    """2026-07-12 recheck P1-C: the LSF is a linear operator on every vector;
    whether the BASELINE happens to be a fixed point of the blur (e.g. an
    exactly flat depth) must not decide whether Jacobian rows are smoothed.
    The binned Jacobian must be identical for a flat and a broad-bump
    baseline, and must differ from the unsmoothed native-R=inf case."""
    mr_flat, model_flat = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr_bump, model_bump = _lsf_mode_inputs(
        lambda wl: 5e-3 * np.exp(-0.5 * ((wl - 1.5) / 0.2) ** 2))
    kw = dict(target_mol=None, R_bin=200.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    r_flat = detect.evaluate_mode("nirspec_prism", mr_flat, model_flat, **kw)
    r_bump = detect.evaluate_mode("nirspec_prism", mr_bump, model_bump, **kw)
    assert np.allclose(r_flat["jac_bins"][0], r_bump["jac_bins"][0],
                       rtol=0, atol=1e-15)
    # and the blur genuinely acts on the narrow feature: an identical setup
    # with no r_native (no blur) must differ by many ppm at the feature
    mr_none, model_none = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr_none["r_native"] = None
    r_none = detect.evaluate_mode("nirspec_prism", mr_none, model_none, **kw)
    assert np.max(np.abs(r_none["jac_bins"][0] - r_flat["jac_bins"][0])) > 5e-6


# --- fail-fast input validation (2026-07-13 recheck 5.1, 5.2) ----------------

def test_detection_significance_rejects_bad_inputs():
    good_s = np.array([3e-4, 1e-4, 2e-4])
    good_sig = np.full(3, 1e-4)
    # baseline still works
    assert np.isfinite(detect.detection_significance(good_s, good_sig))
    with pytest.raises(ValueError, match="signal"):
        detect.detection_significance(np.array([[1.0, 2.0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="signal"):
        detect.detection_significance(np.array([1e-4, np.nan]), np.full(2, 1e-4))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.array([1e-4, 0.0, 1e-4]))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.array([1e-4, np.nan, 1e-4]))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.full(2, 1e-4))       # shape
    with pytest.raises(ValueError, match="nuisance row"):
        detect.detection_significance(good_s, good_sig,
                                      nuisance=[np.ones(2)])
    # covariance: non-square, non-finite, non-symmetric, non-PD all raise
    with pytest.raises(ValueError, match="cov shape"):
        detect.detection_significance(good_s, good_sig, cov=np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        detect.detection_significance(good_s, good_sig,
                                      cov=np.array([[1.0, 2.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 1.0]]) * 1e-8)
    with pytest.raises(ValueError, match="positive-definite"):
        detect.detection_significance(good_s, good_sig, cov=-np.eye(3) * 1e-8)


def test_sigma_and_cov_at_transits_reject_bad_n():
    result = dict(n_transits_eval=1, var_phot=np.full(4, 1e-8),
                  floor=np.zeros(4), wl=np.linspace(3, 4, 4), scenario="random")
    assert detect.sigma_at_transits(result, 3).shape == (4,)   # valid
    for bad in (0, -2, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            detect.sigma_at_transits(result, bad)
        with pytest.raises(ValueError, match="positive integer"):
            detect.cov_at_transits(result, bad)
