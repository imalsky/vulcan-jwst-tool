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
