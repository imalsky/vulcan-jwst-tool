"""2026-07-12 external-audit regression suite.

Three confirmed defects, each pinned here so it cannot come back:
  * the minimum noise floor now has exact PandExo semantics --
    sigma_final = max(sigma_random, floor) on the FINAL bins, with
    none / constant-ppm / wavelength-table choices, constant edge extension,
    no quadrature, no sqrt(R/100) rescaling, and no averaging below the
    floor with added transits;
  * Fisher rank detection is invariant under per-parameter unit rescaling
    (Jacobi-whitened eigendecomposition);
  * the matched-template nuisance projection is invariant under nuisance-row
    rescaling (correlation-form normal matrix).
"""
import numpy as np
import pytest

from jwst_tool import binning, detect, fisher, noise as noise_mod


def _mode_result(n_pix=200, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.sort(rng.uniform(3.0, 5.0, n_pix))
    flux = 5e3 * (1.2 + np.cos(3.0 * wl))
    return dict(wl=wl.tolist(), flux=flux.tolist(),
                noise_1int=np.sqrt(flux / 20.0).tolist(), t_cycle_s=20.0)


def _bins(mode_result, edges, floor_spec, **kw):
    return noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1,
                                      floor_spec, **kw)


# --- floor semantics (PandExo convention) -------------------------------------

def test_no_floor_returns_random_errors_exactly():
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    nz = _bins(mr, edges, None)
    assert np.array_equal(nz["sigma"], np.sqrt(nz["var_phot"]))
    assert np.all(nz["floor"] == 0.0)
    nz0 = _bins(mr, edges, 0.0)          # scalar zero == no floor
    assert np.array_equal(nz0["sigma"], nz["sigma"])


def test_constant_floor_is_hard_max_not_quadrature():
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    free = _bins(mr, edges, None)
    ppm = float(np.median(free["sigma"]) * 1e6)   # floor at the median sigma
    nz = _bins(mr, edges, ppm)
    assert np.array_equal(nz["sigma"],
                          np.maximum(np.sqrt(nz["var_phot"]), ppm * 1e-6))
    # bins already noisier than the floor are untouched (max, not quadrature:
    # quadrature would inflate them by up to sqrt(2))
    above = free["sigma"] > ppm * 1e-6
    assert above.any() and (~above).any()
    assert np.array_equal(nz["sigma"][above], free["sigma"][above])
    assert np.all(nz["sigma"][~above] == ppm * 1e-6)


def test_floor_not_rescaled_by_binning_r():
    """The entered constant floor must arrive unchanged at EVERY binning R --
    the retired convention scaled it by sqrt(R/100) for finer bins."""
    mr = _mode_result()
    for R in (50, 100, 200, 400):
        edges = noise_mod.make_bins(3.0, 5.0, R)
        nz = _bins(mr, edges, 20.0)
        assert np.all(nz["floor"] == 20.0 * 1e-6)


def test_wavelength_table_interpolation_and_edge_extension():
    table = np.array([[3.5, 10.0], [4.0, 30.0], [4.5, 20.0]])
    wl = np.array([3.0, 3.5, 3.75, 4.25, 4.5, 5.0])
    floor = noise_mod.resolve_floor(wl, table)
    assert floor == pytest.approx(
        np.array([10.0, 10.0, 20.0, 25.0, 20.0, 20.0]) * 1e-6)


def test_wavelength_table_unsorted_rows_are_sorted():
    table = np.array([[4.5, 20.0], [3.5, 10.0], [4.0, 30.0]])
    wl = np.linspace(3.0, 5.0, 7)
    ref = noise_mod.resolve_floor(
        wl, np.array([[3.5, 10.0], [4.0, 30.0], [4.5, 20.0]]))
    assert np.array_equal(noise_mod.resolve_floor(wl, table), ref)


@pytest.mark.parametrize("bad", [
    -5.0,                                           # negative scalar
    float("nan"),                                   # non-finite scalar
    np.array([[3.5, 10.0], [4.0, -1.0]]),           # negative floor value
    np.array([[3.5, 10.0], [4.0, np.inf]]),         # non-finite table
    np.array([[3.5, 10.0], [3.5, 20.0]]),           # duplicate wavelength
    np.array([[3.5, 10.0]]),                        # single row
    np.array([3.5, 10.0, 4.0]),                     # wrong shape
])
def test_invalid_floor_specs_raise(bad):
    with pytest.raises(ValueError):
        noise_mod.resolve_floor(np.array([3.0, 4.0]), bad)


def test_transits_approach_floor_from_above():
    """sigma(N) must decrease monotonically toward the floor and never cross
    below it -- the hard-minimum multi-transit behavior."""
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 10)
    nz = _bins(mr, edges, 30.0)
    result = dict(var_phot=nz["var_phot"], floor=nz["floor"], n_transits_eval=1)
    prev = detect.sigma_at_transits(result, 1)
    for n in (2, 5, 20, 100, 10000):
        cur = detect.sigma_at_transits(result, n)
        assert np.all(cur <= prev + 1e-30)
        assert np.all(cur >= nz["floor"])
        prev = cur
    assert np.allclose(detect.sigma_at_transits(result, 10 ** 9), nz["floor"],
                       rtol=1e-3, atol=0)


def test_detect_and_noise_share_the_final_sigma():
    """The sigma the detection score consumes is the SAME clamped sigma the
    noise module quotes (one final uncertainty everywhere)."""
    mr = _mode_result()
    wl = np.asarray(mr["wl"])
    edges = np.geomspace(3.0, 5.0, 10)
    op = binning.build_operator(wl, np.asarray(mr["flux"]), edges)
    nz = _bins(mr, edges, 25.0, op=op)
    sig = detect.detection_significance(np.full(nz["sigma"].size, 1e-4),
                                        nz["sigma"], marginalize_offset=False)
    assert sig == pytest.approx(
        np.sqrt(np.sum((1e-4 / nz["sigma"]) ** 2)), rel=1e-12)


# --- Fisher unit-rescaling invariance ------------------------------------------

def test_fisher_rank_and_sigmas_invariant_under_unit_rescaling():
    """Audit item 4 regression: rescale every parameter by independent factors
    spanning 1e-12..1e12; physical sigmas, rank, and the constrained subspace
    must not change. The raw-eigenvalue threshold flipped an exactly finite
    constraint to 'unconstrained' under a pure unit change."""
    rng = np.random.default_rng(0)
    J0 = rng.standard_normal((5, 60)) * np.array(
        [1e-4, 1.0, 1e3, 1e-7, 5e2])[:, None]
    J0[4] = J0[3] * 3.0                       # one exact degeneracy pair
    s = np.full(60, 1e-4)
    F0 = (J0 / s[None, :] ** 2) @ J0.T
    d0 = {}
    base = fisher._marg_sigmas(F0, 5, diag=d0)
    assert np.isinf(base[3]) and np.isinf(base[4])       # the degenerate pair
    assert np.all(np.isfinite(base[:3]))
    for trial in range(25):
        f = 10.0 ** rng.uniform(-12, 12, 5)
        Js = J0 * f[:, None]
        Fs = (Js / s[None, :] ** 2) @ Js.T
        ds = {}
        sig = fisher._marg_sigmas(Fs, 5, diag=ds) * f    # back to raw units
        assert ds["fisher_rank"] == d0["fisher_rank"]
        assert np.array_equal(np.isinf(sig), np.isinf(base))
        m = np.isfinite(base)
        assert np.allclose(sig[m], base[m], rtol=1e-7, atol=0)


def test_fisher_multi_dim_null_space_invariant_under_rescaling():
    """A 2-D null space (two independent degeneracies), with two parameters
    still constrained. Rank and the finite/inf classification must be invariant
    under per-parameter rescaling -- which requires the null-overlap test to use
    a basis-invariant subspace projection (the null eigenvectors of a degenerate
    eigenspace are only defined up to rotation)."""
    rng = np.random.default_rng(4)
    J0 = rng.standard_normal((6, 80)) * np.array(
        [1.0, 1e3, 1e-5, 1.0, 1e2, 1e-3])[:, None]
    J0[1] = 3.0 * J0[0]          # degeneracy A: params 0,1 unconstrained
    J0[3] = -2.0 * J0[2]         # degeneracy B: params 2,3 unconstrained
    s = np.full(80, 2e-4)
    F0 = (J0 / s[None, :] ** 2) @ J0.T
    d0 = {}
    base = fisher._marg_sigmas(F0, 6, diag=d0)
    assert d0["fisher_rank"] == 4                        # 6 params, 2 null dirs
    assert np.all(np.isinf(base[:4])) and np.all(np.isfinite(base[4:]))
    for _ in range(15):
        f = 10.0 ** rng.uniform(-9, 9, 6)
        Js = J0 * f[:, None]
        Fs = (Js / s[None, :] ** 2) @ Js.T
        ds = {}
        sig = fisher._marg_sigmas(Fs, 6, diag=ds) * f
        assert ds["fisher_rank"] == d0["fisher_rank"]
        assert np.array_equal(np.isinf(sig), np.isinf(base))
        m = np.isfinite(base)
        assert np.allclose(sig[m], base[m], rtol=1e-6, atol=0)


def test_fisher_zero_response_parameter_reads_inf():
    F = np.diag([4.0, 0.0])
    sig = fisher._marg_sigmas(F, 2)
    assert sig[0] == pytest.approx(0.5)
    assert np.isinf(sig[1])


# --- nuisance-row rescaling invariance -----------------------------------------

def test_detection_score_invariant_under_nuisance_row_rescaling():
    """Audit item 5 regression: the profiled score depends only on the SPAN of
    the nuisance rows. The raw-eigenvalue threshold silently dropped a valid
    down-scaled row (changing the score by factors)."""
    rng = np.random.default_rng(7)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    sigma = np.full(n, 1e-4)
    signal = 5e-4 * np.exp(-0.5 * ((wl - 4.0) / 0.1) ** 2)
    seg = (wl >= 4.0).astype(int)
    rows = detect._segment_rows(seg) + detect._slope_rows(seg, wl)
    base = detect.detection_significance(signal, sigma, nuisance=rows)
    C = noise_mod.build_cov(wl, sigma ** 2 * 0.25, sigma, "conservative")
    base_cov = detect.detection_significance(signal, sigma, nuisance=rows,
                                             cov=C)
    for trial in range(25):
        f = 10.0 ** rng.uniform(-12, 12, len(rows))
        scaled = [r * fi for r, fi in zip(rows, f)]
        got = detect.detection_significance(signal, sigma, nuisance=scaled)
        assert got == pytest.approx(base, rel=1e-9)
        got_cov = detect.detection_significance(signal, sigma,
                                                nuisance=scaled, cov=C)
        assert got_cov == pytest.approx(base_cov, rel=1e-9)


def test_detection_score_invariant_under_nuisance_basis_rotation():
    """Audit item 5, second required regression: the profiled score depends only
    on the SPAN of the nuisance rows, so ANY nonsingular remix (rotation + mixing
    + scaling), not just per-row rescaling, must leave it unchanged."""
    rng = np.random.default_rng(9)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    sigma = np.full(n, 1e-4)
    signal = 5e-4 * np.exp(-0.5 * ((wl - 4.0) / 0.1) ** 2)
    seg = (wl >= 4.0).astype(int)
    rows = detect._segment_rows(seg) + detect._slope_rows(seg, wl)
    R = np.stack(rows)
    base = detect.detection_significance(signal, sigma, nuisance=rows)
    C = noise_mod.build_cov(wl, sigma ** 2 * 0.25, sigma, "conservative")
    base_cov = detect.detection_significance(signal, sigma, nuisance=rows, cov=C)
    trials = 0
    while trials < 20:
        M = rng.standard_normal((len(rows), len(rows)))
        if abs(np.linalg.det(M)) < 1e-3:            # keep M well-conditioned
            continue
        trials += 1
        mixed = list(M @ R)                          # arbitrary basis of the span
        assert detect.detection_significance(signal, sigma, nuisance=mixed) \
            == pytest.approx(base, rel=1e-7)
        assert detect.detection_significance(signal, sigma, nuisance=mixed,
                                             cov=C) == pytest.approx(base_cov,
                                                                     rel=1e-7)


def test_zero_nuisance_row_is_ignored():
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1e-4, 30)
    sigma = np.full(30, 1e-4)
    a = detect.detection_significance(signal, sigma)
    b = detect.detection_significance(signal, sigma,
                                      nuisance=[np.zeros(30)])
    assert b == pytest.approx(a, rel=1e-12)
