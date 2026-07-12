"""Closure tests (2026-07 audit item E): the noise model against SYNTHETIC
COUNTS (not Gaussian depth draws), the covariance metric against Monte Carlo,
and -- opt-in, slow -- the autodiff Jacobian against finite differences of the
full forward model.

The default-run tests stay numpy-only. The FD test runs the real VULCAN-JAX ->
ExoJAX forward three times (~5-10 min); enable with JWST_TOOL_RUN_SLOW=1
(heavy validation is scheduled by the owner, never run by default)."""
import os

import numpy as np
import pytest

from jwst_tool import binning, noise as noise_mod


def test_poisson_count_closure():
    """Simulate the MEASUREMENT: in/out-of-transit integrations as Poisson
    counts per pixel, the depth estimator binned through the count-space
    operator. The empirical bin mean/variance must close with the analytic
    depth_error_bins variance built from the same exposure numbers."""
    rng = np.random.default_rng(11)
    n_pix = 300
    wl = np.sort(rng.uniform(3.0, 4.0, n_pix))
    flux = 4e3 * (1.1 + np.sin(4.0 * wl))          # e-/s per pixel
    t_int = 20.0                                    # one integration cycle (s)
    n_in, n_out = 120, 150                          # integrations in the window
    # pure-photon pandeia surrogate: sigma of a 1-integration rate estimate
    noise_1int = np.sqrt(flux / t_int)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise_1int.tolist(), t_cycle_s=t_int)

    edges = np.array([3.0, 3.25, 3.5, 3.75, 4.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=2.9, wl_hi=4.1)
    nz = noise_mod.depth_error_bins(mode_result, edges,
                                    t_in_s=n_in * t_int, t_out_s=n_out * t_int,
                                    n_transits=1, floor_spec=None, op=op)

    depth_true = 0.012 * (1.0 + 0.2 * np.sin(6.0 * wl))
    lam_in = flux * t_int * n_in * (1.0 - depth_true)
    lam_out = flux * t_int * n_out
    n_mc = 3000
    c_in = rng.poisson(lam_in, size=(n_mc, n_pix))
    c_out = rng.poisson(lam_out, size=(n_mc, n_pix))
    d_hat = 1.0 - (c_in / n_in) / (c_out / n_out)
    est = np.stack([binning.bin_values(op, d) for d in d_hat])

    d_true_bin = binning.bin_values(op, depth_true)
    se = np.sqrt(nz["var_phot"] / n_mc)
    # mean closes to the true binned depth (ratio-estimator bias ~1/counts,
    # negligible at ~1e7 counts/pixel); variance closes to the analytic value
    assert np.all(np.abs(est.mean(axis=0) - d_true_bin) < 5.0 * se + 1e-7)
    assert np.allclose(est.var(axis=0), nz["var_phot"], rtol=0.15)


def test_matched_filter_amplitude_variance_closure():
    """The S/N calibration behind sigma_detect: for template u and noise
    covariance C, the profiled amplitude estimate A = (u^T C^-1 y)/(u^T C^-1 u)
    on pure-noise draws must have variance 1/(u^T C^-1 u) -- i.e. the quoted
    sigma_detect is the true S/N unit under the scenario covariance. Also
    pins the failure of the diagonal assumption on correlated noise."""
    rng = np.random.default_rng(13)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    var_phot = np.full(n, 4e-11)
    floor = np.full(n, 1.5e-5)
    C = noise_mod.build_cov(wl, var_phot, floor, "conservative")
    L = np.linalg.cholesky(C)
    u = np.exp(-0.5 * ((wl - 4.0) / 0.15) ** 2)     # smooth-ish template

    ci_u = np.linalg.solve(C, u)
    info = float(u @ ci_u)                          # u^T C^-1 u
    n_mc = 6000
    y = rng.standard_normal((n_mc, n)) @ L.T        # y ~ N(0, C)
    a_hat = (y @ ci_u) / info
    assert np.var(a_hat) == pytest.approx(1.0 / info, rel=0.1)
    # the DIAGONAL metric under-quotes the amplitude variance on this noise:
    w = 1.0 / np.maximum(var_phot, floor ** 2)
    a_diag = (y @ (w * u)) / float(u @ (w * u))
    assert np.var(a_diag) > 2.0 * (1.0 / float(u @ (w * u)))


@pytest.mark.skipif(os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
                    reason="slow: 3 full VULCAN-JAX+ExoJAX forward runs "
                           "(~5-10 min, JAX required); set JWST_TOOL_RUN_SLOW=1")
def test_jacobian_row_matches_finite_difference():
    """One cached warm-started-jvp Jacobian row against a central finite
    difference of two cold re-solved forward models. dT (baseline T-P shift)
    keeps both FD runs on the single-stage chemistry path (met=10x, dco=0).

    Agreement gate is deliberately loose: 'fast' quality certifies chemistry
    at yconv 1e-2, and the FD endpoints re-converge cold while the jvp rides
    the warm continuation -- shape correlation plus ~15% scale is what
    steady-state uniqueness guarantees at this tolerance."""
    from jwst_tool import forward

    def quiet(_s):
        return None

    base = dict(planet="wasp39b", quality="fast", tp_mode="baseline",
                fisher_params=["dT"], use_photo=True)
    if forward.load_result(base) is None:
        forward.run_model(base, log=quiet)
    m0 = forward.load_result(base)
    names = [str(x) for x in m0["jac_names"]]
    row = np.asarray(m0["jac"][names.index("dT")])

    h = 2.0                                          # K, small vs T ~ 1000 K
    d = {}
    for s, tag in ((+h, "p"), (-h, "m")):
        p = dict(base, dT=float(s), fisher_params=[])
        if forward.load_result(p) is None:
            forward.run_model(p, log=quiet)
        d[tag] = np.asarray(forward.load_result(p)["depth"])
    fd = (d["p"] - d["m"]) / (2.0 * h)

    corr = np.corrcoef(row, fd)[0, 1]
    scale = float(np.dot(row, fd) / np.dot(fd, fd))
    assert corr > 0.99
    assert scale == pytest.approx(1.0, abs=0.15)
