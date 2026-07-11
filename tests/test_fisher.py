"""Rank-aware Fisher tests (2026-07 audit 8.1 tests 7-8): duplicated Jacobian
columns must be reported as degenerate, near-singular matrices must not return
arbitrary finite sigmas, and the well-conditioned case must match the analytic
inverse."""
import numpy as np

from jwst_tool import fisher


def test_well_conditioned_matches_analytic():
    J = np.array([[1.0, 0.5, 0.2, 0.9],
                  [0.1, 1.3, 0.7, 0.2]])
    s = np.array([1.0, 2.0, 0.5, 1.5]) * 1e-4
    F = (J / s[None, :] ** 2) @ J.T
    diag = {}
    sig = fisher._marg_sigmas(F, 2, diag=diag)
    cov = np.linalg.inv(F)
    assert np.allclose(sig, np.sqrt(np.diag(cov)), rtol=1e-10)
    assert diag["fisher_rank"] == 2
    assert diag["condition_number"] < 1e6


def test_duplicated_parameters_flagged_unconstrained():
    """Two identical Jacobian rows = a perfect degeneracy: both must come back
    inf, never a finite number."""
    rng = np.random.default_rng(0)
    row = rng.standard_normal(50)
    other = rng.standard_normal(50)
    J = np.stack([row, row, other])
    s = np.full(50, 1e-4)
    F = (J / s[None, :] ** 2) @ J.T
    diag = {}
    sig = fisher._marg_sigmas(F, 3, diag=diag)
    assert np.isinf(sig[0]) and np.isinf(sig[1])
    assert np.isfinite(sig[2])
    assert diag["fisher_rank"] == 2
    assert diag["fisher_dimension"] == 3


def test_near_degenerate_does_not_return_garbage():
    """An almost-duplicated row (relative difference 1e-8) is numerically
    unconstrained: the old np.linalg.inv path returned a huge-but-finite
    'constraint' here without any error."""
    rng = np.random.default_rng(1)
    row = rng.standard_normal(50)
    J = np.stack([row, row * (1 + 1e-8), rng.standard_normal(50)])
    s = np.full(50, 1e-4)
    F = (J / s[None, :] ** 2) @ J.T
    sig = fisher._marg_sigmas(F, 3)
    assert np.isinf(sig[0]) and np.isinf(sig[1])
    assert np.isfinite(sig[2])
    # regression guard: plain inv would have "succeeded" silently
    assert np.all(np.isfinite(np.linalg.inv(F)))


def test_gaussian_prior_reproduces_analytic_posterior():
    """Audit 8.1 test 8: adding a known Gaussian prior as explicit precision
    must reproduce the analytic posterior covariance."""
    J = np.array([[1.0, 0.3], [0.2, 1.1], [0.5, 0.5]]).T
    s = np.array([1e-4, 2e-4, 1.5e-4])
    F = (J / s[None, :] ** 2) @ J.T
    prior_sig = np.array([0.05, 0.2])
    F_post = F + np.diag(1.0 / prior_sig ** 2)
    sig = fisher._marg_sigmas(F_post, 2)
    cov = np.linalg.inv(F + np.diag(1.0 / prior_sig ** 2))
    assert np.allclose(sig, np.sqrt(np.diag(cov)), rtol=1e-10)


def test_mode_forecast_diag_passthrough():
    result = dict(jac_bins=np.array([[1.0, 2.0, 0.5], [0.2, 0.1, 0.9]]),
                  sigma=np.array([1e-4, 1e-4, 1e-4]))
    diag = {}
    out = fisher.mode_forecast(result, ["p0"], diag=diag)
    assert set(out) == {"p0"} and np.isfinite(out["p0"])
    assert diag["fisher_dimension"] == 2 and diag["fisher_rank"] == 2
