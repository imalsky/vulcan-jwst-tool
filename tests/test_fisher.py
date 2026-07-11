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


def test_segment_offset_widens_forecast_and_absorbs_step():
    """A parameter whose signal looks like a per-detector STEP must lose all
    constraint once the segment offsets are floated (two segments -> two
    offsets span the step)."""
    nb = 40
    seg = np.array([0] * 20 + [1] * 20)
    # a science Jacobian row that is a pure detector step + lnR0 (flat)
    step = (seg == 1).astype(float)
    jac = np.stack([step, np.ones(nb)])            # [free=step-like, lnR0]
    s = np.full(nb, 1e-4)
    base = dict(jac_bins=jac, sigma=s)
    # without segment info: the step-like parameter is well constrained
    sig_no = fisher.mode_forecast(dict(base), ["p0"])["p0"]
    assert np.isfinite(sig_no)
    # with two segments: the offset step absorbs it -> unconstrained
    sig_seg = fisher.mode_forecast(dict(base, seg=seg), ["p0"])["p0"]
    assert np.isinf(sig_seg)


def test_single_segment_forecast_unchanged():
    """One detector segment adds no usable offset beyond lnR0, so the forecast
    is identical to the no-seg call (regression guard)."""
    rng = np.random.default_rng(3)
    jac = rng.standard_normal((3, 30))             # 2 free + lnR0
    s = np.full(30, 1e-4)
    seg = np.zeros(30, int)
    a = fisher.mode_forecast(dict(jac_bins=jac, sigma=s), ["p0", "p1"])
    b = fisher.mode_forecast(dict(jac_bins=jac, sigma=s, seg=seg), ["p0", "p1"])
    assert np.allclose([a["p0"], a["p1"]], [b["p0"], b["p1"]], rtol=1e-12)


def test_combined_forecast_counts_offsets_per_segment():
    """Combined Fisher must allocate one offset column per segment of every
    mode (dimension check via the diag)."""
    rng = np.random.default_rng(4)
    r1 = dict(jac_bins=rng.standard_normal((2, 25)), sigma=np.full(25, 1e-4),
              seg=np.array([0] * 12 + [1] * 13))   # 2 segments
    r2 = dict(jac_bins=rng.standard_normal((2, 20)), sigma=np.full(20, 1e-4),
              seg=np.zeros(20, int))               # 1 segment
    diag = {}
    fisher.combined_forecast([r1, r2], ["p0"], diag=diag)
    # 1 free + lnR0 + (2 + 1) segment offsets = 5
    assert diag["fisher_dimension"] == 1 + 1 + 3
