"""Rank-aware Fisher tests (2026-07 audit 8.1 tests 7-8): duplicated Jacobian
columns must be reported as degenerate, near-singular matrices must not return
arbitrary finite sigmas, and the well-conditioned case must match the analytic
inverse."""
import numpy as np
import pytest

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
    # dimension = free + lnR0 + the (always-present) constant offset (P0-A)
    assert diag["fisher_dimension"] == 3 and diag["fisher_rank"] == 3


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
    """A single-segment mode and a no-seg call must agree: both carry exactly
    one constant-offset nuisance (the global offset). The old premise here --
    "one segment adds no usable offset beyond lnR0" -- was recheck bug P0-A:
    lnR0 is a physical derivative, not a constant, so the offset is always
    its own nuisance row now."""
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


def test_mode_forecast_equals_combined_single_result():
    """2026-07-12 recheck P0-A: mode_forecast(r) and combined_forecast([r])
    must implement the SAME statistical model (free params + shared lnR0 +
    one constant offset per segment + slope rows). The old mode_forecast
    omitted the first segment's offset."""
    rng = np.random.default_rng(7)
    nb = 45
    seg = np.array([0] * 22 + [1] * 23)
    jac = rng.standard_normal((3, nb))             # 2 free + lnR0
    slope_rows = rng.standard_normal((2, nb))
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4), seg=seg,
             slope_rows=slope_rows)
    a = fisher.mode_forecast(dict(r), ["p0", "p1"])
    b = fisher.combined_forecast([dict(r)], ["p0", "p1"])
    for k in ("p0", "p1"):
        if np.isinf(a[k]) or np.isinf(b[k]):
            assert np.isinf(a[k]) and np.isinf(b[k])
        else:
            assert a[k] == pytest.approx(b[k], rel=1e-10)
    # and the single-segment / no-seg variants agree too
    r1 = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    a1 = fisher.mode_forecast(dict(r1), ["p0", "p1"])
    b1 = fisher.combined_forecast([dict(r1)], ["p0", "p1"])
    for k in ("p0", "p1"):
        assert a1[k] == pytest.approx(b1[k], rel=1e-10)


def test_constant_science_derivative_unconstrained():
    """Recheck P0-A reproducer: an exactly CONSTANT science derivative must be
    absorbed by the constant calibration offset (unconstrained), even when
    the lnR0 derivative is NOT constant (0.3 + 0.2x): lnR0 cannot stand in
    for the offset."""
    nb = 40
    x = np.linspace(0.0, 1.0, nb)
    jac = np.stack([np.ones(nb), 0.3 + 0.2 * x])   # [free=const, lnR0 nonconst]
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    sig = fisher.mode_forecast(r, ["p0"])["p0"]
    assert np.isinf(sig)
    # a science derivative with real shape stays constrained
    jac2 = np.stack([np.sin(6 * x), 0.3 + 0.2 * x])
    sig2 = fisher.mode_forecast(dict(jac_bins=jac2, sigma=np.full(nb, 1e-4)),
                                ["p0"])["p0"]
    assert np.isfinite(sig2)


def test_per_segment_constant_derivative_unconstrained():
    """A science derivative that is constant WITHIN each segment (any step
    pattern) lies in the span of the per-segment offsets -> unconstrained."""
    nb = 40
    seg = np.array([0] * 20 + [1] * 20)
    jac = np.stack([np.where(seg == 0, 0.7, -0.2), np.linspace(1, 2, nb)])
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4), seg=seg)
    assert np.isinf(fisher.mode_forecast(r, ["p0"])["p0"])


def test_mode_forecast_matches_schur_complement():
    """Marginalized sigma against an independent Schur-complement GLS
    calculation on the full nuisance-augmented design (recheck test 4)."""
    rng = np.random.default_rng(11)
    nb = 60
    seg = np.array([0] * 30 + [1] * 30)
    jac = rng.standard_normal((3, nb))             # 2 free + lnR0
    sigma = np.full(nb, 2e-4)
    r = dict(jac_bins=jac, sigma=sigma, seg=seg)
    got = fisher.mode_forecast(dict(r), ["p0", "p1"])
    # independent construction: rows = [free(2), lnR0, seg0, seg1]
    rows = np.vstack([jac, (seg == 0).astype(float), (seg == 1).astype(float)])
    F = (rows / sigma[None, :] ** 2) @ rows.T
    n_f = 2
    A = F[:n_f, :n_f]
    B = F[:n_f, n_f:]
    D = F[n_f:, n_f:]
    S = A - B @ np.linalg.solve(D, B.T)            # Schur complement
    cov = np.linalg.inv(S)
    assert got["p0"] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-9)
    assert got["p1"] == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-9)


def test_display_sigma_units():
    """Report-unit conventions: metallicity/Kzz in dex (log10), C/O as an
    ABSOLUTE number ratio (sigma_CO = C/O * sigma_lnCO), temperature in K."""
    # internal natural-log sigma of ln(10) -> exactly 1 dex
    assert fisher.display_sigma("lnZ", np.log(10.0)) == pytest.approx(1.0)
    assert fisher.display_sigma("lnKzz", np.log(10.0)) == pytest.approx(1.0)
    # C/O: absolute ratio, scaled by the atmosphere's C/O
    assert fisher.display_sigma("dlnCO", 0.1, co_eval=0.5) == pytest.approx(0.05)
    # dlnCO WITHOUT co_eval is refused loudly -- never a silent wrong scale
    with pytest.raises(ValueError, match="co_eval"):
        fisher.display_sigma("dlnCO", 0.1)
    # a plain temperature is unit-1 (K in, K out)
    assert fisher.display_sigma("T_iso", 42.0) == 42.0


def test_conditional_sigmas_bound_marginalized():
    # conditional (others fixed) <= marginalized (others free), always --
    # both read off the SAME nuisance-augmented Fisher matrix
    rng = np.random.default_rng(7)
    nb = 40
    jac = np.vstack([rng.standard_normal((2, nb)),
                     rng.standard_normal(nb) + 1.0])   # p0, p1, lnR0
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    cond = {}
    marg = fisher.mode_forecast(r, ["p0", "p1"], conditional=cond)
    for n in ("p0", "p1"):
        assert np.isfinite(cond[n]) and cond[n] > 0
        assert cond[n] <= marg[n] * (1 + 1e-12)


def test_conditional_equals_marginalized_when_orthogonal():
    # orthogonal, zero-mean parameter rows (also orthogonal to the constant
    # offset and to lnR0): F is diagonal on the report block, so conditional
    # and marginalized coincide
    nb = 8
    p0 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    p1 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    ln_r0 = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    jac = np.vstack([p0, p1, ln_r0])
    r = dict(jac_bins=jac, sigma=np.ones(nb))
    cond = {}
    marg = fisher.mode_forecast(r, ["p0", "p1"], conditional=cond)
    for n in ("p0", "p1"):
        assert np.isclose(cond[n], marg[n], rtol=1e-12)


def test_conditional_no_response_reads_inf():
    nb = 10
    jac = np.vstack([np.zeros(nb), np.ones(nb)])       # p0 dead, lnR0 alive
    cond = {}
    fisher.mode_forecast(dict(jac_bins=jac, sigma=np.ones(nb)), ["p0"],
                         conditional=cond)
    assert cond["p0"] == np.inf


def test_combined_conditional_accumulates_modes():
    # two modes must beat (or match) either alone, conditionally too
    rng = np.random.default_rng(3)
    r1 = dict(jac_bins=rng.standard_normal((2, 25)), sigma=np.full(25, 1e-4))
    r2 = dict(jac_bins=rng.standard_normal((2, 20)), sigma=np.full(20, 1e-4))
    c1, c12 = {}, {}
    fisher.combined_forecast([r1], ["p0"], conditional=c1)
    fisher.combined_forecast([r1, r2], ["p0"], conditional=c12)
    assert c12["p0"] <= c1["p0"] * (1 + 1e-12)
