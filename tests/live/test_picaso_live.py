"""Env-gated LIVE validation of the PICASO provider + climate mode.

Run with  JWST_TOOL_RUN_PICASO_LIVE=1 python -m pytest tests/live -q
(needs picaso 4.0.1, the reference tree via JWST_TOOL_PICASO_REFDATA, and the
full JAX/ExoJax stack; minutes to ~1 h for the climate matrix -- Isaac
schedules these, never run unprompted).

What each test certifies (first measured 2026-07-20, the v18 release
validation):

* blend-vs-native: the tool's within-node interpolation reproduces picaso's
  own chem_interp to ~4e-15 dex (machine precision).
* leave-one-node-out: cross-node blending predicts a real interior node to
  ~0.03 dex median / ~0.05 dex p95 for the major species (worst-case ~0.2 dex
  CO2; alkali condensation edges can reach ~1 dex locally).
* lnZ FD closure: the reported two-cell-secant row integrates back to the
  actual finite difference of the spectrum.
* provider-vs-vulcan spectrum sanity: differences are attributable to
  disequilibrium physics (SO2 absent in equilibrium; quenching).
* climate smoke matrix: convergence + certification across the advertised
  envelope (W39b default, all three rfacv values, a hot and a cool planet,
  representative nodes). v1 is CERTIFIED around the W39b configuration;
  the rest is exactly what this matrix measures.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_PICASO_LIVE") != "1",
    reason="live PICASO validation: set JWST_TOOL_RUN_PICASO_LIVE=1")


def test_blend_matches_native_picaso_at_a_node():
    from jwst_tool import picaso_chem as pc
    from jwst_tool import picaso_env as pe
    jdi = pe.import_picaso()
    T = np.linspace(400.0, 2500.0, 60)
    P = np.logspace(-5.5, 2.0, 60)
    st = pc.evaluate(10.0, 0.55, T, P)
    case = jdi.inputs(calculation="planet")
    case.add_pt(T=T.copy(), P=P.copy())
    case.chemeq_visscher_2121(0.55, 1.0)
    df = case.inputs["atmosphere"]["profile"]
    worst = 0.0
    for j, sp in enumerate(st.species):
        name = "C-gr" if sp == pc.GRAPHITE_OUT else sp
        mask = st.y[:, j] > 0
        if not mask.any():
            continue
        d = np.max(np.abs(
            np.log10(np.maximum(st.y[mask, j], 1e-300))
            - np.log10(np.maximum(df[name].values[mask], 1e-300))))
        worst = max(worst, d)
    assert worst < 1e-10, f"within-node mismatch {worst:.3e} dex"


def test_leave_one_node_out_blend_error():
    from jwst_tool import picaso_chem as pc
    tabs = [[pc.load_node_table("feh0.3_co0.27"),
             pc.load_node_table("feh0.3_co0.55")],
            [pc.load_node_table("feh0.7_co0.27"),
             pc.load_node_table("feh0.7_co0.55")]]
    truth = pc.load_node_table("feh0.5_co0.46")
    pred = pc.blend_cubes(tabs, 0.5, (0.46 - 0.27) / (0.55 - 0.27))
    Tmask = (truth.T >= 400) & (truth.T <= 3000)
    for m, tol_med in (("H2O", 0.05), ("CO", 0.05), ("CH4", 0.05),
                       ("CO2", 0.06), ("NH3", 0.05), ("H2S", 0.05)):
        j = truth.species.index(m)
        tr = truth.cube[Tmask, :, j]
        pr = pred[Tmask, :, j]
        good = tr > -40
        med = float(np.median(np.abs(pr[good] - tr[good])))
        assert med < tol_med, f"{m}: median LOO error {med:.3f} dex"


def test_picaso_lnz_row_fd_closure():
    from jwst_tool import forward
    h = forward.PICASO_FD_STEPS["lnZ"]
    base = dict(chem_provider="picaso", co_ratio=0.50, met_x_solar=10.0)
    forward.run_model(dict(base, fisher_params=["lnZ"]), log=lambda *a: None)
    m = forward.load_result(dict(base, fisher_params=["lnZ"]))
    j = list(m["jac_names"]).index("lnZ")
    row = m["jac"][j]
    forward.run_model(dict(base, met_x_solar=10.0 * np.exp(h)),
                      log=lambda *a: None)
    forward.run_model(dict(base, met_x_solar=10.0 * np.exp(-h)),
                      log=lambda *a: None)
    dp = forward.load_result(dict(base, met_x_solar=10.0 * np.exp(h)))
    dm = forward.load_result(dict(base, met_x_solar=10.0 * np.exp(-h)))
    secant = (dp["depth"] - dm["depth"]) / (2.0 * h)
    num = float(np.max(np.abs(row - secant)))
    den = float(np.max(np.abs(secant)))
    assert num / den < 0.05, f"lnZ closure {num / den:.3f}"


def test_picaso_vs_vulcan_spectrum_sanity():
    from jwst_tool import forward
    pp = dict(chem_provider="picaso", co_ratio=0.55, met_x_solar=10.0)
    pv = dict(co_ratio=0.55, met_x_solar=10.0)
    forward.run_model(pp, log=lambda *a: None)
    forward.run_model(pv, log=lambda *a: None)
    mp = forward.load_result(pp)
    mv = forward.load_result(pv)
    # same wavelength grid, depths within a plausible disequilibrium band
    assert np.array_equal(mp["wl_um"], mv["wl_um"])
    d = np.abs(mp["depth"] - mv["depth"])
    assert float(np.median(d)) < 2.0e-3     # < ~2000 ppm median
    assert "SO2" not in list(mp["mols"]) and "SO2" in list(mv["mols"])


@pytest.mark.parametrize("planet,met,co,rfacv", [
    ("wasp39b", 10.0, 0.55, 0.5),      # the certified default
    ("wasp39b", 10.0, 0.55, 0.0),
    ("wasp39b", 10.0, 0.55, 1.0),
    ("wasp39b", 1.0, 0.55, 0.5),       # solar node
    ("hd189733b", 1.0, 0.55, 0.5),     # hotter star, high gravity
    ("wasp107b", 10.0, 0.55, 0.5),     # cool, low gravity
])
def test_climate_smoke_matrix(planet, met, co, rfacv):
    from jwst_tool import forward, picaso_climate
    cp = forward.canonical_params(dict(
        planet=planet, tp_mode="picaso_climate", met_x_solar=met,
        co_ratio=co, rfacv=rfacv))
    clim = picaso_climate.get_or_run(cp, lambda *a: None)
    assert clim.cert["converged"]
    assert clim.cert["flux_toa_over_tidal"] < picaso_climate.FLUX_BALANCE_MAX
