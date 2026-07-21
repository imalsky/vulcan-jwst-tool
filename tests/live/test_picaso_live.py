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


@pytest.mark.parametrize("planet,met,co,rfacv,expect", [
    ("wasp39b", 10.0, 0.55, 0.5, "certified"),   # the certified default
    # rfacv 0 (no irradiation): a Tint=200 K interior alone leaves the
    # upper atmosphere far below the 320 K opacity floor; rfacv 1
    # (dayside-only): the 7.6 bar bottom exceeds the 2980 K ceiling.
    # Both are DOCUMENTED envelope refusals (loud, never clipped) --
    # measured 2026-07-21; a silent bad profile is the failure mode here.
    ("wasp39b", 10.0, 0.55, 0.0, "window-refusal"),
    ("wasp39b", 10.0, 0.55, 1.0, "window-refusal"),
    ("wasp39b", 1.0, 0.55, 0.5, "certified"),    # solar node
    ("hd189733b", 1.0, 0.55, 0.5, "certified"),  # hotter star, high gravity
    ("wasp107b", 10.0, 0.55, 0.5, "certified"),  # cool, low gravity
])
def test_climate_smoke_matrix(planet, met, co, rfacv, expect):
    from jwst_tool import forward, picaso_climate
    cp = forward.canonical_params(dict(
        planet=planet, tp_mode="picaso_climate", met_x_solar=met,
        co_ratio=co, rfacv=rfacv))
    if expect == "window-refusal":
        with pytest.raises(RuntimeError, match="modelable window"):
            picaso_climate.get_or_run(cp, lambda *a: None)
        return
    clim = picaso_climate.get_or_run(cp, lambda *a: None)
    assert clim.cert["converged"]
    assert clim.cert["flux_toa_over_tidal"] < picaso_climate.FLUX_BALANCE_MAX


_LOCK_WORKER = """
import sys, time, warnings
warnings.filterwarnings("ignore")
from jwst_tool import forward as f
from jwst_tool import picaso_climate as pcl
tint = float(sys.argv[1])
cp = f.canonical_params({"tp_mode": "picaso_climate", "met_x_solar": 10.0,
                         "co_ratio": 0.55, "tint_cl": tint})
t0 = time.time()
clim = pcl.get_or_run(cp, lambda *a: None)
print("RESULT", clim.key, float(clim.T[-1]), round(time.time() - t0, 1),
      flush=True)
"""


def _spawn_lock_worker(tmp_path, tint):
    import subprocess, sys
    script = tmp_path / "lock_worker.py"
    script.write_text(_LOCK_WORKER)
    return subprocess.Popen([sys.executable, str(script), str(tint)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def test_lock_concurrent_solves_share_one_computation(tmp_path):
    # v18.1 lifecycle (gating): two concurrent processes on one uncached
    # key -> exactly one solves, the other waits on the flock and loads the
    # identical result. Measured pre-rewrite: 80.1 s / 79.4 s identical
    # profiles; this pins the rewritten never-unlink lifecycle.
    import numpy as np
    tint = 200.0 + float(np.random.default_rng().integers(37, 493)) / 10.0
    p1 = _spawn_lock_worker(tmp_path, tint)
    p2 = _spawn_lock_worker(tmp_path, tint)
    out1, _ = p1.communicate(timeout=900)
    out2, _ = p2.communicate(timeout=900)
    r1 = [ln for ln in out1.splitlines() if ln.startswith("RESULT")][-1].split()
    r2 = [ln for ln in out2.splitlines() if ln.startswith("RESULT")][-1].split()
    assert r1[1] == r2[1] and r1[2] == r2[2], (out1, out2)


def test_lock_released_when_holder_is_killed(tmp_path):
    # kill -9 the first (solving) process mid-solve: its flock releases with
    # it and the second process must acquire and complete the solve.
    import numpy as np, time
    tint = 300.0 + float(np.random.default_rng().integers(37, 493)) / 10.0
    holder = _spawn_lock_worker(tmp_path, tint)
    time.sleep(25.0)                     # deep inside the ~80 s solve
    holder.kill()
    holder.wait()
    survivor = _spawn_lock_worker(tmp_path, tint)
    out, _ = survivor.communicate(timeout=900)
    lines = [ln for ln in out.splitlines() if ln.startswith("RESULT")]
    assert lines, out
