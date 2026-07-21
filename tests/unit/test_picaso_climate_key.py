"""picaso_climate cache-key hygiene, certificate revalidation, and the
atm-table writer (numpy-only; no picaso, no reference data, no solves)."""
import json

import numpy as np
import pytest

from jwst_tool import forward
from jwst_tool import picaso_climate as pcl


def _cp(**kw):
    base = dict(tp_mode="picaso_climate", picaso_version="4.0.1",
                picaso_climate_sha1="cafecafecafecafe",
                picaso_ck_node="feh1.0_co0.55", tio_vo=False,
                tint_cl=200.0, rfacv=0.5, climate_rcb=60,
                rp_rjup=1.279, gs_cgs=422.0, rstar_rsun=0.932,
                orbit_au=0.04828, star_teff=5485.0, star_logg=4.5,
                star_feh=0.0,
                # keys that must NOT enter the climate cache
                chem_provider="vulcan", nz=100, nu_pts=4000,
                rt_ptop_bar=1e-8, broadening="air")
    base.update(kw)
    return base


def test_climate_subset_membership():
    sub = pcl.climate_subset(_cp())
    # provider + RT/chem-resolution knobs are EXCLUDED (the converged climate
    # is provider-independent and knows nothing about the RT)
    assert "chem_provider" not in sub
    assert "nz" not in sub and "nu_pts" not in sub
    assert "rt_ptop_bar" not in sub and "broadening" not in sub
    # everything the solve consumes is INCLUDED
    for k in ("picaso_version", "picaso_climate_sha1", "picaso_ck_node",
              "tio_vo", "tint_cl", "rfacv", "climate_rcb", "rp_rjup",
              "gs_cgs", "rstar_rsun", "orbit_au", "star_teff", "star_logg",
              "star_feh"):
        assert k in sub, k
    assert sub["_climate_version"] == pcl._CLIMATE_VERSION


def test_climate_key_fragments_on_inputs_and_tint_override():
    base = pcl.climate_key(_cp())
    assert pcl.climate_key(_cp(tio_vo=True)) != base
    assert pcl.climate_key(_cp(picaso_climate_sha1="0" * 16)) != base
    assert pcl.climate_key(_cp(), tint_override=215.0) != base
    # provider + RT knobs shared: same key
    assert pcl.climate_key(_cp(chem_provider="picaso", nz=150)) == base


def test_climate_subset_refuses_outside_mode():
    with pytest.raises(ValueError):
        pcl.climate_subset(_cp(tp_mode="isothermal"))


# --- certificate revalidation on load ---------------------------------------

def _write_cache(tmp_path, key, cert):
    p = tmp_path / f"{key}.npz"
    np.savez_compressed(p, pressure_bar=np.logspace(-6, 2, 10),
                        temperature_K=np.linspace(900, 2800, 10),
                        dtdp=np.zeros(10), cert_json=json.dumps(cert),
                        provenance_json=json.dumps({}))
    return p


def test_load_accepts_only_matching_certified_entries(tmp_path):
    good = {"climate_key": "k1", "_climate_version": pcl._CLIMATE_VERSION,
            "converged": True}
    p = _write_cache(tmp_path, "k1", good)
    assert pcl._load(p, "k1") is not None
    assert pcl._load(p, "OTHER") is None                  # key mismatch
    stale = dict(good, _climate_version=pcl._CLIMATE_VERSION - 1)
    assert pcl._load(_write_cache(tmp_path, "k2", stale), "k2") is None
    uncert = dict(good, converged=False)
    assert pcl._load(_write_cache(tmp_path, "k3", uncert), "k3") is None
    assert pcl._load(tmp_path / "absent.npz", "k4") is None
    # unreadable/foreign file: recompute, never trust it
    bad = tmp_path / "k5.npz"
    bad.write_bytes(b"not an npz")
    assert pcl._load(bad, "k5") is None


# --- certification gates (pure logic; no solver) ----------------------------

def _out(**kw):
    P = np.logspace(-6, np.log10(300.0), 91)
    T = 900.0 + 2000.0 * (np.log10(P) + 6.0) / 8.5
    base = dict(pressure=P, temperature=T, converged=True,
                cvz_locs=[0, 60, 89, 0, 0, 0],
                flux_balance={"flux_net": np.array([1e-4]),
                              "tidal": np.array([90699.2])})
    base.update(kw)
    return base


def test_certify_passes_a_sane_profile():
    cert = pcl._certify(_out(), "k")
    assert cert["converged"] and cert["flux_toa_over_tidal"] < 1e-6


def test_certify_refuses_each_gate():
    with pytest.raises(RuntimeError, match="did NOT converge"):
        pcl._certify(_out(converged=False), "k")
    with pytest.raises(RuntimeError, match="flux balance"):
        pcl._certify(_out(flux_balance={"flux_net": np.array([1e4]),
                                        "tidal": np.array([9e4])}), "k")
    hot = _out()
    hot["temperature"] = 500.0 * (hot["pressure"] / 1e-6) ** 0.7
    with pytest.raises(RuntimeError, match="gradient"):
        pcl._certify(hot, "k")
    with pytest.raises(RuntimeError, match="model top"):
        pcl._certify(_out(cvz_locs=[0, 2, 89, 0, 0, 0]), "k")


# --- atm-table writer -------------------------------------------------------

def test_write_atm_table_bottom_row_and_window(tmp_path):
    P = np.logspace(-6, np.log10(300.0), 91)
    T = 850.0 + 1900.0 * (np.log10(P) + 6.0) / 8.5      # ~2750 K at 300 bar
    path = tmp_path / "atm.txt"
    pcl._write_atm_table(P, T, path)
    tab = forward._read_tp_table(path)                   # the engine's parser
    assert tab["P_dyn"].max() == pytest.approx(forward.CHEM_P_SPAN_DYN[1])
    assert np.all(np.diff(tab["P_dyn"]) < 0)             # descending (bottom first)
    # the bottom-row T is the log-P interpolant at exactly the chem bottom
    want = float(np.interp(np.log(forward.CHEM_P_SPAN_DYN[1]),
                           np.log(P * 1e6), T))
    assert tab["T"][0] == pytest.approx(want, abs=0.01)


def test_write_atm_table_refuses_out_of_window(tmp_path):
    P = np.logspace(-6, np.log10(300.0), 91)
    T = np.full(91, 1500.0)
    T[P > 1.0] = 3400.0                                  # too hot at depth
    with pytest.raises(RuntimeError, match="modelable window"):
        pcl._write_atm_table(P, T, tmp_path / "atm.txt")


def test_guillot_guess_is_deterministic():
    p = np.logspace(-6, 2, 50)
    a = pcl.guillot_guess(p, 422.0, 200.0, 1643.4)
    b = pcl.guillot_guess(p.copy(), 422.0, 200.0, 1643.4)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) >= 0)                       # monotone with depth
