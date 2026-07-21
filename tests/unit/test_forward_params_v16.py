"""v16 parameter-scope contract (pure Python + numpy, no chemistry stack).

Pins the new surfaces added in forward v16:
  * tp_mode="file": explicit tabulated T-P/Kzz tables, CONTENT-hash keyed
    (tp_file_sha1), loud validation, and -- by design -- NO T-P Fisher rows
    (file-mode forecasts are conditional on the profile).
  * kzz_mode const/Pfunc/JM16/file with per-mode knob validation and
    inert-knob zeroing (cache hygiene).
  * Boundary conditions: use_settling gating (moldiff required, condensation
    refused), diff_esc curated choices, top/bottom flux entry canonicalization.
  * Cloud-deck Fisher marginalization: log_kappa_cloud / alpha_cloud are
    legal fisher_params ONLY with cloud_on.
"""
import numpy as np
import pytest

from jwst_tool import forward


def _table(tmp_path, kzz=True, name="prof.txt", tmin=800.0, tmax=1400.0,
           rows=8, scramble=False):
    P = np.logspace(6.9, -1.0, rows)          # dyne/cm^2, descending (deep first)
    if scramble:
        P = P.copy()
        P[2], P[3] = P[3], P[2]               # break monotonicity
    T = np.linspace(tmax, tmin, rows)
    lines = ["#(dyne/cm2) (K)" + (" (cm2/s)" if kzz else ""),
             "Pressure\tTemp" + ("\tKzz" if kzz else "")]
    for i in range(rows):
        row = f"{P[i]:.6e}\t{T[i]:.1f}"
        if kzz:
            row += "\t1.0e9"
        lines.append(row)
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _pf(path, **kw):
    base = dict(planet="wasp39b", tp_mode="file",
                tp_file=forward.TP_FILE_UPLOAD, tp_file_path=str(path),
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot",
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


# --- tp_mode="file" ---------------------------------------------------------

def test_file_mode_is_content_addressed(tmp_path):
    p1 = _table(tmp_path, name="a.txt")
    p2 = _table(tmp_path, name="b.txt")               # same content, other path
    p3 = _table(tmp_path, name="c.txt", tmax=1500.0)  # different content
    cp1 = forward.canonical_params(_pf(p1))
    assert cp1["tp_file_sha1"] and len(cp1["tp_file_sha1"]) == 16
    assert forward.params_key(_pf(p1)) == forward.params_key(_pf(p2))
    assert forward.params_key(_pf(p1)) != forward.params_key(_pf(p3))


def test_file_mode_has_no_tp_fisher_rows(tmp_path):
    assert forward.TP_PARAM_NAMES["file"] == []
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, fisher_params=["lnZ", "lnKzz"]))
    assert cp["fisher_params"] == ["lnKzz", "lnZ"]
    with pytest.raises(ValueError, match="NO T-P Fisher rows"):
        forward.canonical_params(_pf(p, fisher_params=["Tirr"]))


def test_file_mode_zeroes_parametric_tp_knobs(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, Tirr=1560.0))
    assert cp["Tirr"] == cp["Tint"] == cp["log_kappa"] == cp["log_gamma"] == 0.0
    # outside file mode the file identity is empty (cache hygiene)
    cp_iso = forward.canonical_params(_p())
    assert cp_iso["tp_file"] == "" and cp_iso["tp_file_sha1"] == ""


def test_bad_tables_rejected(tmp_path):
    with pytest.raises(ValueError, match="monotonic"):
        forward.canonical_params(_pf(_table(tmp_path, scramble=True)))
    with pytest.raises(ValueError, match="modelable window"):
        forward.canonical_params(_pf(_table(tmp_path, tmax=3300.0)))
    with pytest.raises(ValueError, match="rows"):
        forward.canonical_params(_pf(_table(tmp_path, rows=3)))
    bad = tmp_path / "nocol.txt"
    bad.write_text("#(dyne/cm2) (K)\nPress\tT\n1e6\t1000\n1e5\t900\n"
                   "1e4\t800\n1e3\t700\n")
    with pytest.raises(ValueError, match="Pressure"):
        forward.canonical_params(_pf(bad))
    with pytest.raises(ValueError, match="not found"):
        forward.canonical_params(_pf(tmp_path / "missing.txt"))


def test_canonical_params_round_trip_in_file_mode(tmp_path):
    # The GUI hands the SUBPROCESS canonical_params(params) as its params
    # file, so canonicalization must be idempotent (given a resolvable path).
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p))
    cp2 = forward.canonical_params({**cp, "tp_file_path": str(p)})
    assert cp2 == cp


# --- kzz_mode ---------------------------------------------------------------

def test_kzz_file_requires_file_tp_and_kzz_column(tmp_path):
    with pytest.raises(ValueError, match="requires tp_mode='file'"):
        forward.canonical_params(_p(kzz_mode="file"))
    no_kzz = _table(tmp_path, kzz=False, name="nokzz.txt")
    with pytest.raises(ValueError, match="Kzz.*column"):
        forward.canonical_params(_pf(no_kzz, kzz_mode="file"))
    cp = forward.canonical_params(_pf(_table(tmp_path), kzz_mode="file"))
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    assert cp["kzz_kdeep"] == 0.0


def test_kzz_parametric_modes_validate_and_zero_inert_knobs():
    cp = forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5,
                                     kzz_plev=0.1))
    assert cp["kzz_kmax"] == 1.0e5 and cp["kzz_plev"] == 0.1
    assert cp["kzz_const"] == 0.0 and cp["kzz_kdeep"] == 0.0
    cp = forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0e6))
    assert cp["kzz_kdeep"] == 1.0e6
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    with pytest.raises(ValueError, match="kzz_kmax"):
        forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e15,
                                    kzz_plev=0.1))
    with pytest.raises(ValueError, match="kzz_plev"):
        forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5,
                                    kzz_plev=1.0e5))
    with pytest.raises(ValueError, match="kzz_kdeep"):
        forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0))
    with pytest.raises(ValueError, match="unknown kzz_mode"):
        forward.canonical_params(_p(kzz_mode="scale"))


# --- boundary conditions ----------------------------------------------------

def test_settling_gating():
    cp = forward.canonical_params(_p(use_settling=True))
    assert cp["use_settling"] is True
    with pytest.raises(ValueError, match="requires use_moldiff"):
        forward.canonical_params(_p(use_settling=True, use_moldiff=False))
    with pytest.raises(ValueError, match="pins settling OFF"):
        forward.canonical_params(_p(use_settling=True, use_condense=True))


def test_diff_esc_curated_choices():
    cp = forward.canonical_params(_p(diff_esc=["H2", "H"]))
    assert cp["diff_esc"] == ["H", "H2"]          # sorted, deduped
    with pytest.raises(ValueError, match="diff_esc"):
        forward.canonical_params(_p(diff_esc=["SO2"]))


def test_diff_esc_requires_moldiff():
    # escape flux ~ TOA molecular-diffusion coefficient: with moldiff off it
    # would silently be zero, so it is refused (not silently kept).
    with pytest.raises(ValueError, match="requires use_moldiff"):
        forward.canonical_params(_p(diff_esc=["H"], use_moldiff=False))
    # empty escape list is fine with moldiff off (nothing to escape)
    cp = forward.canonical_params(_p(use_moldiff=False))
    assert cp["diff_esc"] == []


def test_vm_mol_zeroed_without_moldiff():
    # use_vm_mol is inert when moldiff is off (engine gates use_vm on both);
    # it is zeroed so it cannot fragment the cache into two identical setups.
    on = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=True))
    off = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=False))
    assert on["use_vm_mol"] is False
    assert forward.params_key(on) == forward.params_key(off)
    # with moldiff on the flag is honored
    assert forward.canonical_params(_p(use_vm_mol=True))["use_vm_mol"] is True


def test_bc_entries_canonicalized():
    cp = forward.canonical_params(_p(
        top_flux=[["H2O", 1.0e8], ["CH4", 0.0]],          # zero row dropped
        bot_flux=[["SO2", 1.0e9, 0.1]]))
    assert cp["top_flux"] == [["H2O", 1.0e8]]
    assert cp["bot_flux"] == [["SO2", 1.0e9, 0.1]]
    # defaults stay empty (cache hygiene: v15 keys unchanged by absence)
    cp0 = forward.canonical_params(_p())
    assert cp0["top_flux"] == [] and cp0["bot_flux"] == []
    assert cp0["use_settling"] is False and cp0["diff_esc"] == []
    with pytest.raises(ValueError, match="duplicate"):
        forward.canonical_params(_p(top_flux=[["H2O", 1e8], ["H2O", 2e8]]))
    with pytest.raises(ValueError, match="bad species token"):
        forward.canonical_params(_p(top_flux=[["H2 O", 1e8]]))
    with pytest.raises(ValueError, match="expected 3 fields"):
        forward.canonical_params(_p(bot_flux=[["SO2", 1e9]]))
    with pytest.raises(ValueError, match="vdep"):
        forward.canonical_params(_p(bot_flux=[["SO2", 1e9, -1.0]]))
    with pytest.raises(ValueError, match="beyond"):
        forward.canonical_params(_p(top_flux=[["H2O", 1e30]]))


# --- cloud Fisher marginalization ------------------------------------------

def test_cloud_fisher_params_require_cloud_on():
    cp = forward.canonical_params(_p(cloud_on=True,
                                     fisher_params=["lnZ",
                                                    "log_kappa_cloud",
                                                    "alpha_cloud"]))
    assert set(cp["fisher_params"]) == {"alpha_cloud", "lnZ",
                                        "log_kappa_cloud"}
    with pytest.raises(ValueError, match="cloud"):
        forward.canonical_params(_p(fisher_params=["log_kappa_cloud"]))
    # cloud rows are labeled + stepped like every other parameter
    for n in forward.CLOUD_FISHER_PARAMS:
        assert n in forward.FD_STEPS
        assert n in forward.PARAM_LABELS
        assert n in forward.PARAM_SYMBOLS


def test_cloud_fisher_params_work_in_file_mode(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, cloud_on=True,
                                      fisher_params=["lnZ",
                                                     "log_kappa_cloud"]))
    assert "log_kappa_cloud" in cp["fisher_params"]


# --- Mie condensate deck ----------------------------------------------------

def test_mie_deck_off_by_default_and_zeroed():
    cp = forward.canonical_params(_p())
    assert cp["mie_condensate"] == ""
    # the three continuous knobs are zeroed when the deck is off (cache hygiene)
    assert cp["mie_log_rg"] == cp["mie_sigmag"] == cp["mie_log_mmr"] == 0.0
    # a set deck keys the cache; two condensates never collide
    on = forward.canonical_params(_p(mie_condensate="MgSiO3"))
    assert forward.params_key(on) != forward.params_key(_p())
    fe = forward.canonical_params(_p(mie_condensate="Fe"))
    assert forward.params_key(on) != forward.params_key(fe)


def test_mie_condensate_and_ranges_validated():
    with pytest.raises(ValueError, match="mie_condensate"):
        forward.canonical_params(_p(mie_condensate="Diamond"))
    with pytest.raises(ValueError, match="mie_log_rg"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_log_rg=-1.0))
    with pytest.raises(ValueError, match="mie_sigmag"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_sigmag=5.0))
    with pytest.raises(ValueError, match="mie_log_mmr"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_log_mmr=0.0))


def test_mie_fisher_params_require_a_condensate():
    cp = forward.canonical_params(_p(
        mie_condensate="MgSiO3",
        fisher_params=["lnZ", "mie_log_rg", "mie_sigmag", "mie_log_mmr"]))
    assert set(cp["fisher_params"]) == {"lnZ", "mie_log_rg", "mie_sigmag",
                                        "mie_log_mmr"}
    with pytest.raises(ValueError, match="require a mie_condensate"):
        forward.canonical_params(_p(fisher_params=["mie_log_rg"]))
    # every Mie row is labeled + stepped like any other parameter
    for n in forward.MIE_FISHER_PARAMS:
        assert n in forward.FD_STEPS
        assert n in forward.PARAM_LABELS
        assert n in forward.PARAM_SYMBOLS
    # Mie and the power-law deck are independent (both freeable together)
    cp2 = forward.canonical_params(_p(
        cloud_on=True, mie_condensate="Fe",
        fisher_params=["log_kappa_cloud", "mie_log_mmr"]))
    assert {"log_kappa_cloud", "mie_log_mmr"} <= set(cp2["fisher_params"])


def test_mie_fisher_params_work_in_file_mode(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, mie_condensate="MgSiO3",
                                      fisher_params=["lnZ", "mie_log_rg"]))
    assert "mie_log_rg" in cp["fisher_params"]


def test_every_freeable_param_has_display_metadata():
    # Any parameter that can be freed in the Fisher forecast is also a valid
    # constraint goal and a Jacobian row, so it must carry a display label,
    # unit, symbol, and an FD step -- a missing entry KeyErrors the GUI/run.
    freeable = (set(forward.CHEM_PARAM_NAMES) | set(forward.CLOUD_FISHER_PARAMS)
                | set(forward.MIE_FISHER_PARAMS)
                | {p for ns in forward.TP_PARAM_NAMES.values() for p in ns})
    for m in (forward.PARAM_LABELS, forward.PARAM_UNITS, forward.PARAM_SYMBOLS,
              forward.FD_STEPS):
        assert not (freeable - set(m)), sorted(freeable - set(m))


# --- emission mode (science_mode) ------------------------------------------

def test_science_mode_gating():
    with pytest.raises(ValueError, match="science_mode"):
        forward.canonical_params(_p(science_mode="reflection"))
    # isothermal was removed, so the old emission+isothermal "featureless
    # blackbody" refusal is gone; guillot emission is the normal path
    cp = forward.canonical_params(_p(science_mode="emission",
                                     tp_mode="guillot", Tirr=1560.0))
    assert cp["science_mode"] == "emission"


def test_emission_star_params_and_hygiene(tmp_path):
    cp = forward.canonical_params(_p(science_mode="emission",
                                     tp_mode="guillot", Tirr=1560.0))
    # defaults come from the planet registry star
    assert cp["star_teff"] == 5485.0 and cp["star_logg"] == 4.5
    # transmission zeroes the star identity (it is noise-side only there)
    cp_t = forward.canonical_params(_p())
    assert cp_t["star_teff"] == cp_t["star_logg"] == cp_t["star_feh"] == 0.0
    # rayleigh + chord scheme are transmission-only: normalized in emission
    cp_e = forward.canonical_params(_p(science_mode="emission",
                                       tp_mode="guillot", Tirr=1560.0,
                                       use_rayleigh=True,
                                       rt_integration="trapezoid"))
    assert cp_e["use_rayleigh"] is False
    assert cp_e["rt_integration"] == "simpson"
    with pytest.raises(ValueError, match="star_teff"):
        forward.canonical_params(_p(science_mode="emission",
                                    tp_mode="guillot", Tirr=1560.0,
                                    star_teff=10000.0))
    # emission works with a tabulated profile too
    cp_f = forward.canonical_params(_pf(_table(tmp_path),
                                        science_mode="emission"))
    assert cp_f["science_mode"] == "emission"
    # the two geometries can never share a cache entry
    assert (forward.params_key(_p(science_mode="emission", tp_mode="guillot",
                                  Tirr=1560.0))
            != forward.params_key(_p(tp_mode="guillot", Tirr=1560.0)))


# --- v17 (2026-07-19 audit response) ----------------------------------------

def test_tp_table_must_reach_chemistry_bottom(tmp_path):
    """A table stopping above the chemistry-grid bottom is REFUSED: the engine
    would clamp-extend the last tabulated T isothermally over the quench
    region (v17). The standard fixture reaches 10^6.9 = 7.9e6 dyn/cm^2 and
    passes; a table ending at 1 bar must raise."""
    P = np.logspace(6.0, -1.0, 8)              # 1 bar bottom: too shallow
    lines = ["#(dyne/cm2) (K)", "Pressure\tTemp"]
    for i in range(8):
        lines.append(f"{P[i]:.6e}\t{1200.0 - 40.0 * i:.1f}")
    p = tmp_path / "shallow.txt"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="chemistry-grid bottom"):
        forward.canonical_params(_pf(p))
    # the standard fixture (spans past P_b) still validates
    forward.canonical_params(_pf(_table(tmp_path)))


def test_composition_fd_stencil_envelope():
    """FD Fisher rows for lnZ/dlnCO refuse a baseline within one 2h stencil of
    the validated range edge (v17): the stencil would otherwise silently solve
    the chemistry outside the exercised envelope (met=100 -> 122x solar,
    co=2.0 -> C/O 2.44). AD rows take no stencil and are exempt here (the
    dlnCO AD row has its own run-time b_z margin)."""
    forward.canonical_params(_p(met_x_solar=80.0, fisher_params=["lnZ"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(met_x_solar=100.0, fisher_params=["lnZ"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(met_x_solar=0.1, fisher_params=["lnZ"]))
    forward.canonical_params(_p(co_ratio=1.6, fisher_params=["dlnCO"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(co_ratio=2.0, fisher_params=["dlnCO"]))
    # no stencil under AD; and without fisher_params the value is legal
    forward.canonical_params(_p(met_x_solar=100.0, fisher_params=["lnZ"],
                                jac_method="ad"))
    forward.canonical_params(_p(met_x_solar=100.0))
