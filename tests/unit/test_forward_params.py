"""Pure-Python validation of forward.canonical_params (no chemistry stack).

Covers the 2026-07-13 parameter-scope changes: the WASP-39b GCM special cases
(tp_mode='baseline', kzz_mode='scale', has_gcm_baseline) are REMOVED — only
explicit isothermal/Guillot profiles with constant Kzz exist — and
condensation is supported for BOTH T-P modes (arrays rebuilt on-graph from
the live temperature), gated on molecular diffusion and still exclusive with
a Fisher forecast.
"""
import pytest

from jwst_tool import forward, planets


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="isothermal", T_iso=900.0,
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def test_condensation_off_by_default():
    assert forward.canonical_params(_p())["use_condense"] is False


def test_condensation_ok_isothermal_no_fisher():
    cp = forward.canonical_params(_p(use_condense=True))
    assert cp["use_condense"] is True


def test_condensation_ok_guillot():
    # live-T rebuild makes Guillot condensation self-consistent
    cp = forward.canonical_params(_p(tp_mode="guillot", Tirr=1560.0,
                                     use_condense=True))
    assert cp["use_condense"] is True and cp["tp_mode"] == "guillot"


def test_condensation_rejected_without_moldiff():
    # the growth term IS the molecular-diffusion coefficient
    with pytest.raises(ValueError, match="molecular diffusion"):
        forward.canonical_params(_p(use_condense=True, use_moldiff=False))


def test_condensation_rejected_with_fisher():
    with pytest.raises(ValueError, match="Fisher"):
        forward.canonical_params(_p(use_condense=True, fisher_params=["lnZ"]))
    with pytest.raises(ValueError, match="Fisher"):
        forward.canonical_params(_p(use_condense=True, fisher_params=["T_iso"]))


def test_version_bumped_for_condense_cachebust():
    # a condensing spectrum must never collide with a non-condensing cache key
    on = forward.params_key(_p(use_condense=True))
    off = forward.params_key(_p(use_condense=False))
    assert on != off


def test_gcm_baseline_and_scale_are_removed():
    # no GCM profile may ever be silently substituted -- both modes raise,
    # for WASP-39b just like for every other planet
    with pytest.raises(ValueError, match="baseline"):
        forward.canonical_params(dict(planet="wasp39b", tp_mode="baseline"))
    with pytest.raises(ValueError, match="scale"):
        forward.canonical_params(_p(kzz_mode="scale", kzz_x=1.0))
    assert all("has_gcm_baseline" not in pd for pd in planets.PLANETS.values())


def test_default_tp_mode_is_isothermal():
    # the old default was the removed GCM baseline; it must now be explicit
    assert forward.canonical_params(dict(planet="wasp39b"))["tp_mode"] == "isothermal"


def test_conden_cfg_channel_is_wired():
    # enabling condensation must configure an ACTIVE channel (the pre-v7 bug:
    # use_condense=True with condense_sp=[] condensed nothing, silently)
    assert forward.CONDEN_CFG["condense_sp"] == ["S8"]
    assert forward.CONDEN_CFG["non_gas_sp"] == ["S8_l_s"]
    assert forward.CONDEN_CFG["r_p"]["S8_l_s"] > 0
    assert forward.CONDEN_CFG["rho_p"]["S8_l_s"] > 0
    # conden-window + whole-column fix pin: the upstream methodology that
    # makes condensing solves converge (transport-limited otherwise)
    assert forward.CONDEN_CFG["fix_species"] == ["S8", "S8_l_s"]
    assert forward.CONDEN_CFG["fix_species_from_coldtrap_lev"] is False
    assert forward.CONDEN_CFG["stop_conden_time"] > 0


def test_isothermal_guillot_const_are_accepted():
    for tp, extra in (("isothermal", dict(T_iso=1100.0)),
                      ("guillot", dict(Tirr=1560.0, Tint=100.0,
                                       log_kappa=-2.3, log_gamma=-1.0))):
        cp = forward.canonical_params(_p(tp_mode=tp, **extra))
        assert cp["tp_mode"] == tp
        assert cp["kzz_mode"] == "const"
