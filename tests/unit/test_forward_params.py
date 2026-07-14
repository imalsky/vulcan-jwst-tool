"""Pure-Python validation of forward.canonical_params (no chemistry stack).

Covers the parameter-scope contract: the WASP-39b GCM special cases
(tp_mode='baseline', kzz_mode='scale', has_gcm_baseline) are REMOVED — only
explicit isothermal/Guillot profiles with constant Kzz exist — and
condensation is REMOVED as an option (2026-07-14): use_condense is no longer
a parameter and a truthy request raises, because a condensing VULCAN column
is not reliably differentiable through its window+fix-species pin and so
cannot enter the Fisher forecast this tool is built around.
"""
import pytest

from jwst_tool import forward, planets


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="isothermal", T_iso=900.0,
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def test_condensation_not_a_parameter():
    # use_condense is no longer part of the canonical parameter set
    assert "use_condense" not in forward.canonical_params(_p())


def test_condensation_requested_raises():
    # a truthy use_condense is refused loudly, in every T-P mode and
    # regardless of moldiff / fisher (condensation is simply not offered)
    for extra in (dict(),
                  dict(use_moldiff=True),
                  dict(use_moldiff=False),
                  dict(fisher_params=["lnZ"]),
                  dict(tp_mode="guillot", Tirr=1560.0)):
        with pytest.raises(ValueError, match="condensation .* not supported"):
            forward.canonical_params(_p(use_condense=True, **extra))


def test_condensation_false_is_harmless():
    # an explicit falsy use_condense (leftover config) is not an error and
    # does not leak into the canonical params
    cp = forward.canonical_params(_p(use_condense=False))
    assert "use_condense" not in cp


def test_conden_cfg_constant_is_gone():
    # the S8 condensation config constant was removed with the option
    assert not hasattr(forward, "CONDEN_CFG")


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


def test_isothermal_guillot_const_are_accepted():
    for tp, extra in (("isothermal", dict(T_iso=1100.0)),
                      ("guillot", dict(Tirr=1560.0, Tint=100.0,
                                       log_kappa=-2.3, log_gamma=-1.0))):
        cp = forward.canonical_params(_p(tp_mode=tp, **extra))
        assert cp["tp_mode"] == tp
        assert cp["kzz_mode"] == "const"
