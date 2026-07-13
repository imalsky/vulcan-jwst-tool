"""Pure-Python validation of forward.canonical_params (no chemistry stack).

Covers the 2026-07-12 parameter-scope changes: condensation is gated to the
one self-consistent regime (isothermal T-P, no Fisher forecast), and the
GUI-retired baseline/scale modes stay backend-accepted for the closure test
and scripted reproducibility.
"""
import pytest

from jwst_tool import forward


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


def test_condensation_rejected_for_guillot():
    with pytest.raises(ValueError, match="isothermal"):
        forward.canonical_params(_p(tp_mode="guillot", Tirr=1560.0,
                                    use_condense=True))


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


def test_baseline_and_scale_still_backend_accepted():
    # retired from the GUI, but the validated W39b closure test + scripted
    # reproducibility still drive them directly
    cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="baseline",
                                       dT=0.0, kzz_mode="scale", kzz_x=1.0))
    assert cp["tp_mode"] == "baseline"
    assert cp["kzz_mode"] == "scale"


def test_isothermal_guillot_const_are_accepted():
    for tp, extra in (("isothermal", dict(T_iso=1100.0)),
                      ("guillot", dict(Tirr=1560.0, Tint=100.0,
                                       log_kappa=-2.3, log_gamma=-1.0))):
        cp = forward.canonical_params(_p(tp_mode=tp, **extra))
        assert cp["tp_mode"] == tp
        assert cp["kzz_mode"] == "const"
