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


def test_resolution_defaults_replace_quality():
    # the fidelity "quality" tier is gone; explicit nz/nu_pts/yconv default to
    # the old "fast" tier, and the RT layer count is derived (not a cache field)
    cp = forward.canonical_params(_p())
    assert cp["nz"] == forward.NZ_DEFAULT == 100
    assert cp["nu_pts"] == forward.NU_PTS_DEFAULT == 4000
    assert cp["yconv_cri"] == forward.YCONV_DEFAULT == 1.0e-2
    assert "quality" not in cp        # retired
    assert "art_nlayer" not in cp     # locked to nz in run_model, not cache-keyed


def test_resolution_ceiling_accepted():
    cp = forward.canonical_params(_p(nz=150, nu_pts=8000, yconv_cri=1.0e-3))
    assert (cp["nz"], cp["nu_pts"], cp["yconv_cri"]) == (150, 8000, 1.0e-3)


def test_resolution_out_of_range_raises():
    for bad in (dict(nz=40), dict(nz=200), dict(nu_pts=1000),
                dict(nu_pts=20000), dict(yconv_cri=1.0), dict(yconv_cri=1.0e-6)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_unknown_rt_molecule_points_to_engine():
    # an out-of-set RT molecule is refused loudly with how to add it
    with pytest.raises(ValueError, match="config.MOLECULES"):
        forward.canonical_params(_p(extra_mols=["PH3"]))


def test_vm_mol_is_pinned_explicitly_default_off():
    # VULCAN-JAX flipped its own default to hybrid vm_mol on 2026-07-14; the
    # tool must PIN the scheme in the canonical params (cache-keyed) instead
    # of inheriting the upstream YAML default. Default False = the validated
    # pre-flip baseline chemistry.
    cp = forward.canonical_params(_p())
    assert cp["use_vm_mol"] is False
    assert forward.canonical_params(_p(use_vm_mol=True))["use_vm_mol"] is True


def test_yconv_range_widened_floor():
    # the tolerance ladder reaches 1e-4 (strict, slow); below it still raises
    assert forward.YCONV_RANGE == (1.0e-4, 1.0e-2)
    assert forward.canonical_params(_p(yconv_cri=1.0e-4))["yconv_cri"] == 1.0e-4
    with pytest.raises(ValueError):
        forward.canonical_params(_p(yconv_cri=5.0e-5))


def test_co_baseline_is_the_cfg_elemental_basis():
    # CO_BASELINE must be the network cfg's C_H/O_H (the conserved-column
    # basis, Tsai 2023 10x-solar, C/O ~ 0.549) -- NOT the FastChem EQ-init
    # file's Lodders ratio (0.458), which only seeds the initial guess. The
    # v10 constant used the wrong basis and skewed every absolute-C/O
    # surface by 1.2x; run_model additionally cross-checks the live cfg.
    assert abs(forward.CO_BASELINE - 0.00295 / 0.00537) < 1e-12
    assert abs(forward.CO_BASELINE - 0.549) < 1e-3


def test_composition_is_structural_one_path():
    # v13: composition is ONE structural path -- co_ratio (absolute N_C/N_O)
    # and met_x_solar go straight into the cfg elemental abundances; the
    # legacy differential knobs are gone from the canonical params entirely
    cp = forward.canonical_params(_p())
    assert abs(cp["co_ratio"] - forward.CO_BASELINE) < 1e-5
    assert "dco" not in cp and "co_baseline" not in cp
    # C-rich is the same path, with NO detection-only restriction: FD Fisher
    # rows are certified re-solves, valid at any baseline
    cp = forward.canonical_params(_p(co_ratio=1.5, met_x_solar=30.0,
                                     fisher_params=["lnZ", "dlnCO"]))
    assert cp["co_ratio"] == 1.5 and cp["met_x_solar"] == 30.0


def test_composition_ranges():
    for bad in (dict(co_ratio=0.05), dict(co_ratio=2.5),
                dict(met_x_solar=0.05), dict(met_x_solar=150.0)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_fisher_works_photo_off_and_validates_names():
    # v13: FD Jacobians are certified re-solves -- no photo-on tangent regime,
    # so the old photo-on Fisher gate is gone
    cp = forward.canonical_params(_p(use_photo=False, fisher_params=["lnZ"]))
    assert cp["fisher_params"] == ["lnZ"] and cp["use_photo"] is False
    # unknown Fisher parameters are refused loudly (FD needs a defined step)
    with pytest.raises(ValueError, match="unknown Fisher parameter"):
        forward.canonical_params(_p(fisher_params=["lnFoo"]))
    with pytest.raises(ValueError, match="unknown Fisher parameter"):
        forward.canonical_params(_p(fisher_params=["Tirr"]))  # guillot-only
