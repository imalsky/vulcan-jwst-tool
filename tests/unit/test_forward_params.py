"""Pure-Python validation of forward.canonical_params (no chemistry stack).

Covers the parameter-scope contract. The WASP-39b GCM special cases
(tp_mode='baseline', kzz_mode='scale', has_gcm_baseline) are REMOVED: only
explicit isothermal/Guillot profiles with constant Kzz exist. Condensation
(use_condense, v14) is a canonical parameter again, accepted as a
DETECTION-ONLY forward option; canonical_params refuses it in combination
with fisher_params under ANY jac_method (the pinned condensed reservoir is a
step-sequence-dependent transient, so neither AD nor FD rows through it are
trustworthy), with photochemistry off (no certifiable steady state), and
with molecular diffusion off (the growth term IS the moldiff coefficient).
"""
import pytest

from jwst_tool import forward, planets


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="isothermal", T_iso=900.0,
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def test_condensation_is_detection_only():
    # v14: use_condense is a canonical parameter again (default False), and
    # a detection-only condensing run (photo + moldiff on, no Fisher) is
    # ACCEPTED -- in both T-P modes
    cp = forward.canonical_params(_p())
    assert cp["use_condense"] is False
    cp = forward.canonical_params(_p(use_condense=True))
    assert cp["use_condense"] is True
    cp = forward.canonical_params(_p(use_condense=True, tp_mode="guillot",
                                     Tirr=1560.0))
    assert cp["use_condense"] is True


def test_condensation_refuses_every_derivative_combination():
    # the method-science compatibility matrix: the pinned reservoir is not a
    # reproducible function of the parameters, so condensation + Fisher is
    # refused under EVERY jac_method (FD included, not just AD)
    for jm in ("fd", "ad"):
        with pytest.raises(ValueError, match="ANY Jacobian method"):
            forward.canonical_params(_p(use_condense=True, jac_method=jm,
                                        fisher_params=["lnZ"]))
    # a cold no-photo condensing column has no certifiable steady state
    with pytest.raises(ValueError, match="requires photochemistry ON"):
        forward.canonical_params(_p(use_condense=True, use_photo=False))
    # the condensation growth term IS the molecular-diffusion coefficient
    with pytest.raises(ValueError, match="requires molecular diffusion"):
        forward.canonical_params(_p(use_condense=True, use_moldiff=False))


def test_conden_cfg_is_the_certified_recipe():
    # the S8 channel ships the certified convergence recipe: whole-column
    # pin, mtol_conv floor, sulfur-allotrope conver_ignore, trun_min bound
    c = forward.CONDEN_CFG
    assert c["condense_sp"] == ["S8"]
    assert c["fix_species"] == ["S8", "S8_l_s"]
    assert c["fix_species_from_coldtrap_lev"] is False
    assert c["mtol_conv"] == 1.0e-15
    assert {"S", "S2", "S3", "S4"} <= set(c["conver_ignore"])
    assert c["trun_min"] == c["stop_conden_time"]


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


def test_jac_method_default_fd_and_validated():
    # v14: jac_method joins the canonical params -- certified FD by default,
    # unknown values refused loudly
    cp = forward.canonical_params(_p(fisher_params=["lnKzz"]))
    assert cp["jac_method"] == "fd"
    with pytest.raises(ValueError, match="jac_method"):
        forward.canonical_params(_p(fisher_params=["lnKzz"],
                                    jac_method="magic"))


def test_jac_method_ad_requires_photo_on():
    # the warm-jvp AD rows are validated only in the photo-on regime; FD
    # keeps working photo-off (previous test), AD does not
    cp = forward.canonical_params(_p(fisher_params=["lnKzz", "T_iso"],
                                     jac_method="ad"))
    assert cp["jac_method"] == "ad"
    with pytest.raises(ValueError, match="photo-on"):
        forward.canonical_params(_p(fisher_params=["lnKzz"], jac_method="ad",
                                    use_photo=False))


def test_jac_method_ad_covers_every_row_but_neutralizes_when_inert():
    # v14: 'ad' applies to EVERY requested row, composition directions
    # included (the cross-validated differential map; the C-rich b_z corner
    # refuses at run time) -- so comp-only selections KEEP 'ad'
    cp = forward.canonical_params(_p(fisher_params=["lnZ", "dlnCO"],
                                     jac_method="ad"))
    assert cp["jac_method"] == "ad"
    # with no Jacobian requested at all the knob is inert -- normalized to
    # 'fd' so it cannot fragment the cache key, and photo-off is then fine
    cp = forward.canonical_params(_p(jac_method="ad"))
    assert cp["jac_method"] == "fd"
    cp = forward.canonical_params(_p(jac_method="ad", use_photo=False))
    assert cp["jac_method"] == "fd"


def test_condensation_error_states_91_percent_wrong():
    # the refusal message must be misread-proof: the ~0.91 jvp-vs-FD number
    # is a RELATIVE ERROR (the tangent is ~91% wrong), never presentable as
    # a 0.91 agreement ratio / 9% mismatch
    with pytest.raises(ValueError) as ei:
        forward.canonical_params(_p(use_condense=True, fisher_params=["lnZ"]))
    msg = str(ei.value)
    assert "91% wrong" in msg and "not a 9% mismatch" in msg


def test_rt_knobs_v15_defaults_and_validation():
    # v15: three ExoJAX RT knobs are canonical (cache-keyed). Defaults are
    # the pre-v15 hard-coded values, so a default run reproduces v14 physics.
    cp = forward.canonical_params(_p())
    assert cp["rt_ptop_bar"] == 1.0e-8
    assert cp["rt_integration"] == "simpson"
    assert cp["rt_dit_res"] == 1.0
    # the exercised ranges / choices are validated loudly
    cp = forward.canonical_params(_p(rt_ptop_bar=1.0e-6,
                                     rt_integration="trapezoid",
                                     rt_dit_res=0.2))
    assert (cp["rt_ptop_bar"], cp["rt_integration"], cp["rt_dit_res"]) == \
        (1.0e-6, "trapezoid", 0.2)
    with pytest.raises(ValueError, match="rt_ptop_bar"):
        forward.canonical_params(_p(rt_ptop_bar=1.0e-5))
    with pytest.raises(ValueError, match="rt_ptop_bar"):
        forward.canonical_params(_p(rt_ptop_bar=1.0e-10))
    with pytest.raises(ValueError, match="rt_integration"):
        forward.canonical_params(_p(rt_integration="euler"))
    with pytest.raises(ValueError, match="rt_dit_res"):
        forward.canonical_params(_p(rt_dit_res=0.01))
    with pytest.raises(ValueError, match="rt_dit_res"):
        forward.canonical_params(_p(rt_dit_res=2.0))


def test_rt_knobs_fragment_the_cache_key():
    # changing any RT knob must change the cache key (different physics)
    k0 = forward.params_key(_p())
    assert forward.params_key(_p(rt_ptop_bar=1.0e-7)) != k0
    assert forward.params_key(_p(rt_integration="trapezoid")) != k0
    assert forward.params_key(_p(rt_dit_res=0.5)) != k0
