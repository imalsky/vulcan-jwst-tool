"""JWST instrument selector -- Streamlit GUI.

Launch via the console script ``jwst-tool`` (installed with vulcan-jwst-tool), or
directly:  streamlit run src/jwst_tool/app.py  (from the repo root).

Pipeline per run: VULCAN-JAX photochemistry -> ExoJax transmission spectrum
(local subprocess, disk-cached; ~1.5-2 min at the default 100-layer resolution) ->
Pandeia ETC noise per instrument mode (subprocess in its own conda env, disk-cached) ->
science-goal scoring per mode. Two goal types: DETECT a molecule (nested-model
delta-chi2 significance) or CONSTRAIN a parameter (Fisher forecast from the
autodiff Jacobian, vs a target precision). Planets beyond WASP-39b come from
the registry in planets.py (or a fully custom system).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent   # forward.py subprocess lives here

from jwst_tool import detect, fisher as fisher_mod, forward, noise as noise_mod
from jwst_tool import instruments as ins
from jwst_tool import planets

st.set_page_config(page_title="JWST Instrument Selector",
                   layout="wide")


# ---------------------------------------------------------------------------
# One-time "how it works" gate (plain-language intro + acknowledge button)
# ---------------------------------------------------------------------------
def _intro_gate() -> None:
    """Show a short plain-English explainer the first time the app loads this
    session; the tool proper does not render until the user clicks through.
    The acknowledgment survives the sidebar reset (see _reset_all)."""
    if st.session_state.get("intro_ack"):
        return
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.title("JWST instrument selector")
        st.markdown(
            """
**vulcan-jwst-tool** is a local JWST transmission-spectroscopy planning tool
that ranks JWST time-series instrument modes by how well each one can **detect a
target molecule** or **constrain an atmospheric parameter** for a given
exoplanet, and it reports the number of transits needed to reach a chosen
precision.

It follows the same principle as **PandExo**, a Pandeia exposure-time-calculator
noise forecast for JWST exoplanet spectra, but replaces the assumed input
spectrum with a differentiable forward model whose chemistry follows the
**VULCAN** photochemical kinetics code at
[github.com/exoclime/VULCAN](https://github.com/exoclime/VULCAN) (ported to JAX
as VULCAN-JAX and chained into ExoJAX radiative transfer). It then uses
**automatic differentiation** through that model to obtain the exact spectral
Jacobian, the parameter derivatives of transit depth, that feeds a
**Fisher-information forecast**.

The forecasts are **deliberately optimistic** in three ways:

- Molecule-detection scores are a conditional matched-template signal-to-noise
  ratio at one fixed atmosphere, so a real retrieval does worse.
- The Fisher results are local Cramer-Rao lower bounds rather than posterior
  widths.
- The noise model omits time-correlated systematics, so achieved precision is
  usually poorer.

It is a **planning tool, not an atmospheric retrieval**, and it does not model
time-correlated instrument systematics such as visit-long trends, 1/f residuals,
detrending covariance, or stellar heterogeneity. Condensation is not supported.
            """
        )
        st.write("")
        if st.button("I understand, open the tool", type="primary"):
            st.session_state["intro_ack"] = True
            st.rerun()
    st.stop()


_intro_gate()

st.title("JWST instrument selector")
st.caption(
    "VULCAN-JAX photochemistry → ExoJAX transmission spectrum → Pandeia ETC noise. "
    "Pick a planet and a science goal, run the model locally, and see which "
    "instrument mode achieves it best. Uncertainties are a **Pandeia "
    "instrument-model forecast** with an optional PandExo-style minimum noise "
    "floor, not a full time-series systematics forecast; treat mode rankings "
    "as more robust than absolute ppm."
)
st.caption(f"**{ins.BACKEND_STATUS}**. Every result records the exact "
           "engine + refdata versions in its provenance block.")

_PROG_RE = re.compile(r"\[fwd\] PROG ([0-9.]+) (.*)")

# default target precision per parameter (DISPLAY units: dex / K / absolute C/O)
_TARGET_DEFAULT = {"lnZ": 0.10, "dlnCO": 0.05, "lnKzz": 0.30,
                   "T_iso": 50.0, "Tirr": 50.0, "Tint": 50.0,
                   "log_kappa": 0.30, "log_gamma": 0.30}


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
# Reset = bump a nonce that namespaces EVERY widget key: all widgets are
# re-created at their defaults (session_state.clear() alone does not reset
# keyless widgets, whose state lives outside the exposed dict).
_NONCE = st.session_state.setdefault("reset_nonce", 0)


def _reset_all():
    n = st.session_state.get("reset_nonce", 0) + 1
    ack = st.session_state.get("intro_ack", False)   # a settings reset is not
    st.session_state.clear()                          # a reason to re-show the
    st.session_state["reset_nonce"] = n               # how-it-works intro
    st.session_state["intro_ack"] = ack


def K(name: str) -> str:
    return f"n{_NONCE}_{name}"


with st.sidebar:
    st.markdown("### Planet & star")
    st.caption("The target system (physical identity + host star). Defines the "
               "object; the atmosphere model is set in the sections below.")
    planet_key = st.selectbox(
        "Planet", list(planets.PLANETS) + ["custom"], key=K("planet"),
        format_func=lambda k: planets.PLANETS[k]["label"] if k in planets.PLANETS
        else "Custom planet …",
        help="Every planet runs the same validated chemistry+RT machinery; the "
             "system identity (gravity, radii, star, orbit, UV spectrum) is "
             "swapped in. The T-P profile is set below (isothermal or Guillot).")
    pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
    st.caption(pdef["note"] if planet_key in planets.PLANETS
               else "Starts from WASP-39 b values, edit everything below.")

    def _k(name: str) -> str:            # per-planet widget state
        return K(f"{planet_key}_{name}")

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        teff = st.number_input("Star T_eff (K)", 3000.0, 7000.0,
                               pdef["star"]["teff"], 50.0, key=_k("teff"),
                               help="PHOENIX SED for the ETC (with log g, [Fe/H]).")
        logg = st.number_input("Star log g (cgs, log10 cm/s^2)", 3.5, 5.5,
                               pdef["star"]["log_g"], 0.1, key=_k("logg"),
                               help="Stellar surface gravity as log10(g) in cgs "
                                    "(g in cm/s^2); feeds the PHOENIX SED for the ETC.")
        feh = st.number_input("Star [Fe/H]", -2.0, 0.5,
                              pdef["star"]["metallicity"], 0.1, key=_k("feh"))
        ks_mag = st.number_input("Ks mag (2MASS)", 4.0, 16.0,
                                 pdef["star"]["ks_mag"], 0.1, key=_k("ks"),
                                 help="Sets absolute count rates → saturation & "
                                      "photon noise. Brighter = more saturation.")
        rstar = st.number_input("R_star (R_sun)", 0.2, 3.0, pdef["rstar_rsun"],
                                0.01, key=_k("rstar"), format="%.3f",
                                help="Transit-depth normalization + UV flux at "
                                     "the planet.")
        rp = st.number_input("R_p (R_Jup)", 0.1, 2.5, pdef["rp_rjup"], 0.01,
                             key=_k("rp"), format="%.3f")
        g_ms2 = st.number_input("Surface gravity (m/s²)", 1.0, 100.0,
                                pdef["gs_cgs"] / 100.0, 0.5, key=_k("g"),
                                help="Sets the scale height: lower gravity = "
                                     "bigger spectral features.")
        orbit_au = st.number_input("Semi-major axis (au)", 0.005, 1.0,
                                   pdef["orbit_au"], 0.001, key=_k("a"),
                                   format="%.4f",
                                   help="Scales the stellar UV reaching the "
                                        "planet (photochemistry).")
        t14 = st.number_input("Transit duration T14 (hr)", 0.5, 10.0,
                              pdef["t14_hr"], 0.1, key=_k("t14"))
        sflux = st.selectbox("Stellar UV spectrum (photochemistry)",
                             list(planets.SFLUX_CHOICES),
                             index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
                             format_func=planets.SFLUX_CHOICES.get, key=_k("sflux"),
                             help="Shipped VULCAN spectra, pick the nearest "
                                  "spectral type. Drives photolysis (SO2, CH4 …).")

    teq = float(pdef["teq_k"])
    st.markdown("### VULCAN chemistry")
    st.caption("Inputs to the VULCAN-JAX photochemical-kinetics forward model "
               "(composition + transport + photochemistry → steady-state "
               "abundances). The T-P profile is shared: it also sets the "
               "ExoJAX radiative transfer below.")

    with st.expander("Atmosphere structure, T-P profile (shared with RT)"):
        tp_mode = st.selectbox(
            "T-P profile", ["isothermal", "guillot"], index=0, key=_k("tp"),
            format_func={"isothermal": "Isothermal",
                         "guillot": "Guillot (2010)"}.get,
            help="Sets the temperature the chemistry AND the radiative transfer "
                 "see. Explicit profiles only (the WASP-39b GCM profile was "
                 "removed, isothermal / Guillot).")
        tp_kwargs = {}
        if tp_mode == "isothermal":
            tp_kwargs["T_iso"] = st.slider("T_iso (K)", 400.0, 2500.0,
                                           float(np.clip(teq, 400.0, 2500.0)),
                                           25.0, key=_k("tiso"))
        else:
            tirr0 = float(np.clip(round(teq * np.sqrt(2.0) / 10) * 10,
                                  800.0, 2500.0))
            tp_kwargs["Tirr"] = st.slider("T_irr (K)", 800.0, 2500.0, tirr0, 20.0,
                                          key=_k("tirr"),
                                          help="≈ √2 × equilibrium temperature.")
            tp_kwargs["Tint"] = st.slider("T_int (K)", 50.0, 500.0, 100.0, 25.0,
                                          key=_k("tint"))
            tp_kwargs["log_kappa"] = st.slider("log₁₀ κ_IR (cm²/g)", -4.0, 0.0,
                                               -2.3, 0.1, key=_k("lk"))
            tp_kwargs["log_gamma"] = st.slider("log₁₀ γ (κ_vis/κ_IR)", -2.0, 0.3,
                                               -1.0, 0.05, key=_k("lg"))

    with st.expander("Composition"):
        met = st.select_slider(
            "Metallicity (× solar)",
            options=[1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0],
            value=10.0, key=K("met"),
            help="Uniform metal enrichment; the constraint forecast reports it "
                 "in dex ([M/H]).")
        # Absolute C/O = N_C/N_O (the standard exoplanet-atmosphere quantity, a
        # number ratio -- not a log, not [C/H]/[O/H]). The chemistry knob is
        # fixed-O carbon enrichment with dco = ln(C/O / C/O_baseline) EXACTLY, so
        # this slider maps straight onto it. Options span the validated dco range
        # [-0.5, 0.5]; the baseline option is exactly CO_BASELINE -> dco = 0.
        _co_opts = [0.30, 0.35, 0.40, forward.CO_BASELINE, 0.50, 0.55, 0.60,
                    0.65, 0.70, 0.75]
        co_ratio = st.select_slider(
            "C/O (carbon/oxygen number ratio)", options=_co_opts,
            value=forward.CO_BASELINE, key=K("co"),
            format_func=lambda x: (f"{x:.2f} (solar)"
                                   if abs(x - forward.CO_BASELINE) < 1e-6
                                   else f"{x:.2f}"),
            help="Total carbon/oxygen number ratio N_C/N_O. Baseline 0.46 = "
                 "Lodders 2019 solar (this network's abundance set; the more "
                 "commonly quoted Asplund 2009 solar is 0.55). Fixed-O carbon "
                 "enrichment: O/H is held fixed, C/H scales. C/O = 1 (the "
                 "H2O/CH4-vs-CO transition) is beyond the validated range.")
        dco = float(np.log(co_ratio / forward.CO_BASELINE))

    with st.expander("Vertical mixing (K_zz)"):
        # constant K_zz only (the WASP-39b GCM K_zz profile was removed)
        kzz_mode = "const"
        log_kzz = st.slider("log₁₀ K_zz (cm²/s)", 6.0, 12.0, 9.0, 0.25,
                            key=_k("kzz"),
                            help="Constant eddy-diffusion coefficient: stronger "
                                 "mixing quenches photochemical gradients.")
        kzz_const, kzz_x = 10.0 ** log_kzz, 1.0

    with st.expander("Photochemistry & transport"):
        use_photo = st.checkbox(
            "Photochemistry (UV photolysis)", value=True, key=K("photo"),
            help="Off = thermochemistry + transport only (no photolysis products "
                 "such as SO2). Parameter constraints (the Fisher forecast) "
                 "require this ON: the forecast uses a warm-started forward-mode "
                 "derivative of the steady state, and that tangent only relaxes "
                 "to the true sensitivity in the photo-on regime (validated jvp "
                 "vs finite-difference <0.1% there; with photo off it is "
                 "under-relaxed/unstable). Detection goals work either way.")
        sl_angle_deg = st.slider(
            "Photolysis zenith angle (°)", 0.0, 89.0, 83.0, 1.0, key=K("sza"),
            disabled=not use_photo,
            help="Slant path of the stellar UV. 83° = terminator slant "
                 "(Tsai et al. 2023 W39b); ~57° ≈ dayside average.")
        f_diurnal = st.slider(
            "Diurnal photolysis factor", 0.1, 1.0, 1.0, 0.05, key=K("fdiur"),
            disabled=not use_photo,
            help="Multiplies every photolysis rate. 1.0 = permanent dayside "
                 "(tidally locked); 0.5 mimics day-night averaging.")
        use_moldiff = st.checkbox(
            "Molecular diffusion", value=True, key=K("moldiff"),
            help="Species-dependent molecular diffusion competing with Kzz "
                 "(sets the homopause; matters high up).")

    with st.expander("Numerical grid (layers & convergence)"):
        st.caption("Grid resolution and solver tolerance, same physics, finer "
                   "grids (this replaced the old fast/high fidelity switch).")
        nz = st.slider(
            "Vertical layers (chemistry + RT)", *forward.NZ_RANGE,
            forward.NZ_DEFAULT, 10, key=K("nz"),
            help="VULCAN photochemistry layers; the ExoJAX radiative-transfer "
                 "grid is LOCKED to the same count. More layers resolve steep "
                 "photochemical gradients. Roughly 1.5 min at 100, 3 min at 150.")
        yconv_cri = st.select_slider(
            "Convergence tolerance (yconv)", options=[1.0e-2, 1.0e-3],
            value=forward.YCONV_DEFAULT, key=K("yconv"),
            format_func={1.0e-2: "1e-2 (fast, VULCAN default)",
                         1.0e-3: "1e-3 (strict)"}.get,
            help="Steady-state convergence criterion. 1e-2 is the VULCAN master "
                 "default; use 1e-3, with more layers, for final mid-IR numbers.")

    with st.expander("Condensation (not offered)"):
        st.caption(
            "Condensation through VULCAN is not offered in this tool: a "
            "condensing column's steady state is not reliably differentiable, "
            "so it cannot enter the Fisher forecast.")

    st.divider()
    st.markdown("### ExoJAX radiative transfer")
    st.caption("Turns the VULCAN abundances (at the same T-P) into the "
               "transmission spectrum. Opacity set, scattering, clouds, and "
               "line broadening.")

    with st.expander("Opacity, scattering & clouds"):
        st.caption(
            "RT opacity always includes the base set "
            f"**{' · '.join(forward.MOLECULES)}** (solved on every run). The "
            f"opt-in extras are **{' · '.join(forward.EXTRA_MOLECULES)}**. "
            "Adding more is currently in development.")
        extra_mols = st.multiselect(
            "Extra RT molecules", forward.EXTRA_MOLECULES, default=[],
            key=K("xmols"),
            help="Added to the base opacity set (the chemistry always solves "
                 "them). C2H2/HCN matter at high C/O, H2S at 3.8-4.6 µm, NH3 on "
                 "cool (≲900 K) planets. First use downloads the HITRAN lines "
                 "(~10-15 s each per run).")
        use_rayleigh = st.checkbox(
            "H₂/He Rayleigh scattering", value=True, key=K("rayl"),
            help="Zero-free-parameter known physics; matters shortward of "
                 "~1.5 µm (the SOSS blue end). Leave ON except for comparisons.")
        cloud_on = st.checkbox(
            "Power-law cloud/haze opacity", value=False, key=K("cloud"),
            help="ExoJax power-law retrieval cloud, uniformly mixed: "
                 "κ(ν) = κ₀·(ν/ν₀)^α per gram of atmosphere (no cloud-top "
                 "pressure or particle microphysics). Held FIXED in the Fisher "
                 "forecast (no cloud marginalization), so forecasts with thick "
                 "opacity are best-case.")
        if cloud_on:
            log_kappa_cloud = st.slider(
                "log₁₀ κ_cloud (cm²/g at 3.5 µm)", -4.0, 2.0, -1.0, 0.1,
                key=K("ck"),
                help="Gray amplitude. τ=1 pressure ≈ g/(κ·10⁶) bar: at WASP-39b "
                     "gravity, −1 → ~4 mbar deck, −3 → ~0.4 bar.")
            alpha_cloud = st.slider(
                "Cloud spectral slope α (κ ∝ ν^α)", 0.0, 4.0, 0.0, 0.25,
                key=K("ca"),
                help="0 = gray deck; 4 ≈ Rayleigh-like small-particle haze.")
        else:
            log_kappa_cloud, alpha_cloud = -1.0, 0.0
        broadening = st.selectbox(
            "Line broadening perturber", ["air", "h2he"], index=0,
            key=K("broad"),
            help="'air' = HITRAN terrestrial widths (validated default); "
                 "'h2he' = planetary H₂/He blend, first use downloads "
                 "separate line-list caches, and molecules with no H₂/He "
                 "coverage raise loudly instead of silently falling back.")
        nu_pts = st.select_slider(
            "Native spectral sampling (nu_pts)", options=[4000, 6000, 8000],
            value=forward.NU_PTS_DEFAULT, key=K("nupts"),
            format_func={4000: "4000 (native R≈1500)",
                         6000: "6000 (native R≈2200)",
                         8000: "8000 (native R≈3000)"}.get,
            help="Wavenumber grid points across 1-15 µm, before binning to your "
                 "chosen display R below. Higher sampling sharpens narrow "
                 "features (the weak mid-IR bands) at more runtime.")

    avail_free = forward.CHEM_PARAM_NAMES + forward.TP_PARAM_NAMES[tp_mode]
    mol_options = forward.MOLECULES + [m for m in forward.EXTRA_MOLECULES
                                       if m in extra_mols]

    st.divider()
    st.markdown("### Science goal")

    with st.expander("Goal", expanded=True):
        goal = st.radio(
            "Goal", ["detect", "constrain"], horizontal=True, key=K("goal"),
            format_func={"detect": "Detect a molecule",
                         "constrain": "Constrain a parameter"}.get,
            help="Detect: significance of molecule X being present (Δχ² between "
                 "the spectrum with and without it). Constrain: how tightly a "
                 "parameter (metallicity, C/O, Kzz, …) could be measured, a "
                 "Fisher forecast from the autodiff Jacobian.")
        goal_param, target_prec = None, None
        if goal == "detect":
            target_mol = st.selectbox("Detect molecule", mol_options,
                                      index=mol_options.index("SO2"),
                                      key=K("mol_" + "_".join(sorted(extra_mols))))
        else:
            target_mol = None
            if not use_photo:
                st.warning("Parameter constraints use the Fisher forecast, "
                           "which needs photochemistry ON (VULCAN section).")
            goal_param = st.selectbox(
                "Constrain parameter", avail_free, key=K(f"gp_{tp_mode}"),
                format_func=lambda n: forward.PARAM_LABELS[n],
                help="Constraint is marginalized over the other free parameters "
                     "(Fisher forecast section) and a reference-radius nuisance.")
            unit = forward.PARAM_UNITS[goal_param]
            # label uses the unit when there is one (dex / K), else the bare
            # symbol -- C/O is a dimensionless number ratio
            _tgt_lbl = (f"Target precision (±{unit})" if unit else
                        f"Target precision (±{forward.PARAM_SYMBOLS[goal_param]})")
            if unit == "K":
                target_prec = st.number_input(_tgt_lbl, 5.0, 500.0,
                                              _TARGET_DEFAULT[goal_param], 5.0,
                                              key=K(f"tgt_{goal_param}"))
            else:
                target_prec = st.number_input(_tgt_lbl, 0.01, 3.0,
                                              _TARGET_DEFAULT[goal_param], 0.01,
                                              key=K(f"tgt_{goal_param}"))
        target_sig = st.number_input(
            "Significance level (σ)", 1.0, 10.0, 3.0, 0.5, key=K("tsig"))

    with st.expander("Fisher forecast"):
        st.caption(
            "Expected 1σ constraints on atmosphere parameters from this "
            "observation, a linearized retrieval (Cramér-Rao bound) built "
            "from d(spectrum)/d(parameter), computed by autodiff through the "
            "full chemistry+RT chain. No MCMC, no priors.")
        if goal == "constrain":
            fisher_extra = st.multiselect(
                "Jointly free parameters", avail_free,
                default=[p for p in ("lnZ", "dlnCO", "lnKzz")],
                key=K(f"fx_{tp_mode}"),
                help="The goal parameter is always included. More free "
                     "parameters = a more honest (wider) forecast; each adds "
                     "~20-60 s of Jacobian time.")
            fisher_params = sorted(set(fisher_extra) | {goal_param})
        else:
            do_fisher = st.checkbox(
                "Compute parameter constraints too", value=False,
                key=K("dofish"), disabled=not use_photo,
                help="One warm-started forward-mode jvp per parameter "
                     "(~20-60 s each). Needs photochemistry ON.")
            if not use_photo:
                st.caption("Locked because **photochemistry is OFF** (VULCAN "
                           "chemistry section). The constraint uses a forward-mode "
                           "derivative that is only valid in the photo-on regime, "
                           "turn photochemistry on to unlock it.")
            fisher_params = st.multiselect(
                "Free parameters", avail_free, key=K(f"fp_{tp_mode}"),
                default=["lnZ", "dlnCO", "lnKzz"]) if (do_fisher and use_photo) else []

    st.divider()
    st.markdown("### Instrument & noise")
    st.caption("The JWST measurement itself: which modes, how many transits, "
               "detector saturation, and the Pandeia/PandExo noise model. "
               "Independent of the atmosphere physics above.")

    with st.expander("Observation & instrument modes", expanded=True):
        mode_keys = st.multiselect(
            "Instrument modes",
            options=list(ins.MODES),
            default=ins.DEFAULT_MODES, key=K("modes"),
            help="The ETC computes every mode once per star, so adding modes "
                 "later is instant.",
            format_func=lambda k: (f"{ins.MODES[k]['label']}  "
                                   f"({ins.MODES[k]['wl_min']:g}-"
                                   f"{ins.MODES[k]['wl_max']:g} µm)"))
        r_bin = st.select_slider("Binned resolving power R",
                                 options=[50, 100, 200], value=100, key=K("rbin"))
        n_transits = st.slider("Number of transits", 1, 10, 1, key=K("ntr"))
        t_base = st.number_input("Out-of-transit baseline (hr)", 0.5, 10.0,
                                 float(t14), 0.1, key=_k("tbase"),
                                 help="Sets how well the stellar flux is "
                                      "anchored; PandExo convention is ≈ T14.")
        sat_limit = st.slider("Saturation limit (full-well fraction)",
                              0.5, 0.95, 0.80, 0.05, key=K("sat"),
                              help="Group selection keeps the brightest pixel "
                                   "below this full-well fraction.")

    with st.expander("Noise model"):
        st.markdown("**Minimum noise floor** (PandExo convention)")
        st.caption("Applied as σ_final = max(σ_random, floor) on the final "
                   "binned uncertainties: a hard minimum, never added in "
                   "quadrature, never rescaled by the binning R, and never "
                   "averaged below by adding transits.")
        floor_mode = st.radio(
            "Floor type", ["constant", "none", "file"], horizontal=True,
            key=K("floormode"),
            format_func={"constant": "Constant (ppm)", "none": "No floor",
                         "file": "Wavelength table"}.get)
        floor_table = None
        if floor_mode == "constant":
            floors = {k: st.number_input(ins.MODES[k]["label"], 0.0, 200.0,
                                         ins.MODES[k]["floor_ppm"], 5.0,
                                         key=K(f"floor_{k}"))
                      for k in mode_keys}
        elif floor_mode == "file":
            up = st.file_uploader(
                "Two columns: wavelength (µm), floor (ppm)",
                type=["txt", "csv", "dat"], key=K("floorfile"),
                help="Whitespace- or comma-separated. Linearly interpolated "
                     "to the final bin wavelengths; endpoint values extend "
                     "beyond the supplied range (PandExo behavior). Applied "
                     "to every selected mode.")
            if up is not None:
                try:
                    raw = up.getvalue().decode()
                    delim = "," if "," in raw.splitlines()[0] else None
                    floor_table = np.loadtxt(raw.splitlines(), delimiter=delim,
                                             ndmin=2)
                    noise_mod.resolve_floor(np.array([1.0]),
                                            floor_table)  # validate loudly now
                    st.caption(f"Loaded {floor_table.shape[0]} rows, "
                               f"{floor_table[:, 0].min():g}-"
                               f"{floor_table[:, 0].max():g} µm, "
                               f"{floor_table[:, 1].min():g}-"
                               f"{floor_table[:, 1].max():g} ppm.")
                except Exception as e:
                    st.error(f"Floor table rejected: {e}")
                    floor_table = None
            if floor_table is None:
                st.warning("No valid floor table loaded, runs will use NO "
                           "floor until one is provided.")
        if floor_mode != "constant":
            floors = {k: None for k in mode_keys}
        if floor_mode == "file" and floor_table is not None:
            floors = {k: floor_table for k in mode_keys}

        st.markdown("**Empirical noise sensitivity factor** (× Pandeia σ, "
                    "optional)")
        st.caption("Default **1.0** = the Pandeia prediction as-is. Published "
                   "achieved-vs-predicted ratios (COMPASS/Gordon+2025 G395H "
                   "≈1.05-1.12×; Espinoza+2023 1.2× NIRISS; Bouwman+2023 "
                   "≈1.15× MIRI LRS) are program-specific reference points "
                   "for sensitivity studies, not a calibration. Proportional "
                   "noise, averages down with transits, unlike the floor. "
                   "Recorded in the result metadata.")
        infl = {k: st.number_input(ins.MODES[k]["label"] + " ", 1.0, 3.0,
                                   float(ins.MODES[k].get("noise_infl", 1.0)),
                                   0.05, key=K(f"infl_{k}"))
                for k in mode_keys}

        st.markdown("**Experimental: correlated-floor scenarios**")
        scenario = st.radio(
            "Systematics scenario", list(noise_mod.SCENARIOS),
            format_func=lambda s: noise_mod.SCENARIOS[s]["label"],
            key=K("scenario"), label_visibility="collapsed")
        st.caption("The default (**random**) is the exact diagonal, "
                   "PandExo-style noise model and is what the headline "
                   "numbers use. The correlated presets re-allocate the part "
                   "of the variance the floor adds (the floor *excess*) into "
                   "a spectrally smooth kernel at identical per-bin totals. "
                   "They are stated assumptions, **not calibrated JWST "
                   "systematics models**, for stress-testing how rankings "
                   "move with correlation structure; conservative also "
                   "profiles per-segment slopes.")

    with st.expander("Display"):
        show_noise = st.checkbox("Show simulated noise realization", value=False,
                                 key=K("shownoise"))
        seed = st.number_input("Realization seed", 0, 9999, 0, key=K("seed"))

    st.button("Reset all settings", on_click=_reset_all,
              help="Back to the defaults (also clears the current results).")

params = dict(planet=planet_key, nz=nz, nu_pts=nu_pts, yconv_cri=yconv_cri,
              rp_rjup=rp, gs_cgs=g_ms2 * 100.0, rstar_rsun=rstar,
              orbit_au=orbit_au, sflux=sflux,
              met_x_solar=met, dco=dco,
              kzz_mode=kzz_mode, kzz_x=kzz_x, kzz_const=kzz_const,
              tp_mode=tp_mode, fisher_params=fisher_params,
              use_photo=use_photo, sl_angle_deg=sl_angle_deg,
              f_diurnal=f_diurnal, use_moldiff=use_moldiff,
              use_rayleigh=use_rayleigh, broadening=broadening,
              cloud_on=cloud_on,
              log_kappa_cloud=log_kappa_cloud, alpha_cloud=alpha_cloud,
              extra_mols=extra_mols, **tp_kwargs)
star = dict(teff=teff, log_g=logg, metallicity=feh, ks_mag=ks_mag)
planet_label = (planets.PLANETS[planet_key]["label"]
                if planet_key in planets.PLANETS else "custom planet")

try:
    cached = forward.load_result(params) is not None
    params_error = None
except ValueError as e:          # e.g. stale widget combo mid-rerun
    cached, params_error = False, str(e)

n_jvp = len(fisher_params)
# rough runtime hint keyed off the resolution knobs (old fast ~1.8, high ~2.8 min)
base_min = 0.8 + 0.010 * nz + 0.00005 * (nu_pts - forward.NU_PTS_DEFAULT)
if yconv_cri <= 1.5e-3:              # strict convergence costs extra iterations
    base_min += 0.5
if met != 10.0 or dco != 0.0:        # composition step -> two-stage chemistry
    base_min += 0.6 * (nz / 100.0)
base_min += 0.25 * len(extra_mols)   # opa build + removed spectrum per extra
# cool columns (<~900 K) converge much more slowly (WASP-107b: ~5 min measured)
t_char = {"isothermal": tp_kwargs.get("T_iso", 1100.0),
          "guillot": tp_kwargs.get("Tirr", 1560.0) / np.sqrt(2.0)}.get(tp_mode, 1100.0)
if t_char < 900.0:
    base_min += 2.5
per_jvp = 0.8 if (nz >= 140 or yconv_cri <= 1.5e-3) else 0.5
native_r = int(round(nu_pts * 2950 / 8000 / 50) * 50)
grid_lbl = f"{nz}-layer, native R≈{native_r}"
est = "instant (cached)" if cached else (
    f"~{base_min + per_jvp * n_jvp:.0f} min (local {grid_lbl} run"
    + (f" + {n_jvp} Jacobian directions" if n_jvp else "") + ")")
col_btn, col_note = st.columns([1, 3])
run_clicked = col_btn.button("Run", type="primary", width="stretch")
col_note.caption(f"**{planet_label}**, {grid_lbl}, model spectrum: "
                 f"**{est}**. ETC noise is cached per star.")


# ---------------------------------------------------------------------------
# Compute on click
# ---------------------------------------------------------------------------
def compute():
    if params_error:
        st.error(f"Invalid parameter combination: {params_error}")
        return None
    if not mode_keys:
        st.error("Select at least one instrument mode.")
        return None

    model = forward.load_result(params)
    if model is None:
        with st.status("Running VULCAN-JAX + ExoJAX forward model locally …",
                       expanded=True) as status:
            bar = st.progress(0.0, text="starting …")
            pfile = forward.MODEL_CACHE / f"{forward.params_key(params)}.params.json"
            forward.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(forward.canonical_params(params)))
            proc = subprocess.Popen(
                [sys.executable, str(TOOL_DIR / "forward.py"), str(pfile)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            box = st.empty()
            lines = []
            for line in proc.stdout:
                line = line.rstrip()
                m = _PROG_RE.match(line)
                if m:
                    bar.progress(min(1.0, float(m.group(1))), text=m.group(2))
                else:
                    lines.append(line)
                    box.code("\n".join(lines[-10:]))
            proc.wait()
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                st.error("Forward model failed:\n\n```\n"
                         + "\n".join(lines[-25:]) + "\n```")
                return None
            bar.progress(1.0, text="done")
            status.update(label="Forward model done", state="complete")
        model = forward.load_result(params)
        if model is None:
            st.error("Forward model finished but produced no cache file.")
            return None

    # ETC: always ALL modes (one cache per star; selection changes stay instant)
    all_modes = list(ins.MODES)
    job = noise_mod.noise_job(star, all_modes, sat_limit=sat_limit)
    have_cache = (ins.NOISE_CACHE / f"{noise_mod.job_key(job)}.json").exists()
    if have_cache:
        etc = noise_mod.run_pandeia(job)
    else:
        with st.status(f"Running Pandeia ETC ({ins.BACKEND_STATUS.split(' /')[0]}) …",
                       expanded=True) as status:
            bar = st.progress(0.0, text="starting the ETC …")
            box = st.empty()
            lines = []
            n_started = [0]

            def _cb(s):
                if s.startswith("[pandeia] ") and s.endswith("..."):
                    bar.progress(n_started[0] / len(all_modes),
                                 text=s.removeprefix("[pandeia] ")
                                 .removesuffix("...")
                                 + f" ({n_started[0] + 1}/{len(all_modes)})")
                    n_started[0] += 1
                else:
                    lines.append(s)
                    box.code("\n".join(lines[-8:]))

            etc = noise_mod.run_pandeia(job, progress=_cb)
            bar.progress(1.0, text="done")
            status.update(label="Pandeia ETC done", state="complete")

    t_in_s, t_out_s = t14 * 3600.0, t_base * 3600.0
    results, failed, unusable = [], [], []
    for k in mode_keys:
        if "error" in etc[k]:
            failed.append((k, etc[k]["error"]))
        elif etc[k].get("unusable") or not etc[k].get("wl"):
            unusable.append((k, etc[k].get("reason", "no usable pixels")))
        else:
            try:
                results.append(detect.evaluate_mode(
                    k, etc[k], model, target_mol, r_bin, t_in_s, t_out_s,
                    n_transits, floors[k], noise_inflation=infl[k],
                    scenario=scenario))
            except Exception as e:
                # one bad mode must not kill the whole run -- report it with
                # its label + the actual reason, keep evaluating the rest
                failed.append((k, f"{type(e).__name__}: {e}\n\n"
                                  f"(binning/noise evaluation for {k}; the other "
                                  "modes are unaffected)"))
    return dict(model=model, results=results, failed=failed, unusable=unusable,
                fisher_names=list(fisher_params),
                provenance=etc.get("__provenance__"))


if run_clicked:
    out = compute()
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal, target=target_mol, goal_param=goal_param,
            target_prec=target_prec, target_sig=target_sig,
            n_transits=n_transits, show_noise=show_noise, seed=seed,
            r_bin=r_bin, planet=planet_label, scenario=scenario,
            floor_mode=floor_mode)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
if "out" not in st.session_state:
    st.info("Pick a planet and a science goal in the sidebar, then press **Run**.")
    st.stop()

out = st.session_state["out"]
meta = st.session_state["out_meta"]
model, results = out["model"], out["results"]
goal_r = meta.get("goal", "detect")
# the atmosphere's absolute C/O, for the dlnCO -> absolute-C/O display conversion
# (sigma_CO = C/O * sigma_lnCO); params_json carries this run's dco
_cpj = json.loads(str(model["params_json"]))
co_eval = float(forward.CO_BASELINE * np.exp(float(_cpj.get("dco", 0.0))))

for k, err in out["failed"]:
    first = str(err).strip().splitlines()[-1] if "Traceback" in str(err) else \
        str(err).strip().splitlines()[0]
    st.error(f"{ins.MODES[k]['label']}: failed, {first}")
    with st.expander(f"{ins.MODES[k]['label']} details"):
        st.code(str(err)[-2500:])
for k, reason in out["unusable"]:
    st.warning(f"**{ins.MODES[k]['label']}: unusable on this star**, {reason}.")

if not results:
    st.stop()

if out.get("provenance"):
    _pv = out["provenance"]
    st.caption(
        f"**{ins.BACKEND_STATUS}.** Backend: pandeia.engine "
        f"{_pv['engine_version']} + {_pv['refdata_name']} "
        f"(refdata {_pv['refdata_version']}), worker v{_pv['worker_version']} "
        ",  recorded in every noise cache.")

fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
ok = [r for r in results if not r["saturated"]]

# --- verdict ---------------------------------------------------------------
if goal_r == "detect":
    tsig = float(meta.get("target_sig") or 3.0)
    ranked = sorted(ok or results, key=lambda r: -r["sigma_detect"])
    best = ranked[0]
    bsig = best["sigma_detect"]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for detecting {meta['target']} on "
               f"{meta.get('planet', '?')}: {best['label']}**, "
               f"{bsig:.1f}σ in {ntr} transit{'s' if ntr > 1 else ''} "
               f"(target {tsig:g}σ; median precision "
               f"{best['median_sigma_ppm']:.0f} ppm per R={meta['r_bin']} bin).")
    if bsig >= tsig:
        st.success(verdict + "  Meets the target.")
    elif bsig > 0:
        # floor-aware transit solver: the photon term averages down with N, the
        # (R-anchored) systematic floor does not -- a plain 1/sqrt(N) law was
        # optimistic exactly where it mattered (floor-dominated bright stars)
        tt = detect.transits_to_target(best, tsig)
        if tt["reachable"]:
            st.error(verdict + f"  Missing the target, {tt['n']} transits of "
                     f"{best['label']} would reach it (floor-aware estimate).")
        else:
            st.error(verdict + f"  Missing the target, and NO number of transits "
                     f"reaches it: the systematic floor caps this mode at "
                     f"{tt['sig_inf']:.1f}σ. Lower the floor, choose other modes, "
                     "or relax the target.")
    else:
        st.error(verdict + "  No signal in the selected bands, try other "
                 "modes or a different goal.")
else:
    gp = meta["goal_param"]
    unit = forward.PARAM_UNITS[gp]
    usp = (" " + unit) if unit else ""       # " dex"/" K", or "" for C/O (ratio)
    glabel = forward.PARAM_LABELS[gp]
    target = float(meta["target_prec"])
    tsig = float(meta.get("target_sig") or 3.0)
    with_jac = [r for r in results if r.get("jac_bins") is not None]
    # one saturation policy everywhere: a saturated mode is unusable data, so it
    # is excluded from BOTH the per-mode ranking and the combined forecast (the
    # combined row used to silently include modes the per-mode view dropped)
    usable_jac = [r for r in with_jac if not r["saturated"]]
    per_mode = {}          # tsig-sigma half-widths, display units
    for r in usable_jac:
        s = fisher_mod.display_sigma(gp, fisher_mod.mode_forecast(r, fisher_names)[gp],
                                     co_eval=co_eval)
        if np.isfinite(s):
            per_mode[r["mode_key"]] = tsig * s
    comb = (tsig * fisher_mod.display_sigma(
        gp, fisher_mod.combined_forecast(usable_jac, fisher_names)[gp], co_eval=co_eval)
        if len(usable_jac) >= 2 else np.inf)
    if not per_mode:
        st.error(f"No selected mode constrains {glabel}, its Jacobian has no "
                 "signal in these bands. Try other modes or a different goal.")
        st.stop()
    bk = min(per_mode, key=per_mode.get)
    bs = per_mode[bk]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for constraining {glabel} on "
               f"{meta.get('planet', '?')}: {ins.MODES[bk]['label']}**, "
               f"±{bs:.3g}{usp} at {tsig:g}σ in {ntr} transit"
               f"{'s' if ntr > 1 else ''} (target ±{target:g}{usp} "
               f"at {tsig:g}σ).")
    if bs <= target:
        st.success(verdict + "  Meets the target.")
    elif np.isfinite(comb) and comb <= target:
        st.warning(verdict + f"  No single mode reaches the target, but the "
                   f"combination of all selected modes does "
                   f"(±{comb:.3g}{usp} at {tsig:g}σ).")
    else:
        best_r = next(r for r in usable_jac if r["mode_key"] == bk)
        tt = fisher_mod.transits_to_target(best_r, fisher_names, gp,
                                           target / tsig, detect.sigma_at_transits,
                                           co_eval=co_eval)
        if tt["reachable"]:
            st.error(verdict + f"  Missing the target, {tt['n']} transits of "
                     f"{ins.MODES[bk]['label']} would reach it (floor-aware "
                     "estimate).")
        else:
            st.error(verdict + f"  Missing the target, and NO number of transits "
                     f"reaches it: the systematic floor caps this mode at "
                     f"±{tsig * tt['sig_inf']:.3g}{usp} at {tsig:g}σ. Lower the "
                     "floor, combine modes, or relax the target.")

# --- spectrum figure -------------------------------------------------------
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
ax.plot(wl_s, d_s, color="#555555", lw=0.7, alpha=0.8, zorder=2,
        label="model (native)")
if goal_r == "detect":
    mols = [str(x) for x in model["mols"]]
    d_wo_s = model["depth_wo"][mols.index(meta["target"])][order] * 1e6
    ax.plot(wl_s, d_wo_s, color="#999999", lw=0.9, ls="--", zorder=1,
            label=f"model without {meta['target']}")
rng = np.random.default_rng(int(meta["seed"]))
for r in results:
    c = ins.MODE_COLOR[r["mode_key"]]
    y = r["depth"] * 1e6
    if meta["show_noise"]:
        y = y + rng.normal(0.0, r["sigma"] * 1e6)
    label = r["label"] + (" (saturated!)" if r["saturated"] else "")
    # plot at the response-weighted effective wavelength (matters near detector
    # gaps / steep throughput); falls back to the bin center if absent
    x = r.get("wl_eff", r["wl"])
    ax.errorbar(x, y, yerr=r["sigma"] * 1e6, fmt="o", ms=3.0, lw=1.0,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, zorder=3, label=label)
ax.set_xscale("log")
ticks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
lo = min(min(r["wl"].min() for r in results), 1.0)
hi = max(r["wl"].max() for r in results)
ax.set_xlim(lo * 0.97, hi * 1.03)
sel = (wl_s >= lo * 0.97) & (wl_s <= hi * 1.03)
pad = 0.06 * (d_s[sel].max() - d_s[sel].min())
ax.set_ylim(d_s[sel].min() - pad, d_s[sel].max() + 3 * pad)
ax.set_xlabel("wavelength (μm)")
ax.set_ylabel("transit depth (ppm)")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
st.pyplot(fig, width="stretch")
plt.close(fig)

# --- goal chart + T-P profile ----------------------------------------------
col1, col2 = st.columns([2.6, 1.4])

with col1:
    if goal_r == "detect":
        st.subheader(f"{meta['target']} conditional template S/N (σ_detect)")
        rs = sorted(results, key=lambda r: r["sigma_detect"])
        names = [r["label"] + (" (sat)" if r["saturated"] else "") for r in rs]
        vals = [r["sigma_detect"] for r in rs]
        cols = [ins.MODE_COLOR[r["mode_key"]] for r in rs]
        xrefs, xlabel = (3.0, 5.0), (f"conditional template S/N "
                                     f"({meta['n_transits']} transit"
                                     f"{'s' if meta['n_transits'] > 1 else ''})")
        fmt_v = lambda v: f"{v:.1f}σ"
        vline_target = float(meta.get("target_sig") or 3.0)
    else:
        st.subheader(f"Expected precision on {glabel}")
        items = sorted(per_mode.items(), key=lambda kv: -kv[1])   # best at top
        names = [ins.MODES[k]["label"] for k, _ in items]
        vals = [v for _, v in items]
        cols = [ins.MODE_COLOR[k] for k, _ in items]
        if np.isfinite(comb):
            names.append("ALL SELECTED (combined)")
            vals.append(comb)
            cols.append("#555555")
        xrefs, xlabel = (), (f"expected ±{forward.param_axis(gp)} at {tsig:g}σ "
                             f"({meta['n_transits']} transit"
                             f"{'s' if meta['n_transits'] > 1 else ''}; "
                             "lower is better)")
        fmt_v = lambda v: f"{v:.3g}"
        vline_target = target
    fig2, ax2 = plt.subplots(figsize=(6.4, 0.55 * len(names) + 1.2), dpi=150)
    bars = ax2.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_width() + max(vals) * 0.02,
                 b.get_y() + b.get_height() / 2, fmt_v(v),
                 va="center", fontsize=9, color="#333333")
    for ref in xrefs:
        if ref < max(vals) * 1.15:
            ax2.axvline(ref, color="#bbbbbb", lw=0.8, ls=":")
            ax2.text(ref, len(names) - 0.3, f"{ref:.0f}σ", fontsize=7,
                     color="#888888", ha="center", va="bottom")
    if vline_target is not None:
        ax2.axvline(vline_target, color="#e34948", lw=1.0, ls="--")
        ax2.text(vline_target, len(names) - 0.28, " target", fontsize=7,
                 color="#e34948", ha="left", va="bottom")
    ax2.set_xlim(0, max(max(vals), vline_target or 0) * 1.18 + 1e-12)
    ax2.set_xlabel(xlabel)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

with col2:
    st.subheader("T-P profile")
    cpj = _cpj
    fig3, ax3 = plt.subplots(figsize=(3.4, 3.6), dpi=150)
    ax3.plot(model["T"], model["p_bar"], color="#2a78d6", lw=1.6)
    for tlim in (320.0, 2980.0):
        ax3.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
    ax3.set_yscale("log")
    ax3.invert_yaxis()
    ax3.set_xlabel("temperature (K)")
    ax3.set_ylabel("pressure (bar)")
    ax3.grid(alpha=0.25, lw=0.5)
    fig3.tight_layout()
    st.pyplot(fig3, width="stretch")
    plt.close(fig3)
    st.caption(f"As modeled ({cpj.get('tp_mode', '?')} mode). Dotted lines: "
               "the [320, 2980] K opacity window, profiles outside it are "
               "rejected, never clipped.")

# --- mode details table ------------------------------------------------------
st.subheader("Mode details")
rows = []
key_order = (lambda r: -r["sigma_detect"]) if goal_r == "detect" else (
    lambda r: per_mode.get(r["mode_key"], np.inf))
for r in sorted(results, key=key_order):
    notes = []
    if r["saturated"]:
        notes.append(f"saturates (full-well {r['sat_frac']:.2f} at min groups)")
    n_part = int(np.sum(np.asarray(r.get("n_pix_partial_sat", 0)) > 0))
    if r.get("n_pix_full_sat_dropped"):
        notes.append(f"{r['n_pix_full_sat_dropped']} fully saturated pixels excluded")
    if r.get("n_pix_degenerate_dropped"):
        notes.append(f"{r['n_pix_degenerate_dropped']} degenerate-wavelength "
                     "pixels excluded")
    if n_part:
        notes.append(f"partial saturation in {n_part} bins")
    if r["warnings"]:
        notes.append("; ".join(list(r["warnings"])[:2]))
    row = {"mode": r["label"],
           "band (μm)": f"{r['wl'].min():.2f}-{r['wl'].max():.2f}"}
    # NOTE: this column must stay all-string -- mixing int and str values makes
    # streamlit's Arrow serialization fail (loud pyarrow tracebacks per render)
    if r.get("lsf_applied"):
        notes.append("model blurred to native R (LSF)")
    if r.get("n_segments", 1) > 1:
        notes.append(f"{r['n_segments']} detector segments (offset per segment)")
    if goal_r == "detect":
        row["σ_detect"] = round(r["sigma_detect"], 1)
        _proj = r.get("sigma_detect_proj", float("nan"))
        if np.isfinite(_proj):
            row["σ_detect (proj)"] = round(_proj, 1)
        # experimental correlated scenarios stay OUT of the headline table
        # unless the user explicitly selected one
        if meta.get("scenario", "random") != "random":
            for _sc, _v in r.get("sigma_detect_by_scenario", {}).items():
                if _sc != r.get("scenario"):
                    row[f"σ ({_sc}, exp.)"] = round(_v, 1)
        _t = float(meta.get("target_sig") or 3.0)
        if r["sigma_detect"] > 0:
            _tt = detect.transits_to_target(r, _t)
            row["transits → target"] = (str(_tt["n"]) if _tt["reachable"] else
                                        f"never (floor caps at {_tt['sig_inf']:.1f}σ)")
        else:
            row["transits → target"] = ", "
    else:
        s = per_mode.get(r["mode_key"], np.inf)
        row[f"±{forward.param_axis(gp)} at {tsig:g}σ"] = (
            f"{s:.3g}" if np.isfinite(s)
            else ("saturated" if r["saturated"] else "unconstrained"))
        if np.isfinite(s) and not r["saturated"]:
            _tt = fisher_mod.transits_to_target(r, fisher_names, gp,
                                                target / tsig,
                                                detect.sigma_at_transits,
                                                co_eval=co_eval)
            row["transits → target"] = (str(_tt["n"]) if _tt["reachable"] else
                                        f"never (floor caps at ±{tsig * _tt['sig_inf']:.3g})")
        else:
            row["transits → target"] = ", "
    row.update({"median σ (ppm)": round(r["median_sigma_ppm"]),
                "bins": r["n_bins"], "ngroup": r["ngroup"],
                "cadence (s)": round(r["t_cycle_s"], 1),
                "notes": "; ".join(notes)})
    rows.append(row)
st.dataframe(rows, width="stretch", hide_index=True)
if goal_r == "detect":
    st.caption(
        "**σ_detect is a conditional matched-template S/N at the specified "
        "atmospheric state**, not a retrieval detection: √Δχ² of "
        "(full − without-molecule) over the mode's bins, with a constant depth "
        "offset, plus one step per detector segment (NRS1/NRS2), profiled out. "
        "**σ_detect (proj)** additionally projects out the temperature-structure "
        "and reference-radius (lnR0) Jacobian directions (chemistry and clouds "
        "stay fixed, still conditional); prefer it for narrow margins. σ_bin "
        "is the Pandeia photon+detector noise for in/out-of-transit "
        "integrations (× the optional sensitivity factor, default 1.0), with "
        "the minimum floor applied as a hard maximum on the final bins "
        "(PandExo convention). 'transits → target' averages down the random "
        "term only, the floor is a lower bound at every N, so it can "
        "honestly read 'never'. Groups are chosen and verified against "
        "Pandeia's measured saturation. Because T, clouds, and the other "
        "abundances are not re-fit, a full retrieval detection can only be "
        "weaker."
        + (f" σ_detect is scored under the **{meta.get('scenario')}** "
           "correlated-floor scenario (EXPERIMENTAL, a stated assumption, "
           "not a calibrated systematics model); the σ (…, exp.) columns "
           "re-score it under the other scenarios at identical per-bin "
           "totals." if meta.get("scenario", "random") != "random" else "")
    )
else:
    st.caption(
        f"± per mode is the marginalized Fisher forecast scaled to {tsig:g}σ "
        "(see the table below); 'transits → target' re-solves the Fisher forecast "
        "at each transit count with the random term scaled 1/N and the minimum "
        "floor as a hard lower bound, floor-limited targets read 'never' instead "
        "of an optimistic 1/√N estimate. Saturated modes are excluded from all "
        "forecasts."
    )

# --- Fisher forecast -------------------------------------------------------
# authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
# NOT the multiselect order
if fisher_names and "jac" in model:
    tsig_f = float(meta.get("target_sig") or 3.0)
    st.subheader("Fisher parameter forecast")
    with_jac = [r for r in results if r.get("jac_bins") is not None]

    def _cell(n, s):
        v = tsig_f * fisher_mod.display_sigma(n, s, co_eval=co_eval)
        return "unconstrained" if not np.isfinite(v) or v > 1e4 else f"{v:.3g}"

    frows = []
    usable_f = [r for r in with_jac if not r["saturated"]]
    for r in with_jac:
        if r["saturated"]:
            # shown for completeness, but a saturated mode contributes no usable
            # data -- same exclusion policy as the verdict + combined row
            frows.append({"mode": r["label"] + "  [saturated, excluded]",
                          **{f"±{forward.param_axis(n)} at {tsig_f:g}σ": ", "
                             for n in fisher_names}})
            continue
        sig = fisher_mod.mode_forecast(r, fisher_names)
        frows.append({"mode": r["label"],
                      **{f"±{forward.param_axis(n)} at {tsig_f:g}σ":
                         _cell(n, sig[n]) for n in fisher_names}})
    fdiag = {}
    if len(usable_f) >= 2:
        sig = fisher_mod.combined_forecast(usable_f, fisher_names, diag=fdiag)
        frows.append({"mode": "ALL SELECTED (combined, non-saturated)",
                      **{f"±{forward.param_axis(n)} at {tsig_f:g}σ":
                         _cell(n, sig[n]) for n in fisher_names}})
    st.dataframe(frows, width="stretch", hide_index=True)
    if fdiag:
        rank, dim = fdiag["fisher_rank"], fdiag["fisher_dimension"]
        st.caption(
            f"Numerical health (combined): Fisher rank {rank}/{dim}, condition "
            f"number {fdiag['condition_number']:.2g}."
            + (" **Rank-deficient, degenerate directions are reported as "
               "unconstrained, not as fake finite numbers.**" if rank < dim else ""))
    with st.expander("How to read this table"):
        st.markdown(
            f"- Each cell is the **expected ±uncertainty at {tsig_f:g}σ** "
            f"(= {tsig_f:g} × the Fisher 1σ) on that parameter if you fitted "
            "all listed parameters *simultaneously* to that mode's simulated "
            "data, a linearized best case (Cramér-Rao bound), so real "
            "retrieval posteriors can only be wider.\n"
            "- The sensitivities d(spectrum)/d(parameter) come from **automatic "
            "differentiation through the full VULCAN-JAX chemistry + ExoJAX RT "
            "chain** (photochemistry on), not from finite-difference re-runs.\n"
            "- Each per-mode row also fits (and marginalizes over) a reference-"
            "radius nuisance **lnR0** plus one absolute-depth **offset per "
            "detector segment**, so the two-detector NIRSpec gratings (G395H, "
            "G235H) float independent **NRS1 and NRS2** steps, as every real "
            "G395H fit does (Moran+2023, Madhusudhan+2023). The combined row "
            "shares lnR0 across modes and keeps one offset per segment across "
            "all of them, that's what keeps multi-instrument combinations "
            "honest.\n"
            "- **No priors** are applied: a parameter with no spectral response "
            "in a mode's band reads *unconstrained* rather than a fake number.\n"
            "- Metallicity **[M/H]** and vertical mixing **log Kzz** are reported "
            "in **dex** (factors of 10); **C/O** is the absolute carbon/oxygen "
            "number ratio N_C/N_O (baseline = Lodders 2019 solar ≈ 0.46).\n"
            "- σ is evaluated at the transit count you set. Only the "
            "photon/detector term averages down with more transits; the "
            "systematic floor does not, use the 'transits → target' column, "
            "not a 1/√N extrapolation."
        )
elif out.get("fisher_names"):
    st.info("Fisher forecast requested but the cached model has no Jacobian, "
            "press Run.")
