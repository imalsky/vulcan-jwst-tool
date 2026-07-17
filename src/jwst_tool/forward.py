"""Forward model runner: VULCAN-JAX photochemistry -> ExoJax transmission spectrum.

Two faces:

* Imported by the GUI (light): ``params_key`` / ``cache_path`` / ``load_result``
  touch only the disk cache -- no JAX, no VULCAN, no ExoJax imports.
* Run as a script (heavy):  ``python .../src/jwst_tool/forward.py params.json``
  (equivalently ``python -m jwst_tool.forward params.json``)
  runs the live pipeline and writes the npz cache entry. Progress goes to
  stdout as "[fwd] ..." lines; machine-parsable "[fwd] PROG <frac> <label>"
  lines drive the GUI progress bar.

Planets: every system in ``planets.PLANETS`` (plus "custom") runs on the same
validated W39b SNCHO machinery -- the planet identity (gravity, radius, star,
orbit, UV spectrum) is injected via cfg_overrides for the chemistry and via
profile rp_cm/gs_cgs/rstar_cm for the RT. EVERY planet (including WASP-39b)
gets an isothermal structural baseline at a representative temperature; the
requested T-P profile is evaluated on-graph and drives the chemistry AND the
RT. The WASP-39b GCM T-P/Kzz special cases were REMOVED (2026-07-13): no
tp_mode="baseline", no GCM-scaled Kzz, no has_gcm_baseline branches -- a GCM
profile is never silently substituted.

Numerical resolution (was the GUI "fidelity" switch, now three explicit knobs):
    nz         VULCAN chemistry layers; the ExoJax RT grid (art_nlayer) is LOCKED
               equal to nz in run_model, so chemistry and RT share one layer count
    nu_pts     native wavenumber points over the 1-15 um band (native R ~ nu_pts/2.7)
    yconv_cri  steady-state convergence tolerance
    Defaults reproduce the old "fast" tier (nz=100, nu_pts=4000, yconv 1e-2);
    the validated ceiling is the old "high" tier (nz=150, nu_pts=8000, yconv 1e-3).

Atmosphere-structure knobs (all consumed by the same validated pipeline hooks the
retrieval framework uses):

    T-P profile (tp_mode) -- explicit profiles only, both on-graph tp_eval hooks:
      "isothermal"  T(P) = T_iso
      "guillot"     ExoJax atmprof_Guillot(Tirr, Tint, log10 kappa, log10 gamma)
                    with f=0.25 and the planet's surface gravity
    Kzz: kzz_mode "const" only -- constant Kzz = kzz_const cm^2/s (cfg_overrides
      Kzz_prof="const"), further x kzz_x if given.
    Composition (STRUCTURAL since v13 -- set in the cfg elemental abundances,
    FastChem re-initializes at exactly the requested values; one path for
    every composition, including C-rich):
      met_x_solar  metallicity in x solar -- scales the cfg O/C/N/S together
      co_ratio     absolute C/O = N_C/N_O -- sets C_H = co_ratio * O_H
    Physics (cfg_overrides / RT flags; defaults = the previous hard-coded values):
      use_photo     photochemistry on/off (off = thermochem + transport only;
                    FD Fisher works either way since v13; jac_method="ad"
                    requires ON -- the validated jvp regime)
      sl_angle_deg  photolysis zenith angle (deg; 83 = Tsai 2023 terminator slant)
      f_diurnal     diurnal photolysis factor (1.0 = permanent dayside)
      use_moldiff   molecular diffusion on/off (homopause)
      use_rayleigh  H2/He Rayleigh scattering (ON by default from v4; v3 lacked it)
      cloud_on + log_kappa_cloud + alpha_cloud
                    ExoJax power-law cloud deck (currently fixed, not a Fisher
                    parameter -- see the aerosol-opacity note below)
      extra_mols    opt-in RT molecules beyond the base 5 (C2H2/H2S/HCN/NH3)
      rt_ptop_bar / rt_integration / rt_dit_res  (v15)
                    ExoJAX RT top pressure (band-saturation "wall" knob),
                    ArtTransPure chord-integration scheme (simpson/trapezoid),
                    and PreMODIT broadening-grid spacing; defaults 1e-8 bar /
                    simpson / 1.0 = the pre-v15 hard-coded values, and the
                    engine's echo of each is verified loudly after the build

Any T-P that leaves the modelable premodit window [320, 2980] K on either grid is
REJECTED with a clear error (never clipped) -- same rule as the retrieval.

Condensation is DETECTION-ONLY (v14; fully removed v8-v13): S8 rainout via
the certified conden-window + whole-column fix-species pin recipe
(CONDEN_CFG). It can never meet a derivative. The pin freezes the reservoir
at a step-sequence-dependent transient, so the converged state is not a
reproducible function of the inputs, and the condensing-layer set /
cold-trap index switch discretely in T. That breaks EVERY differentiation
method: the measured jvp-vs-FD relative error is ~0.91 -- the tangent is
about 91% WRONG (an order-unity failure, NOT a 0.91 agreement ratio or a
9% mismatch) -- and finite differences of pinned transients are equally
untrustworthy. `canonical_params` therefore refuses use_condense with
fisher_params (any jac_method), with photo OFF (a cold no-photo condensing
column has no certifiable longdy steady state), and with use_moldiff OFF
(the growth term IS the molecular-diffusion coefficient); adjoint_diag
refuses condensing states too. The open-system "smooth rainout"
replacement that tried to restore differentiability was measured NO-GO
(preserved on the sibling repos' ``research/smooth-rainout-fisher``
branch, not shipped). GUI-documented footgun: a column too hot to condense
still pins S8 at its end-of-window value -- conden-on does NOT reduce to
conden-off; enable it only where sulfur genuinely condenses.

Aerosol opacity, the Fisher-compatible way: represent clouds/haze as a
DIFFERENTIABLE opacity rather than as chemistry. The ExoJax power-law cloud
deck (``cloud_on`` + ``log_kappa_cloud`` + ``alpha_cloud``) is already wired
into the RT and is smooth in its parameters; freeing those (or adding a gray
deck) as Fisher parameters would let an aerosol term enter the forecast
directly -- a natural future addition, and the recommended path instead of
condensation.

Fisher machinery: with ``fisher_params`` set, the runner computes the spectrum
Jacobian d(depth)/d(param) row by row, by one of two methods
(``jac_method``); the method is recorded per row in the npz
(``jac_row_method``) and shown in the GUI:

* "fd" (DEFAULT) -- certified central finite differences ("fd-central"):
  composition rows (lnZ, dlnCO) re-initialize the chemistry per FD point
  (the upstream-VULCAN workflow, valid at ANY composition); lnKzz/T-P rows
  perturb theta on the baseline build; every row passes the h-vs-2h
  consistency gate; lnR0 is an RT-only central difference ("fd-rt").
* "ad" -- one warm-started forward-mode jvp per row ("ad-jvp": the
  validated sensitivity pattern -- continuation from the converged column,
  photochemistry ON required), ~1.7-4x faster per row. Cross-validated
  against the FD rows on W39b defaults: T_iso 0.14%, dlnCO 0.07%, lnKzz
  exact, lnR0 0.9999, lnZ 1.6%. Two stated caveats: the lnZ jvp is the
  FIXED-STRUCTURAL-GRID derivative (the 1.6% gap is the hydrostatic-grid
  rebuild only FD includes), and the dlnCO jvp uses the fixed-O
  differential direction, which is undefined on C-rich baselines where its
  oxygen-reservoir bound b_z <= 0 -- run_model refuses that corner loudly
  (use "fd", which is valid everywhere).

``fisher.py`` turns the Jacobian + the Pandeia noise into parameter forecasts.

Per-molecule "removed" spectra (for detection significance) zero that molecule's
VMR in the RT only -- atmospheric structure (T, mmw) is kept, the standard
nested-model comparison used in observation planning.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# instruments is import-light (os + pathlib only, safe on the GUI's light path)
# and owns the data/output root resolution (env-overridable, loud on failure)
from jwst_tool import instruments as _ins

MODEL_CACHE = _ins.MODEL_CACHE

from jwst_tool import planets   # installed package: works as module AND as a script

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]   # always-on WIDE-profile set
# Opt-in RT additions: the SNCHO network already solves these; adding one costs a
# premodit build (~10-15 s, HITRAN lines downloaded on first use) + one removed
# spectrum. C2H2/HCN carry the high-C/O signal, H2S the 3.8-4.6 um reduced-sulfur
# feature, NH3 the cool (<~900 K) nitrogen chemistry.
EXTRA_MOLECULES = ["C2H2", "H2S", "HCN", "NH3"]
_VERSION = 15  # Bump whenever the physics or the canonical key set changes:
               # invalidates all cached spectra. Full v1-v14 history lives in
               # notes.md. v14: jac_method fd/ad for EVERY row (per-row
               # jac_row_method provenance, b_z guard on the dlnCO AD row) +
               # use_condense back as a DETECTION-ONLY forward option
               # (certified S8 recipe, refused with any derivative).
               # v15: three ExoJAX RT knobs become canonical params --
               # rt_ptop_bar (RT top pressure), rt_integration (transit
               # chord scheme, simpson/trapezoid), rt_dit_res (PreMODIT
               # broadening-grid spacing). Defaults reproduce v14 physics
               # exactly, but the key set changed, so pre-v15 caches are
               # stale. Requires vulcan-retrieval >= 0.10.1 (the engine
               # echoes the knobs; run_model verifies the echo loudly).

# Baseline (unperturbed) carbon-to-oxygen ratio of the shipped network, defined
# the standard way for exoplanet atmospheres: the total-carbon / total-oxygen
# NUMBER ratio  C/O = N_C / N_O  (NOT [C/H]/[O/H], and not a log quantity).
# The basis is the W39b cfg's CUSTOMIZED elemental set (vulcan_jax
# configs/W39b.yaml: use_solar false, C_H 2.95e-3, O_H 5.37e-3 -- the
# Tsai et al. 2023 WASP-39b 10x-solar composition; Asplund-flavored solar
# C/O ~ 0.55). That set defines the CONSERVED atom columns: the elemental
# mode rebuilds every column (and atom_ini) in the cfg basis, and both
# build-time diagnostics confirm it ("[chem] ... baseline C/O = 0.5493").
# It is NOT the FastChem solar_element_abundances.dat set (Lodders 2009
# protosolar, C/O = 0.458) -- that file only seeds the EQUILIBRIUM INITIAL
# GUESS for the non-network trace metals and does not survive into the
# converged column. v10 shipped CO_BASELINE = 0.4579 from the FastChem file;
# that wrong basis skewed every absolute-C/O surface by a factor 1.2.
# Since v13 this constant is only the GUI's DEFAULT co_ratio and display
# baseline (composition itself is structural: run_model pins the cfg
# abundances per request); run_model still cross-checks it against the
# loaded cfg's C_H/O_H and refuses to run on drift.
CO_BASELINE = 0.00295 / 0.00537   # = 0.54935, cfg C_H/O_H (Tsai 2023 10x-solar)

# --- Finite-difference Fisher Jacobians (v13: the ONLY Jacobian path) --------
# Every composition/structure derivative is a CENTRAL difference of fully
# re-initialized, longdy-certified cold solves -- conceptually the upstream-
# VULCAN workflow (set elemental abundances, FastChem re-init, solve), applied
# uniformly. Directions (stated, since any C/O derivative must pick a
# convention): lnZ scales O/C/N/S together (C/O preserved); dlnCO scales C_H
# at fixed O (the same fixed-O direction the retired warm-jvp knob used, so
# rows are 1:1 comparable); lnKzz and the T-P parameters perturb theta on the
# SAME chemistry build (no re-init needed -- Kzz/T enter on-graph).
# Each row is evaluated at step h AND 2h; the two must agree
# (max|J_h - J_2h| / max|J_h| < FD_CONSISTENCY_TOL over the band) or the run
# RAISES -- an FD row dominated by solver convergence noise is never reported.
# The reported row is the Richardson combination (4 J_h - J_2h)/3. Cost: a
# composition row is 4 build+solve cycles (~6-8 min), a theta row 4 cold
# solves (~3-5 min); the price of certified, machinery-free derivatives.
# VALIDATED 2026-07-15 against the warm-jvp AD rows on W39b defaults
# (yconv 1e-3): corr >= 0.9999 and scale within 0.07-1.6% on every row
# (T_iso 0.14%, dlnCO 0.07%, lnKzz exact, lnZ 1.6% -- the lnZ gap is the
# hydrostatic-grid rebuild the FD row includes and the fixed-grid AD chain
# approximated); h-vs-2h consistency 0.004-0.113, all far under the gate.
# Since v14 that warm-jvp AD path is available again as an OPT-IN
# (jac_method="ad") for EVERY row (~1.7-4x faster per row, photo-on only):
# theta rows and lnR0 agree with FD to 0.07-0.14% / 0.9999; the composition
# directions carry the two stated caveats from the module docstring (lnZ =
# the fixed-structural-grid derivative, 1.6% vs FD; dlnCO = the fixed-O
# differential direction, guarded by the b_z bound on C-rich baselines --
# run_model refuses that corner loudly). FD stays the default: certified,
# assumption-free, valid everywhere. Where AD is the ONLY practical tool --
# high-dimensional sensitivities (dL/d ln k over ~800 reactions, dL/dT(P)
# per layer), one adjoint solve vs thousands of FD solves -- see
# VULCAN-JAX steady_state_reaction_sensitivity /
# steady_state_input_sensitivity (validated 0.2-0.8% there).
FD_STEPS = {"lnZ": 0.10, "dlnCO": 0.10, "lnKzz": 0.10,      # ln-space steps
            "T_iso": 10.0, "Tirr": 10.0, "Tint": 10.0,      # Kelvin
            "log_kappa": 0.05, "log_gamma": 0.05}           # dex
FD_COMP_PARAMS = ("lnZ", "dlnCO")     # need a chemistry re-init per FD point
FD_CONSISTENCY_TOL = 0.25
FD_LNR0_STEP = 0.01                   # lnR0 is RT-only (smooth, analytic)
JAC_METHODS = ("fd", "ad")            # certified-FD default / warm-jvp opt-in


def active_molecules(cp: dict) -> list[str]:
    """RT molecule set for canonical params: base set + selected extras."""
    return MOLECULES + [m for m in EXTRA_MOLECULES if m in cp["extra_mols"]]

# Numerical-resolution knobs layered on config.WIDE (1-15 um band unchanged) --
# these replaced the old "fast"/"high" fidelity switch. Defaults reproduce the
# old "fast" tier; the validated ceiling is the old "high" tier. The ExoJax RT
# layer count (art_nlayer) is LOCKED equal to nz in run_model (chemistry and RT
# share one grid), so there is no separate RT-layer knob. Measured fast-vs-high
# agreement (W39b defaults): G395H SO2 3.6 vs 3.8 sigma, F444W 2.8 vs 3.0, Fisher
# sigma(lnZ) 0.027 vs 0.029 dex; the weak mid-IR SO2 bands are the one real
# casualty (MIRI LRS 0.9 vs 1.9 sigma at nz=100 / yconv 1e-2) -- raise nz / tighten
# yconv for final mid-IR numbers.
NZ_DEFAULT, NU_PTS_DEFAULT, YCONV_DEFAULT = 100, 4000, 1.0e-2
NZ_RANGE = (60, 150)            # chemistry (= RT) layers
NU_PTS_RANGE = (4000, 8000)     # native wavenumber points (native R ~ nu_pts/2.7)
YCONV_RANGE = (1.0e-4, 1.0e-2)  # steady-state convergence tolerance (1e-3 is the
                                # validated "high" tier; below it costs runtime
                                # but is safe -- the longdy gate rejects loudly)

# Modelable temperature window (premodit table range, 20 K inset) -- reject, never clip.
T_WINDOW = (320.0, 2980.0)

# Parameters that can be freed in the Fisher forecast, per tp_mode.
CHEM_PARAM_NAMES = ["lnZ", "dlnCO", "lnKzz"]
TP_PARAM_NAMES = {
    "isothermal": ["T_iso"],
    "guillot": ["Tirr", "Tint", "log_kappa", "log_gamma"],
}
# Display SYMBOL, UNIT, and friendly name per parameter for the GUI's constraint
# table / science goals. The symbol is what a reader recognizes and MUST match the
# unit's log base: metallicity and Kzz are reported in dex (log10), so their
# symbols are [M/H] and log Kzz (never "ln", which would mislabel the base); C/O
# is the absolute number ratio N_C/N_O (dimensionless, so no unit bracket).
PARAM_SYMBOLS = {"lnZ": "[M/H]", "dlnCO": "C/O", "lnKzz": "log Kzz",
                 "T_iso": "T_iso", "Tirr": "T_irr", "Tint": "T_int",
                 "log_kappa": "log κ_IR", "log_gamma": "log γ"}
PARAM_UNITS = {"lnZ": "dex", "dlnCO": "", "lnKzz": "dex",
               "T_iso": "K", "Tirr": "K", "Tint": "K",
               "log_kappa": "dex", "log_gamma": "dex"}
PARAM_LABELS = {"lnZ": "Metallicity", "dlnCO": "C/O ratio",
                "lnKzz": "Vertical mixing (Kzz)",
                "T_iso": "Isothermal T", "Tirr": "Guillot T_irr",
                "Tint": "Guillot T_int", "log_kappa": "Guillot log κ_IR",
                "log_gamma": "Guillot log γ"}


def param_axis(name: str) -> str:
    """Axis/column label for a parameter: 'Symbol [unit]', or bare 'Symbol' when
    it is dimensionless (C/O). Keeps every user-facing header on the standard
    representation (e.g. '[M/H] [dex]', 'C/O', 'T_iso [K]')."""
    u = PARAM_UNITS[name]
    return f"{PARAM_SYMBOLS[name]} [{u}]" if u else PARAM_SYMBOLS[name]


# VULCAN condensation channel on the SNCHO network (detection-only, v14; see
# the module docstring for why it can never meet a derivative): the one
# condensation reaction is S8 -> S8_l_s (H2O/NH3 condensation is NOT available
# on this network -- no H2O_l_s/NH3_l_s species). Particle properties:
# rainout-sized 50 um orthorhombic-sulfur particles (rho = 2.07 g/cm^3; r_p
# matches the shipped cfgs' H2O_l_s value) -- smaller aerosol radii make the
# growth term stiffer than Ros2 can resolve to convergence. Convergence
# methodology is upstream VULCAN's conden-window + fix_species pin:
# condensation runs on [start_conden_time, stop_conden_time], then S8 +
# S8_l_s are pinned WHOLE-COLUMN (from_coldtrap_lev=False -- the cold-trap
# argmin degenerates on isothermal columns) and the rest of the chemistry
# converges. Without the pin the steady state is transport-limited (the
# upper S8 reservoir drains through the condensation front on the Kzz
# timescale ~1e9 s while dt stays capped) and every solve would exhaust
# count_max. Caveat, documented: on planets too hot to condense, enabling
# condensation still pins S8/S8_l_s at their t = stop_conden_time transient.
CONDEN_CFG = {
    "use_condense": True,
    "condense_sp": ["S8"],
    "non_gas_sp": ["S8_l_s"],
    "r_p": {"S8_l_s": 5.0e-3},
    "rho_p": {"S8_l_s": 2.07},
    "use_relax": [],
    "use_settling": False,
    "fix_species": ["S8", "S8_l_s"],
    "fix_species_from_coldtrap_lev": False,
    "start_conden_time": 0.0,
    "stop_conden_time": 1.0e6,
    # Convergence mixing-ratio floor for cold (condensing) atmospheres: at
    # the 1e-20 default, kinetically-glacial trace species (e.g. NH3 forming
    # from N2 at ~400 K, drifting at ~1e-18 VMR) gate longdy forever. 1e-15
    # is still orders below any RT-relevant abundance.
    "mtol_conv": 1.0e-15,
    # Default heavy-hydrocarbon conver_ignore list + the trace sulfur
    # allotropes: against a pinned S8 they re-equilibrate on cold-top thermal
    # timescales measured at >=1e15 s (physically unreachable), at abundances
    # far below RT relevance -- none is an RT molecule; the observable sulfur
    # species (SO2, H2S, SO) STAY in the gate. Measured in vulcan-retrieval
    # tests/test_condensation_live_tp.py.
    "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5", "C2H6",
                      "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2", "H2CCO",
                      "S", "S2", "S3", "S4"],
    # Bound certification from below so the conden window + pin always
    # complete before the convergence gate may fire (the certified S8 state
    # is the deterministic end-of-window rainout).
    "trun_min": 1.0e6,
}


def canonical_params(params: dict) -> dict:
    tp_mode = str(params.get("tp_mode", "isothermal"))
    if tp_mode not in TP_PARAM_NAMES:
        raise ValueError(
            f"unknown tp_mode {tp_mode!r} (choose from {list(TP_PARAM_NAMES)}). "
            "The WASP-39b GCM 'baseline' mode was removed -- use an explicit "
            "isothermal or Guillot profile.")
    planet = str(params.get("planet", "wasp39b"))
    if planet not in planets.PLANETS and planet != "custom":
        raise ValueError(f"unknown planet {planet!r}")
    sysd = planets.system_fields(planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS))
    nz = int(params.get("nz", NZ_DEFAULT))
    if not NZ_RANGE[0] <= nz <= NZ_RANGE[1]:
        raise ValueError(f"nz={nz} outside the validated layer range {NZ_RANGE} "
                         "(chemistry layers, also used for the RT grid)")
    nu_pts = int(params.get("nu_pts", NU_PTS_DEFAULT))
    if not NU_PTS_RANGE[0] <= nu_pts <= NU_PTS_RANGE[1]:
        raise ValueError(f"nu_pts={nu_pts} outside the validated range {NU_PTS_RANGE} "
                         "(native wavenumber points; native R ~ nu_pts/2.7)")
    yconv_cri = float(params.get("yconv_cri", YCONV_DEFAULT))
    if not YCONV_RANGE[0] <= yconv_cri <= YCONV_RANGE[1]:
        raise ValueError(f"yconv_cri={yconv_cri:g} outside the validated range "
                         f"{YCONV_RANGE} (steady-state convergence tolerance)")
    sflux = str(params.get("sflux", sysd["sflux"]))
    if sflux not in planets.SFLUX_CHOICES:
        raise ValueError(f"unknown stellar UV spectrum {sflux!r} "
                         f"(choose from {list(planets.SFLUX_CHOICES)})")
    cp = {
        "planet": planet,
        "nz": nz,
        "nu_pts": nu_pts,
        "yconv_cri": round(yconv_cri, 6),
        "rp_rjup": round(float(params.get("rp_rjup", sysd["rp_rjup"])), 4),
        "gs_cgs": round(float(params.get("gs_cgs", sysd["gs_cgs"])), 1),
        "rstar_rsun": round(float(params.get("rstar_rsun", sysd["rstar_rsun"])), 4),
        "orbit_au": round(float(params.get("orbit_au", sysd["orbit_au"])), 5),
        "sflux": sflux,
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        # Composition is fully STRUCTURAL (v13): met_x_solar scales the cfg's
        # O/C/N/S abundances together (He fixed), co_ratio then sets
        # C_H = co_ratio * O_H -- FastChem re-initializes AT the requested
        # composition, the upstream-VULCAN way. One path for every value,
        # including C-rich (> 1); a corner with no certified steady state
        # errors loudly (longdy gate), it never returns a wrong spectrum.
        "co_ratio": round(float(params.get("co_ratio", CO_BASELINE)), 6),
        "kzz_mode": str(params.get("kzz_mode", "const")),
        "kzz_x": round(float(params.get("kzz_x", 1.0)), 4),
        "kzz_const": round(float(params.get("kzz_const", 1.0e9)), 1),
        "tp_mode": tp_mode,
        "T_iso": round(float(params.get("T_iso", 1100.0)), 2),
        "Tirr": round(float(params.get("Tirr", 1560.0)), 2),
        "Tint": round(float(params.get("Tint", 100.0)), 2),
        "log_kappa": round(float(params.get("log_kappa", -2.3)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        # physical VULCAN knobs (all flow through the validated cfg_overrides hook;
        # defaults reproduce the previous hard-coded behavior = the W39b cfg values)
        "use_photo": bool(params.get("use_photo", True)),
        "sl_angle_deg": round(float(params.get("sl_angle_deg", 83.0)), 1),
        "f_diurnal": round(float(params.get("f_diurnal", 1.0)), 3),
        "use_moldiff": bool(params.get("use_moldiff", True)),
        # Upwind molecular-diffusion advection (Shami's vm_branch hybrid scheme).
        # PINNED explicitly since v11: VULCAN-JAX flipped its own default to True
        # on 2026-07-14 and the tool inherited it silently. Default False = the
        # tool's validated pre-flip baseline; True = the upwind scheme, not yet
        # re-baselined for this tool. Only meaningful with use_moldiff on
        # (the engine gates use_vm on use_vm_mol AND use_moldiff).
        "use_vm_mol": bool(params.get("use_vm_mol", False)),
        # RT physics: Rayleigh is known zero-parameter physics, ON by default
        # (v3 and earlier ran without it -- that biased the <1.5 um slope);
        # the cloud deck is the ExoJax power-law retrieval cloud, OFF by default.
        "use_rayleigh": bool(params.get("use_rayleigh", True)),
        # line-broadening perturber: "air" (HITRAN terrestrial widths, the
        # validated default) or "h2he" (planetary H2/He blend; downloads
        # separate h2he/<db> line-list caches on first use, and exojax_rt
        # RAISES for a molecule with no H2/He coverage rather than silently
        # falling back)
        "broadening": str(params.get("broadening", "air")),
        # ExoJAX RT knobs (v15). rt_ptop_bar: the RT column top; above
        # VULCAN's chemistry top the topmost VMR/T are clamped constant
        # (standard transmission convention). Too low a top saturates strong
        # bands into a flat wall (W39b 4.2-5.2 um: ~4.8% of pixels saturated
        # at 1e-6 bar vs 0.1% at the 1e-8 default -- the sibling repo's
        # validation/top_pressure_ladder.py quantifies it). rt_integration:
        # exojax ArtTransPure chord-integration scheme. rt_dit_res: PreMODIT
        # broadening-grid spacing (1.0 = the validated default here, 0.2 =
        # exojax's own default; smaller = finer line wings, slower build).
        "rt_ptop_bar": float(f"{float(params.get('rt_ptop_bar', 1.0e-8)):.6e}"),
        "rt_integration": str(params.get("rt_integration", "simpson")),
        "rt_dit_res": round(float(params.get("rt_dit_res", 1.0)), 3),
        "cloud_on": bool(params.get("cloud_on", False)),
        "log_kappa_cloud": round(float(params.get("log_kappa_cloud", -1.0)), 3),
        "alpha_cloud": round(float(params.get("alpha_cloud", 0.0)), 2),
        # Detection-only condensation (v14): the certified S8 forward recipe.
        # The compatibility matrix below refuses it with ANY derivative.
        "use_condense": bool(params.get("use_condense", False)),
        "extra_mols": sorted(str(m) for m in (params.get("extra_mols") or [])),
        "fisher_params": sorted(str(p) for p in (params.get("fisher_params") or [])),
        # Jacobian method: "fd" (certified central FD, default, valid
        # everywhere) or "ad" (one warm-started jvp per row, photo-on only;
        # see the module docstring for the per-row caveats).
        "jac_method": str(params.get("jac_method", "fd")),
        "version": _VERSION,
    }
    if not 0.0 <= cp["sl_angle_deg"] <= 89.0:
        raise ValueError(f"sl_angle_deg={cp['sl_angle_deg']} outside [0, 89] deg")
    if not 0.0 < cp["f_diurnal"] <= 1.0:
        raise ValueError(f"f_diurnal={cp['f_diurnal']} outside (0, 1]")
    if cp["broadening"] not in ("air", "h2he"):
        raise ValueError(f"broadening={cp['broadening']!r} (choose 'air' or 'h2he')")
    if not 1.0e-9 <= cp["rt_ptop_bar"] <= 1.0e-6:
        raise ValueError(
            f"rt_ptop_bar={cp['rt_ptop_bar']:g} outside [1e-9, 1e-6] bar (the "
            "exercised RT-top range; 1e-8 is the validated default)")
    if cp["rt_integration"] not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={cp['rt_integration']!r}: exojax ArtTransPure "
            "supports 'simpson' (default) or 'trapezoid'")
    if not 0.1 <= cp["rt_dit_res"] <= 1.0:
        raise ValueError(
            f"rt_dit_res={cp['rt_dit_res']:g} outside [0.1, 1.0] (PreMODIT "
            "broadening-grid spacing; 1.0 = this tool's validated default, "
            "0.2 = exojax's own default)")
    bad_mols = set(cp["extra_mols"]) - set(EXTRA_MOLECULES)
    if bad_mols:
        raise ValueError(
            f"unknown RT molecule(s) {sorted(bad_mols)}. This tool ships opacity "
            f"for the always-on base set {MOLECULES} plus the opt-in extras "
            f"{EXTRA_MOLECULES}. To add another molecule you must extend the "
            "forward engine (a cross-repo change in the sibling vulcan-retrieval): "
            "add an entry to retrieval_framework.forward.config.MOLECULES "
            "(HITRAN db id, molmass, VULCAN species name), make sure the SNCHO "
            "network actually solves that species, then list it here in "
            "forward.EXTRA_MOLECULES.")
    if not 0.1 <= cp["co_ratio"] <= 2.0:
        raise ValueError(
            f"co_ratio={cp['co_ratio']} outside [0.1, 2.0] (the network was "
            "never exercised beyond this range)")
    if not 0.1 <= cp["met_x_solar"] <= 100.0:
        raise ValueError(
            f"met_x_solar={cp['met_x_solar']} outside [0.1, 100] x solar")
    allowed_fp = {"lnZ", "dlnCO", "lnKzz"} | set(TP_PARAM_NAMES[tp_mode])
    bad_fp = set(cp["fisher_params"]) - allowed_fp
    if bad_fp:
        raise ValueError(
            f"unknown Fisher parameter(s) {sorted(bad_fp)} for tp_mode="
            f"{tp_mode!r}: choose from ['lnZ', 'dlnCO', 'lnKzz'] + "
            f"{TP_PARAM_NAMES[tp_mode]}")
    if cp["jac_method"] not in JAC_METHODS:
        raise ValueError(
            f"jac_method={cp['jac_method']!r}: choose 'fd' (certified central "
            "finite differences, the default) or 'ad' (one warm-started jvp "
            "per Jacobian row)")
    if cp["jac_method"] == "ad" and not cp["fisher_params"]:
        cp["jac_method"] = "fd"   # no Jacobian requested: inert knob --
        #                           normalize so it can't fragment the cache
    if cp["jac_method"] == "ad" and not cp["use_photo"]:
        raise ValueError(
            "jac_method='ad' (warm-started jvp Jacobian rows) is validated "
            "only in the photo-on regime. Enable photochemistry, or use the "
            "default certified finite differences (jac_method='fd'), which "
            "work photo-off too.")
    # --- condensation: detection-only -- refuse every derivative combo -----
    # (why: module docstring; the raises below carry the full user-facing
    # explanation, and the '91% wrong' wording is test-pinned)
    if cp["use_condense"]:
        if cp["fisher_params"]:
            raise ValueError(
                "condensation (use_condense) cannot be combined with a "
                "Fisher forecast under ANY Jacobian method: the pinned "
                "condensed reservoir is frozen at a step-sequence-dependent "
                "transient (the state is not a reproducible function of the "
                "parameters) and the condensing-layer set switches "
                "discretely in temperature. The AD tangent through it is "
                "about 91% wrong (jvp-vs-FD relative error ~0.91 -- an "
                "order-unity failure, not a 9% mismatch), and finite "
                "differences of pinned transients are equally "
                "untrustworthy. Clear the Fisher parameter list (detection "
                "works), or turn condensation off. For aerosol opacity in a "
                "forecast use the differentiable ExoJax cloud deck "
                "(cloud_on) instead.")
        if not cp["use_photo"]:
            raise ValueError(
                "condensation (use_condense) requires photochemistry ON: a "
                "cold no-photo condensing column has no certifiable longdy "
                "steady state (well-mixed CO2 creeps toward equilibrium on "
                ">= 1e17 s -- the quench regime; upstream integrates those "
                "to a runtime cap, which this tool refuses to present as a "
                "converged spectrum). Enable photochemistry or turn "
                "condensation off.")
        if not cp["use_moldiff"]:
            raise ValueError(
                "condensation (use_condense) requires molecular diffusion "
                "(use_moldiff): the condensation growth term IS the species' "
                "molecular-diffusion coefficient, so with it off every "
                "condensation rate would silently be zero. Enable molecular "
                "diffusion, or turn condensation off.")
    if not cp["use_photo"]:            # photolysis knobs are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = 1.0
    if not cp["cloud_on"]:             # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "isothermal":
        cp["T_iso"] = 0.0
    if tp_mode != "guillot":
        cp["Tirr"] = cp["Tint"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
    if cp["kzz_mode"] != "const":
        raise ValueError(
            f"unknown kzz_mode {cp['kzz_mode']!r}: only 'const' is supported. "
            "The WASP-39b GCM-scaled 'scale' mode was removed -- pass an "
            "explicit kzz_const.")
    bad = set(cp["fisher_params"]) - set(CHEM_PARAM_NAMES + TP_PARAM_NAMES[tp_mode])
    if bad:
        raise ValueError(f"fisher_params {sorted(bad)} not available for tp_mode={tp_mode}")
    return cp


def params_key(params: dict) -> str:
    s = json.dumps(canonical_params(params), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict) -> Path:
    return MODEL_CACHE / f"{params_key(params)}.npz"


def load_result(params: dict):
    """Cached spectrum dict or None.

    Always present: wl_um, depth, depth_wo (nmol, n_nu), mols, ymix, p_bar,
    T, theta, theta_names, params_json, and the convergence certificate
    (conv_stages, conv_accept, conv_longdy, conv_gate). With Fisher
    requested: jac (n_par, n_nu), jac_names, jac_row_method, fd_h, fd_err.
    """
    p = cache_path(params)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _build_tp(cp: dict, gs_cgs: float):
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable) for every mode.
    """
    import jax.numpy as jnp

    mode = cp["tp_mode"]
    if mode == "isothermal":
        def tp_eval(tp, p_bar):
            return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]
        return tp_eval, 1, [cp["T_iso"]], CHEM_PARAM_NAMES + ["T_iso"]
    if mode == "guillot":
        from exojax.atm.atmprof import atmprof_Guillot

        def tp_eval(tp, p_bar):
            p = jnp.asarray(p_bar)
            Tirr, Tint = tp[0], tp[1]
            kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
            return atmprof_Guillot(p, gs_cgs, kappa, gamma, Tint, Tirr, 0.25)
        vals = [cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]]
        return tp_eval, 4, vals, CHEM_PARAM_NAMES + TP_PARAM_NAMES["guillot"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def _make_progress(cp: dict, log):
    """Sequential stage tracker: emits "[fwd] PROG <frac> <label>" lines.

    The stage list MUST mirror run_model's actual stage order (same
    conditionals); weights are rough wall-clock seconds so the GUI bar moves
    honestly. advance() is called at the START of each stage.
    """
    mols = active_molecules(cp)
    stages = [("building chemistry model (compile + warm-up)", 45.0),
              ("building radiative transfer (opacities + CIA)",
               10.0 + 3.0 * len(cp["extra_mols"]))]
    stages += [("solving photochemistry", 35.0)]
    stages += [("full transmission spectrum", 8.0)]
    stages += [(f"spectrum without {m}", 4.0) for m in mols]
    # Jacobian rows: fd = 4 re-init build+solve cycles per composition row /
    # 4 cold solves per lnKzz/T-P row; ad = one warm jvp per row
    _ad = cp["jac_method"] == "ad"
    stages += [((f"AD Jacobian d/d({n})", 110.0) if _ad else
                (f"FD Jacobian d/d({n})",
                 420.0 if n in FD_COMP_PARAMS else 260.0))
               for n in cp["fisher_params"]]
    if cp["fisher_params"]:
        stages += [(("AD" if _ad else "FD") + " Jacobian d/d(lnR0)", 8.0)]
    total = sum(w for _, w in stages)
    state = {"i": 0, "done": 0.0}

    def advance():
        label, w = stages[state["i"]]
        log(f"[fwd] PROG {state['done'] / total:.3f} {label}")
        state["i"] += 1
        state["done"] += w

    def finish():
        log("[fwd] PROG 1.000 done")

    return advance, finish


def _assemble_chem(cp: dict, log):
    """Shared heavy-path assembly (run_model AND adjoint_diag): the resolved
    run profile with the structural composition pinned into cfg_overrides,
    the on-graph T-P hook, theta, and a chemistry-build factory. One code
    path -- the adjoint diagnostics must analyze exactly the model the
    forecasts ran. Imports the engine (import order load-bearing)."""
    # import order is load-bearing: vulcan_chem (env + x64) before jax/exojax
    from types import SimpleNamespace

    from retrieval_framework.forward import config
    from retrieval_framework.forward import vulcan_chem
    import jax

    # Persistent XLA compile cache (shared with the jax_paper adjoint
    # campaign's artifacts): saves the ~40 s runner warm-up on repeat runs
    # and is ESSENTIAL for adjoint_diag, whose step-VJP is a multi-hour
    # cold compile on CPU (measured 2026-07-16).
    jax.config.update("jax_compilation_cache_dir",
                      str(Path.home() / ".cache" / "jax_vulcan"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, cp["gs_cgs"])
    # theta layout [lnZ, dlnCO, lnKzz, tp...] is the vulcan_chem contract; the
    # two composition entries are ALWAYS 0 since v13 -- composition is set
    # STRUCTURALLY in the cfg elemental abundances (below), never as a theta
    # perturbation. Only lnKzz (on-graph multiplier) and the T-P parameters
    # remain live theta directions.
    theta = np.array([0.0, 0.0, np.log(cp["kzz_x"])] + tp_vals,
                     dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")

    profile = dict(config.WIDE)
    # numerical resolution (was the fidelity tier): the ExoJax RT layer count is
    # LOCKED equal to the chemistry layer count -- chemistry and RT share one grid.
    profile["nz"] = cp["nz"]
    profile["art_nlayer"] = cp["nz"]
    profile["nu_pts"] = cp["nu_pts"]
    profile["yconv_cri"] = cp["yconv_cri"]
    # exact-elemental abundance map (lnZ / dlnCO are true column elemental
    # directions; conserved totals rebuilt per theta -- see vulcan_chem docstring).
    # reanchor_atom_ini is moot in this mode but kept for a masks-mode fallback.
    profile["abundance_mode"] = "elemental"
    profile["co_mode"] = "fixed_O"
    profile["broadening"] = cp["broadening"]   # canonical (cache-keyed) knob
    # ExoJAX RT knobs (v15, canonical): the engine validates and ECHOES them
    # on the built rt namespace; run_model verifies the echo so an older
    # engine that ignores unknown profile keys can never return a spectrum
    # that differs from what the cache key claims.
    profile["art_ptop_bar"] = cp["rt_ptop_bar"]
    profile["rt_integration"] = cp["rt_integration"]
    profile["dit_grid_resolution"] = cp["rt_dit_res"]
    profile["reanchor_atom_ini"] = True   # finite-Z steps must re-anchor atom totals
    # step-size cap, validated state-preserving (retrieval case.py): prevents the
    # adaptive-dt ballooning non-convergence at high Kzz the GUI sliders can reach
    profile["dt_max"] = 1.0e11
    mols_active = active_molecules(cp)
    profile["molecules"] = mols_active
    profile["use_photo"] = cp["use_photo"]        # build_chem_model reads this key
    profile["use_rayleigh"] = cp["use_rayleigh"]  # exojax_rt reads this flag

    # --- planet identity ------------------------------------------------------
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    rstar_cm = cp["rstar_rsun"] * planets.R_SUN_CM
    profile["rp_cm"] = rp_cm            # RT geometry (exojax_rt reads these)
    profile["gs_cgs"] = cp["gs_cgs"]
    profile["rstar_cm"] = rstar_cm
    ovr = {                              # chemistry side (applied pre-pre-loop)
        # VULCAN derives gravity as g = G*Mp/Rp^2; convert the tool's gs_cgs knob
        # to the equivalent planet mass at this radius.
        "Mp": cp["gs_cgs"] * rp_cm**2 / planets.G_CGS,
        "Rp": rp_cm, "r_star": cp["rstar_rsun"],
        "orbit_radius": cp["orbit_au"],
        "sflux_file": f"atm/stellar_flux/{cp['sflux']}",
        "use_moldiff": cp["use_moldiff"],
        # pin the vm_mol scheme EXPLICITLY (never inherit the upstream YAML
        # default, which flipped to True on 2026-07-14): hybrid in-loop
        # phase-flip is how vm_mol runs, so the two flags travel together.
        "use_vm_mol": cp["use_vm_mol"],
        "use_hybrid_vm_mol": cp["use_vm_mol"],
    }
    if cp["use_photo"]:                  # photolysis geometry/averaging knobs
        ovr["sl_angle"] = float(np.deg2rad(cp["sl_angle_deg"]))
        ovr["f_diurnal"] = cp["f_diurnal"]
    if cp["use_condense"]:
        # canonical_params confirmed detection-only (no fisher), photo on,
        # moldiff on. The engine rebuilds the condensation arrays on-graph
        # from the live T(P) per solve (vulcan_chem._prep), so isothermal
        # and Guillot are both self-consistent; the channel config
        # (S8 -> S8_l_s + particle properties + the certified convergence
        # recipe) is CONDEN_CFG.
        ovr.update(CONDEN_CFG)
    # Isothermal structural baseline for EVERY planet (including WASP-39b; the
    # GCM structural baseline was removed): the on-graph tp_eval supplies the
    # actual T(P) for chemistry+RT, the structural profile only sets the
    # hydrostatic grid + EQ init. Constant Kzz; lnKzz (theta[2]) still
    # multiplies it on-graph.
    T_struct = (cp["T_iso"] if cp["tp_mode"] == "isothermal"
                else cp["Tirr"] / np.sqrt(2.0))   # ~equilibrium T at f=0.25
    ovr.update({"atm_type": "isothermal", "Tiso": float(T_struct),
                "Kzz_prof": "const", "const_Kzz": cp["kzz_const"]})
    log(f"[fwd] planet {cp['planet']}: isothermal structural baseline "
        f"{T_struct:.0f} K, const Kzz {cp['kzz_const']:.1e} cm2/s, "
        f"UV = {cp['sflux']}")
    profile["cfg_overrides"] = ovr

    # CO_BASELINE must equal the loaded cfg's C_H/O_H (it is the GUI's default
    # co_ratio and the display baseline) -- refuse loudly on drift (the v10
    # bug was exactly a wrong-basis constant here).
    import vulcan_jax as _vj
    _cfg_chk = _vj.load_config(profile.get("vulcan_cfg_name") or config.W39B_CFG_NAME)
    _co_cfg = float(_cfg_chk.C_H) / float(_cfg_chk.O_H)
    if abs(_co_cfg / CO_BASELINE - 1.0) > 1e-9:
        raise RuntimeError(
            f"forward.CO_BASELINE={CO_BASELINE:.5f} no longer matches the "
            f"network cfg's C_H/O_H={_co_cfg:.5f}: the C/O display baseline "
            "would be mislabeled. Update CO_BASELINE to the cfg value (and "
            "bump _VERSION).")

    def _abundance_overrides(met_x_solar: float, co_ratio: float) -> dict:
        # STRUCTURAL composition (v13): scale the cfg's metal abundances
        # together for metallicity (He fixed -- He is not a metal), then set
        # carbon from the requested C/O at the scaled oxygen. FastChem
        # re-initializes at exactly this composition (ini_abun writes the
        # custom O/C/N/S values straight into the FastChem input;
        # fastchem_met_scale only scales the NON-network trace metals
        # (Na/K/Fe/...), so it follows met_x_solar to stay consistent).
        m = met_x_solar / 10.0                 # cfg abundances ARE 10x solar
        o_h = float(_cfg_chk.O_H) * m
        return {"O_H": o_h, "C_H": co_ratio * o_h,
                "N_H": float(_cfg_chk.N_H) * m,
                "S_H": float(_cfg_chk.S_H) * m,
                "fastchem_met_scale": float(met_x_solar)}

    ovr.update(_abundance_overrides(cp["met_x_solar"], cp["co_ratio"]))
    log(f"[fwd] structural composition: {cp['met_x_solar']:g}x solar metals, "
        f"C/O = {cp['co_ratio']:.3f} (C_H {ovr['C_H']:.3e}, O_H {ovr['O_H']:.3e})")

    def _build_chem(extra_abun: dict | None = None, tag: str = "baseline"):
        prof = dict(profile)
        prof["cfg_overrides"] = ({**ovr, **extra_abun} if extra_abun else ovr)
        t_b = time.time()
        chem_b = vulcan_chem.build_chem_model(prof, tp_eval=tp_eval,
                                              n_tp_params=n_tp)
        log(f"[fwd] chemistry model ({tag}) ready in {time.time()-t_b:.0f} s")
        return chem_b

    return SimpleNamespace(
        profile=profile, theta=theta, theta_names=theta_names,
        tp_eval=tp_eval, n_tp=n_tp, build_chem=_build_chem,
        abundance_overrides=_abundance_overrides, config=config)


def _check_t_window(tp_eval, theta, p_bar, log):
    """T-P validity on the chemistry grid: REJECT (never clip) out-of-window
    profiles. Returns the evaluated T(P) as a numpy array."""
    import jax.numpy as jnp

    T_check = np.asarray(tp_eval(jnp.asarray(theta[3:]), jnp.asarray(p_bar)))
    tmin, tmax = float(T_check.min()), float(T_check.max())
    if tmin < T_WINDOW[0] or tmax > T_WINDOW[1]:
        raise RuntimeError(
            f"T-P profile leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K (min {tmin:.0f} K, max {tmax:.0f} K). "
            "Adjust the profile parameters -- out-of-window layers are rejected, "
            "not clipped (opacity tables end there).")
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")
    return T_check


def run_model(params: dict, log=print) -> Path:
    cp = canonical_params(params)
    advance, finish = _make_progress(cp, log)
    A = _assemble_chem(cp, log)
    # heavy imports AFTER _assemble_chem: vulcan_chem must init env/x64 first
    import jax
    import jax.numpy as jnp
    from retrieval_framework.forward import exojax_rt
    from retrieval_framework.forward import interp_map

    config = A.config
    profile, theta, theta_names = A.profile, A.theta, A.theta_names
    tp_eval, _abundance_overrides, _build_chem = (
        A.tp_eval, A.abundance_overrides, A.build_chem)
    mols_active = list(profile["molecules"])

    t0 = time.time()
    advance()
    log("[fwd] building chemistry model (VULCAN-JAX warm-up ~40 s) ...")
    chem = _build_chem()

    T_check = _check_t_window(tp_eval, theta, chem.p_bar, log)

    t0 = time.time()
    advance()
    log("[fwd] building ExoJax RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time()-t0:.0f} s")
    # Echo check on the v15 RT knobs: an engine too old to know these
    # profile keys ignores them silently -- refuse rather than cache a
    # spectrum under a key describing physics the engine did not apply.
    _echo = {"art_ptop_bar": cp["rt_ptop_bar"],
             "rt_integration": cp["rt_integration"],
             "dit_grid_resolution": cp["rt_dit_res"]}
    for k, want in _echo.items():
        got = getattr(rt, k, None)
        if got != want:
            raise RuntimeError(
                f"RT engine did not honor {k}={want!r} (echoed {got!r}). "
                "The installed vulcan-retrieval predates the "
                "profile-overridable RT knobs -- upgrade to >= 0.10.1.")

    p_art_j = jnp.asarray(rt.p_art_bar)

    def art_T(th):
        return tp_eval(th[3:], p_art_j)

    # ExoJax power-law retrieval cloud [log10 kappac0 (cm^2/g at 3.5 um), alphac];
    # held FIXED in the Fisher forecast (no cloud marginalization -- documented).
    cloud_vec = (jnp.asarray([cp["log_kappa_cloud"], cp["alpha_cloud"]])
                 if cp["cloud_on"] else None)

    def make_depth_fn(chem_b):
        """Depth function bound to ONE chemistry build: the interpolation map
        follows that build's hydrostatic grid (composition moves the mean
        molecular weight and hence the pressure grid between the FD re-init
        builds, so the map is never shared across builds)."""
        to_art_b = interp_map.make_to_art(chem_b.p_bar, rt.p_art_bar)
        mol_cols = {k: chem_b.sidx[config.MOLECULES[k]["vulcan"]]
                    for k in rt.molecules}
        h2_b, he_b = chem_b.sidx["H2"], chem_b.sidx["He"]

        def depth_fn(y, th, lnR0=0.0, drop_mol=None):
            ymix = y / jnp.sum(y, axis=1, keepdims=True)
            T_art = art_T(th)
            mmw_art = to_art_b(ymix @ chem_b.species_masses)
            vmr = {k: to_art_b(ymix[:, c]) for k, c in mol_cols.items()}
            if drop_mol is not None:
                vmr[drop_mol] = jnp.zeros_like(vmr[drop_mol])
            return rt.transmission_depth_r(
                vmr, to_art_b(ymix[:, h2_b]), T_art, mmw_art,
                jnp.asarray(lnR0), vmr_he=to_art_b(ymix[:, he_b]),
                cloud=cloud_vec)
        return depth_fn

    depth_from_y = make_depth_fn(chem)

    # --- chemistry: certified cold solves (v13: no warm continuation) --------
    t0 = time.time()
    th0 = jnp.asarray(theta)
    conv_cert = []   # (stage, accept_count, longdy) for every PASSED gate
    def _check_converged(ac, longdy, stage):
        # accept_count < count_max is NOT a convergence test: the hybrid vm_mol phase-flip
        # (and the stall fallback) terminate the runner EARLY -- accept_count ~ count_min+2000,
        # well below count_max -- even when the column is still oscillating and nowhere near a
        # steady state (photo-off W39b: longdy ~ 1-60 with accept_count ~2122). Gate on the
        # runner's own longdy metric against the loose convergence gate (yconv_min); a genuinely
        # converged solve has longdy < yconv_min (photo-on W39b sits at ~0.06 < 0.1).
        ac = int(ac); longdy = float(longdy)
        if not (longdy < chem.yconv_min):
            how = (f"hit the count_max={chem.count_max} cap" if ac >= int(chem.count_max)
                   else f"terminated early at {ac} accepted steps (e.g. hybrid vm_mol "
                        "phase-flip / stall budget) without settling")
            raise RuntimeError(
                f"chemistry did NOT converge ({stage}: longdy={longdy:.3g} >= gate "
                f"yconv_min={chem.yconv_min:g}; {how}). This parameter corner has no "
                "certified steady state -- adjust T-P / Kzz / composition (or the "
                "convergence settings) rather than trusting an unconverged spectrum.")
        conv_cert.append((stage, ac, longdy))

    # Single certified cold solve, always: composition is baked into the build
    # (structural), so there is no composition continuation and no stage 2.
    advance()
    log("[fwd] solving photochemistry (cold, certified) ...")
    y_sol, ac, longdy = chem.converged_y(th0, return_longdy=True)
    _check_converged(ac, longdy, "baseline solve")
    y_np = np.asarray(y_sol)
    if not np.all(np.isfinite(y_np)):
        raise RuntimeError("chemistry solve returned non-finite abundances -- "
                           "parameter set outside the modelable range")
    log(f"[fwd] chemistry solved in {time.time()-t0:.0f} s total")

    # --- RT: full spectrum + one spectrum per removed molecule ---------------
    t0 = time.time()
    advance()
    log("[fwd] radiative transfer: full spectrum (jit compile on first call) ...")
    depth = np.asarray(depth_from_y(y_sol, th0))
    log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

    depth_wo = np.zeros((len(mols_active), depth.shape[0]))
    for i, mol in enumerate(mols_active):
        t1 = time.time()
        advance()
        depth_wo[i] = np.asarray(depth_from_y(y_sol, th0, drop_mol=mol))
        log(f"[fwd] spectrum without {mol} in {time.time()-t1:.0f} s")

    # --- Fisher Jacobian: certified FD (default) / warm-jvp AD (opt-in) ------
    # See the FD_STEPS block at module top. "fd": composition rows
    # re-initialize the chemistry per FD point (the upstream-VULCAN
    # workflow), lnKzz/T-P rows perturb theta on the baseline build; every
    # FD point is longdy-certified, every FD row must pass the h-vs-2h
    # consistency gate, and the reported FD row is the Richardson
    # combination (4 J_h - J_2h)/3. "ad": EVERY row is one warm-started jvp
    # (module docstring has the per-row caveats). The method used for each
    # row is recorded in jac_row_method.
    jac_names = list(cp["fisher_params"])
    jac = np.zeros((len(jac_names) + 1, depth.shape[0])) if jac_names else None
    fd_h, fd_err, row_method = [], [], []
    if jac_names:
        def _certified_depth(chem_b, th, stage):
            y_b, ac_b, ld_b = chem_b.converged_y(jnp.asarray(th),
                                                 return_longdy=True)
            _check_converged(ac_b, ld_b, stage)
            return np.asarray(make_depth_fn(chem_b)(y_b, jnp.asarray(th)))

        def _fd_row(name, d_p1, d_m1, d_p2, d_m2, h):
            j1 = (d_p1 - d_m1) / (2.0 * h)
            j2 = (d_p2 - d_m2) / (4.0 * h)
            if not (np.isfinite(j1).all() and np.isfinite(j2).all()):
                raise RuntimeError(
                    f"FD Jacobian for {name}: non-finite entries")
            scale = float(np.max(np.abs(j1)))
            if scale == 0.0:
                return j1, 0.0     # no spectral response: exact zero row
            err = float(np.max(np.abs(j1 - j2)) / scale)
            if err > FD_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"FD Jacobian for {name} FAILED the step-size consistency "
                    f"check: max|J(h) - J(2h)| / max|J(h)| = {err:.3f} > "
                    f"{FD_CONSISTENCY_TOL} (h = {h:g}). The row is dominated "
                    "by solver convergence noise or curvature -- tighten "
                    "yconv_cri (1e-3 or 1e-4), raise nz, or adjust "
                    "forward.FD_STEPS. An uncertified derivative is never "
                    "reported.")
            return (4.0 * j1 - j2) / 3.0, err   # Richardson, O(h^4)

        def _ad_theta_depth(th):
            # warm continuation from the converged column: the primal is a
            # no-op re-converge, the jvp is the validated steady-state
            # tangent (photo ON -- gated in canonical_params). The baseline
            # theta composition entries are 0 (structural composition), so
            # lnZ_ref/c_o_ref are 0; an lnZ/dlnCO TANGENT direction through
            # them is the validated differential map at this baseline.
            y_w = chem.converged_y(th, warm_y=y_sol, lnZ_ref=0.0, c_o_ref=0.0)
            return depth_from_y(y_w, th)

        if cp["jac_method"] == "ad" and "dlnCO" in jac_names:
            # The dlnCO jvp rides the fixed-O differential direction, which
            # exists only while its oxygen-reservoir bound b_z is positive --
            # on a sufficiently C-rich structural baseline it is not, and no
            # tangent direction exists. FD re-initializes instead and is
            # valid at any composition.
            _bz = getattr(chem, "co_bz_bound", None)
            if _bz is not None and float(_bz) <= 0.0:
                raise RuntimeError(
                    "AD Jacobian for dlnCO is undefined at this composition: "
                    f"the fixed-O differential direction's oxygen-reservoir "
                    f"bound b_z = {float(_bz):.3g} <= 0 (C-rich baseline, "
                    f"C/O = {cp['co_ratio']:g}). Use jac_method='fd' -- the "
                    "certified FD row re-initializes the chemistry and is "
                    "valid at any composition.")

        for j, name in enumerate(jac_names):
            t1 = time.time()
            advance()
            if cp["jac_method"] == "ad":
                # AD row: one warm-started forward-mode jvp along this
                # theta direction (composition directions included -- the
                # cross-validated differential map; lnZ is the fixed-
                # structural-grid derivative, see the module docstring)
                i_par = theta_names.index(name)
                e = np.zeros_like(theta)
                e[i_par] = 1.0
                _, dd = jax.jvp(_ad_theta_depth, (th0,), (jnp.asarray(e),))
                jac[j] = np.asarray(dd)
                if not np.isfinite(jac[j]).all():
                    raise RuntimeError(
                        f"AD Jacobian for {name}: non-finite entries")
                fd_h.append(0.0)          # no FD step: AD row
                fd_err.append(np.nan)     # no h-vs-2h metric: AD row
                row_method.append("ad-jvp")
                log(f"[fwd] AD Jacobian d(depth)/d({name}) in "
                    f"{time.time()-t1:.0f} s (warm-started jvp)")
                continue
            h = FD_STEPS[name]
            dvals = {}
            if name in FD_COMP_PARAMS:
                # composition direction: FastChem re-init + certified cold
                # solve per FD point (4x build+solve)
                for s in (1, -1, 2, -2):
                    f = float(np.exp(s * h))
                    if name == "lnZ":      # all metals together; C/O preserved
                        ab = _abundance_overrides(cp["met_x_solar"] * f,
                                                  cp["co_ratio"])
                    else:                  # dlnCO: carbon at fixed oxygen
                        ab = _abundance_overrides(cp["met_x_solar"],
                                                  cp["co_ratio"] * f)
                    chem_s = _build_chem(ab, tag=f"FD {name} {s:+d}h")
                    dvals[s] = _certified_depth(chem_s, theta,
                                                f"FD {name} {s:+d}h")
            else:
                # theta direction (lnKzz / T-P): baseline build, certified
                # cold solves at theta +- h, +- 2h
                i_par = theta_names.index(name)
                for s in (1, -1, 2, -2):
                    th_s = theta.copy()
                    th_s[i_par] += s * h
                    if i_par >= 3:         # T-P step must stay in the window
                        T_s = np.asarray(tp_eval(jnp.asarray(th_s[3:]),
                                                 jnp.asarray(chem.p_bar)))
                        if T_s.min() < T_WINDOW[0] or T_s.max() > T_WINDOW[1]:
                            raise RuntimeError(
                                f"FD step for {name} ({s:+d}h = {s * h:+g}) "
                                f"leaves the modelable T window {T_WINDOW}: "
                                "move the profile away from the window edge "
                                "or reduce forward.FD_STEPS for it.")
                    dvals[s] = _certified_depth(chem, th_s,
                                                f"FD {name} {s:+d}h")
            jac[j], err = _fd_row(name, dvals[1], dvals[-1],
                                  dvals[2], dvals[-2], h)
            fd_h.append(h)
            fd_err.append(err)
            row_method.append("fd-central")
            log(f"[fwd] FD Jacobian d(depth)/d({name}) in "
                f"{time.time()-t1:.0f} s (h-vs-2h consistency {err:.3f} < "
                f"{FD_CONSISTENCY_TOL})")

        t1 = time.time()
        advance()
        # lnR0 is RT-only (smooth, analytic in lnR0). "fd": one central
        # difference through the radiative transfer, no chemistry and no
        # gate needed; "ad": the RT jvp (the two agree to 0.9999, measured).
        if cp["jac_method"] == "ad":
            _, dd = jax.jvp(lambda r: depth_from_y(y_sol, th0, lnR0=r),
                            (jnp.asarray(0.0),), (jnp.asarray(1.0),))
            jac[-1] = np.asarray(dd)
            fd_h.append(0.0)
            fd_err.append(np.nan)
            row_method.append("ad-jvp")
        else:
            d_rp = np.asarray(depth_from_y(y_sol, th0, lnR0=+FD_LNR0_STEP))
            d_rm = np.asarray(depth_from_y(y_sol, th0, lnR0=-FD_LNR0_STEP))
            jac[-1] = (d_rp - d_rm) / (2.0 * FD_LNR0_STEP)
            fd_h.append(FD_LNR0_STEP)
            fd_err.append(0.0)
            row_method.append("fd-rt")
        jac_names.append("lnR0")
        log(f"[fwd] {cp['jac_method'].upper()} Jacobian d(depth)/d(lnR0) "
            f"[RT-only nuisance] in {time.time()-t1:.0f} s")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params)
    ymix_np = y_np / y_np.sum(axis=1, keepdims=True)
    arrays = dict(
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        depth=depth, depth_wo=depth_wo,
        mols=np.array(mols_active, dtype="U8"),
        ymix=ymix_np, p_bar=np.asarray(chem.p_bar),
        T=np.asarray(T_check), theta=theta,
        theta_names=np.array(theta_names, dtype="U16"),
        params_json=np.array(json.dumps(cp), dtype="U2048"),
        # convergence certificate: the runner's own longdy per gated stage
        # (all strictly below the gate, or run_model would have raised)
        conv_stages=np.array([s for s, _, _ in conv_cert], dtype="U48"),
        conv_accept=np.array([a for _, a, _ in conv_cert], dtype=np.int64),
        conv_longdy=np.array([l for _, _, l in conv_cert], dtype=np.float64),
        conv_gate=np.array([float(chem.yconv_min)], dtype=np.float64),
    )
    if jac is not None:
        arrays["jac"] = jac
        arrays["jac_names"] = np.array(jac_names, dtype="U16")
        # Per-row provenance: the method actually used ("fd-central" /
        # "ad-jvp" / "fd-rt"), the FD step, and the h-vs-2h consistency
        # metric (0 for lnR0 by construction -- RT-only, no gate; NaN for
        # AD rows -- there is no step to vary)
        arrays["jac_row_method"] = np.array(row_method, dtype="U16")
        arrays["fd_h"] = np.array(fd_h, dtype=np.float64)
        arrays["fd_err"] = np.array(fd_err, dtype=np.float64)
    np.savez_compressed(out, **arrays)
    finish()
    log(f"[fwd] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
