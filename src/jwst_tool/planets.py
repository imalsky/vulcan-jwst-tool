"""Planet/system registry for the JWST instrument selector.

Pure data, importable by the light GUI path (no jax/vulcan/exojax imports).

Every planet runs on the SAME validated machinery: the WASP-39b SNCHO photo
network + 10x-solar FastChem baseline (the import-locked network), with the
planet identity injected through the existing hooks --

    chemistry : cfg_overrides {Mp, Rp, r_star, orbit_radius, sflux_file, ...}
                (VULCAN derives gs = G*Mp/Rp^2; gs_cgs is converted to Mp at the
                boundary in forward.py. vulcan_chem.build_chem_model applies them
                before the pre-loop)
    RT        : profile {rp_cm, gs_cgs, rstar_cm}
                (exojax_rt.build_rt_model reads them for geometry/normalization)
    noise     : star dict -> pandeia phoenix SED + Ks normalization
    timing    : t14_hr -> in/out-of-transit integration split

Every planet (including WASP-39b) uses an isothermal structural baseline at a
representative temperature and a user-chosen isothermal / Guillot T-P
evaluated on-graph, with a constant Kzz. The WASP-39b GCM T-P/Kzz baseline
modes were removed 2026-07-13 (no GCM profile is ever silently substituted).

Values are literature defaults for PLANNING (all editable in the GUI):
WASP-39b Mancini+2018/Tsai+2023; HD 189733b Torres+2008 (a: Bouchy+2005);
HD 209458b Torres+2008 (a, gs from Southworth+2010); WASP-107b Piaulet+2021.
(Provenance audited against the NASA Exoplanet Archive 2026-07-15: every
numeric field within the literature spread.) Stellar UV: shipped VULCAN spectra,
nearest available spectral type (shown in the GUI, never silently swapped).
"""
from __future__ import annotations

R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
G_CGS = 6.67430e-8  # gravitational constant (cm^3 g^-1 s^-2); for gs_cgs -> Mp

# Shipped stellar UV spectra usable as photochemistry input (VULCAN-JAX
# atm/stellar_flux/, all same two-column surface-flux format), labeled by type.
SFLUX_CHOICES = {
    "sflux-W39b_Tsai2023.txt": "WASP-39 (G8V, Tsai 2023)",
    "Gueymard_solar.txt": "Sun (G2V, Gueymard 2003)",
    "sflux-HD189_Moses11.txt": "HD 189733 (K1.5V, Moses 2011)",
    "sflux-epseri.txt": "eps Eridani (K2V, MUSCLES)",
    "sflux-GJ436.txt": "GJ 436 (M2.5V, MUSCLES)",
    "sflux-GJ1214.txt": "GJ 1214 (M4.5V, MUSCLES)",
}

PLANETS = {
    "wasp39b": dict(
        label="WASP-39 b",
        star=dict(teff=5485.0, log_g=4.5, metallicity=0.0, ks_mag=10.20),
        rstar_rsun=0.932, rp_rjup=1.279, gs_cgs=422.0,
        orbit_au=0.04828, teq_k=1120.0, t14_hr=2.80,
        sflux="sflux-W39b_Tsai2023.txt",
        note="Inflated ~Saturn-mass planet (Ks = 10.2, G8V host); a well-observed "
             "warm transmission-spectroscopy target.",
    ),
    "hd189733b": dict(
        label="HD 189733 b",
        star=dict(teff=5040.0, log_g=4.5, metallicity=0.0, ks_mag=5.54),
        rstar_rsun=0.756, rp_rjup=1.138, gs_cgs=2190.0,
        orbit_au=0.0313, teq_k=1200.0, t14_hr=1.80,
        sflux="sflux-HD189_Moses11.txt",
        note="Very bright host (Ks = 5.5) with a high-gravity planet: expect "
             "most modes to saturate and small spectral features.",
    ),
    "hd209458b": dict(
        label="HD 209458 b",
        star=dict(teff=6065.0, log_g=4.4, metallicity=0.0, ks_mag=6.31),
        rstar_rsun=1.155, rp_rjup=1.359, gs_cgs=930.0,
        orbit_au=0.0475, teq_k=1450.0, t14_hr=3.07,
        sflux="Gueymard_solar.txt",
        note="The classic inflated hot Jupiter (G0V host; solar UV spectrum, "
             "same proxy the VULCAN HD209 config uses).",
    ),
    "wasp107b": dict(
        label="WASP-107 b",
        star=dict(teff=4430.0, log_g=4.6, metallicity=0.0, ks_mag=8.64),
        rstar_rsun=0.67, rp_rjup=0.94, gs_cgs=270.0,
        orbit_au=0.0553, teq_k=740.0, t14_hr=2.74,
        sflux="sflux-epseri.txt",
        note="Warm Neptune-mass super-puff: very low gravity means huge "
             "spectral features (K6V host; eps Eri UV proxy).",
    ),
}

# The "custom" planet starts from these (WASP-39b) values; everything editable.
CUSTOM_DEFAULTS = PLANETS["wasp39b"]


def system_fields(planet: dict) -> dict:
    """The forward-model parameter fields carried by a registry entry."""
    return dict(rp_rjup=planet["rp_rjup"], gs_cgs=planet["gs_cgs"],
                rstar_rsun=planet["rstar_rsun"], orbit_au=planet["orbit_au"],
                sflux=planet["sflux"])
