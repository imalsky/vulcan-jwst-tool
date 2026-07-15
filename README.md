# vulcan-jwst-tool

A JWST planning tool for exoplanet transmission spectroscopy. Give it a planet
and a science goal, and it ranks JWST time-series modes by how well they meet
that goal and estimates how many transits you need. It runs on a laptop: the
first run of a new setup takes a couple of minutes, and cached re-runs are
instant.

Distribution `vulcan-jwst-tool`, import `jwst_tool`, console script `jwst-tool`.

## How it works

Each run is a three-stage forward model, differentiable end to end:

1. **Chemistry (VULCAN-JAX).** A differentiable 1D photochemical-kinetics solver
   computes the steady-state composition for your planet, temperature structure,
   metallicity, C/O, and mixing. VULCAN-JAX reproduces VULCAN to median
   fractional errors of about 1e-9 to 1e-4 in volume mixing ratio, and is 4.4 to
   6.7x faster on a single CPU.
2. **Spectrum (ExoJAX).** The converged chemistry becomes a transmission
   spectrum through the ExoJAX radiative-transfer engine (the same engine the
   sibling `vulcan-retrieval` package uses).
3. **Noise (Pandeia).** STScI's Pandeia exposure-time calculator, the engine
   behind the JWST web ETC, gives the photon and detector noise for each
   instrument mode on your star, with an optional PandExo-style noise floor.

## What it does

Given the spectrum and the noise, the tool scores each mode against one of two
goals, ranks the modes, and reports the number of transits needed to reach a
target precision (or "never" if the noise floor makes it unreachable).

- **Detect a molecule.** A conditional matched-template S/N: how strongly one
  molecule's opacity stands out above the noise, with calibration nuisances (a
  depth offset, plus per-detector-segment offsets and slopes) profiled out. It
  is conditional on the assumed atmosphere, so it upper-bounds what a real
  retrieval would find.
- **Constrain a parameter.** A Fisher-information forecast built from exact
  parameter derivatives of the spectrum. Because the whole model is
  differentiable, one forward-mode automatic-differentiation pass carries a
  parameter perturbation through the converged chemistry and the transmission
  model and returns the exact derivative of transit depth at every wavelength,
  at a cost of about one extra forward run per freed parameter (no finite
  differences). The reported uncertainties are local Cramer-Rao lower bounds
  under the noise model, marginalized over the calibration nuisances.

## What it cannot do

- **It is a planning tool, not a retrieval.** Trust the ranking of modes more
  than the exact ppm or sigma values. A detection score is a best-case template
  match at one fixed atmosphere; a real retrieval that re-fits temperature,
  clouds, and other gases usually does worse. Fisher forecasts are lower bounds,
  not posterior widths.
- **The noise is an instrument-model forecast, not a reduction.** It does not
  model visit-long trends, 1/f residuals, detrending covariance, or stellar
  heterogeneity. Real reductions can land on either side, though unmodeled
  systematics usually degrade precision.
- **Noise is conservative relative to PandExo, not identical.** This tool
  propagates Pandeia's full extracted noise (correlated ramp noise, background,
  dark, IPC); PandExo's default calculation is close to pure photon noise. Sigmas
  here run larger by about 7 to 12 percent for the near-IR gratings and SOSS, and
  by roughly 35 to 48 percent for MIRI LRS. Parity of everything except the noise
  model is measured in `tests/parity/`.
- **No condensation.** A condensing VULCAN column reaches steady state only
  through a condensation window plus a fix-species pin that is not reliably
  differentiable, so condensation cannot enter the Fisher forecast this tool is
  built on, and it is not offered. For cloud or haze opacity, use the
  differentiable ExoJAX power-law cloud deck (held fixed, not marginalized, in
  Fisher forecasts).
- **Clear-sky by default,** with isothermal or Guillot temperature structure and
  a constant Kzz (no GCM profiles), and a box-transit depth-error model with no
  time-domain light-curve tier.

## What you can set

- **Planet and star.** WASP-39 b (validated against the Tsai et al. 2023 setup),
  HD 189733 b, HD 209458 b, WASP-107 b, or a fully custom system (planet radius,
  gravity, orbit, transit duration; star T_eff, log g, [Fe/H], Ks magnitude).
- **Chemistry (VULCAN).** Temperature structure (isothermal or Guillot 2010),
  metallicity [M/H], C/O ratio, constant Kzz, photochemistry and its geometry,
  molecular diffusion, the stellar UV spectrum, and the numerical grid (layers
  and convergence tolerance).
- **Radiative transfer (ExoJAX).** Opacity molecule set, H2/He Rayleigh
  scattering, an optional power-law cloud deck, a line-broadening perturber, and
  native spectral sampling. The base set (H2O, CO2, CO, CH4, SO2) is always
  solved; C2H2, H2S, HCN, and NH3 are opt-in. Adding more molecules is currently
  in development.
- **Instrument and noise.** NIRSpec PRISM / G395H / G235H (BOTS), NIRISS SOSS
  order 1, NIRCam F322W2 / F444W (grism time series), MIRI LRS slitless; a noise
  floor (none, constant ppm, or a wavelength-vs-ppm table); and the transit
  count.

## Installation

Local development, from this repository's root (a sibling checkout of
`vulcan-retrieval` is assumed):

```
pip install --no-deps -e ../vulcan-retrieval
pip install --no-deps -e .
pip install streamlit pandas
jwst-tool
```

`--no-deps` is required because `vulcan-jax` and `vulcan-retrieval` are published
on TestPyPI, not PyPI. Consumer install:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

`jwst-tool` preflights the chemistry stack and the Pandeia backend with
actionable errors, then launches the Streamlit GUI. Equivalent: `streamlit run
src/jwst_tool/app.py` from the repository root. The first run of a new parameter
set takes about 2 minutes at the default grid (100 layers, native R about 1500),
rising to about 3 minutes at the 150-layer / R about 3000 ceiling, plus 20 to 60
seconds per freed Fisher parameter. Results are disk-cached under `output/`, so
repeat runs are instant.

## Backend (Pandeia noise engine)

Pandeia runs in its own conda environment and is deliberately not a package
dependency. The default backend is `pandeia.engine` 2026.2 with matching JWST
reference data, validated mode by mode against current PandExo. Set these
environment variables (all resolved in `src/jwst_tool/instruments.py`, with loud
failures if missing):

- `JWST_TOOL_PANDEIA_PYTHON`: python of the conda env providing `pandeia.engine`.
- `JWST_TOOL_PANDEIA_REFDATA`: the matching `pandeia_data-*` reference tree (a
  one-time download of about 4.3 GB).
- `JWST_TOOL_PANDEIA_PSF_DIR`: the separate PSF library (2026-era refdata ships
  PSFs apart from the main tree).
- `JWST_TOOL_DATA_DIR` / `JWST_TOOL_OUTPUT_DIR`: input CDBS tree and cache root;
  they default to this repo's `data/` and `output/` in an editable checkout.

Set `JWST_TOOL_BACKEND=legacy` to select the pinned pandeia 3.0 pair
(reproducibility only). The worker refuses a mismatched engine/refdata pair, and
every result records the exact engine, refdata, and worker versions in its cache
key, so switching backends invalidates caches automatically.

## Validation

Fast suite (numpy only, no Pandeia or JAX): `python -m pytest tests -q`. It
covers the binning operator, grid validation, noise-floor semantics, the Fisher
and nuisance-projection invariances, and the count-space noise closures. One
opt-in slow test (`JWST_TOOL_RUN_SLOW=1`) closes an autodiff Jacobian row against
finite differences of the full forward model.

PandExo parity is measured, not assumed (`tests/parity/`, see `outputs/REPORT.md`):
configuration, wavelength grids, extracted count rates, group selection, timing,
and saturation masking all match on the shared 2026.2 engine. The one remaining
difference is the noise model, and it is one-sided: this tool is conservative, as
described above.

## Repository layout

`src/jwst_tool/`: the package (Streamlit GUI `app.py`, forward-model driver
`forward.py`, Pandeia worker `pandeia_worker.py`, and the noise, detect, fisher,
binning, and instrument-registry modules). `data/`: input CDBS tree. `output/`:
generated caches (gitignored). `tests/`: the fast numpy suite plus the PandExo
parity gate. Version history in `notes.md`; operational notes in `CLAUDE.md`.
