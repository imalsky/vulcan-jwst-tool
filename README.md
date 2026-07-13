# vulcan-jwst-tool

JWST instrument selection and information forecasting for exoplanet
transmission spectroscopy, built on a fully differentiable forward model:
steady-state VULCAN-JAX photochemistry coupled to ExoJAX radiative transfer
(the engine from the sibling `vulcan-retrieval` package), with instrument
noise from the STScI Pandeia engine. Distribution name `vulcan-jwst-tool`,
import name `jwst_tool`, console script `jwst-tool`.

Given a planet and a science goal, the tool ranks JWST time-series modes by
how well they achieve it and estimates the number of transits required. Two
goal types are supported:

- **Detect a molecule.** A conditional matched-template signal-to-noise
  ratio: the chi-square distance between the model spectrum and the same
  spectrum with one molecule's opacity removed, with calibration nuisances
  profiled out. This is conditional on the assumed atmospheric state and
  upper-bounds any retrieval detection; it is labeled accordingly throughout.
- **Constrain a parameter.** A Fisher-information forecast built from
  parameter derivatives of the spectrum computed by automatic
  differentiation through the converged chemistry and radiative transfer
  (one warm-started forward-mode JVP per parameter, not finite differences).
  These are machine-precision derivatives of the discretized,
  tolerance-converged numerical model, not a claim of exactness for the
  underlying physics; an opt-in test closes a Jacobian row against finite
  differences of the full stack. Forecast uncertainties are local
  Cramer-Rao lower bounds under the stated noise model, marginalized over
  calibration nuisances; they are not posterior widths.

## Installation

Local development, from this repository's root (sibling checkouts of
`vulcan-retrieval` assumed):

```
pip install --no-deps -e ../vulcan-retrieval
pip install --no-deps -e .
pip install streamlit pandas
jwst-tool
```

`--no-deps` is required because `vulcan-jax` and `vulcan-retrieval` are
published on TestPyPI, not PyPI. Consumer install:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

`jwst-tool` preflights the chemistry stack and the Pandeia backend with
actionable error messages, then launches the Streamlit GUI. Equivalent:
`streamlit run src/jwst_tool/app.py` from the repository root.

The first run of a new parameter set takes about 2 minutes at the default
"fast" fidelity (about 3 minutes at "high"), plus 20 to 60 seconds per freed
Fisher parameter. All results are disk-cached under `output/`; repeat runs
are instant.

## Backend configuration

The Pandeia engine runs in its own conda environment and is deliberately not
a package dependency.

**Backend status: CURRENT (default).** The tool defaults to pandeia.engine
2026.2 with pandeia_data-2026.2-jwst -- the STScI JWST 5.1 release, the pair
validated mode by mode against current PandExo (`tests/parity/`). So a new
user gets current-ETC calibration, and proposal-planning output is
current by default. Set `JWST_TOOL_BACKEND=legacy` to select the pinned
pandeia 3.0 + pandeia_data-3.0rc3 pair instead; it is retained only as an
explicit reproducibility backend. The worker refuses to run a mismatched
engine/refdata pair (the STScI same-release rule), and every result and cache
file records the exact engine, refdata, and worker versions in a
`__provenance__` block hashed into the cache key, so switching backends
invalidates caches automatically.

Environment variables, resolved in `src/jwst_tool/instruments.py` with loud
failures:

- `JWST_TOOL_BACKEND`: `current` (default, pandeia 2026.2) or `legacy`
  (pinned 3.0). Selects the default paths below.
- `JWST_TOOL_PANDEIA_PYTHON`: python of a conda env providing the selected
  pandeia.engine (overrides the backend default).
- `JWST_TOOL_PANDEIA_REFDATA`: the matching `pandeia_data-*` reference tree.
- `JWST_TOOL_PANDEIA_PSF_DIR`: the split PSF library (pandeia_data >= 2026
  ships PSFs separately; set for the current backend, empty for 3.0-era).
- `JWST_TOOL_DATA_DIR`: input data root (minimal synphot CDBS tree); defaults
  to this repository's `data/` in an editable checkout. The same CDBS
  (phoenix grid, CALSPEC Vega, 2MASS Ks bandpass) serves both backends.
- `JWST_TOOL_OUTPUT_DIR`: generated cache root; defaults to this repository's
  `output/`.

Current-backend data (one-time download, ~4.3 GB): pandeia.engine 2026.2 via
`pip install pandeia.engine==2026.2` into its own env, plus the JWST reference
data (`pandeia-data-v2026p2-jwst`, ~15 MiB) and PSF library
(`pandeia-psfs-v2026p2-jwst`, ~4 GiB) from the STScI Pandeia distribution.
Point `JWST_TOOL_PANDEIA_REFDATA`/`JWST_TOOL_PANDEIA_PSF_DIR` at the extracted
trees if they are not at the built-in default location.

## Noise model and scope

The uncertainty calculation is designed to reproduce PandExo-style planning
forecasts. The random-noise term comes from the Pandeia calculation for the
selected instrument configuration (saturation-verified group selection,
per-channel saturation masks) and is propagated through the
in-transit/out-of-transit depth measurement and the final spectral binning.
One count-space measurement operator bins the noise, the model spectrum, and
the Jacobians, so the quoted variance always belongs to the same estimator
as the forecast model.

Instrument configurations pin the TSO conventions PandExo uses rather than
pandeia's generic point-source defaults: rapid time-series readout patterns
(NRSRAPID, NISRAPID, RAPID, FASTR1), PandExo's extraction apertures and sky
annuli per instrument, the ecliptic/medium sky background, and
saturation-driven group selection (only NIRCam carries a hard 100-group
cap, matching PandExo). The 2026-07-12 parity run measured the cost of
leaving these implicit at 8 to 20 percent in extracted flux.

The tool provides the same minimum-noise-floor choices as PandExo:

- no minimum floor;
- a constant minimum uncertainty in ppm; or
- a user-supplied two-column wavelength (micron) versus minimum uncertainty
  (ppm) table.

The floor is evaluated on the final binned wavelength grid and applied as

```
sigma_final(lambda) = max[sigma_random(lambda), floor(lambda)]
```

It is not added in quadrature, does not scale with the requested resolving
power, and does not average below the entered minimum when transits are
added. A wavelength table is linearly interpolated, with endpoint values
continued outside the supplied range, matching PandExo behavior.

An optional empirical noise sensitivity factor multiplies the random term.
Its default is exactly 1.0 for every mode: published achieved-versus-predicted
ratios (for example COMPASS G395H at 1.05 to 1.12, NIRISS conventions at 1.2,
MIRI LRS at roughly 1.15) are program-specific and are provided as reference
points for sensitivity studies (`instruments.LITERATURE_NOISE_FACTORS`), not
as a calibration.

These uncertainties are instrument-model planning forecasts, not a complete
time-domain detector or reduction simulation. The baseline model assumes
diagonal spectral uncertainties and does not model visit-long trends,
residual 1/f structure, pointing-dependent systematics, stellar
heterogeneity, or covariance introduced by a light-curve detrending
pipeline. Real reductions can differ in either direction depending on
extraction and analysis choices, though unmodeled systematics commonly
degrade precision. Treat mode rankings as more robust than absolute ppm
values.

**Experimental correlated-noise scenarios.** The GUI offers optional presets
that re-allocate the variance the floor adds (the floor excess) into a
spectrally smooth kernel at identical per-bin totals, for stress-testing how
rankings respond to correlation structure. These presets are stated
assumptions, not calibrated JWST systematics models; they are excluded from
headline results (the default scenario is the exact diagonal model). A
validated empirical covariance model from real JWST residuals is a possible
future goal, not a current feature.

## Statistics

- The molecule score is a conditional matched-template signal-to-noise
  ratio at the specified atmospheric state, never a formal retrieval
  detection significance. The profiled nuisances are a constant depth
  offset, one offset per detector segment (independent NRS1/NRS2 steps for
  the two-detector NIRSpec gratings), per-segment slopes under the
  conservative scenario, and optionally the temperature-structure and
  reference-radius Jacobian directions (`sigma_detect_proj`).
- Nuisance profiling depends only on the span of the nuisance directions:
  the normal matrix is normalized to correlation form before the
  rank-revealing decomposition, so the score is invariant under any
  rescaling of a nuisance row (regression-tested across 24 decades).
- Fisher rank detection and inversion happen in Jacobi-whitened
  (dimensionless) coordinates, so constraints and ranks are invariant under
  changes of parameter units (regression-tested across 24 decades).
  Degenerate directions are reported as "unconstrained", never as unstable
  finite numbers; rank and condition diagnostics are displayed.
- Transits-to-target calculations scale the random term as 1/N with the
  floor as a hard lower bound at every N, and report "never" when a target
  exceeds the floor-limited ceiling.

## Forward model

Wide-band (1 to 15 micron) transmission spectra from steady-state VULCAN-JAX
photochemistry (SNCHO network, photochemistry on by default) and ExoJAX
radiative transfer. Molecules H2O, CO2, CO, CH4, SO2 always; C2H2, H2S, HCN,
NH3 opt-in. The GUI groups the parameters by the engine they drive, all
validated and cache-keyed:

- **VULCAN chemistry inputs.** Temperature structure: isothermal (T_iso) or
  Guillot (2010) T-P — this profile is shared with the radiative transfer.
  Composition: metallicity (exact elemental scaling about the 10x solar
  baseline), delta ln(C/O). Vertical mixing: a constant Kzz. Photochemistry
  on/off, photolysis zenith angle, diurnal averaging factor, molecular
  diffusion on/off, stellar UV spectrum. Condensation (rainout / cold-trap)
  is available for the isothermal profile only and without a Fisher forecast
  (see Known limitations), off by default.
- **ExoJAX radiative-transfer inputs.** The opacity molecule set, H2/He
  Rayleigh scattering, an optional power-law cloud deck, and the
  line-broadening perturber (terrestrial air or H2/He, cache-keyed; molecules
  without H2/He coverage raise rather than fall back).
- **System / target.** Radii, gravity, orbital distance, transit duration,
  host star.

The WASP-39 b GCM T-P and GCM-scaled Kzz baselines are retired from the tool
(isothermal / Guillot and constant Kzz only); they remain in the backend for
the validated closure test and scripted reproduction. Two fidelity tiers
("fast" and "high") trade grid resolution for runtime at identical physics.
Out-of-window temperature profiles and non-converged chemistry solves raise
errors; nothing is clipped or silently carried.

## Planets

`planets.py` registry: WASP-39 b (validated against the Tsai et al. 2023
setup), HD 189733 b, HD 209458 b, WASP-107 b, or a fully custom system. Every
planet runs the same validated chemistry and radiative-transfer machinery with
the system identity swapped in, and the same isothermal / Guillot T-P and
constant Kzz choices.

## Instrument modes

NIRSpec PRISM, G395H, and G235H (BOTS); NIRISS SOSS order 1; NIRCam F322W2
and F444W (grism time series); MIRI LRS slitless. A mode with no unsaturated
pixels at its shortest ramp is reported unusable with its saturation
numbers. Models are convolved to the instrument's native resolving power
where the binning approaches it (MIRI LRS, PRISM), as the stellar-flux-
weighted count ratio the instrument actually measures (the LSF acts on in-
and out-of-transit counts, not on the depth directly); degenerate-wavelength
reference-data pixels are excluded and counted.

## Validation status

Current test suite: `python -m pytest tests -q` (numpy-only; no Pandeia or
JAX required). It covers the binning operator (conservation, Monte Carlo
estimator closure, Jacobian linearity, segment splitting, native-R
smoothing), strict input-grid validation (non-finite wavelengths, duplicate
and duplicate-majority grids, zero-support operators all raise loudly
instead of degrading), floor semantics (none/constant/table, edge extension, hard-max
behavior, R-independence, multi-transit approach to the floor, invalid-input
rejection), the scale-invariance regressions for Fisher rank and nuisance
projection, scenario covariance properties, rank-aware Fisher behavior, and
Poisson count-space and matched-filter amplitude-variance closures. One
opt-in slow test (`JWST_TOOL_RUN_SLOW=1`) closes an autodiff Jacobian row
against finite differences of the full forward model.

**PandExo parity (measured 2026-07-12,
`tests/parity/outputs/REPORT.md`).** Every instrument mode was run
through both this tool's worker and current PandExo (master, pinned commit)
on the same Pandeia 2026.2 engine and reference data, for a moderate star, a
bright star, and a saturation edge case, with no noise floor. Result:
configuration, wavelength grids, extracted count rates (0.99 to 1.02),
group selection (within one group), integration timing (within 1 percent),
and saturation masking all match. The remaining difference is the noise
model itself and it is one-sided: this tool propagates pandeia's full
extracted noise (correlated ramp noise, background, dark, IPC), while
PandExo's default "fml" calculation is an analytic ramp formula close to
pure photon noise. This tool's sigmas are therefore conservative relative
to PandExo, by roughly 6 to 12 percent for G235H, G395H, NIRISS SOSS, and
NIRCam; by roughly 34 to 50 percent for NIRSpec PRISM under the shipped
two-group minimum (an intentional group-selection policy difference -- PRISM
is not full group-selection parity, and it saturates on bright targets); and
by roughly 33 to 49 percent for MIRI LRS. Margins are quantified per mode in
the report. Published achieved-versus-PandExo ratios (COMPASS G395H 1.05 to
1.12; MIRI LRS roughly 1.15 above random-noise simulations) fall on the
same side. Uncertainties are labeled pandeia-extracted-noise forecasts,
never PandExo-identical output.

Pending release gates, tracked explicitly rather than assumed:

- **Exact in/out propagation.** The box-transit formula uses the
  out-of-transit flux and noise for both sides (documented symmetric
  approximation). It is conservative by 1 + 3d/4 to first order at equal
  in/out baselines (about +0.76 percent sigma at 1 percent depth, +8.2
  percent at 10 percent; the coefficient ranges d/2 to d for unequal
  baselines); exact separate in-transit propagation remains open.
- **Physics sensitivity ladders** (heavy, scheduled on HPC): spectral
  resolution convergence of binned depths and Jacobians, top-pressure and
  extended-chemistry ladder, air versus H2/He broadening A/B
  (`vulcan-retrieval/validation/broadening_ab.py`), and hot line-list
  sensitivity for headline molecules (the default HITRAN main-isotopologue
  lists under-represent hot bands above roughly 1000 K).

## Known limitations

- The model band starts at 1.0 micron (H2-H2 CIA table edge) and MIRI LRS is
  cut at 12 microns.
- Default spectra are clear-sky; the cloud deck is opt-in and is held fixed
  (not marginalized) in Fisher forecasts.
- The box-transit depth-error formula neglects ingress/egress and
  limb-darkening covariance; there is no time-domain light-curve tier.
- All planets use an isothermal or Guillot T-P and a constant Kzz; the
  WASP-39 b GCM baselines are retired from the tool for now.
- Condensation is gated to the isothermal profile with no Fisher forecast: its
  saturation tables are frozen at the structural temperature (which equals the
  chemistry temperature only when isothermal), and the derivative through a
  condensing steady state is not validated. It is off by default; a first
  production use should be spot-checked against a direct VULCAN run.
- Registry values are literature planning defaults; edit them for proposals.

## Repository layout

`src/jwst_tool/` package (GUI `app.py`, forward-model driver `forward.py`,
Pandeia worker `pandeia_worker.py`, noise/detect/fisher/binning modules,
instrument registry); `data/` input CDBS tree; `output/` generated runtime
caches (model spectra + Pandeia noise, gitignored). All tests and validation
live under `tests/`: `tests/unit/` the fast numpy suite (`python -m pytest
tests -q`), `tests/parity/` the PandExo parity gate, split into `scripts/`
(harness), `outputs/` (committed `REPORT.md` + `parity_summary.json`,
git-ignored raw run JSON), and `figs/` (committed figures).
Version history in `notes.md`; operational notes in `CLAUDE.md`.
