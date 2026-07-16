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
  parameter derivatives of the spectrum. By default every derivative is a
  central finite difference of independently converged,
  convergence-certified solves. Composition derivatives (metallicity, C/O)
  re-initialize the chemistry at the perturbed elemental abundances -- the
  standard VULCAN workflow -- and every row is evaluated at two step sizes
  that must agree (the run errors otherwise) before the Richardson-combined
  row is reported. This is slower than automatic differentiation but
  conceptually transparent and self-verifying, and it works at any
  composition, including carbon-rich, with photochemistry on or off.
  Every row can optionally use the faster forward-mode AD path instead
  (measured 1.7x on Kzz/temperature rows, ~4x on composition rows;
  warm-started jvp, photochemistry on; a top-level
  "Differentiation method" menu in the GUI). Cross-validation against FD
  on the WASP-39b defaults: Kzz/temperature rows 0.07-0.14%, lnR0 0.01%,
  C/O 0.07%, metallicity 1.6% (the AD metallicity row is the
  fixed-structural-grid derivative -- the 1.6% is the hydrostatic-grid
  rebuild term only FD includes). Physically invalid method-science
  combinations are refused loudly rather than computed: AD with
  photochemistry off, AD's C/O direction on carbon-rich baselines (the
  fixed-O reservoir bound), and any Fisher forecast through condensation
  (under either method). The method actually used is recorded per row
  (`jac_row_method` in the cache, shown in the GUI provenance line).
  Forecast uncertainties are local Cramer-Rao lower bounds under the
  stated noise model, marginalized over calibration nuisances; they are
  not posterior widths.

After a run, an **adjoint diagnostics** panel answers the high-dimensional
questions finite differences cannot afford: one reverse-mode adjoint solve
(VULCAN-JAX `steady_state_reaction_sensitivity` /
`steady_state_input_sensitivity`, validated upstream to 0.2-0.8% against
finite differences) gives the sensitivity of a molecule's photosphere
abundance to every reaction rate in the network and to every layer's
temperature. Each run is preceded by the adjoint scope audit (states the
adjoint cannot represent are refused, not reported) and ships its numerical
certification: fixed-point tightness, solve residual, twin-ensemble spread
(magnitudes are labeled trustworthy only inside the upstream gates,
otherwise the table is labeled a ranking), plus a delta-method abundance
spread under a stated uniform rate-uncertainty class.

## Installation

The tool spans two python environments plus a set of reference data:

1. **The chemistry/GUI environment** (one env): VULCAN-JAX, the
   vulcan-retrieval forward engine, ExoJAX, and Streamlit.
2. **The Pandeia environment** (its own conda env): the STScI ETC engine,
   deliberately not a package dependency.
3. **Reference data**: see "Data setup" below; `jwst-tool data` reports
   what is present on this machine and how to fetch what is not.

Local install, from this repository's root, with sibling checkouts of
`VULCAN-JAX` and `vulcan-retrieval` next to it (the layout this project
uses; `--no-deps` because those dists live on TestPyPI, not PyPI):

```
pip install --no-deps -e ../VULCAN-JAX
pip install --no-deps -e ../vulcan-retrieval
pip install --no-deps -e .
pip install streamlit pandas
jwst-tool
```

Pandeia environment (once):

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

then point `JWST_TOOL_PANDEIA_PYTHON` at that env's python if it is not at
the built-in default path.

A TestPyPI consumer install (`pip install -i https://test.pypi.org/simple/
--extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'`) works
only once the matching dependency chain (vulcan-jax >= 0.2.0,
vulcan-retrieval >= 0.9.0) is published there; the editable checkout install
above is the supported path today. A site-packages install must also set
`VULCAN_PROJECT_ROOT` (the directory containing the `vulcan-retrieval/`
checkout -- its `data/` tree is required at run time) plus
`JWST_TOOL_DATA_DIR` / `JWST_TOOL_OUTPUT_DIR`.

`jwst-tool` preflights the chemistry stack and the Pandeia backend with
actionable error messages, then launches the Streamlit GUI. Equivalent:
`streamlit run src/jwst_tool/app.py` from the repository root.

The first run of a new parameter set takes about 2 minutes at the default
resolution (100 layers, native R about 1500), rising to about 3 minutes at the
150-layer / R about 3000 ceiling. Fisher rows dominate the runtime when
enabled: with the default FD method, roughly 6 to 8 minutes per composition
parameter (metallicity, C/O) and 3 to 5 minutes per Kzz or temperature
parameter; with the AD method, roughly 1 to 2 minutes per row. The adjoint
diagnostics panel costs one chemistry re-solve plus the adjoint ensemble;
its first run on a machine also compiles the step-VJP (~10-20 minutes,
cached persistently). All results are disk-cached under `output/`; repeat
runs are instant.

Every result is exportable from the GUI: each figure (spectrum, mode
ranking, T-P profile) has a PNG download (200 dpi) and a CSV of the plotted
numbers (binned points per mode, the native model spectrum, the ranking
values, the T-P profile), and the mode-details and Fisher tables download
as CSV.

## Data setup

`jwst-tool data` prints a live report of every item below (present /
missing / downloads on first use) with the exact remedy per item; add
`--deep` to also probe the Pandeia env for its engine version. The GUI shows
the same report in its "Data status" panel and annotates the molecule /
broadening / UV-spectrum widgets with availability, and it refuses loudly at
run time if something required is missing -- nothing degrades silently.

| Data | Size | Where it lives | How you get it |
|---|---|---|---|
| Pandeia engine env | -- | conda env `pandeia_2026` | `pip install pandeia.engine==2026.2` (own env; see above) |
| Pandeia JWST refdata | 15 MiB | `data/pandeia_data-2026.2-jwst/` | download: https://stsci.box.com/v/pandeia-data-v2026p2-jwst |
| Pandeia PSF library | 4 GiB | `data/pandeia_psfs-2026.2-jwst/` | download: https://stsci.box.com/v/pandeia-psfs-v2026p2-jwst |
| PHOENIX stellar grid | 1.9 GB | `data/cdbs/grid/phoenix` | STScI reference-atlases synphot5 tarball (see `data/notes.md` for the exact URL); on the maintainer's machine a symlink to a local copy |
| CALSPEC Vega | 288 KB | `data/cdbs/calspec/` | https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits |
| 2MASS Ks bandpass | 9 KB | `data/cdbs/comp/nonhst/` | ships with this repo (tracked in git) |
| H2-He CIA table | 147 MB | `../vulcan-retrieval/data/opacity_cache/` | manual, once: https://hitran.org/data/CIA/main/H2-He_2011.cia (the `/main/` path segment is required) |
| H2-H2 CIA table | 24 MB | `../vulcan-retrieval/data/opacity_cache/` | auto-fetched by ExoJAX on first use |
| CO line list (ExoMol Li2015) | ~8 MB | `../vulcan-retrieval/data/opacity_cache/CO/` | auto-fetched by ExoJAX on first use |
| HITRAN line lists (per molecule) | ~190 MB total | `../vulcan-retrieval/data/exojax_linelists/` | auto-downloaded on the FIRST run that uses each molecule (~10-15 s; network needed at that moment). H2/He-broadening variants cache separately under `h2he/` |
| Stellar UV spectra | small | inside the `vulcan_jax` package | ship with vulcan-jax; nothing to fetch |
| FastChem binary | -- | inside the `vulcan_jax` package | compiled automatically on the first equilibrium-init run; needs `make` + a C++ compiler |

Everything auto-fetched requires network at that moment; everything manual
is listed with its URL in `jwst-tool data` output and `data/notes.md`. The
generated caches (`output/model_cache/`, `output/noise_cache/`) are safe to
delete at any time and rebuild on demand.

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
The built-in default extracts these to this repository's
`data/pandeia_data-2026.2-jwst/` and `data/pandeia_psfs-2026.2-jwst/`
(gitignored -- not fetched by cloning); point `JWST_TOOL_PANDEIA_REFDATA`/
`JWST_TOOL_PANDEIA_PSF_DIR` elsewhere only if you keep the trees outside the
checkout.

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
  Guillot (2010) T-P, shared with the radiative transfer.
  Composition: metallicity (reported in dex, [M/H]; scales the network's
  O/C/N/S abundances together) and the C/O ratio (the absolute carbon/oxygen
  number ratio N_C/N_O; baseline 0.55, the network cfg's WASP-39b elemental
  set from Tsai et al. 2023). Both are structural: the chemistry
  re-initializes at exactly the requested elemental abundances, so any
  value works, including carbon-rich C/O > 1. Vertical
  mixing: a constant
  Kzz. Photochemistry on/off, photolysis zenith angle, diurnal averaging
  factor, molecular diffusion on/off, upwind molecular-diffusion advection
  (vm_mol, off by default), stellar UV spectrum, and the numerical
  grid (chemistry layers -- shared with the RT grid -- and convergence
  tolerance). Condensation (S8 rainout) is available for detection goals
  only, never with a derivative-based forecast (see Known limitations);
  for aerosol opacity in a forecast use the differentiable ExoJAX cloud
  deck instead.
- **ExoJAX radiative-transfer inputs.** The opacity molecule set, H2/He
  Rayleigh scattering, an optional power-law cloud deck, and the
  line-broadening perturber (terrestrial air or H2/He, cache-keyed; molecules
  without H2/He coverage raise rather than fall back).
- **System / target.** Radii, gravity, orbital distance, transit duration,
  host star.

The WASP-39 b GCM T-P and GCM-scaled Kzz baseline modes were REMOVED
(2026-07-13): every planet, including WASP-39 b, uses an isothermal
structural baseline with the explicit isothermal / Guillot profile evaluated
on-graph, and a constant Kzz. No GCM profile is ever silently substituted.
Numerical resolution is set by explicit knobs (the old fast/high fidelity
switch was retired): the number of vertical layers (chemistry and radiative
transfer share one layer count, in the VULCAN section) and the convergence
tolerance, plus the native spectral sampling in the ExoJAX section -- all trade
grid resolution for runtime at identical physics. Out-of-window temperature
profiles and non-converged chemistry solves raise errors; nothing is clipped or
silently carried.

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
opt-in slow test (`JWST_TOOL_RUN_SLOW=1`) closes a production Jacobian row
against an independent smaller-step finite difference of the full forward
model.

**PandExo parity (measured 2026-07-12,
`tests/parity/outputs/REPORT.md`).** Every instrument mode was run
through both this tool's worker and current PandExo (master, pinned commit)
on the same Pandeia 2026.2 engine and reference data, for a moderate star, a
bright saturation edge case, and a faint (Ks = 13) star, with no noise
floor. Result: configuration, wavelength grids, extracted count rates
(flux-ratio medians 0.987 to 1.03), group selection (within one group on
the moderate and bright stars; within about 1 percent relative, up to 5
groups absolute, on the faint star's 500-to-1000-group ramps), per-group
integration timing (within 0.1 percent; the total inherits the group
choice, up to about 5.5 percent where a single group of 16 dominates), and
saturation masking all match. The remaining difference is the noise
model itself and it is one-sided: this tool propagates pandeia's full
extracted noise (correlated ramp noise, background, dark, IPC), while
PandExo's default "fml" calculation is an analytic ramp formula close to
pure photon noise. This tool's sigmas are therefore conservative relative
to PandExo, by roughly 2 to 24 percent for G235H, G395H, NIRISS SOSS, and
NIRCam on matched configurations (up to about 31 percent under the policy
configurations on the faint star); by roughly 34 to 50 percent for NIRSpec
PRISM under the shipped two-group minimum (an intentional group-selection
policy difference -- PRISM is not full group-selection parity, and it
saturates on bright targets); and by roughly 33 to 56 percent for MIRI LRS.
Margins are quantified per mode in the report. Published achieved-versus-PandExo ratios (COMPASS G395H 1.05 to
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
- Ultra-hot-Jupiter opacity sources are absent: no H- bound-free/free-free
  continuum (the SNCHO network has no ionization chemistry, so H-/e- cannot
  be produced) and no atomic or metal-oxide/hydride species (Na, K, Fe, TiO,
  VO, FeH). Because the band starts at 1.0 micron, the Na/K resonance wings
  and most TiO/VO bands fall out of band anyway; the in-band gaps are the H-
  continuum (bound-free edge at 1.64 micron plus free-free longward), FeH
  and VO near 1 to 1.6 micron, and the hot line-list under-representation
  already noted. Forecasts for profiles hotter than roughly 2000 K are
  unreliable and overstate molecular detectability; the GUI warns when a
  profile exceeds that.
- Default spectra are clear-sky; the cloud deck is opt-in and is held fixed
  (not marginalized) in Fisher forecasts.
- The box-transit depth-error formula neglects ingress/egress and
  limb-darkening covariance; there is no time-domain light-curve tier.
- All planets use an isothermal or Guillot T-P and a constant Kzz; the
  WASP-39 b GCM baseline modes were removed (no GCM profile is silently
  substituted).
- Condensation (S8 sulfur rainout, the SNCHO network's one condensation
  channel) is offered for DETECTION goals only, via the certified VULCAN
  recipe: a condensation window followed by a whole-column fix-species pin,
  then a longdy-certified converge of the remaining chemistry
  (photochemistry and molecular diffusion required). It can never be
  combined with a derivative: the pin freezes the condensed reservoir at a
  step-sequence-dependent transient, so the converged state is not a
  reproducible function of the input parameters -- the measured
  forward-mode tangent is about 91% wrong against finite differences (a
  relative error of ~0.91, an order-unity failure -- not a 0.91 agreement
  ratio and not a 9% mismatch), and finite differences of pinned
  transients are equally untrustworthy. The active-condensation layers and
  cold-trap level also switch discretely with temperature. Requesting
  condensation together with a Fisher forecast therefore raises, under
  either differentiation method, as does condensation with photochemistry
  off (no certifiable steady state) or molecular diffusion off. Known
  caveat (in the GUI help): a column too hot to condense still pins S8 at
  its end-of-window value -- enable condensation only where sulfur
  genuinely condenses. An open-system "smooth rainout"
  replacement built to restore differentiability was measured NO-GO (it could
  not reach a strict flux-balanced steady state within its gate); that
  campaign is preserved on the `research/smooth-rainout-fisher` branch of the
  sibling repos, not shipped here. For aerosol / haze opacity, the
  Fisher-compatible route is a **differentiable ExoJAX cloud**: the existing
  power-law cloud deck is already wired into the RT; freeing its parameters
  (or adding a gray deck) as Fisher parameters is a natural future addition.
- Registry values are literature planning defaults; edit them for proposals.

## Repository layout

`src/jwst_tool/` package (GUI `app.py`, forward-model driver `forward.py`,
adjoint diagnostics `adjoint_diag.py`, Pandeia worker `pandeia_worker.py`,
noise/detect/fisher/binning modules, instrument registry, data-availability
detector `datacheck.py` backing `jwst-tool data` and the GUI status panel); `data/` input CDBS tree; `output/` generated runtime
caches (model spectra + Pandeia noise, gitignored). All tests and validation
live under `tests/`: `tests/unit/` the fast numpy suite (`python -m pytest
tests -q`), `tests/parity/` the PandExo parity gate, split into `scripts/`
(harness), `outputs/` (committed `REPORT.md` + `parity_summary.json`,
git-ignored raw run JSON), and `figs/` (committed figures).
Version history in `notes.md`; operational notes in `CLAUDE.md`.
