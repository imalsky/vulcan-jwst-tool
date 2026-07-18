# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST exoplanet spectroscopy observations with a
live forward model: VULCAN-JAX photochemistry feeds ExoJAX radiative
transfer, with instrument noise from the STScI Pandeia engine. Given a
planet and a science goal, it ranks JWST time-series modes and estimates
the transits required. Two geometries: transmission (transit depth) and
thermal emission (secondary-eclipse depth Fp/Fs x (Rp/Rs)^2, with a
PHOENIX stellar SED for Fs). Import name `jwst_tool`, console script
`jwst-tool`.

## Installation

1. Install from TestPyPI. The sibling packages resolve automatically:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]>=0.10.3'
```

2. Create the Pandeia engine environment (once):

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

3. Tell the tool where to keep data and caches. Add these lines to your
shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux) so they persist
across sessions, then open a new terminal or `source` the file:

```
export VULCAN_PROJECT_ROOT="$HOME/vulcan"
export JWST_TOOL_DATA_DIR="$HOME/vulcan/jwst_data"
export JWST_TOOL_OUTPUT_DIR="$HOME/vulcan/jwst_output"
export JWST_TOOL_PANDEIA_PYTHON="$(conda run -n pandeia_2026 which python)"
```

4. Fetch the reference data:

```
jwst-tool fetch
```

This downloads every dataset with a public URL (CIA tables, the PHOENIX
grid, CALSPEC pieces) and prints the two STScI Box downloads it cannot
script, with the exact paths to extract them to. Everything else fetches
itself on first use. `jwst-tool data` shows a live status report with a
remedy per item at any time.

## Running

`jwst-tool` preflights the stack and launches the Streamlit GUI. A new
parameter set takes about 2 minutes; the app shows a runtime estimate
before each run. Results are disk-cached and downloadable as PNG or CSV.

## Physics choices and conventions

Composition scaling (stated because any metallicity / C/O knob must pick a
convention, and papers that leave it implicit are the literature's main
complaint, Drummond et al. 2019, MNRAS 486, 1123): metallicity scales the
network's O/C/N/S abundances together with He/H held fixed (the universal
practice), and C/O moves CARBON at fixed, metallicity-scaled OXYGEN. This
is VULCAN's own published convention (Tsai et al. 2017, ApJS 228, 20).
Note when comparing across codes: petitRADTRANS, the ATMO/Goyal grids, and
GGchem instead anchor metallicity on carbon and vary oxygen, and the
Sonora/PICASO family preserves C+O, so at non-solar C/O the same physical
atmosphere maps to different (Z, C/O) coordinates in different codes, and
different molecules carry the C/O signature (C-bearing species here;
H2O/CO2 in O-varied codes). Near-solar C/O the conventions agree.

T-P profiles are explicit only: isothermal, Guillot, or a tabulated table
(the shipped WASP-39b evening-terminator profile or an upload; the cache
key carries the table's content hash). A tabulated profile has no
temperature parameter, so file-mode Fisher forecasts carry NO temperature
row: they are conditional on the profile being exactly right, and the
reported sigmas are optimistic by the amount temperature uncertainty
would add. Kzz profiles: constant, two parametric forms (Pfunc, JM16),
or the table's Kzz column; the Fisher lnKzz row is a multiplicative scale
of the whole profile in every mode.

Boundary conditions (all off by default): gravitational settling,
diffusion-limited escape (H/H2/He), and constant top/bottom per-species
fluxes with deposition velocities.

Clouds come in two forms. The analytic power-law deck (a gray-to-sloped
opacity per gram) and a physically-grounded Mie condensate deck (real
refractive-index optics from the ExoJAX virga database, a column-uniform
lognormal size distribution) are independent and can be combined. Either
deck's parameters can be freed in the Fisher forecast (marginalized) when
the deck is on: the power-law amplitude and slope, and the Mie particle
radius, size dispersion, and abundance. The Mie radius and dispersion ride
the piecewise-linear Mie lookup grid, so their finite-difference rows carry
a step-size consistency check that refuses a step straddling a grid node;
the Mie abundance is exactly linear. Each Mie condensate needs a one-time
lookup grid built with `tools/generate_miegrid.py`.

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed
atmosphere. The Fisher forecast is linear and local, so real posteriors
can only be wider. The noise model omits time-correlated systematics and
is conservative against PandExo by roughly 2 to 24 percent in the near
infrared and 33 to 56 percent for MIRI LRS. Stellar contamination (the
transit light source effect) is NOT modeled: unocculted spots and faculae
can dominate transit-depth systematics for active hosts, strongest
shortward of ~3 um (Rackham et al. 2018, ApJ 853, 122; Lim et al. 2023,
ApJL 955, L22), so treat short-wavelength depths around active stars with
care. Emission mode is pure-absorption thermal emission (ExoJAX
ArtEmisPure): no scattering in the emergent flux, no reflected light, and
the run refuses atmospheres whose RT column is not optically thick at its
bottom (there is no interior flux term).

ExoJAX capabilities present upstream but NOT wired here: reflected-light
spectra (ArtReflectPure/ArtReflectEmis), scattering emission (ArtEmisScat;
the Mie deck here enters as extinction in the pure-absorption solvers,
without a forward-scattering source term), correlated-k opacities (OpaCKD),
H-minus continuum, atomic/FeH line lists, rotational and instrumental
broadening operators, and GP noise kernels.

## Layout

```
src/jwst_tool/
├── app.py             Streamlit GUI
├── forward.py         forward-model driver: chemistry, RT, Jacobians
├── adjoint_diag.py    reverse-mode adjoint diagnostics
├── fisher.py          Fisher forecasts
├── detect.py          detection statistics
├── noise.py           ETC noise model and floors
├── binning.py         the single measurement operator
├── pandeia_worker.py  Pandeia subprocess
├── instruments.py     mode registry and path roots
├── datacheck.py       data-availability detection
├── planets.py         planet registry
└── cli.py             console entry point
tests/unit/            numpy-only suite: python -m pytest tests -q
tests/parity/          PandExo parity harness and report
```

## Science goals

Detection scores how strongly one molecule imprints on the spectrum: the
chi-square distance between the model and the same model without that
molecule's opacity, with calibration nuisances profiled out. It
upper-bounds any real retrieval detection. Constraint builds a
Fisher-information forecast from the spectrum's parameter derivatives and
reports local Cramer-Rao lower bounds, not posterior widths.
