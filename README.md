# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST transmission-spectroscopy observations with a
live forward model instead of an assumed input spectrum. Steady-state
VULCAN-JAX photochemistry feeds ExoJAX radiative transfer, and instrument
noise comes from the STScI Pandeia engine. Given a planet and a science goal,
the tool ranks JWST time-series modes by how well they achieve it and
estimates the number of transits required. The distribution name is
`vulcan-jwst-tool`, the import name is `jwst_tool`, and the console script is
`jwst-tool`.

Two goal types are supported. Detecting a molecule scores a conditional
matched-template signal-to-noise ratio: the chi-square distance between the
model spectrum and the same spectrum with one molecule's opacity removed,
with calibration nuisances profiled out. This is conditional on the assumed
atmosphere and upper-bounds any retrieval detection. Constraining a parameter
runs a Fisher-information forecast built from the spectrum's parameter
derivatives. Forecast uncertainties are local Cramer-Rao lower bounds under
the stated noise model, not posterior widths.

Derivatives are computed by the method suited to each question, and every
reported number is labeled with the method that produced it. The default is
certified central finite differences: independently re-converged solves,
evaluated at two step sizes that must agree before a row is reported.
Composition rows re-initialize the chemistry at the perturbed elemental
abundances, the standard VULCAN workflow, valid at any composition. A
forward-mode AD path (warm-started jvp) is available for every row from a
menu in the GUI. It is 1.7 to 4 times faster per row, requires
photochemistry, and refuses physically invalid corners rather than reporting
them. A post-run adjoint panel answers the high-dimensional questions
finite differences cannot afford: one reverse-mode solve gives the
sensitivity of a molecule's abundance to every reaction rate in the network
and to every layer's temperature, with a scope audit and numerical
certification attached. Condensation (S8 rainout) is offered for detection
goals only. The condensation pin freezes a step-history-dependent state, so
no differentiation method is valid through it, and the tool refuses the
combination under either method.

## Installation

The packages are published on TestPyPI. Install the tool and its sibling
dependencies (`vulcan-retrieval`, `vulcan-jax`) in one step:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]>=0.10.0'
```

The Pandeia noise backend runs in its own conda environment, deliberately not
a package dependency:

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

A site-packages install needs four environment variables so the tool can find
its data and write its caches: `VULCAN_PROJECT_ROOT` (the directory holding
the `vulcan-retrieval` data tree), `JWST_TOOL_DATA_DIR`,
`JWST_TOOL_OUTPUT_DIR`, and `JWST_TOOL_PANDEIA_PYTHON` (that conda
environment's python). Reference data (opacity caches, line lists, CDBS
pieces, Pandeia refdata) is not in the wheels. Run `jwst-tool data` for a
live report of every dataset with the exact remedy for anything missing.

Developers working from checkouts can instead install the three sibling
repositories editable, in dependency order, each with
`pip install --no-deps -e .` from its root (`VULCAN-JAX`, then
`vulcan-retrieval`, then this repository).

## Running

`jwst-tool` preflights the chemistry stack and the Pandeia backend, then
launches the Streamlit GUI (equivalently `streamlit run
src/jwst_tool/app.py`). A new parameter set takes about 2 minutes at the
default resolution. Fisher rows dominate when enabled: with finite
differences, about 6 to 8 minutes per composition parameter and 3 to 5
minutes per mixing or temperature parameter, or 1 to 2 minutes per row with
AD. The first adjoint-panel run on a machine also compiles the solver's
step-VJP, which can take hours on CPU and is then cached persistently. All
results are disk-cached, so repeat runs are instant, and every figure and
table has PNG or CSV downloads.

The default noise backend is Pandeia 2026.2. Noise forecasts are
Pandeia-extracted noise with an optional PandExo-style minimum floor. They
are not PandExo-identical sigmas: measured parity is documented in
`tests/parity/outputs/REPORT.md`, with the tool conservative by roughly 2 to
24 percent (NIR) and 33 to 56 percent (MIRI LRS). Mode rankings are more
trustworthy than absolute ppm.

## Scope and limits

The tool is a planner, not a retrieval. Detection scores are conditional on
one fixed atmosphere. The noise model omits time-correlated systematics
(visit-long trends, detrending covariance, stellar heterogeneity), so
achieved precision is usually poorer than forecast. T-P profiles are
isothermal or Guillot only, with constant eddy diffusion. Every planet in
the registry runs the same validated WASP-39b SNCHO machinery with the
system identity swapped in, and registry values are literature planning
defaults meant to be edited. Solves that cannot reach the convergence gate
raise loudly instead of returning an unconverged spectrum. The full
version-by-version history, measured validation numbers, and physics-audit
log live in `notes.md`; operational rules live in `CLAUDE.md`.

## Layout

`src/jwst_tool/` holds the package: the GUI (`app.py`), the forward-model
driver (`forward.py`), adjoint diagnostics (`adjoint_diag.py`), the Pandeia
worker, and the noise, detection, Fisher, binning, instrument-registry, and
data-availability modules. `tests/unit/` is the fast numpy-only suite
(`python -m pytest tests -q`); `tests/parity/` is the PandExo parity
harness. Generated caches live under `output/` and are gitignored.
