# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST transmission-spectroscopy observations with a
live forward model: VULCAN-JAX photochemistry feeds ExoJAX radiative
transfer, with instrument noise from the STScI Pandeia engine. Given a
planet and a science goal, it ranks JWST time-series modes and estimates
the transits required. Import name `jwst_tool`, console script `jwst-tool`.

## Installation

From TestPyPI, with the sibling packages resolved automatically:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]>=0.10.1'
```

The Pandeia backend lives in its own conda environment:

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

Set `VULCAN_PROJECT_ROOT`, `JWST_TOOL_DATA_DIR`, `JWST_TOOL_OUTPUT_DIR`,
and `JWST_TOOL_PANDEIA_PYTHON`. Reference data is not in the wheels. Run
`jwst-tool data` for a live report with the remedy for anything missing.

## Running

`jwst-tool` preflights the stack and launches the Streamlit GUI. A new
parameter set takes about 2 minutes; the app shows a runtime estimate
before each run. Results are disk-cached and downloadable as PNG or CSV.

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed
atmosphere. The Fisher forecast is linear and local, so real posteriors
can only be wider. The noise model omits time-correlated systematics and
is conservative against PandExo by roughly 2 to 24 percent in the near
infrared and 33 to 56 percent for MIRI LRS
(`tests/parity/outputs/REPORT.md`), so mode rankings are more trustworthy
than absolute ppm. Planet registry values are literature planning defaults
meant to be edited. Solves that miss the convergence gate raise instead of
returning a bad spectrum. History and validation numbers are in
`notes.md`; operational rules are in `CLAUDE.md`.

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

## Derivatives

Every reported number is labeled with the method that produced it. The
default is central finite differences of independently re-converged
solves, checked at two step sizes, valid at any composition. A
forward-mode AD path is available for every row and is 1.7 to 4 times
faster, but requires photochemistry and refuses invalid corners. A
post-run adjoint panel gives the sensitivity of a molecule's abundance to
every reaction rate and every layer's temperature from one reverse-mode
solve, with a scope audit run first. Its first run per machine compiles
the solver's step-VJP, which can take hours and is then cached.
Condensation (S8 rainout) is detection-only. Its pin freezes a
step-history-dependent state, so no derivative is valid through it.
