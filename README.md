# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST transmission-spectroscopy observations with a
live forward model: VULCAN-JAX photochemistry feeds ExoJAX radiative
transfer, with instrument noise from the STScI Pandeia engine. Given a
planet and a science goal, it ranks JWST time-series modes and estimates
the transits required. Import name `jwst_tool`, console script `jwst-tool`.

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

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed
atmosphere. The Fisher forecast is linear and local, so real posteriors
can only be wider. The noise model omits time-correlated systematics and
is conservative against PandExo by roughly 2 to 24 percent in the near
infrared and 33 to 56 percent for MIRI LRS.

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
