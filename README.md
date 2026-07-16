# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST transmission-spectroscopy observations with a
live forward model instead of an assumed input spectrum. Steady-state
VULCAN-JAX photochemistry feeds ExoJAX radiative transfer. Instrument noise
comes from the STScI Pandeia engine. Given a planet and a science goal, the
tool ranks JWST time-series modes by how well they achieve it and estimates
the number of transits required. The distribution name is
`vulcan-jwst-tool`, the import name is `jwst_tool`, and the console script
is `jwst-tool`.

## Installation

The packages are published on TestPyPI. Install the tool and its sibling
dependencies (`vulcan-retrieval`, `vulcan-jax`) in one step:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]>=0.10.0'
```

The Pandeia noise backend runs in its own conda environment. It is
deliberately not a package dependency:

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

A site-packages install needs four environment variables so the tool can
find its data and write its caches: `VULCAN_PROJECT_ROOT` (the directory
holding the `vulcan-retrieval` data tree), `JWST_TOOL_DATA_DIR`,
`JWST_TOOL_OUTPUT_DIR`, and `JWST_TOOL_PANDEIA_PYTHON` (that conda
environment's python). Reference data such as opacity caches and Pandeia
refdata is not in the wheels. Run `jwst-tool data` for a live report of
every dataset with the exact remedy for anything missing.

## Running

`jwst-tool` preflights the chemistry stack and the Pandeia backend and then
launches the Streamlit GUI. A new parameter set takes about 2 minutes at
the default resolution, and each freed Fisher parameter adds minutes (the
app shows a runtime estimate before you press Run). Results are
disk-cached, so repeat runs are instant, and every figure and table has a
PNG or CSV download.

## Scope and limits

The tool is a planner, not a retrieval. Detection scores are conditional
on one fixed atmosphere. The Fisher forecast is linear and local: it
expands the spectrum to first order around the assumed atmosphere, so the
quoted bounds hold only near that point and real posteriors can only be
wider. The noise model is Pandeia-extracted noise with an optional
PandExo-style minimum floor. It omits time-correlated systematics, so
achieved precision is usually poorer than forecast. Measured parity with
PandExo is documented in `tests/parity/outputs/REPORT.md`. The tool is
conservative by roughly 2 to 24 percent in the near infrared and 33 to 56
percent for MIRI LRS, so mode rankings are more trustworthy than absolute
ppm. T-P profiles are isothermal or Guillot only, with constant eddy
diffusion. Every planet in the registry runs the same validated WASP-39b
SNCHO machinery with the system identity swapped in. Registry values are
literature planning defaults meant to be edited. Solves that cannot reach
the convergence gate raise loudly instead of returning an unconverged
spectrum. Version history and measured validation numbers live in
`notes.md`. Operational rules live in `CLAUDE.md`.

## Layout

```
vulcan-jwst-tool/
├── src/jwst_tool/
│   ├── app.py             Streamlit GUI
│   ├── forward.py         forward-model driver: chemistry, RT, Jacobians
│   ├── adjoint_diag.py    reverse-mode adjoint diagnostics
│   ├── fisher.py          Fisher forecasts
│   ├── detect.py          detection statistics
│   ├── noise.py           ETC noise model and floors
│   ├── binning.py         the single measurement operator
│   ├── pandeia_worker.py  Pandeia subprocess (runs in its own conda env)
│   ├── instruments.py     instrument-mode registry and path roots
│   ├── datacheck.py       data-availability detection
│   ├── planets.py         planet registry
│   └── cli.py             console entry point
├── tests/
│   ├── unit/              fast numpy-only suite: python -m pytest tests -q
│   └── parity/            PandExo parity harness and committed report
├── data/                  input reference data
└── output/                generated caches (gitignored)
```

## Science goals

The tool supports two goals. The detection goal asks how strongly a single
molecule imprints on the spectrum. It computes the chi-square distance
between the model and the same model with that molecule's opacity removed,
after profiling out calibration nuisances. The score assumes the rest of
the atmosphere is known and it upper-bounds any real retrieval detection.
The constraint goal asks how tightly an observation could measure a
parameter such as metallicity or C/O. It builds a Fisher-information
forecast from the spectrum's parameter derivatives. The result is a local
Cramer-Rao lower bound under the stated noise model. It is not a posterior
width.

## Derivatives

Derivatives are computed by the method suited to each question. Every
reported number is labeled with the method that produced it. The default
is central finite differences of independently re-converged solves. Each
row is evaluated at two step sizes that must agree before it is reported.
Composition rows re-initialize the chemistry at the perturbed elemental
abundances. That is the standard VULCAN workflow and it works at any
composition. A forward-mode AD path is available for every row from a menu
in the GUI. It is 1.7 to 4 times faster per row. It requires
photochemistry and refuses physically invalid corners instead of reporting
them. A post-run adjoint panel answers the high-dimensional questions that
finite differences cannot afford. One reverse-mode solve gives the
sensitivity of a molecule's abundance to every reaction rate in the
network and to every layer's temperature. A scope audit runs first and the
numbers that qualify each result are reported with it. The first
adjoint-panel run on a machine compiles the solver's step-VJP. That can
take hours on CPU and is then cached, so later runs take minutes.
Condensation (S8 rainout) is offered for detection goals only. The
condensation pin freezes a state that depends on the solver's step
history. No differentiation method is valid through it and the tool
refuses the combination.
