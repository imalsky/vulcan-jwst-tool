"""JWST instrument-selection tool: VULCAN-JAX chemistry -> ExoJax RT -> Pandeia noise.

A live-forward-model instrument selector (PandExo-style GUI): pick a science
goal (detect molecule X on a WASP-39b-like planet), and the tool runs the
live VULCAN-JAX + ExoJax forward model locally, builds parameter Jacobians as
certified central finite differences of independently converged solves (each
row ships its own step-size consistency bound -- see forward.py's FD block),
simulates each JWST instrument mode's transit-depth precision with the real
STScI Pandeia ETC engine, and ranks the modes by conditional
matched-template S/N (not a retrieval detection significance -- see
detect.py) plus rank-aware Fisher forecasts.

Lives in src/jwst_tool/ (dist: vulcan-jwst-tool); the shared
forward-model modules come from the sibling vulcan-retrieval package
(retrieval_framework.forward.*).

Entry point: the console script ``jwst-tool``, or
``streamlit run src/jwst_tool/app.py`` from the repo root.
"""
from jwst_tool._version import __version__  # noqa: F401
