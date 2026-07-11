"""JWST instrument-selection tool: VULCAN-JAX chemistry -> ExoJax RT -> Pandeia noise.

A PandExo-style planning GUI: pick a science goal (detect molecule X on a
WASP-39b-like planet), and the tool runs the live VULCAN-JAX + ExoJax forward
model locally, simulates each JWST instrument mode's transit-depth precision
with the real STScI Pandeia ETC engine, and ranks the modes by detection
significance.

Lives in vulcan-jwst-tool/src/jwst_tool/ (dist: vulcan-jwst-tool); the shared
forward-model modules come from the sibling vulcan-retrieval package
(retrieval_framework.forward.*).

Entry point: the console script ``jwst-tool``, or
``streamlit run vulcan-jwst-tool/src/jwst_tool/app.py`` from the repo root.
"""
from jwst_tool._version import __version__  # noqa: F401
