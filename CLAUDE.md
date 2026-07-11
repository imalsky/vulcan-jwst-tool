# CLAUDE.md — vulcan-jwst-tool operational notes

- Dist `vulcan-jwst-tool`, import `jwst_tool`, console script `jwst-tool`. Depends on
  the sibling `vulcan-retrieval` dist for the forward-model engine
  (`retrieval_framework.forward`). Local install: `pip install --no-deps -e .`
  (--no-deps because vulcan-jax/vulcan-retrieval live on TestPyPI, not PyPI).
- Path roots (all resolved in `src/jwst_tool/instruments.py`, loud on failure):
  `data/` = INPUTS (minimal synphot cdbs; env `JWST_TOOL_DATA_DIR` overrides),
  `output/` = GENERATED model_cache/ + noise_cache/ (env `JWST_TOOL_OUTPUT_DIR`).
  A site-packages install must set both env vars.
- The Pandeia noise backend runs in its OWN conda env (pandeia.engine 3.0 via
  `picaso_base`): `JWST_TOOL_PANDEIA_PYTHON` + `JWST_TOOL_PANDEIA_REFDATA` are
  machine-specific; `noise.run_pandeia` refuses loudly if missing.
- `data/cdbs/grid/phoenix` is a symlink to an external picaso tree; dangling on other
  machines (the pandeia preflight fails loudly). Do not replace it with a copy.
- `forward._VERSION` is the cache-buster for model_cache spectra (v5 = 2026-07-11
  exact-elemental map); bump it whenever the physics changes.
- The forward model imports must keep the order config -> vulcan_chem -> exojax
  (guard-enforced in retrieval_framework.forward.vulcan_chem).
- Heavy HPC/retrieval operational rules live in the sibling vulcan-retrieval repo's
  CLAUDE.md; this tool runs locally only.
- Historical version log: `notes.md`. Single README per repo (current usage only).
