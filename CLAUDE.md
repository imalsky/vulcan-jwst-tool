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
- `noise.noise_job`'s `worker_version` (3 as of the 2026-07-11 v6 estimator pass)
  + the backend fingerprint (engine version, refdata VERSION hashes) bust the
  noise_cache; bump worker_version whenever `pandeia_worker.py` output changes.
- ONE measurement operator (`binning.py`, count-space/flux weights) bins noise,
  model, and Jacobians — never reintroduce separate binning paths (that was the
  2026-07-11 external audit's headline finding; history in `notes.md` v6).
  Degenerate-wavelength pandeia pixels (G395H red-edge pileup) are excluded there.
- Fisher inversion must stay rank-aware (`fisher._marg_sigmas`: unconditional
  eigh + relative threshold; degenerate directions read `inf`) — no
  `np.linalg.inv` on Fisher matrices.
- Suite: `python -m pytest tests -q` (numpy-only, fast, no pandeia/JAX needed).
- The forward model imports must keep the order config -> vulcan_chem -> exojax
  (guard-enforced in retrieval_framework.forward.vulcan_chem).
- Heavy HPC/retrieval operational rules live in the sibling vulcan-retrieval repo's
  CLAUDE.md; this tool runs locally only.
- Historical version log: `notes.md`. Single README per repo (current usage only).
