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
- `noise.noise_job`'s `worker_version` (**5** as of 2026-07-12) + the backend
  fingerprint (engine version, refdata VERSION hashes) bust the noise_cache;
  bump worker_version whenever `pandeia_worker.py` output changes. v4 =
  photsys Ks normalization + `r_native` export; v5 = engine/refdata
  release-match gate (worker refuses a mismatched pair) + `__provenance__`
  block (exact engine/refdata/python versions) in every result/cache file
  (noise caches stale again; model cache `_VERSION=5` NOT).
- Pandeia stays PINNED at engine 3.0 + pandeia_data-3.0rc3 (dated decision
  2026-07-12, rationale + upgrade checklist in README "Known limits" and
  instruments.py). Do not upgrade casually; on upgrade re-check
  `n_pix_degenerate_dropped` and re-baseline sigma/ngroup on one star.
- Star normalization is band-integrated **2MASS Ks vegamag** (synphot
  `2mass,ks` + local CALSPEC Vega in `data/cdbs/calspec/`), NOT the retired
  monochromatic at_lambda/666.7 Jy shortcut (mis-scaled 0.4–3.1% by Teff). The
  worker points `synphot.conf.vega_file` at the local copy so it runs offline;
  preflight checks both files.
- ONE measurement operator (`binning.py`, count-space/flux weights) bins noise,
  model, and Jacobians — never reintroduce separate binning paths (that was the
  2026-07-11 external audit's headline finding; history in `notes.md` v6).
  Degenerate-wavelength pandeia pixels (G395H red-edge pileup) are excluded there.
- **Detector-segment offsets**: `binning.segment_ids` splits the pixel grid at
  wavelength gaps (NRS1|NRS2). One depth-offset nuisance per segment is
  profiled in BOTH `detect.detection_significance` and `fisher.mode_forecast`/
  `combined_forecast`. Never drop the per-segment offset for the two-detector
  gratings — a single mode-wide offset lets an NRS1/NRS2 step masquerade as
  signal.
- **Native-R LSF**: `binning.smooth_to_native_r` blurs the model to the
  worker's `r_native` before binning; it auto-no-ops when the kernel is
  unresolved by the model grid (high-R gratings), so it only bites on MIRI
  LRS / PRISM. `r_native` comes from the refdata dispersion files.
- **Noise inflation**: per-mode `noise_infl` (instruments.MODES, editable in
  GUI) multiplies the Pandeia σ — literature-calibrated, proportional (averages
  down with N). Passed through `detect.evaluate_mode(noise_inflation=...)`.
- **σ_detect labeling**: it is a *conditional matched-template S/N*, never a
  retrieval detection. `sigma_detect_proj` additionally profiles T-P + lnR0
  Jacobian directions. Keep the GUI/README wording honest.
- `noise.pixel_depth_variance` RAISES on a sub-integration-cycle window or
  n_transits < 1 (retired the silent `max(1, ...)`).
- Fisher inversion must stay rank-aware (`fisher._marg_sigmas`: unconditional
  eigh + relative threshold; degenerate directions read `inf`) — no
  `np.linalg.inv` on Fisher matrices.
- **Noise scenarios** (2026-07-12): `noise.SCENARIOS` re-allocates the floor
  budget between white and ln-λ-smooth (SE kernel) parts — `noise.build_cov`
  returns None for "random" (exact diagonal fast path) or a PD covariance.
  INVARIANT: diag(C) = var_phot + floor² in every scenario (scenarios change
  correlation, never totals — tested). detect/fisher/transits-to-target all
  consume the stored `cov`/`slope_rows`; "conservative" adds per-segment
  slope nuisances. Kernel presets are stated assumptions, not measurements.
- Suite: `python -m pytest tests -q` (numpy-only, fast, no pandeia/JAX
  needed). One env-gated slow test (`JWST_TOOL_RUN_SLOW=1`) FD-closes a
  Jacobian row with 3 real forward runs — Isaac schedules it, never run it
  unprompted.
- **σ_detect terminology is settled** (2026-07-12 sweep): "conditional
  template S/N" everywhere user-facing (chart header/axis, pyproject
  description, __init__, README) — never reintroduce bare "detection
  significance" for σ_detect.
- The forward model imports must keep the order config -> vulcan_chem -> exojax
  (guard-enforced in retrieval_framework.forward.vulcan_chem).
- Heavy HPC/retrieval operational rules live in the sibling vulcan-retrieval repo's
  CLAUDE.md; this tool runs locally only.
- Historical version log: `notes.md`. Single README per repo (current usage only).
