# CLAUDE.md — vulcan-jwst-tool operational notes

- **Fail fast and loud (standing rule, same as the sibling repos):** no
  behavior-changing fallback paths, ever. Missing backends/data/refdata
  raise with the offending path and the remedy; a mismatched engine/refdata
  pair is refused, not attempted; a check that cannot run must SAY so
  (RuntimeWarning / explicit "SKIPPED"), never pass silently; unknown
  scenario/parameter values raise instead of defaulting. When adding a
  feature, prefer a loud error over a degraded result.
- Dist `vulcan-jwst-tool`, import `jwst_tool`, console script `jwst-tool`. Depends on
  the sibling `vulcan-retrieval` dist for the forward-model engine
  (`retrieval_framework.forward`). Local install: `pip install --no-deps -e .`
  (--no-deps because vulcan-jax/vulcan-retrieval live on TestPyPI, not PyPI).
- Path roots (all resolved in `src/jwst_tool/instruments.py`, loud on failure):
  `data/` = INPUTS (minimal synphot cdbs; env `JWST_TOOL_DATA_DIR` overrides),
  `output/` = GENERATED model_cache/ + noise_cache/ (env `JWST_TOOL_OUTPUT_DIR`).
  A site-packages install must set both env vars.
- The Pandeia noise backend runs in its OWN conda env: `JWST_TOOL_PANDEIA_PYTHON`
  + `JWST_TOOL_PANDEIA_REFDATA` are machine-specific; `noise.run_pandeia` refuses
  loudly if missing. DEFAULT engine is 2026.2 (`pandeia_2026` env); the pinned 3.0
  (`picaso_base`) is the `legacy` backend only — see the "Backend DEFAULT" bullet
  below.
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
- **Backend DEFAULT is "current" = Pandeia 2026.2** (2026-07-13, was pinned
  3.0). `JWST_TOOL_BACKEND` selects `current` (default: engine 2026.2 +
  pandeia_data-2026.2-jwst, validated vs PandExo in tests/parity/) or `legacy`
  (pinned 3.0 + 3.0rc3, reproducibility only); the two default path sets live
  in `instruments._BACKENDS`, and the explicit JWST_TOOL_PANDEIA_* env vars
  override per-path. Switching backends self-invalidates caches (engine +
  refdata in every cache key). The current backend uses the split refdata
  layout (`JWST_TOOL_PANDEIA_PSF_DIR` for the separate PSF library; `psf_dir`
  in the job, preflighted). NIRCam mode rename ssgrism→lw_tsgrism is a
  2026-engine thing the parity harness patches; the registry keeps the 3.0
  name (build_default_calc maps it). On any backend change re-check
  `n_pix_degenerate_dropped` (the 3.0rc3 red-edge pileup is fixed in 2026.2 —
  the degenerate mask drops 0 there).
- **TSO instrument configs are pinned EXPLICITLY (2026-07-12 parity run)**:
  readout_pattern on every mode (NRSRAPID/NISRAPID/RAPID/FASTR1 — engine
  defaults are the WRONG, non-TSO patterns and drift between releases),
  PandExo's extraction strategy (apertures/sky annuli), and
  background="ecliptic" + background_level="medium" (BOTH keys or the
  engine fails). ngroup_max follows PandExo policy: NIRCam hard-capped at
  100, everything else saturation-limited (PANDEXO_UNBOUNDED_NGROUP).
  Leaving any of these implicit cost 8–20% in extracted flux and picked
  frame-averaged BOTS ramps. Never add a mode without all of them.
- **PandExo parity is MEASURED, not pending** (`tests/parity/`, organized
  `scripts/` + `outputs/` (REPORT.md + parity_summary.json committed, raw run
  JSON git-ignored) + `figs/`): config/grid/flux/ngroup/timing parity achieved
  on engine 2026.2 both sides; the residual sigma gap is the NOISE MODEL
  (pandeia full extracted noise vs PandExo's analytic fml ≈ photon-only):
  ours conservative by ~7–12% (NIR) / ~35–48% (MIRI LRS). NEVER label
  sigmas "PandExo-identical"; they are pandeia-extracted-noise forecasts.
  Figures show ONLY the 1:1 parity (config/timing + extracted flux); the
  noise-model difference is documented in REPORT.md, not plotted. Re-run:
  `tests/parity/scripts/run_parity.py` (5 env vars, loud) then `make_report.py`
  + `make_parity_plots.py`; PARITY_REUSE_PANDEXO=1 reuses the PandExo side
  when its job is unchanged.
- Star normalization is band-integrated **2MASS Ks vegamag** (synphot
  `2mass,ks` + local CALSPEC Vega in `data/cdbs/calspec/`), NOT the retired
  monochromatic at_lambda/666.7 Jy shortcut (mis-scaled 0.4–3.1% by Teff). The
  worker points `synphot.conf.vega_file` at the local copy so it runs offline;
  preflight checks both files.
- ONE measurement operator (`binning.py`, count-space/flux weights) bins noise,
  model, and Jacobians — never reintroduce separate binning paths (that was the
  2026-07-11 external audit's headline finding; history in `notes.md` v6).
  Degenerate-wavelength pandeia pixels (G395H red-edge pileup) are excluded there.
- **Exact cell-edge integration (2026-07-12 re-audit V2, item 2)**: model
  cell-averages go through `binning._pl_antideriv` (EXACT piecewise-quadratic
  antiderivative of the piecewise-linear model). NEVER `np.interp` a
  cumulative-trapezoid integral to cell edges — that is only O(h²) and was off
  by ~1–2 ppm for edges between model nodes. Applies to BOTH `bin_model` and
  `smooth_to_native_r`; pinned by machine-precision tests in `test_binning.py`.
- **PandExo group caps (re-audit V2, item 5)**: `instruments.PANDEXO_NGROUP_MAX`
  bounds each instrument's `ngroup_max` (NIRCam grism = 100, not 180); an
  import-time guard refuses any mode above its cap and the worker clamps its
  selected ramp via `pandeia_worker._clamp_ngroup`. Never raise a NIRCam mode
  above 100 without a matching PandExo/Pandeia change.
- **Strict grid validation (2026-07-12 re-audit, item 7)**: `binning` raises
  on non-finite wavelengths/weights/model values, non-ascending model grids,
  and bad bin edges — NEVER silently drops invalid samples. Median pixel
  spacing is computed over strictly POSITIVE gaps (`_positive_gap_median`);
  exact-duplicate wavelengths are degenerate pixels (zero spectral support),
  flagged by `degenerate_wl_mask` even on duplicate-majority grids. An
  operator left with zero usable pixels raises with a per-criterion exclusion
  breakdown; `detect.evaluate_mode` raises when saturation+degeneracy exclude
  every pixel. Regression tests: the "strict grid validation" block in
  `test_binning.py`.
- **Detector-segment offsets**: `binning.segment_ids` splits the pixel grid at
  wavelength gaps (NRS1|NRS2). One depth-offset nuisance per segment is
  profiled in BOTH `detect.detection_significance` and `fisher.mode_forecast`/
  `combined_forecast`. Never drop the per-segment offset for the two-detector
  gratings — a single mode-wide offset lets an NRS1/NRS2 step masquerade as
  signal. **INCLUDING segment 0** (2026-07-12 recheck P0-A): lnR0 is a
  physical derivative, NOT a constant, so it never stands in for the first
  segment's offset — `fisher._segment_offset_rows` returns n_seg rows and
  `mode_forecast(r)` must equal `combined_forecast([r])` (pinned). In
  `detect`, the constant row is profiled even for a SINGLE bin (one bin +
  free offset = zero shape information, score 0, never |s|/σ — recheck P2-D).
- **The LSF operator never gates on data** (recheck P1-C): when `r_native`
  exists, `smooth_to_native_r` is applied to the depth, the removed-molecule
  depth, AND every Jacobian row unconditionally — `lsf_applied` is display
  metadata only. A flat baseline is a fixed point of the blur while a narrow
  Jacobian feature is not; gating on `lsf_applied` left Jacobians unsmoothed
  by ~59 ppm on the reproducer (pinned in `test_detect.py`).
- **Public noise APIs fail fast** (recheck P2-E): `noise_inflation` must be
  finite >0 (it is squared, so −1 silently acted as +1), `build_cov`/
  `pixel_depth_variance`/`make_bins` validate shapes/finiteness/signs and
  raise. Never re-loosen these for convenience.
- **Native-R LSF**: `binning.smooth_to_native_r` blurs the model to the
  worker's `r_native` before binning; it auto-no-ops when the kernel is
  unresolved by the model grid (high-R gratings), so it only bites on MIRI
  LRS / PRISM. `r_native` comes from the refdata dispersion files.
- **LSF is the FLUX-WEIGHTED count ratio, not a flat depth blur** (2026-07-12
  maximal-audit item 2): the instrument forms `d_obs = 1 - L[F(1-d)]/L[F] =
  L[F d]/L[F]`, the stellar-flux-weighted LSF mean of the depth. `detect`
  passes the worker's per-pixel stellar flux (interpolated onto the model grid)
  as `smooth_to_native_r(..., weight=)` for the depth, removed-molecule, AND
  every Jacobian row (one weight → operator linear in d). NEVER revert to the
  flat `L[d]` blur — it mislocated depth by ~120 ppm near a structured stellar
  spectrum (F only pixel-resolved, so sub-pixel stellar lines are a documented
  limit). `weight=None` reproduces the old flat blur exactly; pinned in
  `test_binning.py` (ratio-reference + >10× closer than flat).
- **Minimum noise floor (2026-07-12 external audit)**: exact PandExo
  semantics, `noise.resolve_floor` + `sigma = max(sigma_random, floor)` on
  the FINAL bins. Three modes: none / constant ppm / wavelength-vs-ppm table
  (linear interp, constant edge extension; invalid tables raise). NEVER
  quadrature, NEVER sqrt(R/100)-rescaled (the retired R-anchor), NEVER
  averaged below by transits (`detect.sigma_at_transits` clamps at every N).
  `depth_error_bins`/`evaluate_mode` take `floor_spec`, not `floor_ppm`.
- **Noise sensitivity factor** (`noise_infl`): DEFAULT 1.0 for every mode --
  the Pandeia prediction as-is. Literature achieved-vs-predicted ratios live
  in `instruments.LITERATURE_NOISE_FACTORS` as reference points only; never
  reintroduce them as defaults or call them a calibration. Passed through
  `detect.evaluate_mode(noise_inflation=...)`, recorded in results.
- **Scale invariance (2026-07-12 audit, both CONFIRMED bugs)**: Fisher rank
  detection runs on the Jacobi-whitened (unit-diagonal) matrix
  (`fisher._marg_sigmas`); nuisance profiling normalizes the normal matrix
  to correlation form (`detect.detection_significance`). Never threshold raw
  eigenvalues of a mixed-unit matrix -- both regressions are pinned in
  `tests/test_floor_and_invariance.py` (24-decade rescaling sweeps). Null-space
  overlap uses the basis-invariant L2 projection norm over null eigenvectors
  (not a single eigenvector's max component); the nuisance score is pinned
  invariant under an arbitrary nonsingular remix of the rows, not just per-row
  rescaling (re-audit V2, item 3/5 tightening).
- **Backend label**: `instruments.BACKEND_STATUS` (backend-selected string)
  must stay on every user-facing surface (GUI captions, README). It now reads
  positively for the default current 2026.2 backend and as a LEGACY warning
  when `JWST_TOOL_BACKEND=legacy` -- a 3.0 run must never present as
  current-ETC output.
- **σ_detect labeling**: it is a *conditional matched-template S/N*, never a
  retrieval detection. `sigma_detect_proj` additionally profiles T-P + lnR0
  Jacobian directions. Keep the GUI/README wording honest.
- `noise.pixel_depth_variance` RAISES on a sub-integration-cycle window or
  n_transits < 1 (retired the silent `max(1, ...)`). It uses the OUT-of-transit
  flux/σ for BOTH in/out terms (symmetric shallow-transit approximation) — kept
  model-INDEPENDENT so one operator bins noise and model; this is CONSERVATIVE
  (excess ~**3d/4** for equal in/out baselines — NOT d/2; +0.76% at d=1%, +8.2%
  at 10%; d/2–d for unequal baselines). Exact separate in/out propagation
  is a remaining refinement, NOT a same-answer refactor (2026-07-12
  maximal-audit item 3, docstring has the numbers).
- Fisher inversion must stay rank-aware (`fisher._marg_sigmas`: unconditional
  eigh + relative threshold; degenerate directions read `inf`) — no
  `np.linalg.inv` on Fisher matrices.
- **Noise scenarios are EXPERIMENTAL** (2026-07-12 audit): `noise.SCENARIOS`
  re-allocates the floor EXCESS (max(0, floor² − var_phot)) between white
  and ln-λ-smooth (SE kernel) parts — `noise.build_cov` returns None for
  "random" or when the floor binds nowhere (exact diagonal fast path), else
  a PD covariance. INVARIANT: diag(C) = max(var_phot, floor²) = σ_final² in
  every scenario (tested with atol=0 — the original allclose was vacuous at
  ~1e-10 variances; always pass atol=0 when asserting on variances).
  Excluded from headline results: default scenario "random"; the GUI shows
  cross-scenario columns only when a correlated preset is selected, labeled
  experimental. detect/fisher/transits-to-target consume the stored
  `cov`/`slope_rows`; "conservative" adds per-segment slope nuisances.
  Kernel presets are stated assumptions, never calibrated covariances.
- **Parameter scope (2026-07-13)**: ONLY explicit isothermal / Guillot T-P
  and constant Kzz exist — the WASP-39 b GCM `baseline` T-P mode, `scale` Kzz
  mode, and `has_gcm_baseline` were REMOVED end-to-end (canonical_params
  raises on them; defaults are isothermal/const; every planet including W39b
  gets an isothermal structural baseline; no GCM profile is ever silently
  substituted). The slow `test_closure.py` FD test now drives `T_iso`
  (same single-scalar theta[3] design the retired `dT` had). GUI is organized
  into five clearly-labeled sections: 🪐 Planet & star, 🧪 VULCAN chemistry
  (T-P, composition, Kzz, photochem; condensation is a why-not note), 🌈
  ExoJAX RT (opacity/scattering/clouds/broadening), 🎯 Science goal (goal,
  fidelity, Fisher), 🔭 Instrument & noise (modes, transits, saturation,
  noise model). Verified with Streamlit `AppTest`
  (`session_state['intro_ack']=True` skips the how-it-works gate).
- **Condensation is REMOVED as an option (2026-07-14).** `use_condense` is no
  longer a parameter; `canonical_params` RAISES on a truthy `use_condense`
  (do not re-add a silent path), `CONDEN_CFG` is gone, and the GUI shows a
  read-only "why it's not offered" note instead of a checkbox. Why: a
  condensing VULCAN column reaches steady state only via a condensation
  window + whole-column fix-species pin that freezes the condensed reservoir
  (S8 / S8_l_s) at a step-sequence-dependent transient, which is not reliably
  differentiable (forward-mode jvp vs FD ~0.91) and switches discretely in T
  — unusable for this tool's Fisher forecast. The open-system smooth-rainout
  replacement that tried to fix this was measured B0C NO-GO and is preserved
  on the sibling repos' `research/smooth-rainout-fisher` branch (+ tag
  `smooth-rainout-b0c-no-go-2026-07-14`), NOT on main and NOT here. For
  aerosol opacity the Fisher-compatible route is a differentiable ExoJAX
  cloud (the power-law `cloud_on` deck is already wired; freeing it or adding
  a gray deck as a Fisher parameter is the recommended future addition —
  never re-introduce condensation as the answer). `forward._VERSION` bumped
  to 8 (use_condense dropped from the cache key). The engine's live-T(P)
  `vulcan_chem._prep` condensation rebuild still exists in the sibling
  retrieval repo for forward-model use; this tool simply does not expose it.
- Suite: `python -m pytest tests -q` (numpy-only, fast, no pandeia/JAX
  needed). One env-gated slow test (`JWST_TOOL_RUN_SLOW=1`) FD-closes a
  Jacobian row with 3 real forward runs — Isaac schedules it, never run it
  unprompted. `test_forward_params.py` pins the condensation-unsupported /
  T-P / Kzz gating (pure Python, no stack).
- **σ_detect terminology is settled** (2026-07-12 sweep): "conditional
  template S/N" everywhere user-facing (chart header/axis, pyproject
  description, __init__, README) — never reintroduce bare "detection
  significance" for σ_detect.
- The forward model imports must keep the order config -> vulcan_chem -> exojax
  (guard-enforced in retrieval_framework.forward.vulcan_chem).
- Heavy HPC/retrieval operational rules live in the sibling vulcan-retrieval repo's
  CLAUDE.md; this tool runs locally only.
- Historical version log: `notes.md`. Single README per repo (current usage only).
