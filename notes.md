# jwst_tool: version history and physics-audit log

Historical log for the instrument selector, moved out of `README.md` (which
keeps only current reference content). The quoted fragments below are verbatim
from the pre-0.6 README, where they were written into the reference text as
each version shipped.

## v1: initial tool

The README's "Known limits" section was originally labeled "(v1)": the model
band (1.0 um H2-H2 CIA edge, MIRI LRS cut at 12 um), the literature-default
planet registry, and the no-partial-saturation limit date from the first
version.

## v4: physics audit (Rayleigh + exposed physics knobs)

Verbatim: "H2/He Rayleigh scattering (ON by default as of v4 — earlier versions
omitted it, biasing the <1.5 µm slope)".

## v5: 2026-07-11 audit response (noise, statistics, elemental abundance map)

All cached spectra were invalidated (`_VERSION = 5`). Verbatim fragments:

- "*Detect a molecule*: `σ = √Δχ²` of (full − without-X) on each mode's bins
  with a free constant depth offset profiled out (v5 — removing a molecule's
  flat continuum no longer counts as signal; matches the Fisher offset
  treatment)."
- "Floors are quoted **per R=100 bin** and anchored there: finer bins scale the
  per-bin floor by √(R/100), so the bin slider cannot manufacture floor-limited
  significance (v5). The photon and floor terms are returned separately, so
  multi-transit predictions average down only the photon term."
- "Saturated modes are excluded from BOTH the per-mode ranking and the combined
  row (v5 — previously the combined row silently included them)."
- "\"Transits → target\" is floor-aware as of v5: the photon term scales 1/N,
  the R-anchored floor is fixed, and the solver reports **never** when the
  target exceeds the floor-limited ceiling (the old 1/√N scaling was optimistic
  exactly where the floor dominated)."
- "Abundance knobs are exact **elemental** directions as of v5
  (`abundance_mode="elemental"`): lnZ and dlnCO move conserved column
  elemental ratios exactly (H/He fixed), the column sums to P/(k_B T) per
  layer, and the chemistry's conserved atom totals match the requested gas.
  v4 and earlier used species-mask scalings with documented elemental leakage
  — all cached spectra were invalidated (`_VERSION = 5`)."
- "As of v5 the on-graph T-P drives the FULL chemistry structure per evaluation
  (rates, n₀, hydrostatic geometry via the runner's in-loop refresh, Dzz/vm) —
  the one remaining baseline-T bake is the photolysis cross-section
  T-interpolation (upstream host-side step, second-order)."

The v5 statistics/noise changes are the jwst_tool arm of the project-wide
2026-07-11 scientific-correctness pass; the full audit summary is preserved in
the sibling vulcan-retrieval repo's `notes.md` and the operational
consequences in that repo's `CLAUDE.md`.

## v6: 2026-07-11 estimator-consistency pass (revised external audit, P0 items)

Response to the revised end-to-end audit (same day as v5). Model spectra were
NOT invalidated (`forward._VERSION` stays 5 — the physics is untouched); the
NOISE cache was (`worker_version = 3` + backend fingerprint in the job key).

- **One measurement operator (audit P0.1, the headline).** New `binning.py`:
  count-space (flux-weighted) binning shared by noise, model, removed-molecule
  model, and every Jacobian row. Previously noise.py combined pixels
  inverse-variance while detect.py binned the model with trapezoid wavelength
  weights — two different estimators, so the forecast model point was not the
  expectation of the statistic whose variance was quoted. Measured impact on
  the HD 189733 b cache: sigmas essentially unchanged (photon-dominated ⇒
  count ≈ ivar weights), binned-model shifts of 2–15 ppm rms per mode (up to
  ~47 ppm in coarse-pixel MIRI bins). The model is cell-averaged over pixel
  wavelength cells (piecewise-linear integral, gap-capped) — no extra LSF blur
  on top of Pandeia's extraction.
- **Degenerate-wavelength pixel exclusion.** pandeia_data-3.0rc3's G395H
  extracted grid piles ~700 samples within <1e-4 µm at the NRS2 red edge
  (spacing down to 3.7e-6 µm vs 6.6e-4 median). Counting them as independent
  samples made the 5.10 µm bin's sigma ~2.7x too small and skewed its model
  value by ~115 ppm. `binning.degenerate_wl_mask` (cut: local spacing <
  0.02x median) drops them loudly (`n_pix_degenerate_dropped`, GUI note).
- **Rank-aware Fisher (P0.2).** `_marg_sigmas` now ALWAYS eigendecomposes
  (never `np.linalg.inv`, which returns misleading finite numbers for
  ill-conditioned F without raising), applies a relative eigenvalue threshold
  (1e-10), flags parameters loaded on null directions as unconstrained, and
  exposes rank / condition number / eigenvalues (GUI caption on the combined
  forecast). Regression test: a Jacobian row duplicated to 1e-8 must read
  unconstrained.
- **Saturation verified per channel (P0.4).** The worker now cross-checks the
  probe-linear group estimate against pandeia's `sat_ngroups`, re-runs and
  steps down until the MEASURED fraction_saturation respects the limit, and
  exports per-pixel `n_partial_saturated`/`n_full_saturated` curves; fully
  saturated pixels are excluded from the estimator, partial ones counted per
  bin. (Verified: G395H sat_ngroups=5 → ngroup 4 at 0.786 full-well; MIRI 52
  → 41 at 0.784.)
- **Reproducible cache key (P0.3, partial).** The noise cache key now hashes
  the pandeia engine version (queried from the backend python) + refdata
  VERSION-file checksums; the worker records `engine_version` per mode and
  preflights the refdata/CDBS paths (fails with the offending PATH, not a
  synphot traceback). Upgrading to the current Pandeia (2026.2 / JWST 5.1 as
  of 2026-07-11) remains open — it needs the matching refdata download and a
  regression pass; noted in README "Known limits".
- **Honest labels (P0.5).** GUI: ETC noise labeled "random-noise lower bound
  + floor scenario", σ_detect labeled "fixed-model distinguishability (upper
  bound)", Fisher already carried the Cramér–Rao best-case explainer, now
  with rank/condition reporting. Checked against STScI docs: Pandeia's
  extracted noise DOES include the HgCdTe quantum-yield/Fano excess variance
  (engine signal.py: Ve = qy(qy+fano)Rp), so the "pipeline ERR underestimate
  below 2 µm" caution applies to real-data reductions, not to these
  forecasts.
- **First test suite** (`tests/`): 13 numpy-only operator + Fisher tests
  (constant-depth conservation, MC estimator mean/variance, same-estimator
  identity, linearity, gap containment, nested rebinning, degenerate-pixel
  flagging, Fisher degeneracy/prior/analytic).
- **Failure hygiene** (user request): one bad mode no longer kills the whole
  evaluation (reported per mode with its reason), missing target molecule
  raises with the fix, worker preflights paths, and the mode-details
  "transits → target" column is all-string (mixed int/str made streamlit's
  Arrow serialization spew tracebacks on every render).

Deferred (audit P1/P2, documented in README "Known limits"): full wavelength
covariance input, empirically calibrated per-mode systematics scenarios,
time-domain injection-recovery tier, box-transit → light-curve-fit noise
propagation, cloud/stellar-contamination marginalization in detectability.

## v7: 2026-07-11 audit-response pass (P1 fixes + literature calibration)

Second external audit (collaborator report + independent literature/source
research: PandExo `jwst.py`, pandeia 2026.2 engine source, COMPASS/Gordon+2025,
Holmberg & Madhusudhan 2023, Eureka!/POSEIDON source, Jakobsen QY technote).
Model physics untouched (`forward._VERSION` stays 5); noise cache busted
(`worker_version = 4`). Several audit P0/P1 items were verified already-correct
and NOT changed; the rest were fixed.

**Fixed:**
- **Detector-segment offsets (audit: "one offset per mode is too coarse").**
  `binning.segment_ids` splits the extracted grid at wavelength gaps
  (SEGMENT_GAP_FACTOR × median spacing) → NRS1|NRS2 for G395H/G235H, one
  segment for everything else. One depth-offset nuisance per segment is now
  profiled in the detection score (`detect.detection_significance` rank-aware
  offset+step projection) AND every Fisher forecast (`fisher.mode_forecast` /
  `combined_forecast`). Justified by universal practice: Moran+2023 (78 ppm
  NRS2 step on GJ 486 b), Madhusudhan+2023 (NRS1/NRS2 offsets flipped a K2-18 b
  DMS claim from 2.4σ to insignificant).
- **Native-R LSF convolution (audit: "no LSF, narrow features mis-assigned").**
  `binning.smooth_to_native_r` blurs the model to the worker-exported
  `r_native(λ)` (from the refdata dispersion files) before binning, on a
  flux-conserving uniform ln-λ grid finer than the kernel. Auto-no-ops when the
  kernel is unresolved by the model grid — so it is a no-op for the high-R
  gratings (G395H native R≈2700 ≫ the ~R1000 forward model, sub-ppm edge
  regime, matching the research verdict) and only bites on MIRI LRS (native
  R≈40–160) and PRISM (R≈30–300), exactly where R_bin≈100 approaches native R.
- **Literature-calibrated noise inflation (audit: "achieved ≠ predicted").**
  Per-mode `noise_infl` multiplies the Pandeia σ: G395H/G235H ×1.10
  (COMPASS/Gordon+2025: NRS1 1.05×, NRS2 1.12× vs PandExo), SOSS ×1.20
  (Espinoza+2023), MIRI LRS ×1.15 (Bouwman+2023), PRISM ×1.0 (photon-limited,
  Rustamkulov+2023), NIRCam ×1.05. Proportional noise, so it averages down with
  transits (unlike the floor). Editable in the GUI; threaded through
  `detect.evaluate_mode(noise_inflation=)`.
- **σ_detect relabeled + nuisance-projected variant (audit: "template, not
  detection").** σ_detect is now called a *conditional matched-template S/N at
  the specified atmospheric state* in code, GUI, and README. New
  `sigma_detect_proj` additionally profiles the T-P + lnR0 Jacobian directions
  (chemistry/clouds fixed — still conditional), shown as a second column for
  narrow margins.
- **Band-integrated Ks normalization (audit: "monochromatic Ks is
  color-biased").** The worker switched from at_lambda(2.159 µm, 666.7 Jy) to
  synphot photsys `2mass,ks` vegamag with a LOCAL CALSPEC Vega
  (`data/cdbs/calspec/alpha_lyr_stis_011.fits`) + the 2MASS Ks bandpass, so it
  still runs offline (preflighted). MEASURED old/new flux ratio over 3–5 µm:
  +0.9% (3200 K), +0.4% (4500 K), +1.4% (5500 K), +3.1% (6500 K) — worst for
  hot stars (Brackett-γ wing), and it fed saturation/ngroup choices at full
  amplitude.
- **Loud sub-cycle window (audit: "max(1,…) silently invents an integration").**
  `noise.pixel_depth_variance` RAISES when the in/out window is shorter than
  one integration cycle, or n_transits < 1 — no silent single integration.
- **Response-weighted effective wavelength.** `evaluate_mode` returns `wl_eff`
  (count-weighted) alongside bin edges; the spectrum plot uses it (matters at
  detector gaps / steep throughput).

**Verified already-correct, NOT changed (avoids churn the audit expected):**
- Umbrella-repo "old estimator" P0 is MOOT — `vulcan_exojax_run` was deleted in
  the sibling-repo restructure; the standalone repo is the only copy.
- Quantum-yield excess variance IS in Pandeia's `extracted_noise`
  (engine `signal.py`: Ve = qy·(qy+fano)·Rp = (1+3p)·Rp, matches Jakobsen
  Eq. 12/15; refdata conversion curves confirmed: NIRSpec→1.88, NIRISS→1.84).
  No parity fix needed for these forecasts; PandExo instead *divides* flux by
  QY, a different (mixed-units) convention.
- Flux-weighted count-space binning is exactly what Eureka!/ExoTiC-JEDI measure
  (verified in Eureka! `s4_genLC.py` — unweighted count mean); the deterministic
  star-only weights dodge the bias of data-estimated inverse-variance weights.
- Temperature IS freeable in Fisher (`fisher_params` ⊃ dT / T_iso / 4-param
  Guillot). Clouds + stellar contamination remain the genuinely fixed
  directions (documented).

**8 new tests** (21 → 24, still numpy-only): segment splitting, bin-segment
majority, native-R flux conservation + high-R no-op, constant/step offset
profiling, real-feature survival, sub-cycle raise, inflation scaling, Fisher
segment-offset absorption, combined per-segment offset counting.

**Still deferred (audit P1/P2, README "Known limits"):** full wavelength
covariance matrix (GP/empirical — the modern best practice per Holmberg &
Madhusudhan 2023, Rotman+2025; the √(R/100) floor here is a conservative
white-noise envelope, not measured covariance), empirical per-mode systematics
scenarios, time-domain injection-recovery, 2-D Pandeia response matrix, cloud
marginalization in the Fisher forecast. Pandeia 2026.2/JWST 5.1 upgrade
(current 3.0rc3) still deferred — needs matching refdata + regression pass, and
ETC 6.0/Cycle 6 lands ~2026-07-16 anyway.

## v8: 2026-07-12 triad-audit response (provenance, scenarios, closure, exposure)

**Backend identity (worker_version 4 → 5, noise caches stale):**
- `pandeia_worker` now ENFORCES the STScI same-release rule before any
  calculation (`_check_backend_match`: engine `__version__` vs the refdata
  VERSION / VERSION_PSF / dir-name version; one clear error instead of seven
  identical per-mode tracebacks) and writes a top-level `__provenance__`
  block (engine/refdata/python/numpy versions, worker_version) into every
  result → every cache file. GUI shows it as a caption. Dated decision
  recorded (instruments.py + README): STAY on engine 3.0 + pandeia_data-3.0rc3;
  upgrade checklist documented (re-check `n_pix_degenerate_dropped`,
  re-baseline sigma/ngroup on one star).

**Correlated-noise scenario tier (the audit's checklist-B gap):**
- `noise.SCENARIOS` (random / moderate / conservative) re-allocates the
  R-anchored floor budget between white and ln-λ-smooth (SE kernel) parts;
  `noise.build_cov` returns None (random: exact diagonal fast path) or a PD
  covariance. INVARIANT: diag(C) = var_phot + floor² in every scenario —
  ranking changes are correlation structure, never bigger error bars.
- `detect.detection_significance(..., cov=)` full-covariance metric;
  per-segment SLOPE nuisance rows (`_slope_rows`, conservative scenario);
  `evaluate_mode(..., scenario=)` stores `cov`/`slope_rows`/`scenario` and
  reports `sigma_detect_by_scenario` (all three shown in the GUI table).
- Fisher: `F = J C⁻¹ Jᵀ` per mode (block-diagonal across modes in
  `combined_forecast`), slope rows marginalized like offsets;
  transits-to-target rebuilds C at every N (photon diagonal scales 1/N,
  floor kernel fixed) in both detect and fisher solvers.

**Closure tests + terminology (checklist D/E residuals):**
- Poisson synthetic-COUNT closure of the binned estimator (mean + variance vs
  the analytic noise model), matched-filter amplitude-variance closure under
  the scenario covariance (diagonal metric pinned as miscalibrated on
  correlated noise), and an opt-in slow FD test (`JWST_TOOL_RUN_SLOW=1`,
  3 real forward runs) closing a cached autodiff Jacobian row against central
  finite differences.
- σ_detect terminology sweep finished: chart subheader/x-axis, pyproject
  description, `__init__` docstring, README all say "conditional template
  S/N" now.

**Parameter exposure + README:**
- The dead broadening knob (audit's one confirmed bug) is a real canonical
  cache-keyed parameter: `broadening="air"|"h2he"` validated in
  `canonical_params`, flows to the RT profile, GUI selectbox in Physics.
  (Adding the key re-keys all model caches; old entries are orphaned, not
  wrong.)
- README lead rewritten differentiability/Fisher-first (intuitive-then-formal:
  derivative currency → template S/N tier → Fisher tier with CRLB caveats),
  PandExo-style GUI demoted to the closing sentence.

**28 new tests** (24 → 52: 13 worker backend-identity, 12 scenario/covariance
+ slopes, 2 closure, 1 opt-in FD): suite still numpy-only by default.
