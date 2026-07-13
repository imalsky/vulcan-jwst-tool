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

## v9: 2026-07-12 strict end-to-end audit response (floor semantics, scale invariance)

Response to the same-day "final strict end-to-end scientific audit". Five
confirmed items fixed; two deferred with explicit labeling.

**Confirmed and fixed:**
- **PandExo minimum-floor semantics** (audit item 2): the quadrature +
  sqrt(R/100)-anchored floor is GONE. `noise.resolve_floor` implements
  none / constant ppm / wavelength-vs-ppm table with linear interpolation and
  constant edge extension; applied as `sigma_final = max(sigma_random, floor)`
  on the final bins. Multi-transit paths clamp at every N
  (`detect.sigma_at_transits`); invalid tables (negative, non-finite,
  duplicate wavelengths, wrong shape) raise. `depth_error_bins` /
  `evaluate_mode` now take `floor_spec` (None | ppm scalar | (n,2) table).
- **Fisher rank unit-invariance** (item 4, CONFIRMED numerically: a pure
  rescaling flipped rank 1→2): `_marg_sigmas` whitens to the unit-diagonal
  correlation form before the eigen-threshold, transforms sigmas back, and
  reports the whitened (scale-free) spectrum in the diagnostics.
- **Nuisance-projection row-scale invariance** (item 5, CONFIRMED: x1e10 on
  one row moved sigma_detect 10.30→10.79): `detection_significance`
  normalizes the nuisance normal matrix to correlation form; score now
  depends only on the nuisance span (both metric paths).
- **Noise multipliers default 1.0** (item 7): literature ratios moved to
  `instruments.LITERATURE_NOISE_FACTORS` (reference points, not defaults);
  GUI relabels the knob "empirical noise sensitivity factor".
- **Legacy backend labeling** (item 1): `instruments.BACKEND_STATUS` shown in
  the GUI header + provenance caption + README; the 3.0/3.0rc3 pin decision
  stands (provenance block + same-release gate unchanged, worker still v5).

**Also done:** correlated scenarios re-based on the floor EXCESS
(diag(C) = max(var_phot, floor²) = sigma_final², invariant now tested with
atol=0 — the v8 assertion was vacuous at ~1e-10 variances) and marked
EXPERIMENTAL / excluded from headline results; GUI grew a dedicated "Noise
model" sidebar section (floor type, per-mode floors or table upload,
sensitivity factors, experimental scenarios) and split "Physics" into
"Chemistry (VULCAN-JAX)" vs "Spectrum & clouds (ExoJAX RT)"; "can only be
equal or worse" wording replaced with the balanced either-direction phrasing;
README rewritten (professional register, PandExo-compatible scope language,
validation-status section with the pending gates).

**Deferred with explicit labels (not silently):** PandExo parity suite
(item 3) — random-noise path labeled "Pandeia-extracted-noise box-transit
approximation" in noise.py + README until a mode-by-mode parity matrix
against current PandExo passes; Pandeia engine upgrade (item 1 alternative
branch) — stays pinned with the LEGACY label. Item 6 (warm/cold gate) was
already a hard per-run gate in vulcan-retrieval (validate_warm, exits
nonzero); no change needed.

**Tests 52 → 70** (+18: floor semantics ×8, invariance sweeps ×4 incl.
24-decade rescalings, zero-row/zero-response guards, updated scenario
invariants): `python -m pytest tests -q`, numpy-only, all green.

## v10: 2026-07-12 strict re-audit V2 (exact sub-cell integration, NIRCam groups)

Response to the "final strict re-audit (V2)". Re-audit re-confirmed items 1/3/4
as already fixed in v9 (PandExo floor semantics, Fisher unit-invariance,
nuisance row-scale invariance — verified against the actual code, not the
docs). Two genuinely-remaining code defects fixed; two invariance requirements
tightened to the audit's exact wording.

**Confirmed and fixed:**
- **Exact piecewise-linear antiderivative** (audit item 2): `binning.bin_model`
  and `binning.smooth_to_native_r` built a cumulative-*trapezoid* integral at
  the model nodes and then LINEARLY interpolated it to arbitrary cell edges.
  The antiderivative of a piecewise-linear spectrum is piecewise-*quadratic*,
  so that was wrong for any cell edge between nodes (the audit's y=x, [0.1,0.2]
  counterexample returned 0.5 instead of 0.15; ~1–2 ppm on realistic narrow
  features). New `binning._pl_antideriv` carries the quadratic term
  explicitly — `I(x) = icum_k + y_k·dx + ½·slope_k·dx²` — so every cell-edge
  integral (final bins AND the native-R regrid) is exact to machine precision.
- **NIRCam group cap** (item 5): NIRCam grism `ngroup_max` was 180, above
  PandExo's hard NIRCam maximum of 100 — on a faint target the optimizer could
  pick an unsupported ramp. Both NIRCam modes capped at 100;
  `instruments.PANDEXO_NGROUP_MAX` documents the cap and an import-time guard
  refuses any mode above its instrument's cap; the worker's group selection
  goes through one `pandeia_worker._clamp_ngroup`, so a selected ramp can never
  leave `[ngroup_min, ngroup_max]`.

**Tightened to the audit's exact regression wording:**
- Fisher null-subspace overlap (item 3) now uses the basis-invariant L2
  projection norm over the null eigenvectors, not a single eigenvector's
  largest component (arbitrary for a degenerate null eigenspace).
- Added the second required item-5 regression: `detection_significance` is now
  pinned invariant under an arbitrary nonsingular remix (rotation + mixing) of
  the nuisance rows, not only per-row rescaling.

**Deferred, unchanged, still explicitly labeled (validation gates, not code):**
items 6 (upgrade to the current matched Pandeia data stack — the 3.0/3.0rc3 pin
is a dated decision with a written upgrade checklist and LEGACY label), 7
(automated current-PandExo parity fixtures), and 8 (recorded physical
convergence/history passes — run on HPC, Isaac's to schedule). These need heavy
backend/HPC runs, not source changes, and remain the documented pre-production
release gates.

**Tests 70 → 76** (+7: exact-antiderivative counterexample + linear-submodel
machine-precision closure ×2, NIRCam/instrument group-cap + optimizer-clamp
×3, multi-dim Fisher null-space invariance + nuisance-basis rotation
invariance ×2): `python -m pytest tests -q`, numpy-only, all green
(76 passed, 1 slow test skipped by default).

## v11: 2026-07-12 maximal cross-repo audit response (LSF flux ratio, in/out labeling)

Response to the "maximally intensive scientific and numerical audit" spanning
all three repos. Verified each of its five confirmed errors against the current
code (not the report). Verdicts + actions:

**ERROR 1 — stale vendored jwst_tool in `vulcan_exojax_run`: already resolved.**
The umbrella repo is archived on GitHub and absent locally; `build/` is
gitignored (0 tracked files). Nothing to run/diverge. No action.

**ERROR 2 — LSF convolved the depth directly instead of the count ratio: FIXED.**
The instrument measures LSF-averaged in/out COUNTS and ratios them:
`d_obs = 1 - L[F(1-d)]/L[F] = L[F d]/L[F]` — the FLUX-WEIGHTED LSF mean of the
depth, not the flat `L[d]`. `binning.smooth_to_native_r` gained a `weight`
argument (stellar flux, cell-averaged onto the same ln-λ grid via the exact
antiderivative); `detect.evaluate_mode` interpolates the worker's per-pixel
stellar flux onto the model grid and passes it to the depth, removed-molecule,
and every Jacobian-row blur (same weight → operator stays linear in d).
`weight=None`/constant reduces exactly to the old flat blur. Measured on the
audit's stellar-line + planetary-feature case: flux-weighting tracks the true
ratio to ~5 ppm (grid/4σ-truncation residual) while the flat blur was ~120 ppm
biased — the audit's failure, closed. F is only pixel-resolved, so sub-pixel
stellar lines remain a documented limitation. Pinned by 3 new tests
(none==constant, constant-depth preservation, ratio-reference match + >10×
closer than flat).

**ERROR 3 — box-transit variance uses one (out-of-transit) spectrum for both
in/out terms: documented (audit's own P2 disposition).** `pixel_depth_variance`
assumes F_in≈F_out, σ_in≈σ_out; exact propagation carries a (1-d) factor, so
this is CONSERVATIVE (never under-predicts σ) with the excess growing with
depth (~+0.08% at d=0.1%, +0.76% at 1%, +8% at 10%). Kept model-INDEPENDENT on
purpose (one operator bins noise and model consistently — the 2026-07-11 P0
fix); exact separate in/out propagation stays a pending PandExo-parity gate.
Docstring now states the approximation and its bias magnitude explicitly.

**ERRORs 4 & 5 live in the sibling vulcan-retrieval repo** (set_observations
validation; box-prior evidence separating the physical T-P-window support from
solver-dependent convergence attrition) — both fixed there the same day; see
that repo's CLAUDE.md / notes.

**Tests 76 → 79** (+3 flux-weighted-LSF): `python -m pytest tests -q`,
numpy-only, all green (79 passed, 1 slow test skipped by default).

## v12: 2026-07-12 maximal adversarial re-audit response (grid validation, retirement)

Response to the superseding maximal adversarial re-audit (report dated
2026-07-12, tests 2026-07-13 UTC). The audit ran against main head 67ef1be,
FOUR commits behind local main: its Errors 1-5 (cumulative-integral interp,
Fisher unit rank, nuisance basis scaling, quadrature/R-scaled floor, flat
depth-space LSF) were already fixed in 0f31753 / 6706934 / 26a65f0 (v9-v11).
All five were re-verified against the current head with the audit's own
reproducers before this pass: linear cell average exact to float64 (the
audit's "exact 0.0102" reference is itself an arithmetic slip; the true value
is 0.0103 and the code returns it exactly), Fisher sigmas/rank invariant
under a 24-decade unit rescale, nuisance score invariant under row rescaling
and invertible remixes, 20 ppm floor exactly 20 ppm at every R_bin and
transit count, flux-weighted LSF within ~1.4 ppm of the brute-force count
ratio where the flat blur erred by ~98 ppm.

**ERROR 7 (the one confirmed outstanding code defect) — invalid wavelength
grids degraded silently: FIXED.** `binning` now validates loudly at every
entry point: non-finite wavelengths/weights/model values raise (never
argsort-parked and dropped); median pixel spacing uses strictly POSITIVE gaps
(`_positive_gap_median`), so exact-duplicate wavelengths are flagged
degenerate even on duplicate-majority grids (all-gaps median collapsed to 0
and disabled the mask entirely — the audit's reproducer); bin edges must be
finite/ascending; `build_operator` raises with a per-criterion exclusion
breakdown when no usable pixel survives, instead of returning an empty
operator that failed later with a bare numpy min/max error;
`detect.evaluate_mode` raises when saturation + degeneracy exclude every
pixel; `smooth_to_native_r` rejects non-finite weights. Duplicate policy
documented on `build_operator`: exact duplicates carry zero spectral support,
are flagged by `degenerate_wl_mask`, and are excluded + COUNTED via
n_pix_degenerate.

**ERROR 6 — divergent public monorepo (vulcan_exojax_run): RETIRED for
real.** The standalone repos were already the authoritative source (the
2026-07-11 restructure), but the public monorepo still carried installable
stale package copies. Its README now carries a deprecation notice pointing
to the standalone repos at exact SHAs and the repo is archived (read-only).

**Docs:** README qualifies the derivative claim (machine-precision AD of the
discretized, tolerance-converged model — not physical exactness), states the
LSF count-ratio form, and lists grid validation in the test coverage;
CLAUDE.md gains the strict-grid-validation invariant.

**Not adopted from the audit:** its "exact result 0.0102" for the E1
reproducer (arithmetic slip, see above), and its claimed 5.86/7.67-sigma
nuisance failures and rank-1-vs-2 Fisher flip, which do not reproduce on the
current head (they targeted the pre-v9 code).

**Tests 79 → 86** (+7 strict-grid-validation): `python -m pytest tests -q`,
numpy-only, all green (86 passed, 1 slow test skipped by default).
