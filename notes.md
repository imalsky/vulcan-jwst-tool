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
