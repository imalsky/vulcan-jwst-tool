# PICASO provider + climate T-P mode: scope, limits, and roadmap

Status: v18 (tool 0.12.0, 2026-07-20). This is the versioned record of what
the PICASO integration ships, the science limits it states, the measured
findings behind its design decisions, and the features deliberately deferred
(with re-entry sketches). The GUI links here wherever a limit bites.

## What shipped (v18)

- **`chem_provider="picaso"`**: PICASO 4.0.1 thermochemical-equilibrium
  chemistry (Visscher 2121 grid, 101 T x 21 P x 50 species per node file,
  13 [M/H] x 6 C/O nodes) as a second forward-model engine. Everything
  downstream is SHARED with the VULCAN-JAX engine: ExoJax RT, the one
  binning operator, Pandeia noise, detect/fisher. Equilibrium-vs-kinetics
  on identical machinery is the science axis.
- **Composition blending**: the stock `chemeq_visscher_2121` picks the
  NEAREST node file (no composition interpolation -- FD rows through it
  would be exactly zero). The provider blends the 2x2 bracketing node files
  bilinearly in ([M/H] dex, C/O) of the log10 abundances, then interpolates
  (T, P) with exactly picaso's own `chem_interp` convention (bilinear in 1/T
  and log10 P; verified to 4e-15 dex against native picaso). Outside the
  tables the provider REFUSES where picaso would silently extrapolate.
- **`tp_mode="picaso_climate"`**: the PICASO radiative-convective climate
  solver (preweighted correlated-k tables, 196 bins x 8 g-points) as a T-P
  mode for BOTH providers -- including full VULCAN kinetics (photochemistry,
  SO2) running on the PICASO RCE profile. Certified, cached
  (`output/picaso_climate_cache/`, atomic writes + process-safe locking with
  stale-lock recovery), shared between providers.
- **Fisher rows** (`jac_method="fd"` only): lnZ / dlnCO as symmetric
  two-cell interpolant secants with a one-sided-secant kink gate
  (`picaso_chem.FD_KINK_TOL = 0.5`, hard error); T-P rows by table
  re-equilibration; `Tint_cl` (climate mode) as a full-climate-re-solve row
  ("fd-climate", 4 certified solves, h = 15 K). Provenance per row:
  `jac_row_method`, `fd_h`, `fd_err`, `fd_kink`, `fd_grid_cell`.
- **Certificates**: the provider writes `picaso_cert_json` (blend nodes +
  weights, per-layer pre-normalization gas sums, realized gas C/O, floored
  entries, suspect-cell hits); climate mode adds `climate_provenance_json`
  (convergence + flux metric + gradient envelope + convective-zone
  structure). `jwst-tool data` gains a "PICASO provider data" section;
  reference data is selected ONLY by `JWST_TOOL_PICASO_REFDATA` (no baked-in
  path) and fingerprinted by CONTENT into every cache key.
- **Native-RT parity harness**: `tests/parity_picaso/` compares picaso's own
  `get_transit_1d` against the tool's ExoJax RT on one identical state --
  offline validation only, never a production path. MEASURED (2026-07-20,
  W39b isothermal 1100 K, shared absorbers H2O/CO2/CO/CH4, R = 100 bins):
  broadband offset -2207 ppm (reference-radius conventions; removed), then
  median |residual| 688 ppm, p95 1540 ppm -- OUTSIDE the up-front targets
  (150/400 ppm), dominated by the opacity sources (the native DB is the
  resampled R=15,000 'default_3.3' product; the tool uses HITRAN through
  exojax PreMODIT) plus the g(z)-vs-constant-gravity conventions. This is
  the honest cross-model envelope, reported in
  tests/parity_picaso/outputs/REPORT.md; it is exactly why the production
  path never mixes the two RTs.

## Stated science limits (intrinsic, not bugs)

- **No SO2 / S2 / S8** under the picaso provider: equilibrium sulfur sits in
  H2S / OCS. The WASP-39b photochemical-sulfur headline science stays
  VULCAN-only. The GUI removes SO2 from the menus; `canonical_params`
  refuses it loudly.
- **C/O hard-capped at 1.10** by the Visscher grid (VULCAN handles 2.0
  structurally). Metallicity spans 0.1-100x solar ([M/H] in [-1, 2]).
- **Composition derivatives are TWO-CELL INTERPOLANT SECANTS**, not local
  derivatives: the grid nodes are kinks of the interpolant. MEASURED
  (2026-07-20, W39b defaults): at the C/O = 0.55 NODE the one-sided dlnCO
  secants disagree by 152% of the symmetric row (left cell [0.46, 0.55] vs
  right cell [0.55, 0.82] carry genuinely different chemistry response), so
  the kink gate HARD-ERRORS there -- by design. At the mid-cell C/O = 0.50
  (the GUI default for the provider) the whole stencil stays inside one
  cell: kink 0.089, h-vs-2h 0.003. lnZ at the [M/H] = +1.0 node passes
  (kink 0.17; the metallicity cells are symmetric). Cross-node blend
  accuracy (leave-one-node-out at feh0.5/co0.46): median ~0.01-0.03 dex,
  p95 <~ 0.05 dex for major species; worst ~0.2 dex (CO2); ~1 dex locally
  at the K condensation edge.
- **Climate composition is EXACT-CK-NODE only**: the correlated-k tables
  carry no composition interpolation, so climate mode accepts only shipped
  nodes (extreme metallicities +-1.5/+-2.0 ship only C/O 0.27-0.82).
  Consequence: at a node, the dlnCO Fisher row can hard-error (see above) --
  climate-mode C/O constraints are a known v1 gap.
- **One-way coupling**: climate T-P is solved with PICASO equilibrium CK
  opacities, then post-processed with either engine's chemistry and ExoJax
  RT. The chemistry NEVER feeds back into the climate opacity. This is not
  radiative-chemical self-consistency and is never labeled as such.
- **`climate_rcb` is a model assumption**: MEASURED on W39b defaults,
  rcb_guess 60 vs 65 both pass every certification but differ by up to
  341 K below ~0.4 bar with the convective boundary parked at the guess
  (layers above agree to ~2 K) -- the weakly-constrained deep-adiabat
  degeneracy of strongly irradiated planets; PICASO's stratification search
  does not resolve it. It is cache-keyed, surfaced in the GUI with the
  measured numbers, and the Tint_cl row differentiates at FIXED rcb. The
  climate solve itself is bit-deterministic (repeat and fresh-opacity
  reruns: exactly 0 K).
- **Pressure policy**: the equilibrium tables and the climate grid start at
  1e-6 bar; above it the topmost layer is held constant (the sibling
  interp_map's documented edge clamp -- it logs the clamped layer count on
  every run). The provider chemistry grid spans exactly 1e-6 bar to the
  chemistry bottom (7.6 bar); the VULCAN+climate path goes through the
  file-mode top-clamp logging. Nothing extrapolates silently.
- **Certified domain**: v1 climate mode is certified around the WASP-39b
  configuration. Other planets / nodes / rfacv values are dynamically
  convergence-gated (the certification refuses anything unconverged,
  flux-imbalanced, gradient-pathological, or top-convective) and should be
  treated as experimental until `tests/live/test_picaso_live.py`'s smoke
  matrix has been run for them.
- **T-window interaction**: climate profiles are truncated/interpolated to
  end exactly at the 7.6-bar chemistry bottom (W39b default: 2832 K there,
  inside the 320-2980 K opacity window with ~150 K margin). Hotter
  planets/Tint may legitimately REFUSE at the window -- a stated envelope
  limit, never clipped.

## Measured data-quality findings (upstream-reportable)

- **One corrupted cell** in `sonora_2121grid_feh1.0_co0.55.txt` at
  (T = 900 K, logP = -5.523): H2 and He are under-normalized by ~0.75 with
  their ratio preserved while trace species keep normal absolute values
  (gas sum 0.746; every other cell in that file >= 0.998). The W39b default
  profile passes near this cell; the provider does NOT repair table data --
  the loader flags it (`SUSPECT_SUM = 0.90`) and the certificate reports any
  suspect cell inside the evaluated span.
- **Extreme-metallicity gas sums**: the |feh| >= 1.5 files sum to 0.86-0.98
  at T <~ 500 K (documented missing-species behavior). The provider
  renormalizes per layer (the upstream-recommended treatment), records
  pre-normalization sums, refuses below `GAS_SUM_MIN = 0.70`, and flags
  below `GAS_SUM_WARN = 0.98`.
- **Gas accounting**: ions and electrons are COUNTED in the gas total and
  the mean molecular weight (they are gas-phase number density; e- mass
  5.49e-4 amu); only graphite is excluded as a condensate (renamed
  `C-gr_l_s` so the shared RT condensate mask handles it exactly like
  VULCAN's reservoir columns).
- **Realized gas C/O != the file label below ~1700 K** (silicate
  condensation sequesters O: label 0.46 -> gas-phase 0.55 at 800-1200 K;
  matches at 2000 K). Real physics, recorded in the certificate; label
  comparisons only above `CO_CHECK_T_K = 2000 K`.
- **`chemeq_visscher_2121`'s docstring says 20 pressures; the files carry
  21** (2121 = 21 x 101). The loader validates the real shape.
- **Native transmission returns all-NaN when `gravity()` is given bare
  gravity**: the altitude integration needs planet.mass (g = GM/z^2). The
  parity script documents this trap; always pass mass + radius.

## Deferred features (why, and how to re-enter)

1. **Quench / lnKzz row** (the reason there is NO lnKzz under picaso: it
   has no effect in equilibrium, so the row would be identically zero).
   Re-entry: PICASO 4's `atmosphere(quench=...)` / `find_kzz` /
   `adjust_quench_chemistry` machinery restores a physical lnKzz direction
   (quench approximation vs VULCAN's full kinetics -- a scientifically
   interesting comparison axis). Needs its own FD smoothness study (quench
   levels move discretely with Kzz) and compatibility rules before any
   Fisher row is certified.
2. **Cloudy climate (virga)**: the reference tree's `virga/` directory is
   EMPTY; cloudy climate solves would download condensate files and need
   their own certification. Re-entry: populate virga refdata, extend
   `climate_refdata_fingerprint`, add a virga toggle with its own
   compatibility matrix.
3. **Sonora guess profiles**: `sonora_grids/` is empty, so the climate
   guess is a deterministic analytic Guillot profile (measured: converges
   in ~1 min on W39b). Re-entry only if some configuration cannot converge
   from the analytic guess (then: a guess ladder, never warm-starting from
   previous solves -- determinism is a certification property).
4. **PICASO-native RT as a GUI backend**: rejected for production (decision
   2026-07-20): the local opacities.db is R=15,000 with only 10 line
   species (no NH3/HCN), and a second RT path would break the
   one-measurement-operator rule. The parity harness
   (`tests/parity_picaso/`) is the supported use.
5. **Off-node climate composition**: would require blending correlated-k
   TABLES (not log abundances) or on-the-fly k-table mixing; out of scope
   for v1. Climate mode stays exact-node.
6. **Per-side (left/right) composition derivatives at nodes**: the kink
   gate currently refuses; reporting both one-sided secants as an interval
   is a possible v2 presentation.
7. **`jwst-tool fetch` for PICASO refdata**: the reference tree is
   user-supplied science data (Zenodo: chemistry/CK 10.5281/zenodo.13733116,
   opacities 10.5281/zenodo.14861730); datacheck reports it, fetch does not
   download it.
8. **AD through climate mode**: refused in v1 (`jac_method="ad"` +
   picaso_climate); the VULCAN warm-jvp rows on a fixed climate T-P would
   be well-defined, but the combination is uncertified.

## Live validation

`JWST_TOOL_RUN_PICASO_LIVE=1 python -m pytest tests/live -q` runs the
measured-2026-07-20 battery: within-node native parity, leave-one-node-out
blend accuracy, lnZ FD closure, picaso-vs-vulcan spectrum sanity, and the
climate smoke matrix (W39b x rfacv {0, 0.5, 1}, solar node, HD 189733 b,
WASP-107 b). The native-RT parity report lives in
`tests/parity_picaso/outputs/REPORT.md`.
