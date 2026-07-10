# jwst_tool — JWST instrument selector (VULCAN-JAX × ExoJAX × Pandeia)

A PandExo-style planning GUI built on the live pipeline in this bundle: pick a
science goal (detect molecule X, forecast parameter constraints), run the
VULCAN-JAX photochemistry → ExoJAX transmission model **locally**, simulate each
JWST time-series mode's transit-depth precision with the **real STScI Pandeia ETC
engine**, and rank the modes.

## Launch

```
cd vulcan_exojax_run
streamlit run jwst_tool/app.py
```

First run of a new parameter set: ~2–3 min (chemistry warm-up ~50 s, ~40 s per
chemistry solve, ~10 s per RT eval), plus ~20–60 s per freed Fisher parameter.
Everything is disk-cached (`data/jwst_tool/`): repeat runs are instant.

## What it computes

- **Forward model** (`forward.py`, subprocess): WIDE-band (1–15 µm, native
  R≈3000) transmission spectrum of the WASP-39b column, photochemistry ON,
  molecules H2O/CO2/CO/CH4/SO2. Knobs: metallicity (about the 10× solar
  baseline, `reanchor_atom_ini` + two-stage solve for finite steps), Δln(C/O),
  Kzz (GCM profile × factor, or constant via `cfg_overrides`), and the T-P
  profile — baseline+ΔT, isothermal, or Guillot (ExoJax `atmprof_Guillot`,
  the same hook the retrieval uses). Out-of-window T-P ([320, 2980] K) and
  count_max-exhausted solves **raise**, they are never clipped/carried.
- **Noise** (`pandeia_worker.py` in the `picaso_base` conda env — pandeia.engine
  3.0 matching the on-disk `pandeia_data-3.0rc3` refdata): per-native-pixel
  extracted flux + noise for a PHOENIX star normalized to the entered Ks mag
  (at_lambda; 2MASS zeropoint), groups auto-chosen to stay under the saturation
  limit (PandExo-style). Transit-depth error per bin:
  `var = (noise/flux)² (1/n_in + 1/n_out) / n_transits`, inverse-variance
  binned, then a **non-averaging** systematic floor in quadrature (defaults per
  mode, editable; Greene+2016-ish, in-flight performance is often better).
- **Detection significance**: `σ = √Σ((full − without-X)/σ_bin)²` on each
  mode's bins (nested-model Δχ² proxy).
- **Fisher forecast** (`fisher.py`, opt-in): one warm-started forward-mode jvp
  per freed parameter through the full chain (the validated sensitivity
  pattern) + an RT-only lnR0 column. Per-mode rows marginalize lnR0; the
  combined row shares lnR0 and adds one absolute-depth offset nuisance per mode.

## Modes

NIRSpec PRISM / G395H / G235H (BOTS), NIRISS SOSS order 1, NIRCam F322W2 /
F444W (grism time series), MIRI LRS slitless. A mode with no unsaturated pixels
at its shortest ramp (e.g. PRISM on WASP-39, Ks=10.2) is reported **unusable**
with the saturation numbers, matching the known PRISM brightness limit.

## Backend wiring (this machine)

- Pandeia engine: `picaso_base` conda env (engine 3.0). The base env's engine
  2026.1 rejects the 3.0rc3 refdata (`nsuperstripe` KeyError).
- `pandeia_refdata`: `~/Documents/Important_Docs/JWST_CYCLE5/picaso_ian/data/pandeia_data-3.0rc3`
- `PYSYN_CDBS`: `data/jwst_tool/cdbs` — a minimal tree: `grid/phoenix` symlinked
  from `RT-Project/picaso/reference/stellar_grids`, `comp/nonhst/johnson_j_003_syn.fits`
  fetched from ssb.stsci.edu/trds (pandeia's extinction module needs it).
  Paths live in `instruments.py`.

## Known limits (v1)

- Model band starts at 1.0 µm (H2-H2 CIA table edge), so SOSS order 1 loses
  0.85–1.0 µm and order 2 is not offered; MIRI LRS is cut at 12 µm.
- One planet baseline (WASP-39b column, its gravity/radius/star). Other planets
  need a baked VULCAN baseline + `rp_cm`/`gs_cgs`/`rstar_cm` in the profile.
- HITRAN line lists (main isotopologue), adequate for planning; not HITEMP/ExoMol.
- No partial-saturation strategy (pandeia group optimization only).
