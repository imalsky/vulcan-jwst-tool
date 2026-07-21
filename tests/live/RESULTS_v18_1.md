# Live validation record: v18.1 (tool 0.12.1, model-cache v19)

Run 2026-07-21 on the development Mac (base env, picaso 4.0.1, reference
tree via JWST_TOOL_PICASO_REFDATA), committed as the durable evidence the
2026-07-21 review asked for. Reproduce with:

    JWST_TOOL_RUN_PICASO_LIVE=1 python -m pytest tests/live -q

## Battery outcome: 12/12

| test | result | measured |
|---|---|---|
| blend vs native picaso at a node | pass | max 4e-15 dex |
| leave-one-node-out blend accuracy | pass | medians <= 0.05 dex (majors) |
| picaso lnZ row FD closure | pass | closure < 5% of the secant |
| picaso vs vulcan spectrum sanity | pass | median gap < 2000 ppm; SO2 vulcan-only |
| climate matrix: W39b 10x/0.55 rfacv 0.5 | pass (certified) | flux metric ~1.5e-6 |
| climate matrix: W39b rfacv 0.0 | pass (ENVELOPE REFUSAL) | top below the 320 K opacity floor -- loud, never clipped |
| climate matrix: W39b rfacv 1.0 | pass (ENVELOPE REFUSAL) | bottom 3106 K above the 2980 K ceiling |
| climate matrix: W39b solar node | pass (certified) | |
| climate matrix: HD 189733 b | pass (certified) | |
| climate matrix: WASP-107 b | pass (certified) | |
| lock: concurrent solves share one computation | pass | solver 80.1 s, waiter 79.4 s, bit-identical (pre-rewrite baseline); re-verified against the never-unlink lifecycle |
| lock: kill -9'd holder releases to the survivor | pass | flock releases on death; survivor completes |

## Key measured numbers this record pins

- Climate solve: bit-deterministic (repeat + fresh-opacity reruns differ by
  exactly 0 K); W39b default converges in ~60-90 s from the analytic guess.
- Tint_cl FD row (fd-climate, h = 15 K): h-vs-2h 1.6%; W39b transmission
  responds at ~1.2e-7 per K (irradiation-dominated -- weak by physics).
- dlnCO node-kink at C/O = 0.55: 1.52 (hard-errors by design); mid-cell
  0.50: kink 0.089, h-vs-2h 0.003. lnZ at the metallicity node: kink 0.17.
- rcb certification bounds (W39b): 45/50 refused by the flux gate
  (4.2e-2 / 2.8e-3), 55 refused by the T-window (3074 K at 7.6 bar);
  certified 60-75 all Schwarzschild-consistent; observables in
  docs/picaso_roadmap.md.
- Corrupt-cell correction (feh1.0_co0.55 @ 900 K, logP -5.523):
  content-guard sha 29c1396fb5d35401 matches the shipped file; correction
  vs renormalize-through bound <= 2.19 ppm (900 K profile), 0.00 ppm
  (1100 K default).
- Native-RT parity envelope (offline harness): offset -2207 ppm removed,
  median 688 ppm, p95 1540 ppm vs ExoJax (opacity-source dominated;
  tests/parity_picaso/outputs/REPORT.md).

The fast numpy-only suite at this commit: 247 passed, 13 skipped
(the 13 = these env-gated live tests plus the pre-existing slow closure).
