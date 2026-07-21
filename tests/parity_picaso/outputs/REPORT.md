# Native-PICASO RT vs tool ExoJax RT: one-state parity

Generated 2026-07-20 16:50 by scripts/run_native_rt_parity.py. OFFLINE validation only; the production path is always provider chemistry + ExoJax. See the script docstring for the method and why exact agreement is not expected (different opacity sources + reference-radius conventions).

State: W39b geometry, isothermal 1100 K, blended equilibrium at 10x solar / C/O 0.55, absorbers ['H2O', 'CO2', 'CO', 'CH4'] on H2/He. Native DB: opacities_0.3_15_R15000.db.

| metric | value |
|---|---|
| broadband offset (removed) | -2207 ppm |
| median abs residual | 688 ppm |
| p95 abs residual | 1540 ppm |
| max abs residual | 2019 ppm |
| bins (R=100, 1.0-12.0 um) | 250 |

Targets (stated in the script docstring, findings not CI gates):

- OUTSIDE TARGET: broadband offset |x| < 2000 ppm
- OUTSIDE TARGET: median |resid| < 150 ppm
- OUTSIDE TARGET: p95 |resid| < 400 ppm

Figure: ../figs/parity_native_rt.png
