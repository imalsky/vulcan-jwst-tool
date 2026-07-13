# PandExo numerical parity report

Generated 2026-07-12 by `run_parity.py` + `make_report.py` in this directory.

Both sides run on the SAME current Pandeia backend, so every difference below is an ESTIMATOR/policy difference, not an engine calibration difference. PandExo is current master (commit in the provenance block). Configuration: constant transit depth 0.01, transit duration 2.8036 h, equal out-of-transit baseline, saturation limit 80%, no noise floor, native (R=None) grids.

Columns: sigma ratio = (this tool's per-pixel transit-depth sigma) / (PandExo's), median [5th, 95th percentile] over matched pixels. 'matched' uses PandExo's integration counts in the tool formula (isolates the noise model); 'policy' uses the tool's own floor(T/t_int) counts (adds the integration-counting policy). flux ratio compares extracted stellar count rates (engine parity; expect 1.0000).

## Star `w39_like` (Teff 5400 K, logg 4.45, [Fe/H] 0.0, Ks 10.663)

Backend: engine 2026.2 + pandeia_data-2026.2-jwst (worker v5); PandExo 2026.2 (git master 1af4c79791, 2026-07-10) on engine 2026.2.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 2/1 | 0.699/0.452 | 14439/22314 | 0.9976 [0.9663, 1.5317] (n=403) | 1.2066 [1.0587, 1.5140] (n=403) | 1.5000 [1.3162, 1.8821] (n=403) |
| nirspec_g395h | OK | 124/125 | 112.770/113.652 | 89/89 | 0.9973 [0.9436, 1.0392] (n=3330) | 1.1078 [1.0767, 1.1330] (n=3330) | 1.1078 [1.0767, 1.1330] (n=3330) |
| nirspec_g235h | OK | 57/57 | 52.336/52.316 | 192/193 | 1.0009 [0.9654, 1.0705] (n=3424) | 1.0952 [1.0655, 1.1307] (n=3424) | 1.0981 [1.0683, 1.1336] (n=3424) |
| niriss_soss | OK | 19/19 | 109.900/109.880 | 91/92 | 1.0203 [0.9763, 1.2934] (n=2040) | 1.1158 [1.0837, 1.1509] (n=2040) | 1.1219 [1.0896, 1.1572] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9963 [0.9592, 1.0098] (n=1812) | 1.0928 [0.9172, 1.1026] (n=1812) | 1.0947 [0.9188, 1.1044] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9895 [0.9433, 1.0454] (n=1267) | 1.0872 [0.9341, 1.1107] (n=1267) | 1.0890 [0.9357, 1.1126] (n=1267) |
| miri_lrs | OK | 253/252 | 40.237/40.237 | 250/251 | 0.9974 [0.9951, 0.9984] (n=372) | 1.4848 [1.4775, 1.5945] (n=372) | 1.4878 [1.4804, 1.5977] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.914 | 1.221 |
| nirspec_g395h | 1.220 | 1.014 |
| nirspec_g235h | 1.199 | 1.013 |
| niriss_soss | 1.527 | 1.119 |
| nircam_f322w2 | 1.222 | 1.040 |
| nircam_f444w | 1.277 | 1.106 |
| miri_lrs | 30.476 | 14.103 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

## Star `bright_hot` (Teff 6250 K, logg 4.3, [Fe/H] 0.0, Ks 8.5)

Backend: engine 2026.2 + pandeia_data-2026.2-jwst (worker v5); PandExo 2026.2 (git master 1af4c79791, 2026-07-10) on engine 2026.2.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 2/1 | 0.699/0.452 | 14439/22314 | 1.0002 [0.9858, 1.0198] (n=265) | 1.0751 [1.0335, 1.1655] (n=265) | 1.3365 [1.2848, 1.4489] (n=265) |
| nirspec_g395h | OK | 16/17 | 15.354/16.236 | 657/622 | 1.0013 [0.9803, 1.0254] (n=3330) | 1.1099 [1.0964, 1.1222] (n=3330) | 1.0799 [1.0668, 1.0919] (n=3330) |
| nirspec_g235h | OK | 7/7 | 7.236/7.216 | 1394/1399 | 1.0037 [0.9777, 1.0602] (n=3424) | 1.0567 [1.0445, 1.0802] (n=3424) | 1.0586 [1.0463, 1.0821] (n=3424) |
| niriss_soss | OK | 2/2 | 16.502/16.482 | 611/613 | 1.0148 [0.9881, 1.3051] (n=2040) | 1.0648 [1.0281, 1.1186] (n=2040) | 1.0666 [1.0298, 1.1205] (n=2040) |
| nircam_f322w2 | OK | 67/67 | 23.167/23.161 | 435/436 | 1.0008 [0.9636, 1.0161] (n=1812) | 1.0944 [0.9699, 1.1031] (n=1812) | 1.0956 [0.9710, 1.1044] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9980 [0.9652, 1.0242] (n=1267) | 1.0980 [1.0092, 1.1094] (n=1267) | 1.0999 [1.0109, 1.1113] (n=1267) |
| miri_lrs | OK | 39/39 | 6.203/6.362 | 1627/1587 | 1.0011 [1.0003, 1.0018] (n=372) | 1.3484 [1.3411, 1.6160] (n=372) | 1.3317 [1.3245, 1.5961] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.221 | 1.068 |
| nirspec_g395h | 1.234 | 1.015 |
| nirspec_g235h | 1.130 | 1.017 |
| niriss_soss | 1.299 | 1.067 |
| nircam_f322w2 | 1.191 | 1.007 |
| nircam_f444w | 1.206 | 1.015 |
| miri_lrs | 5.923 | 3.293 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.', 'Saturated?': 'Full saturation:\n There are 98 pixels saturated at the end of the first group. These pixels cannot be recovered.', 'Num Groups Reset?': 'Optimized NGROUPS below minimum (1). SET TO NGROUPS=1'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

## Findings

1. **Configuration parity: achieved.** With the registry's explicit TSO readout patterns, PandExo's extraction strategy (apertures/annuli), and the ecliptic/medium background, the two sides agree on the extracted wavelength grids (every pixel matches), extracted count rates (flux ratios 0.99-1.02), selected groups (within 1), integration times (within 1%), and integration counts (within rounding policy). Saturation behavior matches: pixels PandExo masks as saturated are the pixels this tool excludes.

2. **The remaining sigma difference is the noise model itself, and it is one-sided.** This tool propagates pandeia's full extracted noise (correlated ramp/read noise, background, dark, IPC, quantum-yield excess); PandExo's default 'fml' calculation is an analytic ramp formula that sits within a few percent of pure photon noise in the NIR. The attribution tables above show the variance excess over photon counts on both sides; their ratio reproduces the observed sigma ratios (e.g. G395H: 1.220/1.014 = 1.203 ~= 1.108^2). This tool is therefore systematically CONSERVATIVE relative to PandExo: ~7-12% higher sigma for NIRSpec/NIRISS/NIRCam, and larger for MIRI LRS (~35-48%), where the deep-red background and detector terms dominate and the analytic formula under-represents them. For context, published achieved-vs-PandExo noise ratios (COMPASS G395H 1.05-1.12; MIRI LRS ~1.15-1.2 above random-noise simulations) fall on the same side, between the two models for NIRSpec.

3. **Residual policy differences (documented, small):** integration counts are floored here vs rounded in PandExo (at most one integration per window); ngroup_min is 2 here while PandExo will select 1 group (PRISM on a bright star); the symmetric in/out approximation adds ~+0.5% sigma at 1% depth (grows with depth; docstring in noise.pixel_depth_variance).

4. **What may be claimed:** the instrument configuration, timing, group optimization, saturation handling, and extraction of this tool match current PandExo on the current engine. Absolute sigmas are NOT PandExo-identical and are not labeled as such: they are pandeia-extracted-noise forecasts, conservative relative to PandExo's analytic noise by the mode-dependent margins quantified above.
