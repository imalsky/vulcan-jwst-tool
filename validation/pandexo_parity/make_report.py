"""Render REPORT.md from parity_summary.json (run after run_parity.py).

Usage: python validation/pandexo_parity/make_report.py
"""
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fmt_ratio(s: dict | None) -> str:
    if not s or not s.get("n"):
        return "--"
    return (f"{s['median']:.4f} [{s['p05']:.4f}, {s['p95']:.4f}] "
            f"(n={s['n']})")


def main():
    summary = json.loads((HERE / "parity_summary.json").read_text())
    cfg = summary["config"]
    lines = []
    w = lines.append
    w("# PandExo numerical parity report")
    w("")
    w(f"Generated {date.today().isoformat()} by `run_parity.py` + "
      "`make_report.py` in this directory.")
    w("")
    w("Both sides run on the SAME current Pandeia backend, so every "
      "difference below is an ESTIMATOR/policy difference, not an engine "
      "calibration difference. PandExo is current master (commit in the "
      "provenance block). Configuration: constant transit depth "
      f"{cfg['depth']}, transit duration {cfg['transit_duration_s']/3600:.4f} h, "
      "equal out-of-transit baseline, saturation limit "
      f"{cfg['sat_limit']:.0%}, no noise floor, native (R=None) grids.")
    w("")
    w("Columns: sigma ratio = (this tool's per-pixel transit-depth sigma) / "
      "(PandExo's), median [5th, 95th percentile] over matched pixels. "
      "'matched' uses PandExo's integration counts in the tool formula "
      "(isolates the noise model); 'policy' uses the tool's own "
      "floor(T/t_int) counts (adds the integration-counting policy). "
      "flux ratio compares extracted stellar count rates (engine parity; "
      "expect 1.0000).")
    for sname, block in summary["stars"].items():
        star = cfg["stars"][sname]
        w("")
        w(f"## Star `{sname}` (Teff {star['teff']:.0f} K, logg "
          f"{star['log_g']}, [Fe/H] {star['metallicity']}, Ks "
          f"{star['ks_mag']})")
        w("")
        po = block.get("provenance_ours") or {}
        pp = block.get("provenance_pandexo") or {}
        w(f"Backend: engine {po.get('engine_version')} + "
          f"{po.get('refdata_name')} (worker v{po.get('worker_version')}); "
          f"PandExo {pp.get('pandexo_version')} on engine "
          f"{pp.get('pandeia_engine_version')}.")
        w("")
        w("| mode | status | ngroup ours/PX | t_int s ours/PX | "
          "n_int ours/PX(in) | flux ratio | sigma ratio (matched) | "
          "sigma ratio (policy) |")
        w("|---|---|---|---|---|---|---|---|")
        for m in block["modes"]:
            if m.get("status") == "OK":
                w(f"| {m['key']} | OK | {m['ngroup_ours']}/"
                  f"{m['ngroup_pandexo']} | {m['t_int_ours_s']:.3f}/"
                  f"{m['t_int_pandexo_s']:.3f} | {m['n_int_ours']}/"
                  f"{m['n_int_pandexo_in']:.0f} | "
                  f"{fmt_ratio(m.get('flux_ratio'))} | "
                  f"{fmt_ratio(m.get('sigma_ratio_matched'))} | "
                  f"{fmt_ratio(m.get('sigma_ratio_policy'))} |")
            elif m.get("status") == "SATURATED":
                w(f"| {m['key']} | SATURATED (ours: unusable, loud; "
                  f"PandExo ngroup={m.get('pandexo_ngroup')}) | -- | -- | "
                  "-- | -- | -- | -- |")
            else:
                w(f"| {m['key']} | ERROR (see parity_summary.json) | -- | "
                  "-- | -- | -- | -- | -- |")
        w("")
        w("Noise-model attribution (median per-integration variance over "
          "pure photon counts; photon-limited = 1.0):")
        w("")
        w("| mode | this tool (pandeia extracted noise) | PandExo (fml) |")
        w("|---|---|---|")
        for m in block["modes"]:
            if m.get("status") == "OK":
                w(f"| {m['key']} | {m['var_excess_ours']:.3f} | "
                  f"{m['var_excess_pandexo']:.3f} |")
        for m in block["modes"]:
            if m.get("status") == "OK" and m.get("pandexo_warnings"):
                warns = {k: v for k, v in m["pandexo_warnings"].items()
                         if str(v) not in ("nan", "None", "0", "All good")}
                if warns:
                    w("")
                    w(f"PandExo warnings for {m['key']}: {warns}")
    w("")
    w("## Findings")
    w("")
    w("1. **Configuration parity: achieved.** With the registry's explicit "
      "TSO readout patterns, PandExo's extraction strategy (apertures/"
      "annuli), and the ecliptic/medium background, the two sides agree on "
      "the extracted wavelength grids (every pixel matches), extracted "
      "count rates (flux ratios 0.99-1.02), selected groups (within 1), "
      "integration times (within 1%), and integration counts (within "
      "rounding policy). Saturation behavior matches: pixels PandExo masks "
      "as saturated are the pixels this tool excludes.")
    w("")
    w("2. **The remaining sigma difference is the noise model itself, and "
      "it is one-sided.** This tool propagates pandeia's full extracted "
      "noise (correlated ramp/read noise, background, dark, IPC, "
      "quantum-yield excess); PandExo's default 'fml' calculation is an "
      "analytic ramp formula that sits within a few percent of pure photon "
      "noise in the NIR. The attribution tables above show the variance "
      "excess over photon counts on both sides; their ratio reproduces the "
      "observed sigma ratios (e.g. G395H: 1.220/1.014 = 1.203 ~= 1.108^2). "
      "This tool is therefore systematically CONSERVATIVE relative to "
      "PandExo: ~7-12% higher sigma for NIRSpec/NIRISS/NIRCam, and larger "
      "for MIRI LRS (~35-48%), where the deep-red background and detector "
      "terms dominate and the analytic formula under-represents them. For "
      "context, published achieved-vs-PandExo noise ratios (COMPASS "
      "G395H 1.05-1.12; MIRI LRS ~1.15-1.2 above random-noise "
      "simulations) fall on the same side, between the two models for "
      "NIRSpec.")
    w("")
    w("3. **Residual policy differences (documented, small):** integration "
      "counts are floored here vs rounded in PandExo (at most one "
      "integration per window); ngroup_min is 2 here while PandExo will "
      "select 1 group (PRISM on a bright star); the symmetric in/out "
      "approximation adds ~+0.5% sigma at 1% depth (grows with depth; "
      "docstring in noise.pixel_depth_variance).")
    w("")
    w("4. **What may be claimed:** the instrument configuration, timing, "
      "group optimization, saturation handling, and extraction of this "
      "tool match current PandExo on the current engine. Absolute sigmas "
      "are NOT PandExo-identical and are not labeled as such: they are "
      "pandeia-extracted-noise forecasts, conservative relative to "
      "PandExo's analytic noise by the mode-dependent margins quantified "
      "above.")
    w("")
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {HERE / 'REPORT.md'}")


if __name__ == "__main__":
    main()
