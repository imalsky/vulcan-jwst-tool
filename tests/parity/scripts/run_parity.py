"""PandExo numerical parity harness (2026-07-12 external audit, release gate).

Runs MATCHED star/instrument configurations through BOTH noise paths on the
SAME current Pandeia backend and compares them mode by mode:

  * this package's worker (src/jwst_tool/pandeia_worker.py) + its box-transit
    depth-error propagation (noise.pixel_depth_variance), and
  * current PandExo (master; pandexo_worker.py in this directory).

Running both sides on one engine/refdata generation isolates ESTIMATOR
differences (timing policy, in/out propagation, saturation handling) from
engine-calibration differences -- the point of the audit's parity gate. The
tool's pinned legacy 3.0 backend is unaffected; this harness points the
worker at the current backend explicitly via environment variables.

Required environment (all loud, no defaults -- machine paths stay out of git):
  JWST_TOOL_PANDEIA_PYTHON   python of a conda env with pandeia.engine 2026.2
                             AND pandexo (master) installed
  JWST_TOOL_PANDEIA_REFDATA  extracted pandeia_data-2026.2-jwst tree
  JWST_TOOL_PANDEIA_PSF_DIR  extracted pandeia_psfs-2026.2-jwst tree
  JWST_TOOL_DATA_DIR         directory whose cdbs/ holds the phoenix grid,
                             calspec Vega, and comp/nonhst bandpasses
  JWST_TOOL_OUTPUT_DIR       the tool's own worker noise cache (model_cache /
                             noise_cache); unrelated to the parity artifacts

Usage: python tests/parity/run_parity.py
Everything parity lives in THIS directory (tests/parity/): the raw per-run
JSON (git-ignored, see .gitignore) is written here alongside the committed
artifacts (parity_summary.json, REPORT.md, the figures). The tool's Pandeia
noise cache still goes under JWST_TOOL_OUTPUT_DIR (that is the app's cache,
not a parity output).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent        # tests/parity/scripts
OUTPUTS = HERE.parent / "outputs"             # raw JSON + parity_summary.json
REPO = HERE.parents[2]                         # scripts -> parity -> tests -> repo
sys.path.insert(0, str(REPO / "src"))

for var in ("JWST_TOOL_PANDEIA_PYTHON", "JWST_TOOL_PANDEIA_REFDATA",
            "JWST_TOOL_PANDEIA_PSF_DIR", "JWST_TOOL_DATA_DIR",
            "JWST_TOOL_OUTPUT_DIR"):
    if not os.environ.get(var):
        raise SystemExit(f"run_parity: {var} must be set (see module docstring)")

from jwst_tool import instruments as ins            # noqa: E402
from jwst_tool import noise                          # noqa: E402

# One transit's worth of time on each side of the transit (baseline
# fraction 1.0 in PandExo terms); WASP-39 b duration.
T_TRANSIT_S = 2.8036 * 3600.0
DEPTH = 0.01            # constant-depth planet (rp^2/r*^2)
SAT_LIMIT = 0.80

STARS = {
    "w39_like": dict(teff=5400.0, log_g=4.45, metallicity=0.0, ks_mag=10.663),
    "bright_hot": dict(teff=6250.0, log_g=4.30, metallicity=0.0, ks_mag=8.5),
    # a faint K dwarf so NIRSpec PRISM is UNSATURATED and gets a valid parity
    # point (it saturates on the two brighter stars -- there both tools flag
    # it and only the unusable-regime ngroup floor differs, ngroup 2 vs 1)
    "faint_k": dict(teff=4500.0, log_g=4.60, metallicity=0.0, ks_mag=13.0),
}

# PandExo template name + the overrides that pin BOTH sides to one
# configuration (subarray / readout / filter), per mode key. PandExo's SOSS
# template defaults to substrip96 and its MIRI template to the FAST pattern;
# real TSO programs (and this tool) use substrip256 / FASTR1.
PANDEXO_MODES = {
    "nirspec_prism": ("NIRSpec Prism",
                      {"detector": {"subarray": "sub512",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g395h": ("NIRSpec G395H",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g235h": ("NIRSpec G235H",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "niriss_soss": ("NIRISS SOSS",
                    {"detector": {"subarray": "substrip256",
                                  "readout_pattern": "nisrapid",
                                  "readmode": "nisrapid"}}),
    "nircam_f322w2": ("NIRCam F322W2",
                      {"instrument": {"filter": "f322w2"}}),
    "nircam_f444w": ("NIRCam F444W",
                     {"instrument": {"filter": "f444w"}}),
    "miri_lrs": ("MIRI LRS",
                 {"detector": {"readout_pattern": "fastr1",
                               "readmode": "fastr1"}}),
}

def run_ours(star: dict, keys: list[str]) -> dict:
    # noise_job now resolves engine-generation mode renames (NIRCam
    # ssgrism->lw_tsgrism on the 2026 engine) via instruments.engine_mode(), so
    # parity exercises the SAME production path as a normal run -- no separate
    # rename here (that was the bug: the old parity-only patch let NIRCam pass
    # the gate while the production path silently sent the rejected token).
    job = noise.noise_job(star, keys, sat_limit=SAT_LIMIT)
    eng = noise.backend_fingerprint()["engine_version"]
    if not eng.startswith("2026"):
        raise SystemExit(
            f"run_parity: JWST_TOOL_PANDEIA_PYTHON resolves engine {eng!r}; "
            "the parity gate compares on the CURRENT backend (2026.x)")
    return noise.run_pandeia(job, progress=lambda s: print("  " + s, flush=True))


def run_pandexo(star: dict, keys: list[str], workdir: Path,
                tag: str = "") -> dict:
    job = {
        "refdata": os.environ["JWST_TOOL_PANDEIA_REFDATA"],
        "psf_dir": os.environ["JWST_TOOL_PANDEIA_PSF_DIR"],
        "cdbs": ins.PYSYN_CDBS,
        "vega_file": str(Path(ins.PYSYN_CDBS) / "calspec"
                         / "alpha_lyr_stis_011.fits"),
        "star": {"teff": star["teff"], "logg": star["log_g"],
                 "metal": star["metallicity"], "kmag": star["ks_mag"]},
        "transit_duration_s": T_TRANSIT_S,
        "sat_level_pct": SAT_LIMIT * 100.0,
        "depth": DEPTH,
        "modes": [{"key": k, "pandexo_name": PANDEXO_MODES[k][0],
                   "config_overrides": PANDEXO_MODES[k][1]} for k in keys],
    }
    jf = workdir / f"{tag}pandexo_job.json"
    rf = workdir / f"{tag}pandexo_result.json"
    if rf.exists() and os.environ.get("PARITY_REUSE_PANDEXO") == "1":
        prev = json.loads(rf.read_text())
        prev_job = json.loads(jf.read_text()) if jf.exists() else None
        if prev_job == job:
            print(f"  [pandexo] REUSING {rf} (PARITY_REUSE_PANDEXO=1, "
                  "identical job)", flush=True)
            return prev
        print("  [pandexo] job changed; re-running despite "
              "PARITY_REUSE_PANDEXO=1", flush=True)
    jf.write_text(json.dumps(job))
    r = subprocess.run([ins.PICASO_PYTHON, str(HERE / "pandexo_worker.py"),
                        str(jf), str(rf)], text=True, capture_output=True)
    print(r.stdout)
    if r.returncode != 0 or not rf.exists():
        raise SystemExit(f"pandexo worker failed (rc={r.returncode}):\n"
                         f"{r.stderr[-3000:]}")
    return json.loads(rf.read_text())


def _stats(ratio: np.ndarray) -> dict:
    r = ratio[np.isfinite(ratio)]
    if r.size == 0:
        return dict(n=0)
    return dict(n=int(r.size), median=float(np.median(r)),
                p05=float(np.percentile(r, 5)),
                p95=float(np.percentile(r, 95)),
                max_abs_dev=float(np.max(np.abs(r - 1.0))))


def compare_mode(key: str, ours: dict, px: dict) -> dict:
    out = {"key": key}
    # a FAILED side carries a traceback STRING under "error"; a successful
    # PandExo mode also has an "error" key -- its sigma ARRAY -- so the
    # failure test must be on the type, not key presence
    if isinstance(ours.get("error"), str) or isinstance(px.get("error"), str):
        out["status"] = "ERROR"
        out["ours_error"] = str(ours.get("error", ""))[-400:]
        out["pandexo_error"] = str(px.get("error", ""))[-400:]
        return out
    if ours.get("unusable"):
        out["status"] = "SATURATED"
        out["ours_reason"] = ours["reason"]
        out["pandexo_ngroup"] = px.get("ngroup")
        out["pandexo_warnings"] = px.get("warnings")
        return out

    wl_o = np.asarray(ours["wl"])
    flux_o = np.asarray(ours["flux"])
    noise_o = np.asarray(ours["noise_1int"])
    o = np.argsort(wl_o)             # MIRI LRS disperses red-to-blue; sort so
    wl_o, flux_o, noise_o = wl_o[o], flux_o[o], noise_o[o]  # searchsorted works
    wl_p = np.asarray(px["wave"])
    err_p = np.asarray(px["error"])
    tim = px["timing"]

    # grid identity: match pixels by wavelength (the two sides run the same
    # engine, so the extraction grids should agree exactly; PandExo NaNs
    # fully saturated pixels, ours drops non-finite ones)
    ok_p = np.isfinite(err_p) & (err_p > 0)
    ii = np.searchsorted(wl_o, wl_p[ok_p])
    ii = np.clip(ii, 0, wl_o.size - 1)
    exact = np.abs(wl_o[ii] - wl_p[ok_p]) < 1e-9 * np.maximum(wl_p[ok_p], 1e-9)
    io, ip = ii[exact], np.where(ok_p)[0][exact]

    # integration counts: our floor policy vs PandExo's timing
    t_cyc = float(ours["t_cycle_s"])
    n_ours = int(T_TRANSIT_S / t_cyc)
    n_p_in = float(tim["Num Integrations In Transit"])
    n_p_out = float(tim["Num Integrations Out of Transit"])

    def sigma_ours(n_in, n_out):
        return (noise_o[io] / flux_o[io]) * np.sqrt(1.0 / n_in + 1.0 / n_out)

    # noise-model attribution: per-integration variance over pure photon
    # counts (photon-limited == 1.0). Ours uses pandeia's full extracted
    # noise; PandExo's default "fml" formula is analytic ramp noise.
    tm = float(tim["Measurement Time per Integration (sec)"])
    excess_ours = float(np.median(noise_o[io] ** 2 * tm / flux_o[io]))
    e_out = np.asarray(px["electrons_out"])[ip]
    v_out = np.asarray(px["var_out"])[ip]
    excess_px = float(np.median(v_out[e_out > 0] / e_out[e_out > 0])) \
        if (e_out > 0).any() else float("nan")

    out.update(
        status="OK",
        npix_ours=int(wl_o.size), npix_pandexo=int(wl_p.size),
        npix_matched=int(io.size),
        ngroup_ours=int(ours["ngroup"]), ngroup_pandexo=int(px["ngroup"]),
        sat_frac_ours=float(ours["sat_frac"]),
        t_int_ours_s=t_cyc,
        t_int_pandexo_s=float(tim["Time/Integration incl reset (sec)"]),
        t_frame_pandexo_s=float(tim["Seconds per Frame"]),
        n_int_ours=n_ours, n_int_pandexo_in=n_p_in, n_int_pandexo_out=n_p_out,
        config_ours={
            "subarray": ins.MODES[key]["config"].get(
                "detector", {}).get("subarray"),
            "readout": ins.MODES[key]["config"].get(
                "detector", {}).get("readout_pattern"),
            "filter": ins.MODES[key]["config"].get(
                "instrument", {}).get("filter"),
            "disperser": ins.MODES[key]["config"].get(
                "instrument", {}).get("disperser")},
        config_pandexo={
            "subarray": px["config"]["detector"].get("subarray"),
            "readout": px["config"]["detector"].get("readout_pattern"),
            "mode": px["config"]["instrument"].get("mode"),
            "filter": px["config"]["instrument"].get("filter"),
            "disperser": px["config"]["instrument"].get("disperser")},
        flux_ratio=_stats(flux_o[io] / np.asarray(px["e_rate_out"])[ip]),
        sigma_ratio_matched=_stats(sigma_ours(n_p_in, n_p_out) / err_p[ip]),
        sigma_ratio_policy=_stats(sigma_ours(n_ours, n_ours) / err_p[ip]),
        var_excess_ours=excess_ours, var_excess_pandexo=excess_px,
        pandexo_warnings=px.get("warnings"),
    )
    return out


def main():
    # raw per-run JSON goes in tests/parity/outputs/ (git-ignored there,
    # alongside the committed parity_summary.json and REPORT.md)
    out_root = OUTPUTS
    out_root.mkdir(parents=True, exist_ok=True)
    keys = list(PANDEXO_MODES)
    summary = {"stars": {}, "config": dict(
        transit_duration_s=T_TRANSIT_S, depth=DEPTH, sat_limit=SAT_LIMIT,
        stars=STARS)}
    for sname, star in STARS.items():
        print(f"=== {sname}: jwst_tool worker (current backend) ===", flush=True)
        ours = run_ours(star, keys)
        (out_root / f"{sname}_ours.json").write_text(json.dumps(ours))
        print(f"=== {sname}: PandExo ===", flush=True)
        px = run_pandexo(star, keys, out_root, tag=f"{sname}_")
        (out_root / f"{sname}_pandexo.json").write_text(json.dumps(px))
        rows = [compare_mode(k, ours.get(k, {"error": "missing"}),
                             px.get(k, {"error": "missing"})) for k in keys]
        summary["stars"][sname] = {
            "provenance_ours": ours.get("__provenance__"),
            "provenance_pandexo": px.get("__provenance__"),
            "modes": rows,
        }
        (OUTPUTS / "parity_summary.json").write_text(
            json.dumps(summary, indent=1))
        print(f"=== {sname}: done ===", flush=True)
    print(f"summary -> {OUTPUTS / 'parity_summary.json'}")


if __name__ == "__main__":
    main()
