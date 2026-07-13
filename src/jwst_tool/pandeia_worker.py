"""Pandeia ETC worker -- runs INSIDE the picaso_base conda env (pandeia.engine 3.0).

Standalone on purpose: no imports from the rest of the tool, stdlib + pandeia only.

    python pandeia_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata path>, "cdbs": <PYSYN_CDBS path>,
     "star": {"teff":.., "log_g":.., "metallicity":.., "ks_mag":..},
     "sat_limit": 0.80,
     "modes": [{"key":.., "instrument":.., "mode":.., "config": {...},
                "strategy": {...}, "ngroup_min":.., "ngroup_max":..}, ...]}

result.json: one entry per mode key, plus a reserved "__provenance__" entry
(engine/refdata/python versions -- the exact backend identity of this run;
written before any mode runs so even an all-failed result is attributable).
Per mode key:
    {"wl": [...um], "flux": [...e-/s], "noise_1int": [...e-/s, sigma for 1 integ],
     "n_part_sat": [...], "n_full_sat": [...],   per-pixel saturated-group counts
     "r_native": [...] | null,      native resolving power R(lambda) on the wl
                                    grid (refdata dispersion file; null when the
                                    disperser has no such file -- host then skips
                                    the LSF blur, safe for high-R modes only),
     "r_native_source": "..",
     "t_cycle_s": .., "ngroup": .., "sat_frac": .., "sat_ngroups": ..,
     "saturated": bool, "engine_version": "..",
     "warnings": {...}}      -- or {"error": "..."} if that mode failed.

Star normalization (worker_version >= 4): band-integrated synphot photsys
normalization to the 2MASS Ks magnitude in vegamag ("2mass,ks" bandpass from
the minimal CDBS comp/nonhst tree, Vega = local CALSPEC alpha_lyr) -- the same
convention as the STScI web ETC. The old at_lambda shortcut (monochromatic
666.7 Jy zero point AT 2.159 um) mis-scales the flux by ~1-4% depending on
spectral type (CO bandhead for cool stars, Brackett-gamma wing for warm ones;
Cohen 2003 isophotal calibration holds only for A0V shapes), which fed
saturation/ngroup choices at full amplitude.

Group selection: probe at ngroup_min, form two candidates -- the PandExo-style
linear extrapolation of the brightest-pixel full-well fraction, and pandeia's
own scalar sat_ngroups (min groups-to-saturation over extraction pixels) scaled
by sat_limit -- take the smaller, then VERIFY: run the chosen ngroup and step
down until the measured fraction_saturation is actually under sat_limit (the
linear assumption alone can overshoot). If even ngroup_min busts the limit the
mode is flagged saturated (kept, with its degraded ngroup_min numbers, so the
GUI can say WHY it's bad). Channel-level saturation comes from the report's 1d
n_partial_saturated / n_full_saturated curves so the host can exclude or flag
affected pixels instead of trusting one mode-wide boolean.
"""
import copy
import glob
import json
import math
import os
import sys
import traceback

import numpy as np


def _make_calc(build_default_calc, m, star):
    calc = build_default_calc("jwst", m["instrument"], m["mode"])
    for section, kv in (m.get("config") or {}).items():
        calc["configuration"][section].update(kv)
    calc["configuration"]["detector"]["nint"] = 1
    calc["configuration"]["detector"]["nexp"] = 1
    for k, v in (m.get("strategy") or {}).items():
        calc["strategy"][k] = v
    # sky background model (pandeia's default is "minzodi"/"benchmark"); the
    # registry pins PandExo's TSO convention ("ecliptic" at "medium"). BOTH
    # keys are required together -- the engine resolves them to one canned
    # file (<background>_<level>.fits) and fails on a partial pair.
    if m.get("background"):
        calc["background"] = m["background"]
        if not m.get("background_level"):
            raise ValueError(
                f"mode {m['key']}: background={m['background']!r} without "
                "background_level -- the engine needs both (e.g. 'medium')")
        calc["background_level"] = m["background_level"]
    calc["scene"][0]["spectrum"]["sed"] = {
        "sed_type": "phoenix", "teff": float(star["teff"]),
        "log_g": float(star["log_g"]), "metallicity": float(star["metallicity"])}
    # band-integrated 2MASS Ks normalization in vegamag (web-ETC convention);
    # see module docstring for why the at_lambda shortcut was retired
    calc["scene"][0]["spectrum"]["normalization"] = {
        "type": "photsys", "bandpass": "2mass,ks",
        "norm_flux": float(star["ks_mag"]), "norm_fluxunit": "vegamag"}
    return calc


def _native_r(refdata, m, wl):
    """Native resolving power R(lambda) interpolated onto the extracted grid,
    from the mode's refdata dispersion file. Returns (list|None, source str).

    Disperser token: config disperser where present; MIRI LRS is the p750l
    prism; SOSS picks the order-specific file. Missing file -> (None, note):
    the host then applies no LSF blur, which is only safe for high-R modes --
    every low-R mode used here (PRISM, LRS, SOSS) has a dispersion file in
    pandeia_data (verified for 3.0rc3).
    """
    disp = (m.get("config", {}).get("instrument", {}) or {}).get("disperser")
    if m["instrument"] == "miri" and m["mode"] == "lrsslitless":
        disp = "p750l"
    if m["instrument"] == "niriss" and m["mode"] == "soss":
        order = int((m.get("strategy") or {}).get("order", 1))
        disp = f"gr700xd-ord{order}"
    if not disp:
        return None, "no disperser token for this mode"
    pat = os.path.join(refdata, "jwst", m["instrument"], "dispersion",
                       f"*{disp}*disp*.fits")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None, f"no dispersion file matching {pat}"
    from astropy.io import fits
    with fits.open(hits[0]) as h:
        cols = {c.upper(): c for c in h[1].columns.names}
        if "R" not in cols or "WAVELENGTH" not in cols:
            return None, f"{os.path.basename(hits[0])} lacks WAVELENGTH/R columns"
        w = np.asarray(h[1].data[cols["WAVELENGTH"]], float)
        r = np.asarray(h[1].data[cols["R"]], float)
    order_ix = np.argsort(w)
    r_i = np.interp(np.asarray(wl, float), w[order_ix], r[order_ix])
    return r_i.tolist(), os.path.basename(hits[0])


def _clamp_ngroup(ng, ng_min, ng_max):
    """Clamp a candidate group count to the mode's supported [ng_min, ng_max].

    ng_max is the PandExo-compatible hard maximum for the mode (NIRCam grism =
    100; see instruments.PANDEXO_NGROUP_MAX). This is the ONE place the group
    optimizer's output is bounded, so a selected ramp can never exceed the
    supported range on a faint target (2026-07-12 audit item 5)."""
    return max(int(ng_min), min(int(ng_max), int(ng)))


def _run(perform_calculation, calc, ngroup):
    c = copy.deepcopy(calc)
    c["configuration"]["detector"]["ngroup"] = int(ngroup)
    return perform_calculation(c)


def _sat_curve(rpt, key, n_pix):
    """Per-pixel saturated-group counts from the 1d report (zeros if absent or
    on a different grid -- additive diagnostic, never load-bearing)."""
    try:
        wave, curve = rpt["1d"][key]
        curve = np.asarray(curve, dtype=float)
        if curve.shape[0] == n_pix:
            return np.nan_to_num(curve, nan=0.0)
    except Exception:
        pass
    return np.zeros(n_pix)


def _one_mode(build_default_calc, perform_calculation, m, star, sat_limit,
              refdata):
    calc = _make_calc(build_default_calc, m, star)
    ng_min, ng_max = int(m["ngroup_min"]), int(m["ngroup_max"])

    probe = _run(perform_calculation, calc, ng_min)
    sat_probe = float(probe["scalar"]["fraction_saturation"])
    sat_ng = probe["scalar"].get("sat_ngroups")

    # candidate group counts: linear extrapolation of the probe's full-well
    # fraction, and pandeia's own groups-to-saturation estimate; take the
    # smaller (more conservative), then verify empirically below.
    cands = []
    if sat_probe > 0:
        cands.append(int(math.floor(ng_min * sat_limit / sat_probe)))
    if sat_ng is not None and np.isfinite(sat_ng) and sat_ng > 0:
        cands.append(int(math.floor(sat_limit * float(sat_ng))))
    ng_best = min(cands) if cands else ng_max
    saturated = ng_best < ng_min      # even the shortest ramp busts the limit
    ng_best = _clamp_ngroup(ng_best, ng_min, ng_max)

    rpt = probe if ng_best == ng_min else _run(perform_calculation, calc, ng_best)
    # verify the CHOSEN ramp: the linear full-well model can overshoot, so step
    # down until the measured saturation fraction actually respects the limit
    for _ in range(4):
        frac = float(rpt["scalar"]["fraction_saturation"])
        if frac <= sat_limit or ng_best <= ng_min:
            break
        ng_best = max(ng_min, min(ng_best - 1,
                                  int(math.floor(ng_best * sat_limit / frac))))
        rpt = _run(perform_calculation, calc, ng_best)
    if float(rpt["scalar"]["fraction_saturation"]) > sat_limit:
        saturated = True

    wl, _sn = rpt["1d"]["sn"]
    flux = np.asarray(rpt["1d"]["extracted_flux"][1], dtype=float)
    noise = np.asarray(rpt["1d"]["extracted_noise"][1], dtype=float)
    wl = np.asarray(wl, dtype=float)
    good = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(noise) & (flux > 0) & (noise > 0)

    if not good.any():
        # pandeia returned no usable pixels (e.g. NIRSpec PRISM on a bright star:
        # saturation within the shortest ramp NaNs the extracted noise everywhere)
        return {
            "unusable": True,
            "reason": (f"no unsaturated pixels at the shortest ramp "
                       f"(ngroup={ng_best}, max full-well fraction "
                       f"{float(rpt['scalar']['fraction_saturation']):.1f}) -- "
                       "target too bright for this mode"),
            "ngroup": int(ng_best),
            "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
            "saturated": True,
            "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
        }

    n_part = _sat_curve(rpt, "n_partial_saturated", wl.shape[0])
    n_full = _sat_curve(rpt, "n_full_saturated", wl.shape[0])
    r_native, r_src = _native_r(refdata, m, wl[good])

    import pandeia.engine
    return {
        "wl": wl[good].tolist(),
        "flux": flux[good].tolist(),
        "noise_1int": noise[good].tolist(),
        "n_part_sat": n_part[good].tolist(),
        "n_full_sat": n_full[good].tolist(),
        "r_native": r_native,
        "r_native_source": r_src,
        "t_cycle_s": float(rpt["scalar"]["total_exposure_time"]),
        "ngroup": int(ng_best),
        "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
        "sat_ngroups": (float(sat_ng) if sat_ng is not None
                        and np.isfinite(sat_ng) else None),
        "saturated": bool(saturated),
        "engine_version": str(getattr(pandeia.engine, "__version__", "unknown")),
        "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
    }


def _release(version):
    """Leading dotted-numeric release segment of a version string
    ("3.0rc3" -> "3.0", "2026.2" -> "2026.2"); None if it has none."""
    import re
    m = re.match(r"(\d+(?:\.\d+)*)", str(version).strip())
    return m.group(1) if m else None


def _refdata_version(refdata):
    """Best-available refdata version: the VERSION file (some pandeia_data
    releases), VERSION_DATA (the 2026+ split data trees), else VERSION_PSF's
    first line (3.0-era trees have only that), else the pandeia_data-<ver>
    directory name. Returns (version|None, source)."""
    for name in ("VERSION", "VERSION_DATA", "VERSION_PSF"):
        p = os.path.join(refdata, name)
        if os.path.isfile(p):
            with open(p) as f:
                first = f.readline().strip()
            if first:
                return first, name
    base = os.path.basename(os.path.normpath(refdata))
    if base.startswith("pandeia_data-"):
        return base[len("pandeia_data-"):], "directory name"
    return None, "no VERSION/VERSION_PSF file or pandeia_data-<ver> dir name"


def _check_backend_match(engine_version, refdata):
    """Enforce STScI's matching rule (engine and refdata versions must be the
    same release) BEFORE any calculation: a mismatched pair otherwise fails
    deep inside the engine (or worse, runs with wrong calibrations). Returns
    the provenance fields; raises RuntimeError on mismatch/undeterminable."""
    ref_ver, source = _refdata_version(refdata)
    if ref_ver is None:
        raise RuntimeError(
            f"cannot determine the pandeia_data version of {refdata} ({source}); "
            "refusing to run against unidentifiable reference data. Point "
            "JWST_TOOL_PANDEIA_REFDATA at an intact pandeia_data tree.")
    eng_rel, ref_rel = _release(engine_version), _release(ref_ver)
    if eng_rel is None or ref_rel is None or eng_rel != ref_rel:
        raise RuntimeError(
            f"pandeia.engine {engine_version} does not match pandeia_data "
            f"{ref_ver} (from {source}) at {refdata}. STScI requires the "
            "engine and refdata releases to be the same. Fix the pair: point "
            "JWST_TOOL_PANDEIA_PYTHON at an env whose engine matches "
            "JWST_TOOL_PANDEIA_REFDATA (this repo's validated pair is engine "
            "3.0 + pandeia_data-3.0rc3 in the picaso_base env).")
    return {"refdata_version": ref_ver, "refdata_version_source": source}


def _preflight(job):
    """Fail with the offending PATH, not a deep synphot traceback, when the
    reference trees are missing (e.g. a stale job/cache from an old layout)."""
    problems = []
    if not os.path.isdir(job["refdata"]):
        problems.append(f"pandeia_refdata does not exist: {job['refdata']}")
    psf_dir = job.get("psf_dir")
    if psf_dir:
        # split-layout (pandeia_data >= 2026) PSF library; must be an intact
        # tree, not just an existing directory
        if not os.path.isfile(os.path.join(psf_dir, "VERSION_PSF")):
            problems.append(
                f"PSF_DIR has no VERSION_PSF file: {psf_dir} (point "
                "JWST_TOOL_PANDEIA_PSF_DIR at the extracted pandeia_psfs tree)")
    cdbs = job["cdbs"]
    if not os.path.isdir(cdbs):
        problems.append(f"PYSYN_CDBS does not exist: {cdbs}")
    else:
        phx = os.path.join(cdbs, "grid", "phoenix")
        if not os.path.isdir(os.path.realpath(phx)):
            problems.append(
                f"PHOENIX grid missing or dangling symlink: {phx} "
                f"-> {os.path.realpath(phx)} (the star SED cannot be built)")
        for rel, why in (
                (os.path.join("comp", "nonhst", "2mass_ks_001_syn.fits"),
                 "the 2MASS Ks bandpass (photsys normalization)"),
                (os.path.join("calspec", "alpha_lyr_stis_011.fits"),
                 "the Vega spectrum (vegamag normalization)")):
            if not os.path.isfile(os.path.join(cdbs, rel)):
                problems.append(f"missing {rel} in PYSYN_CDBS -- {why}; fetch "
                                "it from https://ssb.stsci.edu/trds/")
    if problems:
        raise RuntimeError(
            "pandeia worker preflight failed:\n  " + "\n  ".join(problems)
            + "\n(check JWST_TOOL_PANDEIA_REFDATA / JWST_TOOL_DATA_DIR, or "
            "regenerate a stale cached job)")


def main():
    job = json.load(open(sys.argv[1]))
    _preflight(job)
    os.environ["pandeia_refdata"] = job["refdata"]
    os.environ["PYSYN_CDBS"] = job["cdbs"]
    if job.get("psf_dir"):
        os.environ["PSF_DIR"] = job["psf_dir"]
    import warnings as _w
    _w.filterwarnings("ignore")
    # synphot's default vega_file is an ssb.stsci.edu URL: point it at the
    # local CALSPEC copy so vegamag normalization works offline (preflighted)
    import synphot
    synphot.conf.vega_file = os.path.join(
        job["cdbs"], "calspec", "alpha_lyr_stis_011.fits")
    from pandeia.engine.calc_utils import build_default_calc
    from pandeia.engine.perform_calculation import perform_calculation

    import pandeia.engine
    engine_version = str(getattr(pandeia.engine, "__version__", "unknown"))
    match = _check_backend_match(engine_version, job["refdata"])
    out = {
        "__provenance__": {
            "engine_version": engine_version,
            "refdata_path": job["refdata"],
            "refdata_name": os.path.basename(os.path.normpath(job["refdata"])),
            **match,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "worker_version": job.get("worker_version"),
        },
    }
    print(f"[pandeia] engine {engine_version} + {out['__provenance__']['refdata_name']} "
          f"(refdata {match['refdata_version']} from {match['refdata_version_source']})",
          flush=True)
    for m in job["modes"]:
        key = m["key"]
        print(f"[pandeia] {key} ...", flush=True)
        try:
            out[key] = _one_mode(build_default_calc, perform_calculation,
                                 m, job["star"], float(job.get("sat_limit", 0.8)),
                                 job["refdata"])
            if out[key].get("unusable"):
                print(f"[pandeia] {key}: UNUSABLE ({out[key]['reason']})", flush=True)
            else:
                print(f"[pandeia] {key}: ngroup={out[key]['ngroup']} "
                      f"sat={out[key]['sat_frac']:.2f} npix={len(out[key]['wl'])}",
                      flush=True)
        except Exception:
            out[key] = {"error": traceback.format_exc()}
            print(f"[pandeia] {key}: FAILED", flush=True)

    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    print("[pandeia] done", flush=True)


if __name__ == "__main__":
    main()
