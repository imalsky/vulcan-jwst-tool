"""Host-side noise interface: subprocess to the Pandeia worker + transit-depth error math.

The worker gives, per instrument mode, the per-native-pixel extracted stellar
flux and its 1-integration sigma. This module turns that into a transit-depth
uncertainty per spectral bin:

    depth = 1 - F_in/F_out
    var(depth)_pixel = (sigma_1int/flux)^2 * (1/n_int_in + 1/n_int_out) / n_transits
    var(depth)_bin   = sum(w^2 var_pixel) / (sum w)^2,  w = flux   (count-space)
    sigma_bin        = sqrt(var_bin + floor^2)           (floor does NOT average down)

The count-space combination is the variance of the SAME estimator detect.py
uses to bin the model (binning.build_operator) -- the sum-of-extracted-counts
bin depth every real reduction produces. The old inverse-variance combination
quoted the variance of a different (optimal-weights) estimator than the one
the model was binned with; in the photon-dominated limit the two agree.

What this is, and is not: an ETC RANDOM-NOISE LOWER BOUND under the selected
extraction/detector configuration, plus an editable systematic-floor scenario.
Pandeia's extracted noise includes photon (with the HgCdTe quantum-yield/Fano
excess variance below ~2 um), background, dark, and correlated read noise +
IPC -- it is NOT a time-series systematics model (1/f residuals, visit-long
trends, pointing/tilt events, limb-darkening and detrending covariance,
stellar heterogeneity). Real final uncertainties can only be equal or worse.

Results are cached by a hash of (star, modes, sat_limit, engine + refdata
versions) so the ETC runs once per star/instrument set and stale caches are
invalidated when the Pandeia backend changes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import binning
from . import instruments as ins

_BACKEND_FINGERPRINT = None


def backend_fingerprint() -> dict:
    """Pandeia backend identity baked into every cache key (queried once per
    process): engine version from the picaso_base python, refdata version-file
    contents. "unavailable" (still cache-key-stable) when the env is missing --
    run_pandeia raises loudly on an actual run attempt."""
    global _BACKEND_FINGERPRINT
    if _BACKEND_FINGERPRINT is not None:
        return _BACKEND_FINGERPRINT
    engine = "unavailable"
    py = Path(ins.PICASO_PYTHON)
    if py.exists():
        try:
            r = subprocess.run(
                [str(py), "-c",
                 "import pandeia.engine; print(pandeia.engine.__version__)"],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                engine = r.stdout.strip().splitlines()[-1]
        except Exception:
            pass
    ref = Path(ins.PANDEIA_REFDATA)
    refver = []
    for name in ("VERSION", "VERSION_PSF"):
        f = ref / name
        if f.exists():
            refver.append(f"{name}:{hashlib.sha1(f.read_bytes()).hexdigest()[:12]}")
    _BACKEND_FINGERPRINT = {
        "engine_version": engine,
        "refdata_name": ref.name,
        "refdata_version_files": sorted(refver),
    }
    return _BACKEND_FINGERPRINT


def noise_job(star: dict, mode_keys: list[str], sat_limit: float = 0.80) -> dict:
    modes = []
    for key in mode_keys:
        m = dict(ins.MODES[key])
        modes.append({
            "key": key, "instrument": m["instrument"], "mode": m["mode"],
            "config": m.get("config", {}), "strategy": m.get("strategy", {}),
            "ngroup_min": m["ngroup_min"], "ngroup_max": m["ngroup_max"],
        })
    return {
        "refdata": ins.PANDEIA_REFDATA, "cdbs": ins.PYSYN_CDBS,
        "backend": backend_fingerprint(),
        "star": {k: float(star[k]) for k in ("teff", "log_g", "metallicity", "ks_mag")},
        "sat_limit": float(sat_limit),
        "modes": modes,
        "worker_version": 3,   # cache-buster: bump when pandeia_worker output changes
    }


def job_key(job: dict) -> str:
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def run_pandeia(job: dict, progress=None, force: bool = False) -> dict:
    """Run the worker in picaso_base (or return the cached result).

    ``progress``: optional callable(str) receiving worker stdout lines live.
    Raises RuntimeError (loudly, with stderr) if the worker process itself dies;
    per-mode pandeia failures come back as {"error": traceback} entries.
    """
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ins.NOISE_CACHE / f"{job_key(job)}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    py = Path(ins.PICASO_PYTHON)
    if not py.exists():
        raise RuntimeError(
            f"Pandeia backend python not found at {py} (the picaso_base conda env "
            "with pandeia.engine 3.0). The noise model cannot run without it.")

    in_json = ins.NOISE_CACHE / f"{job_key(job)}.job.json"
    out_json = ins.NOISE_CACHE / f"{job_key(job)}.out.json"
    in_json.write_text(json.dumps(job))
    worker = ins.TOOL_DIR / "pandeia_worker.py"

    proc = subprocess.Popen([str(py), str(worker), str(in_json), str(out_json)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if progress:
            progress(line.rstrip())
    proc.wait()
    if proc.returncode != 0 or not out_json.exists():
        err = proc.stderr.read()
        raise RuntimeError(f"pandeia worker failed (rc={proc.returncode}):\n{err[-3000:]}")

    result = json.loads(out_json.read_text())
    cache.write_text(json.dumps(result))
    return result


def make_bins(wl_lo: float, wl_hi: float, R: float) -> np.ndarray:
    """Log-spaced bin EDGES at resolving power R over [wl_lo, wl_hi]."""
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


# The quoted per-mode systematic floors (instruments.MODES / Greene+2016-style
# numbers) are per-bin values AT R=100 bins. A real systematic is spectrally
# correlated, so it cannot be made to shrink by slicing the band into more bins:
# treating a fixed per-bin floor as white noise inflated a floor-dominated
# detection significance by ~sqrt(R_bin/100) when the bin slider moved 100->200.
# Anchoring here keeps the floor-limited information content of the band
# R-independent: per-bin floor = floor_ppm * sqrt(R_bin/R_REF) for finer bins.
# Coarser-than-reference bins KEEP the full floor (no sqrt averaging-down --
# conservative, systematics don't integrate out).
FLOOR_REF_R = 100.0


def pixel_depth_variance(mode_result: dict, t_in_s: float, t_out_s: float,
                         n_transits: int) -> np.ndarray:
    """Per-native-pixel transit-depth variance (box-depth approximation).

    var = (sigma_1int/flux)^2 (1/n_in + 1/n_out) / n_transits, with the
    integration counts from the mode's measured cycle time. Neglects
    ingress/egress, limb darkening, and depth-detrending covariance -- the
    fast lower-bound mode; a time-domain information calculation would need
    the full light-curve derivative set.
    """
    flux = np.asarray(mode_result["flux"])
    noise = np.asarray(mode_result["noise_1int"])
    t_cycle = float(mode_result["t_cycle_s"])
    n_in = max(1, int(t_in_s / t_cycle))
    n_out = max(1, int(t_out_s / t_cycle))
    return (noise / flux) ** 2 * (1.0 / n_in + 1.0 / n_out) / max(1, int(n_transits))


def depth_error_bins(mode_result: dict, edges: np.ndarray,
                     t_in_s: float, t_out_s: float, n_transits: int,
                     floor_ppm: float, op: dict | None = None) -> dict:
    """Per-bin transit-depth sigma from a worker mode result.

    ``op`` is the mode's count-space measurement operator (binning.build_operator);
    pass the SAME operator used to bin the model so noise and model describe one
    estimator. Built here from the pixel grid alone when omitted (noise-only use).

    Returns dict(wl_center, sigma, n_pix, var_phot, floor, n_transits) over the
    operator's kept bins. ``var_phot`` is the photon/detector bin variance AT the
    evaluated ``n_transits`` (it scales as 1/N); ``floor`` is the per-bin
    R-anchored systematic (N-independent; see FLOOR_REF_R). sigma =
    sqrt(var_phot + floor^2). Returning the two components separately is what
    lets callers extrapolate to other transit counts CORRECTLY -- a plain
    1/sqrt(N) scaling of sigma is optimistic wherever the floor contributes.
    """
    if op is None:
        op = binning.build_operator(np.asarray(mode_result["wl"]),
                                    np.asarray(mode_result["flux"]), edges)
    var_pix = pixel_depth_variance(mode_result, t_in_s, t_out_s, n_transits)
    var_phot = binning.bin_variance(op, var_pix)
    centers = 0.5 * (edges[:-1] + edges[1:])
    r_bin = (centers / np.diff(edges))[op["keep"]]
    floor = (floor_ppm * 1e-6) * np.sqrt(np.maximum(r_bin, FLOOR_REF_R) / FLOOR_REF_R)
    sigma = np.sqrt(var_phot + floor ** 2)
    return dict(wl_center=op["wl_center"], sigma=sigma, n_pix=op["n_pix"],
                var_phot=var_phot, floor=floor,
                n_transits=int(max(1, int(n_transits))))
