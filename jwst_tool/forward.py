"""Forward model runner: VULCAN-JAX photochemistry -> ExoJax transmission spectrum.

Two faces:

* Imported by the GUI (light): ``params_key`` / ``cache_path`` / ``load_result``
  touch only the disk cache -- no JAX, no VULCAN, no ExoJax imports.
* Run as a script (heavy):  ``python jwst_tool/forward.py params.json``
  runs the live pipeline (~2-3 min on this machine: chemistry build+warmup ~50 s,
  RT build ~10 s, ~40 s per chemistry solve, ~10 s per RT eval) and writes the
  npz cache entry. Progress goes to stdout as "[fwd] ..." lines for the GUI.

Atmosphere-structure knobs (all consumed by the same validated pipeline hooks the
retrieval framework uses):

    T-P profile (tp_mode):
      "baseline"    WASP-39b GCM T(P) + uniform shift dT          (tp_eval=None)
      "isothermal"  T(P) = T_iso                                  (tp_eval hook)
      "guillot"     ExoJax atmprof_Guillot(Tirr, Tint, log10 kappa, log10 gamma)
                    with f=0.25 and the W39b surface gravity      (tp_eval hook)
    Kzz:
      kzz_mode "scale"  baseline GCM Kzz profile x kzz_x
      kzz_mode "const"  constant Kzz = kzz_const cm^2/s (cfg_overrides
                        Kzz_prof="const"), further x kzz_x if given
    Composition:
      met_x_solar  metallicity in x solar -> lnZ = ln(met/10) about the 10x baseline
      dco          Delta ln(C/O) carbon-enrichment proxy

Any T-P that leaves the modelable premodit window [320, 2980] K on either grid is
REJECTED with a clear error (never clipped) -- same rule as the retrieval.

Fisher machinery: with ``fisher_params`` set, the runner also computes the
spectrum Jacobian d(depth)/d(param) with one warm-started forward-mode jvp per
parameter (the validated sensitivity pattern: continuation from the converged
column, photochemistry ON), plus an RT-only lnR0 (reference-radius nuisance)
column. ``fisher.py`` turns that + the Pandeia noise into parameter forecasts.

Per-molecule "removed" spectra (for detection significance) zero that molecule's
VMR in the RT only -- atmospheric structure (T, mmw) is kept, the standard
nested-model comparison used in observation planning.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = TOOL_DIR.parent
MODEL_CACHE = BUNDLE_DIR / "data" / "jwst_tool" / "model_cache"

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]   # the WIDE-profile set
_VERSION = 2   # bump to invalidate all cached spectra

# Modelable temperature window (premodit table range, 20 K inset) -- reject, never clip.
T_WINDOW = (320.0, 2980.0)

# Parameters that can be freed in the Fisher forecast, per tp_mode.
CHEM_PARAM_NAMES = ["lnZ", "dlnCO", "lnKzz"]
TP_PARAM_NAMES = {
    "baseline": ["dT"],
    "isothermal": ["T_iso"],
    "guillot": ["Tirr", "Tint", "log_kappa", "log_gamma"],
}
# Display units for the GUI's constraint table.
PARAM_UNITS = {"lnZ": "dex(Z)", "dlnCO": "ln(C/O)", "lnKzz": "dex(Kzz)",
               "dT": "K", "T_iso": "K", "Tirr": "K", "Tint": "K",
               "log_kappa": "dex", "log_gamma": "dex"}


def canonical_params(params: dict) -> dict:
    tp_mode = str(params.get("tp_mode", "baseline"))
    cp = {
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        "dco": round(float(params.get("dco", 0.0)), 4),
        "kzz_mode": str(params.get("kzz_mode", "scale")),
        "kzz_x": round(float(params.get("kzz_x", 1.0)), 4),
        "kzz_const": round(float(params.get("kzz_const", 1.0e9)), 1),
        "tp_mode": tp_mode,
        "dT": round(float(params.get("dT", 0.0)), 2),
        "T_iso": round(float(params.get("T_iso", 1100.0)), 2),
        "Tirr": round(float(params.get("Tirr", 1560.0)), 2),
        "Tint": round(float(params.get("Tint", 100.0)), 2),
        "log_kappa": round(float(params.get("log_kappa", -2.3)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        "fisher_params": sorted(str(p) for p in (params.get("fisher_params") or [])),
        "version": _VERSION,
    }
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "baseline":
        cp["dT"] = 0.0
    if tp_mode != "isothermal":
        cp["T_iso"] = 0.0
    if tp_mode != "guillot":
        cp["Tirr"] = cp["Tint"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
    if cp["kzz_mode"] != "const":
        cp["kzz_const"] = 0.0
    bad = set(cp["fisher_params"]) - set(CHEM_PARAM_NAMES + TP_PARAM_NAMES[tp_mode])
    if bad:
        raise ValueError(f"fisher_params {sorted(bad)} not available for tp_mode={tp_mode}")
    return cp


def params_key(params: dict) -> str:
    s = json.dumps(canonical_params(params), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict) -> Path:
    return MODEL_CACHE / f"{params_key(params)}.npz"


def load_result(params: dict):
    """Cached spectrum dict or None. Keys: wl_um, depth, mols, depth_wo (nmol, n_nu),
    and (if Fisher was requested) jac (n_par, n_nu) + jac_names."""
    p = cache_path(params)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _build_tp(cp: dict, gs_cgs: float):
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable); None for baseline mode
    (where the runner's own validated uniform-shift knob theta[3]=dT is used).
    """
    import jax.numpy as jnp

    mode = cp["tp_mode"]
    if mode == "baseline":
        return None, 0, [cp["dT"]], CHEM_PARAM_NAMES + ["dT"]
    if mode == "isothermal":
        def tp_eval(tp, p_bar):
            return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]
        return tp_eval, 1, [cp["T_iso"]], CHEM_PARAM_NAMES + ["T_iso"]
    if mode == "guillot":
        from exojax.atm.atmprof import atmprof_Guillot

        def tp_eval(tp, p_bar):
            p = jnp.asarray(p_bar)
            Tirr, Tint = tp[0], tp[1]
            kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
            return atmprof_Guillot(p, gs_cgs, kappa, gamma, Tint, Tirr, 0.25)
        vals = [cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]]
        return tp_eval, 4, vals, CHEM_PARAM_NAMES + TP_PARAM_NAMES["guillot"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def run_model(params: dict, log=print) -> Path:
    sys.path.insert(0, str(BUNDLE_DIR))
    import config                                    # noqa: E402
    import vulcan_chem                               # noqa: E402  (env + x64 first)
    import jax                                       # noqa: E402
    import jax.numpy as jnp                          # noqa: E402
    import exojax_rt                                 # noqa: E402
    import interp_map                                # noqa: E402

    cp = canonical_params(params)
    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, config.GS_CGS)
    theta = np.array([np.log(cp["met_x_solar"] / 10.0), cp["dco"],
                      np.log(cp["kzz_x"])] + tp_vals, dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")

    profile = dict(config.WIDE)
    profile["reanchor_atom_ini"] = True   # finite-Z steps must re-anchor atom totals
    # step-size cap, validated state-preserving (retrieval case.py): prevents the
    # adaptive-dt ballooning non-convergence at high Kzz the GUI sliders can reach
    profile["dt_max"] = 1.0e11
    if cp["kzz_mode"] == "const":
        # constant eddy-diffusion profile via the same cfg_overrides hook the
        # fisher_zco tiers use; lnKzz (theta[2]) still multiplies it on-graph.
        profile["cfg_overrides"] = {"Kzz_prof": "const", "const_Kzz": cp["kzz_const"]}

    t0 = time.time()
    log("[fwd] building chemistry model (VULCAN-JAX warm-up ~50 s) ...")
    chem = vulcan_chem.build_chem_model(profile, tp_eval=tp_eval, n_tp_params=n_tp)
    log(f"[fwd] chemistry ready in {time.time()-t0:.0f} s")

    # --- T-P validity: REJECT (never clip) out-of-window profiles ------------
    if tp_eval is not None:
        T_check = np.asarray(tp_eval(jnp.asarray(theta[3:]), jnp.asarray(chem.p_bar)))
    else:
        T_check = np.asarray(chem.T_base) + cp["dT"]
    tmin, tmax = float(T_check.min()), float(T_check.max())
    if tmin < T_WINDOW[0] or tmax > T_WINDOW[1]:
        raise RuntimeError(
            f"T-P profile leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K (min {tmin:.0f} K, max {tmax:.0f} K). "
            "Adjust the profile parameters -- out-of-window layers are rejected, "
            "not clipped (opacity tables end there).")
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")

    t0 = time.time()
    log("[fwd] building ExoJax RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time()-t0:.0f} s")

    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
    mol_cols = {k: chem.sidx[config.MOLECULES[k]["vulcan"]] for k in rt.molecules}
    h2 = chem.sidx["H2"]
    he = chem.sidx["He"]
    p_art_j = jnp.asarray(rt.p_art_bar)

    def art_T(th):
        if tp_eval is None:
            return to_art(jnp.asarray(chem.T_base) + th[3])
        return tp_eval(th[3:], p_art_j)

    def depth_from_y(y, th, lnR0=0.0, drop_mol=None):
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        T_art = art_T(th)
        mmw_art = to_art(ymix @ chem.species_masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in mol_cols.items()}
        if drop_mol is not None:
            vmr[drop_mol] = jnp.zeros_like(vmr[drop_mol])
        return rt.transmission_depth_r(
            vmr, to_art(ymix[:, h2]), T_art, mmw_art, jnp.asarray(lnR0),
            vmr_he=to_art(ymix[:, he]))

    # --- chemistry: two-stage for composition steps (validated pattern) ------
    t0 = time.time()
    th0 = jnp.asarray(theta)
    def _check_converged(ac, stage):
        ac = int(ac)
        if ac >= int(chem.count_max):
            raise RuntimeError(
                f"chemistry did NOT converge ({stage}: {ac} accepted steps hit the "
                f"count_max={chem.count_max} cap). This parameter corner has no "
                "certified steady state -- adjust T-P / Kzz / composition rather "
                "than trusting an unconverged spectrum.")

    if cp["met_x_solar"] == 10.0 and cp["dco"] == 0.0:
        log("[fwd] solving chemistry (single stage, baseline composition) ...")
        y_sol, ac = chem.converged_y(th0, return_diag=True)
        _check_converged(ac, "single stage")
    else:
        log("[fwd] solving chemistry stage 1/2 (T/Kzz relaxation) ...")
        th_relax = th0.at[0].set(0.0).at[1].set(0.0)
        y_relaxed, ac1 = chem.converged_y(th_relax, return_diag=True)
        _check_converged(ac1, "stage 1, T/Kzz relaxation")
        log(f"[fwd] stage 1 done ({time.time()-t0:.0f} s); "
            "stage 2/2 (composition, warm continuation) ...")
        y_sol, ac2 = chem.converged_y(th0, warm_y=y_relaxed, lnZ_ref=0.0,
                                      c_o_ref=0.0, return_diag=True)
        _check_converged(ac2, "stage 2, composition continuation")
    y_np = np.asarray(y_sol)
    if not np.all(np.isfinite(y_np)):
        raise RuntimeError("chemistry solve returned non-finite abundances -- "
                           "parameter set outside the modelable range")
    log(f"[fwd] chemistry solved in {time.time()-t0:.0f} s total")

    # --- RT: full spectrum + one spectrum per removed molecule ---------------
    t0 = time.time()
    log("[fwd] radiative transfer: full spectrum (jit compile on first call) ...")
    depth = np.asarray(depth_from_y(y_sol, th0))
    log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

    depth_wo = np.zeros((len(MOLECULES), depth.shape[0]))
    for i, mol in enumerate(MOLECULES):
        t1 = time.time()
        depth_wo[i] = np.asarray(depth_from_y(y_sol, th0, drop_mol=mol))
        log(f"[fwd] spectrum without {mol} in {time.time()-t1:.0f} s")

    # --- Fisher Jacobian: warm-started jvp per free parameter + lnR0 ---------
    jac_names = list(cp["fisher_params"])
    jac = np.zeros((len(jac_names) + 1, depth.shape[0])) if jac_names else None
    if jac_names:
        lnZ0, co0 = float(theta[0]), float(theta[1])

        def f_theta(th):
            # continuation from the converged column: primal is a no-op re-converge,
            # the jvp is the validated warm-started steady-state tangent
            y = chem.converged_y(th, warm_y=y_sol, lnZ_ref=lnZ0, c_o_ref=co0)
            return depth_from_y(y, th)

        for j, name in enumerate(jac_names):
            t1 = time.time()
            i_par = theta_names.index(name)
            e = np.zeros_like(theta)
            e[i_par] = 1.0
            _, dd = jax.jvp(f_theta, (th0,), (jnp.asarray(e),))
            jac[j] = np.asarray(dd)
            log(f"[fwd] Jacobian d(depth)/d({name}) in {time.time()-t1:.0f} s")

        t1 = time.time()
        _, dd = jax.jvp(lambda r: depth_from_y(y_sol, th0, lnR0=r),
                        (jnp.asarray(0.0),), (jnp.asarray(1.0),))
        jac[-1] = np.asarray(dd)
        jac_names.append("lnR0")
        log(f"[fwd] Jacobian d(depth)/d(lnR0) [RT-only nuisance] in {time.time()-t1:.0f} s")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params)
    ymix_np = y_np / y_np.sum(axis=1, keepdims=True)
    arrays = dict(
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        depth=depth, depth_wo=depth_wo,
        mols=np.array(MOLECULES, dtype="U8"),
        ymix=ymix_np, p_bar=np.asarray(chem.p_bar),
        T=np.asarray(T_check), theta=theta,
        theta_names=np.array(theta_names, dtype="U16"),
        params_json=np.array(json.dumps(cp), dtype="U1024"),
    )
    if jac is not None:
        arrays["jac"] = jac
        arrays["jac_names"] = np.array(jac_names, dtype="U16")
    np.savez_compressed(out, **arrays)
    log(f"[fwd] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
