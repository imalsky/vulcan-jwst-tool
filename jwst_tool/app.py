"""JWST instrument selector -- Streamlit GUI.

    cd vulcan_exojax_run
    streamlit run jwst_tool/app.py

Pipeline per run: VULCAN-JAX photochemistry -> ExoJax transmission spectrum
(local subprocess, ~2-3 min per new parameter set, disk-cached) -> Pandeia ETC
noise per instrument mode (picaso_base subprocess, disk-cached) -> per-mode
detection significance of the target molecule + optional Fisher parameter
forecast from the autodiff Jacobian.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR.parent))

from jwst_tool import detect, fisher as fisher_mod, forward, noise as noise_mod
from jwst_tool import instruments as ins

st.set_page_config(page_title="JWST Instrument Selector", page_icon="🔭",
                   layout="wide")

st.title("JWST instrument selector")
st.caption(
    "VULCAN-JAX photochemistry → ExoJAX transmission spectrum → Pandeia ETC noise. "
    "Pick a science goal, run the model locally, and see which instrument mode "
    "detects it best. Baseline atmosphere: WASP-39b (10× solar, Tsai et al. 2023 setup)."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Star & transit")
    teff = st.number_input("T_eff (K)", 3000.0, 7000.0, ins.STAR_W39["teff"], 50.0)
    logg = st.number_input("log g", 3.5, 5.5, ins.STAR_W39["log_g"], 0.1)
    feh = st.number_input("[Fe/H]", -2.0, 0.5, ins.STAR_W39["metallicity"], 0.1)
    ks_mag = st.number_input("Ks mag (2MASS)", 4.0, 16.0, ins.STAR_W39["ks_mag"], 0.1)
    t14 = st.number_input("Transit duration T14 (hr)", 0.5, 10.0, ins.TRANSIT_T14_HR, 0.1)
    t_base = st.number_input("Out-of-transit baseline (hr)", 0.5, 10.0,
                             ins.TRANSIT_T14_HR, 0.1)

    st.header("Atmosphere structure")
    tp_mode = st.selectbox(
        "T-P profile", ["baseline", "isothermal", "guillot"], index=0,
        format_func={"baseline": "WASP-39b GCM profile + ΔT",
                     "isothermal": "Isothermal",
                     "guillot": "Guillot (2010)"}.get)
    tp_kwargs = {}
    if tp_mode == "baseline":
        tp_kwargs["dT"] = st.slider("ΔT (K, uniform shift)", -200.0, 200.0, 0.0, 25.0)
    elif tp_mode == "isothermal":
        tp_kwargs["T_iso"] = st.slider("T_iso (K)", 400.0, 2500.0, 1100.0, 25.0)
    else:
        tp_kwargs["Tirr"] = st.slider("T_irr (K)", 800.0, 2500.0, 1560.0, 20.0)
        tp_kwargs["Tint"] = st.slider("T_int (K)", 50.0, 500.0, 100.0, 25.0)
        tp_kwargs["log_kappa"] = st.slider("log₁₀ κ_IR (cm²/g)", -4.0, 0.0, -2.3, 0.1)
        tp_kwargs["log_gamma"] = st.slider("log₁₀ γ (κ_vis/κ_IR)", -2.0, 0.3, -1.0, 0.05)

    kzz_mode = st.radio("K_zz profile", ["scale", "const"], horizontal=True,
                        format_func={"scale": "GCM profile × factor",
                                     "const": "constant"}.get)
    if kzz_mode == "scale":
        kzz_x = st.select_slider("K_zz multiplier",
                                 options=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                                 value=1.0)
        kzz_const = 1.0e9
    else:
        log_kzz = st.slider("log₁₀ K_zz (cm²/s)", 6.0, 12.0, 9.0, 0.25)
        kzz_const, kzz_x = 10.0 ** log_kzz, 1.0

    st.header("Composition")
    met = st.select_slider("Metallicity (× solar)",
                           options=[1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0],
                           value=10.0)
    dco = st.slider("Δ ln(C/O) (carbon enrichment)", -0.5, 0.5, 0.0, 0.05)

    st.header("Science goal")
    target_mol = st.selectbox("Detect molecule", forward.MOLECULES,
                              index=forward.MOLECULES.index("SO2"))
    n_transits = st.slider("Number of transits", 1, 10, 1)
    r_bin = st.select_slider("Binned resolving power R", options=[50, 100, 200], value=100)
    mode_keys = st.multiselect(
        "Instrument modes",
        options=list(ins.MODES),
        default=list(ins.MODES),
        format_func=lambda k: ins.MODES[k]["label"])

    st.header("Fisher forecast")
    avail_free = forward.CHEM_PARAM_NAMES + forward.TP_PARAM_NAMES[tp_mode]
    do_fisher = st.checkbox("Parameter constraints (autodiff Jacobian)", value=False,
                            help="One warm-started forward-mode jvp per parameter "
                                 "(~20–60 s each) through the full chemistry+RT chain.")
    fisher_params = st.multiselect("Free parameters", avail_free,
                                   default=["lnZ", "dlnCO", "lnKzz"]) if do_fisher else []

    with st.expander("Advanced"):
        sat_limit = st.slider("Saturation limit (full-well fraction)", 0.5, 0.95, 0.80, 0.05)
        show_noise = st.checkbox("Show simulated noise realization", value=False)
        seed = st.number_input("Realization seed", 0, 9999, 0)
        st.markdown("**Systematic noise floors (ppm)**")
        floors = {k: st.number_input(ins.MODES[k]["label"], 0.0, 200.0,
                                     ins.MODES[k]["floor_ppm"], 5.0, key=f"floor_{k}")
                  for k in mode_keys}

params = dict(met_x_solar=met, dco=dco,
              kzz_mode=kzz_mode, kzz_x=kzz_x, kzz_const=kzz_const,
              tp_mode=tp_mode, fisher_params=fisher_params, **tp_kwargs)
star = dict(teff=teff, log_g=logg, metallicity=feh, ks_mag=ks_mag)

cached = forward.load_result(params) is not None
n_jvp = len(fisher_params)
est = "instant (cached)" if cached else (
    f"~{2.5 + 0.7 * n_jvp:.0f} min (local model run"
    + (f" + {n_jvp} Jacobian directions" if n_jvp else "") + ")")
col_btn, col_note = st.columns([1, 3])
run_clicked = col_btn.button("Run", type="primary", use_container_width=True)
col_note.caption(f"Model spectrum for these settings: **{est}**. "
                 "ETC noise is cached per star + instrument set.")


# ---------------------------------------------------------------------------
# Compute on click
# ---------------------------------------------------------------------------
def compute():
    if not mode_keys:
        st.error("Select at least one instrument mode.")
        return None

    model = forward.load_result(params)
    if model is None:
        with st.status("Running VULCAN-JAX + ExoJAX forward model locally …",
                       expanded=True) as status:
            pfile = forward.MODEL_CACHE / f"{forward.params_key(params)}.params.json"
            forward.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(forward.canonical_params(params)))
            proc = subprocess.Popen(
                [sys.executable, str(TOOL_DIR / "forward.py"), str(pfile)],
                cwd=str(TOOL_DIR.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            box = st.empty()
            lines = []
            for line in proc.stdout:
                lines.append(line.rstrip())
                box.code("\n".join(lines[-12:]))
            proc.wait()
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                st.error("Forward model failed:\n\n```\n" + "\n".join(lines[-25:]) + "\n```")
                return None
            status.update(label="Forward model done", state="complete")
        model = forward.load_result(params)
        if model is None:
            st.error("Forward model finished but produced no cache file.")
            return None

    job = noise_mod.noise_job(star, mode_keys, sat_limit=sat_limit)
    have_cache = (ins.NOISE_CACHE / f"{noise_mod.job_key(job)}.json").exists()
    if have_cache:
        etc = noise_mod.run_pandeia(job)
    else:
        with st.status("Running Pandeia ETC (STScI engine, picaso_base env) …",
                       expanded=True) as status:
            box = st.empty()
            lines = []
            etc = noise_mod.run_pandeia(job, progress=lambda s: (
                lines.append(s), box.code("\n".join(lines[-8:]))))
            status.update(label="Pandeia ETC done", state="complete")

    t_in_s, t_out_s = t14 * 3600.0, t_base * 3600.0
    results, failed, unusable = [], [], []
    for k in mode_keys:
        if "error" in etc[k]:
            failed.append((k, etc[k]["error"]))
        elif etc[k].get("unusable") or not etc[k].get("wl"):
            unusable.append((k, etc[k].get("reason", "no usable pixels")))
        else:
            results.append(detect.evaluate_mode(
                k, etc[k], model, target_mol, r_bin, t_in_s, t_out_s,
                n_transits, floors[k]))
    return dict(model=model, results=results, failed=failed, unusable=unusable,
                fisher_names=list(fisher_params))


if run_clicked:
    out = compute()
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(target=target_mol, n_transits=n_transits,
                                            show_noise=show_noise, seed=seed,
                                            r_bin=r_bin)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
if "out" not in st.session_state:
    st.info("Set the science goal in the sidebar and press **Run**.")
    st.stop()

out = st.session_state["out"]
meta = st.session_state["out_meta"]
model, results = out["model"], out["results"]

for k, err in out["failed"]:
    st.error(f"{ins.MODES[k]['label']}: Pandeia calculation failed — see details.")
    with st.expander(f"{ins.MODES[k]['label']} traceback"):
        st.code(err[-2500:])
for k, reason in out["unusable"]:
    st.warning(f"**{ins.MODES[k]['label']}: unusable on this star** — {reason}.")

if not results:
    st.stop()

ok = [r for r in results if not r["saturated"]]
ranked = sorted(ok or results, key=lambda r: -r["sigma_detect"])
best = ranked[0]

verdict = (f"**Best mode for detecting {meta['target']}: {best['label']}** — "
           f"{best['sigma_detect']:.1f}σ in {meta['n_transits']} transit"
           f"{'s' if meta['n_transits'] > 1 else ''} "
           f"(median precision {best['median_sigma_ppm']:.0f} ppm per "
           f"R={meta['r_bin']} bin).")
if best["sigma_detect"] >= 5:
    st.success(verdict)
elif best["sigma_detect"] >= 3:
    st.warning(verdict + "  Marginal (3–5σ): consider more transits.")
else:
    st.error(verdict + "  Below 3σ with this setup — more transits or a different goal.")

# --- spectrum figure -------------------------------------------------------
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6
mols = [str(x) for x in model["mols"]]
d_wo_s = model["depth_wo"][mols.index(meta["target"])][order] * 1e6

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
ax.plot(wl_s, d_s, color="#555555", lw=0.7, alpha=0.8, zorder=2,
        label="model (native R≈3000)")
ax.plot(wl_s, d_wo_s, color="#999999", lw=0.9, ls="--", zorder=1,
        label=f"model without {meta['target']}")
rng = np.random.default_rng(int(meta["seed"]))
for r in results:
    c = ins.MODE_COLOR[r["mode_key"]]
    y = r["depth"] * 1e6
    if meta["show_noise"]:
        y = y + rng.normal(0.0, r["sigma"] * 1e6)
    label = r["label"] + (" (saturated!)" if r["saturated"] else "")
    ax.errorbar(r["wl"], y, yerr=r["sigma"] * 1e6, fmt="o", ms=3.0, lw=1.0,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, zorder=3, label=label)
ax.set_xscale("log")
ticks = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
lo = min(min(r["wl"].min() for r in results), 1.0)
hi = max(r["wl"].max() for r in results)
ax.set_xlim(lo * 0.97, hi * 1.03)
sel = (wl_s >= lo * 0.97) & (wl_s <= hi * 1.03)
pad = 0.06 * (d_s[sel].max() - d_s[sel].min())
ax.set_ylim(d_s[sel].min() - pad, d_s[sel].max() + 3 * pad)
ax.set_xlabel("wavelength (μm)")
ax.set_ylabel("transit depth (ppm)")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# --- significance bar chart + table ---------------------------------------
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader(f"{meta['target']} detection significance")
    fig2, ax2 = plt.subplots(figsize=(5.5, 0.55 * len(results) + 1.2), dpi=150)
    rs = sorted(results, key=lambda r: r["sigma_detect"])
    names = [r["label"] + (" ⚠" if r["saturated"] else "") for r in rs]
    vals = [r["sigma_detect"] for r in rs]
    cols = [ins.MODE_COLOR[r["mode_key"]] for r in rs]
    bars = ax2.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_width() + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                 f"{v:.1f}σ", va="center", fontsize=9, color="#333333")
    for ref in (3.0, 5.0):
        if ref < max(vals) * 1.15:
            ax2.axvline(ref, color="#bbbbbb", lw=0.8, ls=":")
            ax2.text(ref, len(rs) - 0.3, f"{ref:.0f}σ", fontsize=7, color="#888888",
                     ha="center", va="bottom")
    ax2.set_xlim(0, max(vals) * 1.18 + 0.5)
    ax2.set_xlabel(f"detection significance ({meta['n_transits']} transit"
                   f"{'s' if meta['n_transits'] > 1 else ''})")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with col2:
    st.subheader("Mode details")
    rows = []
    for r in sorted(results, key=lambda r: -r["sigma_detect"]):
        notes = []
        if r["saturated"]:
            notes.append(f"saturates (full-well {r['sat_frac']:.2f} at min groups)")
        if r["warnings"]:
            notes.append("; ".join(list(r["warnings"])[:2]))
        rows.append({
            "mode": r["label"],
            "band (μm)": f"{r['wl'].min():.2f}–{r['wl'].max():.2f}",
            "σ_detect": round(r["sigma_detect"], 1),
            "median σ (ppm)": round(r["median_sigma_ppm"]),
            "bins": r["n_bins"],
            "ngroup": r["ngroup"],
            "cadence (s)": round(r["t_cycle_s"], 1),
            "notes": "; ".join(notes),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "σ_detect = √Σ((full − without-molecule)/σ_bin)² over the mode's bins; "
        "σ_bin combines Pandeia photon+detector noise for in/out-of-transit "
        "integrations with the (non-averaging) systematic floor. "
        "Groups are chosen to stay under the saturation limit, PandExo-style."
    )

# --- Fisher forecast -------------------------------------------------------
# authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
# NOT the multiselect order
fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
if fisher_names and "jac" in model:
    st.subheader("Fisher parameter forecast")
    with_jac = [r for r in results if r.get("jac_bins") is not None]

    def fmt(name, s):
        v = fisher_mod.display_sigma(name, s)
        return "unconstrained" if not np.isfinite(v) or v > 1e3 else f"{v:.3g}"

    frows = []
    for r in with_jac:
        sig = fisher_mod.mode_forecast(r, fisher_names)
        frows.append({"mode": r["label"],
                      **{f"σ({n}) [{forward.PARAM_UNITS[n]}]": fmt(n, sig[n])
                         for n in fisher_names}})
    if len(with_jac) >= 2:
        sig = fisher_mod.combined_forecast(with_jac, fisher_names)
        frows.append({"mode": "ALL SELECTED (combined)",
                      **{f"σ({n}) [{forward.PARAM_UNITS[n]}]": fmt(n, sig[n])
                         for n in fisher_names}})
    st.dataframe(frows, use_container_width=True, hide_index=True)
    st.caption(
        "Marginalized 1σ forecasts from the forward-mode autodiff Jacobian of the "
        "full VULCAN-JAX + ExoJAX chain (warm-started jvp per parameter, photochemistry "
        "on). Per-mode rows marginalize a reference-radius (lnR0) nuisance; the "
        "combined row shares lnR0 and adds one absolute-depth offset nuisance per "
        "mode. No priors — a flat direction reads 'unconstrained'. lnZ and lnKzz are "
        "reported in dex."
    )
elif out.get("fisher_names"):
    st.info("Fisher forecast requested but the cached model has no Jacobian — press Run.")
