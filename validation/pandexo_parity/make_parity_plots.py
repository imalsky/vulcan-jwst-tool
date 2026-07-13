"""Render the PandExo-parity figures from the committed parity_summary.json and
(where available) the raw per-wavelength run outputs.

Usage:
    python validation/pandexo_parity/make_parity_plots.py

Aggregate figures (1, 3, 4) read only parity_summary.json (committed, always
available). The per-wavelength overlay figure (2) additionally reads the raw
$JWST_TOOL_OUTPUT_DIR/pandexo_parity/{star}_{ours,pandexo}.json produced by
run_parity.py, and is skipped with a notice if that directory is absent.

Design: validated categorical palette (dataviz skill). Two-series overlays use
blue = this tool, orange = PandExo. Per-mode panels color by mode in the fixed
palette order. One axis per panel, thin marks, recessive grid, PNG @ 200 dpi.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

# --- validated categorical palette (light mode) ------------------------------
TOOL = "#2a78d6"       # this tool (blue, slot 1)
PANDEXO = "#eb6834"    # PandExo (orange, slot 8) -- CVD-safe against blue
MODE_HUES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
             "#4a3aa7", "#e34948", "#e87ba4"]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"

MODES = ["nirspec_prism", "nirspec_g395h", "nirspec_g235h", "niriss_soss",
         "nircam_f322w2", "nircam_f444w", "miri_lrs"]
LABEL = {"nirspec_prism": "PRISM", "nirspec_g395h": "G395H",
         "nirspec_g235h": "G235H", "niriss_soss": "SOSS",
         "nircam_f322w2": "F322W2", "nircam_f444w": "F444W",
         "miri_lrs": "MIRI LRS"}
MCOL = dict(zip(MODES, MODE_HUES))
STAR_MARK = {"w39_like": "o", "bright_hot": "s"}
STAR_LABEL = {"w39_like": "W39-like (Ks 10.7)", "bright_hot": "bright (Ks 8.5)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summary():
    return json.loads((HERE / "parity_summary.json").read_text())


def ok_rows(summary, star):
    return {m["key"]: m for m in summary["stars"][star]["modes"]
            if m.get("status") == "OK"}


# =============================================================================
# Figure 1 -- configuration & timing parity: ours vs PandExo on the 1:1 line
# =============================================================================
def fig_config_parity(summary):
    quantities = [
        ("ngroup_pandexo", "ngroup_ours", "groups / integration"),
        ("t_int_pandexo_s", "t_int_ours_s", "integration time"),
        ("n_int_pandexo_in", "n_int_ours", "integrations in transit"),
    ]
    unit = {"integration time": " (s)"}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.5))
    for ax, (kx, ky, title) in zip(axes, quantities):
        xs, ys = [], []
        for star in summary["stars"]:
            rows = ok_rows(summary, star)
            for key in MODES:
                if key not in rows:
                    continue
                x, y = rows[key][kx], rows[key][ky]
                xs.append(x)
                ys.append(y)
                ax.scatter(x, y, s=46, marker=STAR_MARK[star],
                           color=MCOL[key], edgecolor="white", linewidth=0.7,
                           zorder=3)
        lo = min(xs + ys) * 0.7
        hi = max(xs + ys) * 1.4
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        u = unit.get(title, "")
        ax.set_xlabel(f"PandExo  {title}{u}")
        ax.set_ylabel(f"this tool  {title}{u}")
        ax.set_title(title)
        _style(ax)
    mode_handles = [Line2D([], [], marker="o", ls="", color=MCOL[m],
                           markeredgecolor="white", label=LABEL[m])
                    for m in MODES]
    star_handles = [Line2D([], [], marker=STAR_MARK[s], ls="", color=INK2,
                           markeredgecolor="white", label=STAR_LABEL[s])
                    for s in STAR_MARK]
    line_handle = [Line2D([], [], color=INK2, ls="--", label="1:1 parity")]
    fig.legend(handles=mode_handles + star_handles + line_handle,
               loc="lower center", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Configuration & timing parity: this tool vs current PandExo "
                 "on the same Pandeia 2026.2 engine", fontsize=11.5, y=1.0)
    fig.text(0.5, 0.905, "points on the dashed line agree exactly; the only "
             "off-diagonal case is PRISM's group floor (this tool's "
             "ngroup_min=2 vs PandExo's ngroup=1 on a bright star)",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=[0, 0.16, 1, 0.885])
    out = FIGS / "fig1_config_timing_parity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 2 -- per-wavelength extracted-flux and depth-sigma parity (G395H)
# =============================================================================
def fig_perwavelength(summary, out_root, mode="nirspec_g395h",
                      star="w39_like"):
    of = out_root / f"{star}_ours.json"
    pf = out_root / f"{star}_pandexo.json"
    if not (of.exists() and pf.exists()):
        print(f"  [fig2] raw run outputs not in {out_root} -- skipping "
              "per-wavelength overlay (set JWST_TOOL_OUTPUT_DIR and re-run "
              "run_parity.py to regenerate)")
        return None
    o = json.loads(of.read_text())[mode]
    p = json.loads(pf.read_text())[mode]
    wl_o = np.asarray(o["wl"])
    flux_o = np.asarray(o["flux"])
    n1 = np.asarray(o["noise_1int"])
    order = np.argsort(wl_o)
    wl_o, flux_o, n1 = wl_o[order], flux_o[order], n1[order]
    wl_p = np.asarray(p["wave"])
    erate = np.asarray(p["e_rate_out"])
    err_p = np.asarray(p["error"])           # PandExo per-pixel depth sigma
    # pair on the shared extraction grid (identical wavelengths)
    ok = np.isfinite(err_p) & (err_p > 0)
    idx = np.clip(np.searchsorted(wl_o, wl_p[ok]), 0, wl_o.size - 1)
    ex = np.abs(wl_o[idx] - wl_p[ok]) < 1e-9 * np.maximum(wl_p[ok], 1e-9)
    io, ip = idx[ex], np.where(ok)[0][ex]
    row = ok_rows(summary, star)[mode]
    n_in, n_out = row["n_int_pandexo_in"], row["n_int_pandexo_out"]
    sig_o = (n1[io] / flux_o[io]) * np.sqrt(1.0 / n_in + 1.0 / n_out)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 5.6),
                             gridspec_kw={"height_ratios": [3, 1]})
    # --- extracted stellar count rate ---
    ax = axes[0, 0]
    ax.plot(wl_p, erate, color=PANDEXO, lw=1.3, label="PandExo", zorder=2)
    ax.plot(wl_o, flux_o, color=TOOL, lw=1.3, ls=(0, (4, 2)),
            label="this tool", zorder=3)
    ax.set_ylabel("extracted stellar\ncount rate  (e$^-$/s)")
    ax.set_title("Extracted stellar flux (engine product)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    axr = axes[1, 0]
    axr.plot(wl_o[io], flux_o[io] / erate[ip], color=INK2, lw=1.0)
    axr.axhline(1.0, color=PANDEXO, lw=1.0, ls=":")
    axr.set_ylim(0.9, 1.1)
    axr.set_ylabel("ratio\ntool / PandExo")
    axr.set_xlabel("wavelength (micron)")
    _style(axr)
    # --- per-pixel depth uncertainty ---
    ax = axes[0, 1]
    ax.plot(wl_p[ip], err_p[ip] * 1e6, color=PANDEXO, lw=1.3,
            label="PandExo (analytic fml)", zorder=2)
    ax.plot(wl_o[io], sig_o * 1e6, color=TOOL, lw=1.3, ls=(0, (4, 2)),
            label="this tool (pandeia noise)", zorder=3)
    ax.set_ylabel("per-pixel depth\nuncertainty  (ppm)")
    ax.set_title("Depth uncertainty (matched integrations, no floor)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    axr = axes[1, 1]
    axr.plot(wl_o[io], sig_o / err_p[ip], color=INK2, lw=1.0)
    axr.axhline(1.0, color=PANDEXO, lw=1.0, ls=":")
    axr.set_ylabel("ratio\ntool / PandExo")
    axr.set_xlabel("wavelength (micron)")
    _style(axr)
    med = np.median(sig_o / err_p[ip])
    axr.annotate(f"median {med:.3f} x -- conservative", xy=(0.5, 1.12),
                 xycoords="axes fraction", ha="center", va="bottom",
                 fontsize=8.5, color=INK2)

    fig.suptitle(f"Per-wavelength parity, {LABEL[mode]} on a "
                 f"{STAR_LABEL[star]} star (Pandeia 2026.2, both sides)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIGS / "fig2_perwavelength_g395h.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 3 -- noise-model attribution: variance excess over pure photon
# =============================================================================
def fig_noise_attribution(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, star in zip(axes, ["w39_like", "bright_hot"]):
        rows = ok_rows(summary, star)
        keys = [m for m in MODES if m in rows]
        x = np.arange(len(keys))
        w = 0.38
        ours = [rows[k]["var_excess_ours"] for k in keys]
        px = [rows[k]["var_excess_pandexo"] for k in keys]
        ax.bar(x - w / 2, ours, w, color=TOOL, zorder=3,
               label="this tool (pandeia full extracted noise)")
        ax.bar(x + w / 2, px, w, color=PANDEXO, zorder=3,
               label="PandExo (analytic fml ramp)")
        ax.axhline(1.0, color=INK2, lw=1.0, ls="--", zorder=2)
        ax.annotate("pure photon limit", xy=(len(keys) - 0.5, 1.0),
                    xytext=(len(keys) - 0.5, 1.18), ha="right", fontsize=8,
                    color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[k] for k in keys], rotation=35, ha="right",
                           fontsize=8.5)
        ax.set_title(STAR_LABEL[star])
        _style(ax)
    axes[0].set_ylabel("per-integration variance\n/ pure photon counts")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Why the depth-uncertainty residual exists: the noise MODEL, "
                 "not the configuration", fontsize=11.5)
    fig.text(0.5, 0.005, "This tool propagates Pandeia's full extracted noise "
             "(correlated ramp/read, background, dark, IPC); PandExo's default "
             "is an analytic ramp formula near photon-only. The ratio of these "
             "bars reproduces the depth-sigma ratio.", ha="center",
             fontsize=8.5, color=INK2, wrap=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = FIGS / "fig3_noise_model_attribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Figure 4 -- depth-sigma ratio summary across modes & stars
# =============================================================================
def fig_sigma_ratio_summary(summary):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    stars = ["w39_like", "bright_hot"]
    dx = {"w39_like": -0.15, "bright_hot": 0.15}
    x = np.arange(len(MODES))
    ax.axhspan(1.0, 1.75, color=TOOL, alpha=0.05, zorder=0)
    ax.axhline(1.0, color=INK2, lw=1.1, ls="--", zorder=1)
    for star in stars:
        rows = ok_rows(summary, star)
        for i, key in enumerate(MODES):
            if key not in rows:
                continue
            s = rows[key]["sigma_ratio_matched"]
            med, lo, hi = s["median"], s["p05"], s["p95"]
            ax.errorbar(i + dx[star], med, yerr=[[med - lo], [hi - med]],
                        marker=STAR_MARK[star], ms=8, color=MCOL[key],
                        ecolor=MCOL[key], elinewidth=1.3, capsize=3,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODES])
    ax.set_ylabel("depth-uncertainty ratio\nthis tool / PandExo")
    ax.set_title("Depth-uncertainty parity: medians one-sided & conservative "
                 "(matched integrations, no floor)")
    ax.annotate("medians conservative\n(this tool >= PandExo;\nMIRI LRS "
                "background-dominated)", xy=(0.30, 1.44), fontsize=8.5,
                color=INK2)
    star_handles = [Line2D([], [], marker=STAR_MARK[s], ls="", color=INK2,
                           markeredgecolor="white", label=STAR_LABEL[s])
                    for s in stars]
    parity_h = [Line2D([], [], color=INK2, ls="--", label="exact parity")]
    ax.legend(handles=star_handles + parity_h, frameon=False, fontsize=9,
              loc="upper right")
    ax.set_ylim(0.9, 1.72)
    _style(ax)
    fig.tight_layout()
    out = FIGS / "fig4_sigma_ratio_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    summary = load_summary()
    out_root = Path(os.environ.get(
        "JWST_TOOL_OUTPUT_DIR", str(HERE))) / "pandexo_parity"
    made = [fig_config_parity(summary),
            fig_perwavelength(summary, out_root),
            fig_noise_attribution(summary),
            fig_sigma_ratio_summary(summary)]
    for pth in made:
        if pth is not None:
            print(f"wrote {pth}")


if __name__ == "__main__":
    main()
