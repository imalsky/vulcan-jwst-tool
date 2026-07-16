"""Data-availability detection for vulcan-jwst-tool.

One module that KNOWS every external dataset the tool touches, probes the
filesystem for each, and reports a structured, honest status: present,
missing, or fetched-on-first-use. Consumed three ways:

* ``jwst-tool data``            CLI report with per-item remedies
  (``--deep`` also asks the Pandeia env for its engine version)
* the GUI "Data status" panel + the availability annotations on the
  molecule / broadening / UV-spectrum widgets
* the unit tests (every check takes explicit paths, so they run on tmp dirs)

Detection never REPLACES the loud runtime failures (repo standing rule): the
workers still raise on missing data at run time; this module only makes the
state visible up front, with the exact remedy, before a 2-minute run dies on
a missing file.

Import discipline: stdlib only (no numpy/jax/streamlit); the one non-stdlib
touch is an optional ``import retrieval_framework.forward.config``, which is
documented import-light and raises a clear RuntimeError when its data tree is
missing -- that exception text IS the status detail.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jwst_tool import instruments as ins
from jwst_tool import planets

# ---------------------------------------------------------------------------
# Report structure
# ---------------------------------------------------------------------------

#: status values: "ok" (present), "missing" (absent, manual fetch),
#: "on-first-use" (absent, but the stack fetches it automatically when first
#: needed -- requires network at that moment)
OK, MISSING, AUTO = "ok", "missing", "on-first-use"


@dataclass
class Item:
    key: str            # stable machine key, e.g. "linelist:NH3"
    label: str          # human name, e.g. "NH3 HITRAN line list"
    status: str         # OK / MISSING / AUTO
    required: bool      # required for ANY run (vs opt-in feature data)
    detail: str         # what was found / which path is missing
    remedy: str = ""    # exact command / URL / pointer to get it
    section: str = ""   # filled by full_report()


def _found(path: Path) -> str:
    return f"found: {path}"


# ---------------------------------------------------------------------------
# Python stack (this environment)
# ---------------------------------------------------------------------------

_STACK = (
    ("vulcan_jax", True,
     "pip install -e <PROJECT_ROOT>/VULCAN-JAX --no-deps   (or from TestPyPI: "
     "pip install -i https://test.pypi.org/simple/ vulcan-jax)"),
    ("retrieval_framework", True,
     "pip install -e <PROJECT_ROOT>/vulcan-retrieval --no-deps   (dist name "
     "vulcan-retrieval; provides the forward-model engine)"),
    ("exojax", True, "pip install exojax"),
    ("jax", True, "pip install jax"),
    ("streamlit", False, "pip install streamlit pandas   (GUI only)"),
    ("pandas", False, "pip install streamlit pandas   (GUI only)"),
)


def check_python_stack() -> list[Item]:
    items = []
    for mod, required, remedy in _STACK:
        try:
            present = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            present = False
        items.append(Item(
            key=f"pkg:{mod}", label=f"python package {mod}",
            status=OK if present else MISSING, required=required,
            detail=("importable" if present else
                    "not importable in this environment"),
            remedy="" if present else remedy))
    return items


# ---------------------------------------------------------------------------
# Chemistry / RT engine data (the sibling vulcan-retrieval data tree)
# ---------------------------------------------------------------------------

def _engine_config():
    """The retrieval engine's config module, or an exception instance.

    ``retrieval_framework.forward.config`` is documented import-light (stdlib
    only) and raises a RuntimeError naming the missing data tree -- that
    message is exactly what the report should show.
    """
    try:
        return importlib.import_module("retrieval_framework.forward.config")
    except Exception as e:                      # ImportError or its RuntimeError
        return e


def linelist_path(mol: str, broadening: str = "air") -> Path | None:
    """Cache file ExoJAX/radis writes for ``mol``'s HITRAN list (None = not a
    HITRAN-sourced molecule, or engine config unavailable).

    Layouts mirror exojax_rt._build_opa: air lists live directly under
    DEMO_DATABASE, h2he lists in the h2he/ subdir (the path stem must stay a
    radis-parseable molecule token -- pinned in the engine repo's
    test_opacity_cache_paths).
    """
    cfg = _engine_config()
    if isinstance(cfg, Exception):
        return None
    spec = cfg.MOLECULES.get(mol)
    if spec is None or spec["source"] != "hitran":
        return None
    base = Path(cfg.DEMO_DATABASE)
    if broadening == "h2he":
        base = base / "h2he"
    return base / f"{spec['db']}.h5"


def molecule_linelist_status(mols: list[str],
                             broadening: str = "air") -> dict[str, str]:
    """{molecule: OK/AUTO/MISSING} for the GUI widget annotations.

    HITRAN-sourced molecules are AUTO when absent (ExoJAX downloads on first
    use); the cached-ExoMol CO reports its offline cache directory.
    """
    cfg = _engine_config()
    out = {}
    for m in mols:
        if isinstance(cfg, Exception):
            out[m] = MISSING
            continue
        spec = cfg.MOLECULES.get(m)
        if spec is None:
            out[m] = MISSING
        elif spec["source"] == "hitran":
            p = linelist_path(m, broadening)
            out[m] = OK if (p is not None and p.exists()) else AUTO
        else:                                   # exomol_cached CO
            out[m] = OK if Path(spec["db"]).is_dir() else AUTO
    return out


def check_engine_data(base_mols: list[str], extra_mols: list[str]) -> list[Item]:
    cfg = _engine_config()
    if isinstance(cfg, Exception):
        return [Item(
            key="engine:config", label="forward-model engine data root",
            status=MISSING, required=True, detail=str(cfg),
            remedy="Install vulcan-retrieval as an editable sibling checkout, "
                   "or set VULCAN_PROJECT_ROOT to the directory containing "
                   "the vulcan-retrieval/ checkout.")]

    items = [Item(
        key="engine:config", label="forward-model engine data root",
        status=OK, required=True, detail=_found(Path(cfg.DATA_DIR)))]

    co_dir = Path(cfg.CO_CACHED_DIR)
    items.append(Item(
        key="opacity:CO", label="CO line list (ExoMol Li2015, offline cache)",
        status=OK if co_dir.is_dir() and any(co_dir.iterdir()) else AUTO,
        required=True,
        detail=_found(co_dir) if co_dir.is_dir() else f"absent: {co_dir}",
        remedy="ExoJAX fetches the ExoMol CO/12C-16O/Li2015 tables on first "
               "use (network required), or copy data/opacity_cache/CO/ from "
               "another checkout."))

    h2h2 = Path(cfg.CIA_H2H2_FILE)
    items.append(Item(
        key="cia:H2-H2", label="H2-H2 collision-induced absorption table",
        status=OK if h2h2.is_file() else AUTO, required=True,
        detail=_found(h2h2) if h2h2.is_file() else f"absent: {h2h2}",
        remedy="ExoJAX fetches it on first use (~24 MB), or download "
               f"https://hitran.org/data/CIA/main/H2-H2_2011.cia to {h2h2}"))

    h2he = Path(cfg.CIA_H2HE_FILE)
    items.append(Item(
        key="cia:H2-He", label="H2-He collision-induced absorption table",
        status=OK if h2he.is_file() else MISSING, required=True,
        detail=_found(h2he) if h2he.is_file() else f"absent: {h2he}",
        remedy="Manual, one-time, ~147 MB (the RT refuses to build without "
               "it): download https://hitran.org/data/CIA/main/H2-He_2011.cia "
               f"to {h2he} (the /main/ path segment is required)."))

    for mol in base_mols + extra_mols:
        spec = cfg.MOLECULES.get(mol)
        if spec is None or spec["source"] != "hitran":
            continue                             # CO handled above
        p = linelist_path(mol)
        required = mol in base_mols
        items.append(Item(
            key=f"linelist:{mol}", label=f"{mol} HITRAN line list",
            status=OK if p.exists() else AUTO, required=required,
            detail=_found(p) if p.exists() else f"absent: {p}",
            remedy="Downloaded automatically from hitran.org on the first "
                   "run that uses it (~10-15 s; network required)."))
    return items


# ---------------------------------------------------------------------------
# VULCAN-JAX stellar UV spectra (ship inside the vulcan_jax package)
# ---------------------------------------------------------------------------

def _vulcan_pkg_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("vulcan_jax")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def vulcan_atm_dir() -> Path | None:
    pkg = _vulcan_pkg_dir()
    return None if pkg is None else pkg / "atm"


def check_fastchem() -> list[Item]:
    """The FastChem equilibrium-init binary: compiled on demand (make + C++)."""
    pkg = _vulcan_pkg_dir()
    if pkg is None:
        return []                    # covered by the python-stack section
    binary = pkg / "fastchem_vulcan" / "fastchem"
    return [Item(
        key="fastchem:binary", label="FastChem binary (equilibrium initializer)",
        status=OK if binary.is_file() else AUTO, required=True,
        detail=_found(binary) if binary.is_file() else f"absent: {binary}",
        remedy="Compiled automatically on the first equilibrium-init run "
               "(every shipped planet config); needs `make` and a C++ "
               "compiler on PATH. No download involved.")]


def uv_spectra_status() -> dict[str, bool]:
    """{sflux filename: present} for every registry stellar UV spectrum."""
    atm = vulcan_atm_dir()
    return {f: (atm is not None and (atm / "stellar_flux" / f).is_file())
            for f in planets.SFLUX_CHOICES}


def check_stellar_uv() -> list[Item]:
    atm = vulcan_atm_dir()
    if atm is None:
        return [Item(
            key="uv:package", label="stellar UV spectra (vulcan_jax/atm)",
            status=MISSING, required=True,
            detail="vulcan_jax is not importable, so its shipped spectra "
                   "cannot be located",
            remedy="Install vulcan-jax (see the python-stack section).")]
    items = []
    for fname, label in planets.SFLUX_CHOICES.items():
        p = atm / "stellar_flux" / fname
        items.append(Item(
            key=f"uv:{fname}", label=f"UV spectrum {label}",
            status=OK if p.is_file() else MISSING, required=True,
            detail=_found(p) if p.is_file() else f"absent: {p}",
            remedy="" if p.is_file() else
                   "Ships inside the vulcan-jax package (atm/stellar_flux/); "
                   "a missing file means a broken/partial install -- "
                   "reinstall vulcan-jax."))
    return items


# ---------------------------------------------------------------------------
# Pandeia noise backend + synphot CDBS
# ---------------------------------------------------------------------------

def _refdata_version(refdata: Path) -> str | None:
    for name in ("VERSION", "VERSION_DATA"):
        f = refdata / name
        if f.is_file():
            try:
                return f.read_text().strip().splitlines()[0]
            except OSError:
                return None
    return None


def check_pandeia_backend(python: str | Path = None,
                          refdata: str | Path = None,
                          psf_dir: str | Path | None = None) -> list[Item]:
    """The ACTIVE backend's three path roots (mirrors the worker preflight)."""
    python = Path(python if python is not None else ins.PICASO_PYTHON)
    refdata = Path(refdata if refdata is not None else ins.PANDEIA_REFDATA)
    psf_dir = (ins.PANDEIA_PSF_DIR if psf_dir is None else psf_dir) or ""

    items = [Item(
        key="pandeia:python", label="Pandeia engine environment (python)",
        status=OK if python.exists() else MISSING, required=True,
        detail=(_found(python) if python.exists() else f"absent: {python}"),
        remedy="Create a conda env with the matching engine, e.g.  "
               "conda create -n pandeia_2026 python=3.11  then  "
               "<env>/bin/pip install pandeia.engine==2026.2  and point "
               "JWST_TOOL_PANDEIA_PYTHON at its python.")]

    ver = _refdata_version(refdata)
    items.append(Item(
        key="pandeia:refdata", label="Pandeia JWST reference data",
        status=OK if refdata.is_dir() else MISSING, required=True,
        detail=(f"{_found(refdata)}" + (f" (version {ver})" if ver else ""))
               if refdata.is_dir() else f"absent: {refdata}",
        remedy="Download the release matching the engine (current pair: "
               "pandeia_data-2026.2-jwst, ~15 MiB) from "
               "https://stsci.box.com/v/pandeia-data-v2026p2-jwst and "
               "extract it to the path above (or point "
               "JWST_TOOL_PANDEIA_REFDATA elsewhere)."))

    if psf_dir:                                  # split-PSF layout (>= 2026)
        pv = Path(psf_dir) / "VERSION_PSF"       # worker preflight checks this
        items.append(Item(
            key="pandeia:psf", label="Pandeia PSF library (split layout)",
            status=OK if pv.is_file() else MISSING, required=True,
            detail=(_found(Path(psf_dir)) if pv.is_file() else
                    f"no VERSION_PSF under: {psf_dir}"),
            remedy="Download pandeia_psfs-2026.2-jwst (~4 GiB) from "
                   "https://stsci.box.com/v/pandeia-psfs-v2026p2-jwst and "
                   "extract it to the path above (or point "
                   "JWST_TOOL_PANDEIA_PSF_DIR elsewhere)."))
    return items


def probe_pandeia_engine(python: str | Path = None,
                         timeout: float = 60.0) -> str:
    """Ask the backend env for its pandeia.engine version (slow: subprocess).

    Returns the version string, or an 'unavailable (...)' explanation.
    Never raises -- this is a report, the run path has its own loud gate.
    """
    python = Path(python if python is not None else ins.PICASO_PYTHON)
    if not python.exists():
        return f"unavailable (no python at {python})"
    try:
        r = subprocess.run(
            [str(python), "-c",
             "import pandeia.engine; print(pandeia.engine.__version__)"],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"unavailable (probe timed out after {timeout:g} s)"
    if r.returncode != 0 or not r.stdout.strip():
        tail = (r.stderr or "").strip().splitlines()
        return ("unavailable (pandeia.engine not importable in that env"
                + (f": {tail[-1]}" if tail else "") + ")")
    return r.stdout.strip().splitlines()[-1]


def check_synphot_cdbs(cdbs: str | Path = None) -> list[Item]:
    """The minimal synphot tree the worker preflights (star SED + Ks norm)."""
    cdbs = Path(cdbs if cdbs is not None else ins.PYSYN_CDBS)
    items = []
    phx = cdbs / "grid" / "phoenix"
    phx_real = Path(os.path.realpath(phx))
    phx_ok = phx_real.is_dir()
    items.append(Item(
        key="cdbs:phoenix", label="PHOENIX stellar grid (synphot)",
        status=OK if phx_ok else MISSING, required=True,
        detail=(f"found: {phx} -> {phx_real}" if phx_ok else
                f"missing or dangling symlink: {phx} -> {phx_real}"),
        remedy="Fetch the STScI reference-atlases PHOENIX tarball (~1.9 GB): "
               "https://archive.stsci.edu/hlsps/reference-atlases/"
               "hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_"
               "synphot5.tar ('pheonix' spelling is STScI's own), extract, "
               f"and place its grp/redcat/trds/grid/phoenix tree at {phx} "
               "(a real directory is fine; the shipped symlink just points "
               "at an existing local copy)."))
    for rel, label, remedy in (
            (Path("comp") / "nonhst" / "2mass_ks_001_syn.fits",
             "2MASS Ks bandpass (Ks normalization)",
             "Ships with the repo (data/cdbs/comp/); restore it from git or "
             "fetch https://ssb.stsci.edu/trds/comp/nonhst/"
             "2mass_ks_001_syn.fits (8.6 KB)."),
            (Path("calspec") / "alpha_lyr_stis_011.fits",
             "Vega spectrum (vegamag normalization)",
             "Fetch https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits"
             f" (288 KB) to {cdbs / 'calspec'}/")):
        p = cdbs / rel
        items.append(Item(
            key=f"cdbs:{rel.name}", label=label,
            status=OK if p.is_file() else MISSING, required=True,
            detail=_found(p) if p.is_file() else f"absent: {p}",
            remedy="" if p.is_file() else remedy))
    return items


# ---------------------------------------------------------------------------
# Generated caches (informational)
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    def _stat(d: Path, glob: str):
        files = list(d.glob(glob)) if d.is_dir() else []
        return {"n": len(files),
                "mb": round(sum(f.stat().st_size for f in files) / 2**20, 1)}
    return {"model_cache": _stat(ins.MODEL_CACHE, "*.npz"),
            "noise_cache": _stat(ins.NOISE_CACHE, "*.json")}


# ---------------------------------------------------------------------------
# Full report + rendering
# ---------------------------------------------------------------------------

def full_report(base_mols: list[str] = None, extra_mols: list[str] = None,
                deep: bool = False) -> dict:
    """Every section's items (each Item.section filled), plus cache stats.

    ``base_mols``/``extra_mols`` default to the forward model's sets (passed
    in to keep this module import-independent of forward.py for tests).
    ``deep`` adds a subprocess probe of the Pandeia env's engine version.
    """
    if base_mols is None or extra_mols is None:
        from jwst_tool import forward
        base_mols = forward.MOLECULES if base_mols is None else base_mols
        extra_mols = forward.EXTRA_MOLECULES if extra_mols is None else extra_mols

    sections = {
        "Python stack (this environment)": check_python_stack(),
        "Chemistry / RT engine data": check_engine_data(base_mols, extra_mols),
        "Chemistry equilibrium initializer": check_fastchem(),
        "Stellar UV spectra (photochemistry)": check_stellar_uv(),
        f"Pandeia noise backend ({ins.JWST_TOOL_BACKEND})":
            check_pandeia_backend(),
        "Star normalization data (synphot CDBS)": check_synphot_cdbs(),
    }
    for name, items in sections.items():
        for it in items:
            it.section = name
    report = {"backend": ins.JWST_TOOL_BACKEND,
              "backend_status": ins.BACKEND_STATUS,
              "sections": sections, "caches": cache_stats()}
    if deep:
        report["engine_probe"] = probe_pandeia_engine()
    return report


def all_items(report: dict) -> list[Item]:
    return [it for items in report["sections"].values() for it in items]


def required_ok(report: dict) -> bool:
    """True when every REQUIRED item is present or auto-fetches on first use."""
    return all(it.status != MISSING for it in all_items(report)
               if it.required)


def missing_required(report: dict) -> list[Item]:
    return [it for it in all_items(report)
            if it.required and it.status == MISSING]


_STATUS_TEXT = {OK: "OK", MISSING: "MISSING",
                AUTO: "downloads on first use"}


def format_report(report: dict) -> str:
    """Plain-text report for the CLI (one line per item + remedies)."""
    lines = [f"jwst-tool data status  (backend: {report['backend_status']})"]
    if "engine_probe" in report:
        lines.append(f"pandeia.engine version probe: {report['engine_probe']}")
    for name, items in report["sections"].items():
        lines += ["", name, "-" * len(name)]
        for it in items:
            flag = _STATUS_TEXT[it.status]
            opt = "" if it.required else "  [optional]"
            lines.append(f"  [{flag}]{opt} {it.label}")
            lines.append(f"      {it.detail}")
            if it.remedy and it.status != OK:
                lines.append(f"      how to get it: {it.remedy}")
    c = report["caches"]
    lines += ["", "Generated caches (safe to delete; rebuilt on demand)",
              "----------------------------------------------------",
              f"  model spectra: {c['model_cache']['n']} files, "
              f"{c['model_cache']['mb']} MB",
              f"  pandeia noise: {c['noise_cache']['n']} files, "
              f"{c['noise_cache']['mb']} MB"]
    miss = missing_required(report)
    lines += ["", ("All required data present." if not miss else
                   f"{len(miss)} required item(s) MISSING -- runs that need "
                   "them will refuse loudly. Remedies above.")]
    return "\n".join(lines)
