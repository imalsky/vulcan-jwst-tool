# data: provenance

Inputs for this repo live here (env `JWST_TOOL_DATA_DIR` overrides; an
editable checkout infers the root, a site-packages install must set it). The
model and noise caches are GENERATED and live in `../output/`
(`JWST_TOOL_OUTPUT_DIR`): `model_cache/` spectra and `noise_cache/` Pandeia
results, regenerated per run and cache-busted by `forward._VERSION` /
the backend fingerprint in the code.

Run `jwst-tool data` for a live status report of every item below (and the
sibling-repo data the forward model needs), with per-item download remedies.

## Tracked in git (ships with a clone)

- `cdbs/comp/nonhst/2mass_ks_001_syn.fits` -- the 2MASS Ks bandpass used for
  the photsys vegamag normalization (required; worker preflight checks it).
  Source: https://ssb.stsci.edu/trds/comp/nonhst/2mass_ks_001_syn.fits
- `cdbs/comp/nonhst/johnson_j_003_syn.fits` -- UNUSED leftover of the retired
  J-band normalization (tracked but referenced by no code path).
- `cdbs/grid/phoenix` -- a SYMLINK into an external local stellar-grid tree
  (`RT-Project/picaso/reference/stellar_grids/...`); it dangles on other
  machines and the pandeia worker preflight fails loudly there. Do not
  replace it with a copy on THIS machine; on another machine, fetch the
  STScI reference-atlases PHOENIX tarball (~1.9 GB) and place its
  `grp/redcat/trds/grid/phoenix` tree at this path (a real directory works):
  https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
  (the 'pheonix' spelling in the filename is STScI's own).
- `notes.md` -- this file.

## Gitignored (fetch once per machine)

- `cdbs/calspec/alpha_lyr_stis_011.fits` -- CALSPEC Vega for the vegamag
  normalization (288 KB; required, preflighted). Source:
  https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits
- `pandeia_data-2026.2-jwst/` -- Pandeia JWST reference data for the default
  "current" backend (~15 MiB download / 30 MB extracted; must carry
  VERSION_DATA matching the engine release).
  Source: https://stsci.box.com/v/pandeia-data-v2026p2-jwst
- `pandeia_psfs-2026.2-jwst/` -- the split PSF library (pandeia_data >= 2026;
  ~4 GiB; must contain VERSION_PSF).
  Source: https://stsci.box.com/v/pandeia-psfs-v2026p2-jwst

The forward model's own data (HITRAN line lists, CIA tables, CO ExoMol cache,
stellar UV spectra) lives in the SIBLING repos (vulcan-retrieval `data/`,
vulcan_jax package data) -- see the README "Data setup" table and
`jwst-tool data`.
