# data: provenance

Inputs for this repo live here (env `JWST_TOOL_DATA_DIR` overrides; an
editable checkout infers the root, a site-packages install must set it). The
model and noise caches are GENERATED and live in `../output/`
(`JWST_TOOL_OUTPUT_DIR`): `model_cache/` spectra and `noise_cache/` Pandeia
results, regenerated per run and cache-busted by `forward._VERSION` in the
code (v5 invalidated all earlier cached spectra).

## Tracked in git

- `cdbs/`: the minimal synphot CDBS tree the Pandeia backend needs
  (`grid/phoenix` symlinked from an external stellar-grid tree,
  `comp/nonhst/johnson_j_003_syn.fits` fetched from ssb.stsci.edu/trds).
  The phoenix symlink points into the external picaso tree
  (`RT-Project/picaso/reference/stellar_grids`) and dangles on other machines;
  the pandeia preflight fails loudly there. Do not replace it with a copy.
