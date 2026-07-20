# vulcan-jwst-tool server image.
#
# Build context = a directory containing the THREE repo checkouts side by side
# (the clone name VULCAN-JAX is load-bearing for the engine's path inference):
#   VULCAN-JAX/  vulcan-retrieval/  vulcan-jwst-tool/  .dockerignore
# Build:  docker build -t jwst-tool:latest -f vulcan-jwst-tool/deploy/Dockerfile .
#
# The image carries CODE + the two python environments only. All datasets
# (pandeia refdata/PSFs, CDBS incl. the phoenix grid, exojax line lists,
# opacity cache) and all caches live on volumes -- see docker-compose.yml.
#
# After the first successful build, consider pinning this tag to its digest.
FROM condaforge/miniforge3:latest

# build-essential: FastChem is vendored C++ compiled with plain `make`
# (vulcan_jax.ini_abun runs it on demand; we pre-compile below).
# curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Two environments, mirroring the validated Mac setup: the app env runs the
# GUI + JAX forward model; the pandeia env is ONLY entered by the noise
# worker subprocess (jwst_tool.noise points at it via JWST_TOOL_PANDEIA_PYTHON).
RUN conda create -y -n app python=3.12 \
    && conda create -y -n pandeia python=3.12 \
    && conda clean -afy

# Pandeia env first (cheap layer, rarely changes).
COPY vulcan-jwst-tool/deploy/requirements-pandeia.txt /srv/build/requirements-pandeia.txt
RUN /opt/conda/envs/pandeia/bin/pip install --no-cache-dir -r /srv/build/requirements-pandeia.txt

# Source trees. data/ and output/ are excluded by .dockerignore; they are
# volume mounts at runtime (the engine's import-time data marker check runs
# then, against the mounted data).
COPY VULCAN-JAX /srv/vulcan/VULCAN-JAX
COPY vulcan-retrieval /srv/vulcan/vulcan-retrieval
COPY vulcan-jwst-tool /srv/vulcan/vulcan-jwst-tool
COPY vulcan-jwst-tool/deploy/constraints-app.txt /srv/build/constraints-app.txt

# Editable installs, same pattern as the NAS deployment: all three in ONE
# resolve so the local checkouts satisfy the vulcan-* version floors and
# PyPI provides the rest, pinned to the versions validated on the Mac.
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir \
        -c /srv/build/constraints-app.txt \
        -e /srv/vulcan/VULCAN-JAX \
        -e /srv/vulcan/vulcan-retrieval \
        -e "/srv/vulcan/vulcan-jwst-tool[gui]"

# Pre-compile the FastChem equilibrium-init binary for THIS platform.
# (ini_abun would compile it on the first run; doing it here keeps first user
# runs fast and fails the build loudly if the toolchain is broken. rm first:
# a mac-arm64 binary copied in from a working tree would otherwise be kept.)
RUN rm -f /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem \
    && make -C /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan \
    && test -x /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem

# Writable HOME for streamlit/matplotlib/astropy caches (volume-mounted for
# persistence); output dirs exist even when run without mounts.
RUN mkdir -p /srv/home \
        /srv/vulcan/vulcan-jwst-tool/output \
        /srv/vulcan/vulcan-retrieval/output \
    && chmod -R 777 /srv/home

ENV VULCAN_PROJECT_ROOT=/srv/vulcan \
    JWST_TOOL_DATA_DIR=/srv/vulcan/vulcan-jwst-tool/data \
    JWST_TOOL_OUTPUT_DIR=/srv/vulcan/vulcan-jwst-tool/output \
    JWST_TOOL_PANDEIA_PYTHON=/opt/conda/envs/pandeia/bin/python \
    JWST_TOOL_BACKEND=current \
    HOME=/srv/home \
    PATH=/opt/conda/envs/app/bin:$PATH
# JWST_TOOL_PANDEIA_REFDATA / _PSF_DIR are NOT set: the "current" backend
# resolves both under JWST_TOOL_DATA_DIR (portable layout). The "legacy"
# 3.0 backend is not deployed in this image.

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["jwst-tool", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
