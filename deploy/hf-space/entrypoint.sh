#!/usr/bin/env bash
# Space entrypoint: verify persistent storage, seed /data from the HF dataset
# repo if needed, wire the engine's fixed data/output paths, launch the GUI.
set -euo pipefail

if [ ! -d /data ]; then
    echo "ERROR: /data does not exist -- this Space has no persistent storage." >&2
    echo "Enable it in Settings -> Storage (Small 20 GB tier is sufficient):" >&2
    echo "the ~8 GB dataset seed and all model/noise caches live there." >&2
    exit 1
fi

mkdir -p /data/jwst-data /data/retrieval-data /data/output /data/retrieval-output /data/home

python /srv/app/bootstrap_data.py

# The retrieval engine reads REPO_DIR/data and REPO_DIR/output by contract
# (not env-overridable); the image removed the cloned stubs for these links.
ln -sfn /data/retrieval-data /srv/vulcan/vulcan-retrieval/data
ln -sfn /data/retrieval-output /srv/vulcan/vulcan-retrieval/output

# CORS/XSRF off: required for uploads (T-P tables, noise-floor tables) to
# work behind the Spaces proxy. Keep the Space PRIVATE.
exec jwst-tool \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
