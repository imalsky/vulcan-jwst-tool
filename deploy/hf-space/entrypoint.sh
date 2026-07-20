#!/usr/bin/env bash
# Space entrypoint. Preferred layout (HF volumes API, 2026): the dataset repo
# is mounted READ-ONLY at /srv/hub-data and a writable bucket volume at /data
# holds caches -- no download at boot. Fallback: seed /data from the dataset
# repo via bootstrap_data.py (needs HF_TOKEN) when no dataset mount exists.
set -euo pipefail

if [ -d /srv/hub-data/jwst-data ]; then
    echo "[entrypoint] dataset volume found at /srv/hub-data (no download)"
    export JWST_TOOL_DATA_DIR=/srv/hub-data/jwst-data
    ln -sfn /srv/hub-data/retrieval-data /srv/vulcan/vulcan-retrieval/data
    # Read-only data: anything that tries to WRITE into the data trees
    # (e.g. first-use h2he line-list downloads) fails loudly -- expected.
else
    if [ ! -d /data ]; then
        echo "ERROR: no dataset volume at /srv/hub-data and no storage at" >&2
        echo "/data. Mount the dataset repo as a volume (Settings -> " >&2
        echo "Storage/Volumes, or HfApi.set_space_volumes) or add a" >&2
        echo "writable volume so bootstrap_data.py can seed it." >&2
        exit 1
    fi
    echo "[entrypoint] no dataset mount -- seeding /data (download path)"
    mkdir -p /data/jwst-data /data/retrieval-data
    python /srv/app/bootstrap_data.py
    export JWST_TOOL_DATA_DIR=/data/jwst-data
    ln -sfn /data/retrieval-data /srv/vulcan/vulcan-retrieval/data
fi

# Writable state: the bucket volume at /data when present, else container
# disk (ephemeral -- caches lost on restart, everything still works).
STATE=/data
if [ ! -d /data ] || ! touch /data/.rwtest 2>/dev/null; then
    STATE=/tmp/state
    echo "[entrypoint] WARNING: no writable /data volume; caches are" \
         "EPHEMERAL (lost on restart/rebuild)"
fi
rm -f /data/.rwtest 2>/dev/null || true
mkdir -p "$STATE/output" "$STATE/retrieval-output" "$STATE/home" "$STATE/cwd"
export JWST_TOOL_OUTPUT_DIR="$STATE/output"
export HOME="$STATE/home"
ln -sfn "$STATE/retrieval-output" /srv/vulcan/vulcan-retrieval/output
# VULCAN-JAX's legacy IO writes a RELATIVE output/ dir in the process CWD
# (harmless junk, but the CWD must be writable -- the container default is
# root-owned and the forward subprocess inherits CWD from here).
cd "$STATE/cwd"

# CORS/XSRF off: required for uploads (T-P tables, noise-floor tables) to
# work behind the Spaces proxy. Keep the Space PRIVATE.
exec jwst-tool \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
