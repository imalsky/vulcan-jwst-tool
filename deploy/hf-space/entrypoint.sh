#!/usr/bin/env bash
# Space entrypoint. Preferred layout (HF volumes API, 2026): the dataset repo
# is mounted READ-ONLY at /srv/hub-data and a writable bucket volume at /data
# holds caches + anything the engines insist on writing. Fallback: seed /data
# from the dataset repo via bootstrap_data.py (needs HF_TOKEN) when no
# dataset mount exists.
set -euo pipefail

# Writable state root: the bucket volume at /data when present, else
# container disk (ephemeral -- caches lost on restart, everything works).
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

if [ -d /srv/hub-data/jwst-data ]; then
    echo "[entrypoint] dataset volume found at /srv/hub-data (no download)"
    # jwst-data is a pure READ consumer (cdbs/refdata/PSFs/mie): serve it
    # straight from the read-only mount.
    export JWST_TOOL_DATA_DIR=/srv/hub-data/jwst-data
    # retrieval-data must be WRITABLE: radis creates a tempdir + lock files
    # inside exojax_linelists even for pure cache reads, and h2he line
    # lists download on first use. Sync the mount to the bucket (~360 MB
    # once); cp -au makes later boots a cheap stat pass that also picks up
    # dataset files that landed AFTER an earlier partial sync (the mount
    # live-updates as commits land).
    echo "[entrypoint] syncing retrieval-data to writable storage ..."
    mkdir -p "$STATE/retrieval-data"
    cp -au /srv/hub-data/retrieval-data/. "$STATE/retrieval-data/"
    ln -sfn "$STATE/retrieval-data" /srv/vulcan/vulcan-retrieval/data
else
    if [ ! -d /data ]; then
        echo "ERROR: no dataset volume at /srv/hub-data and no storage at" >&2
        echo "/data. Mount the dataset repo as a volume (Settings ->" >&2
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

# VULCAN-JAX's legacy IO writes a RELATIVE output/ dir in the process CWD
# (harmless junk, but the CWD must be writable -- the container default is
# root-owned and the forward subprocess inherits CWD from here).
cd "$STATE/cwd"

# CORS/XSRF off: required for uploads (T-P tables, noise-floor tables) to
# work behind the Spaces proxy.
exec jwst-tool \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
