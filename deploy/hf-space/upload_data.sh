#!/usr/bin/env bash
# Run on the Mac. Stages a symlink-dereferenced copy of the two data trees
# (~7.5 GB; the phoenix grid symlink must be materialized) and uploads them
# to the private HF dataset repo the Space seeds /data from.
#
# Prereqs:
#   pip install -U "huggingface_hub[cli]"
#   hf auth login              (or: huggingface-cli login)
#   dataset repo created at hf.co/new-dataset (private)
#
# Usage:  ./upload_data.sh [staging-dir]     (default ~/Desktop/hf_data_stage)
#   env DATASET_REPO overrides the target (default imalsky/vulcan-jwst-tool-data)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAGE="${1:-$HOME/Desktop/hf_data_stage}"
REPO="${DATASET_REPO:-imalsky/vulcan-jwst-tool-data}"

if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' on PATH." >&2
    echo "Install with: pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

if [ ! -e "$ROOT/vulcan-jwst-tool/data/cdbs/grid/phoenix/catalog.fits" ]; then
    echo "ERROR: phoenix grid not resolvable through the cdbs symlink" >&2
    exit 1
fi

# -RL: follow symlinks so the staged copy is self-contained.
if [ ! -e "$STAGE/jwst-data/cdbs/grid/phoenix/catalog.fits" ]; then
    echo "Staging jwst-data (~7 GB copy, needs the disk space) ..."
    mkdir -p "$STAGE"
    cp -RL "$ROOT/vulcan-jwst-tool/data" "$STAGE/jwst-data"
fi
if [ ! -d "$STAGE/retrieval-data/cm24_wasp39b" ]; then
    echo "Staging retrieval-data ..."
    cp -RL "$ROOT/vulcan-retrieval/data" "$STAGE/retrieval-data"
fi

# Resumable uploader (safe to re-run after an interrupted upload). Uploads the
# staging dir's CONTENTS, giving jwst-data/ + retrieval-data/ at the repo root
# -- exactly the layout bootstrap_data.py expects.
echo "Uploading to $REPO (resumable; re-run this script if interrupted) ..."
$HF upload-large-folder "$REPO" --repo-type dataset "$STAGE"

echo "Done. The staging copy at $STAGE can be deleted once the Space boots."
