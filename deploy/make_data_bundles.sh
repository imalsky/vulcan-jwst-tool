#!/usr/bin/env bash
# Run on the Mac. Packs the two data trees the server needs into tarballs,
# DEREFERENCING symlinks (-L) so data/cdbs/grid/phoenix (a symlink into the
# local picaso tree, ~2.7 GB) is materialized -- it dangles on any other
# machine otherwise.
#
#   jwst-data.tar       vulcan-jwst-tool/data   (~7 GB: pandeia PSFs 4.3G,
#                       phoenix 2.7G, pandeia refdata, cdbs, exojax_mie)
#   retrieval-data.tar  vulcan-retrieval/data   (~360 MB: exojax line lists,
#                       opacity cache, cm24_wasp39b marker)
#
# Usage:  ./make_data_bundles.sh [output-dir]   (default: ~/Desktop)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$HOME/Desktop}"

for d in "$ROOT/vulcan-jwst-tool/data" "$ROOT/vulcan-retrieval/data"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: $d not found -- run this from a full working checkout" >&2
        exit 1
    fi
done
if [ ! -e "$ROOT/vulcan-jwst-tool/data/cdbs/grid/phoenix/catalog.fits" ]; then
    echo "ERROR: phoenix grid not resolvable through the cdbs symlink" >&2
    echo "(data/cdbs/grid/phoenix must point at a live picaso tree)" >&2
    exit 1
fi

echo "Packing jwst-data.tar (about 7 GB, several minutes) ..."
tar -c -L -f "$OUT/jwst-data.tar" -C "$ROOT/vulcan-jwst-tool" data
echo "Packing retrieval-data.tar ..."
tar -c -L -f "$OUT/retrieval-data.tar" -C "$ROOT/vulcan-retrieval" data

ls -lh "$OUT/jwst-data.tar" "$OUT/retrieval-data.tar"
echo
echo "Upload both to the instance home directory, one command per transfer:"
echo "  scp -i YOUR_KEY.pem $OUT/jwst-data.tar ubuntu@SERVER_IP:~"
echo "  scp -i YOUR_KEY.pem $OUT/retrieval-data.tar ubuntu@SERVER_IP:~"
