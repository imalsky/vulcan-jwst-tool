#!/usr/bin/env bash
# Run ON the Ubuntu 24.04 instance (as the default ubuntu user). Idempotent.
# Steps: install docker -> clone the three repos -> stage data bundles ->
# build the image -> start (if .env is configured).
#
# Expects (optional but recommended before running): jwst-data.tar and
# retrieval-data.tar in ~ (uploaded via scp; see make_data_bundles.sh).
set -euo pipefail

SRC=~/vulcan-src
DATA=/srv/vulcan-data

# --- docker ---------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-v2 git
    sudo usermod -aG docker "$USER" || true
fi

# --- sources --------------------------------------------------------------
# Clone target name VULCAN-JAX is load-bearing (the engine's path contract);
# the GitHub repo for it is named jax-vulcan.
mkdir -p "$SRC"
cd "$SRC"
[ -d VULCAN-JAX ] || git clone https://github.com/imalsky/jax-vulcan.git VULCAN-JAX
[ -d vulcan-retrieval ] || git clone https://github.com/imalsky/vulcan-retrieval.git
[ -d vulcan-jwst-tool ] || git clone https://github.com/imalsky/vulcan-jwst-tool.git
cp vulcan-jwst-tool/deploy/dockerignore .dockerignore

# --- data volumes ---------------------------------------------------------
sudo mkdir -p "$DATA"/jwst-data "$DATA"/retrieval-data "$DATA"/output "$DATA"/retrieval-output "$DATA"/home
for t in jwst-data retrieval-data; do
    if [ -f ~/"$t".tar ]; then
        echo "Unpacking $t.tar ..."
        sudo tar -xf ~/"$t".tar -C "$DATA/$t" --strip-components=1
        rm ~/"$t".tar
    fi
done
sudo chmod -R a+rwX "$DATA"
if [ ! -d "$DATA/retrieval-data/cm24_wasp39b" ]; then
    echo "WARNING: $DATA/retrieval-data is not seeded yet (engine will refuse" >&2
    echo "to start until retrieval-data.tar is uploaded and unpacked)." >&2
fi

# --- image ----------------------------------------------------------------
echo "Building image (first build downloads ~2-3 GB of packages; 15-30 min) ..."
sudo docker build -t jwst-tool:latest -f "$SRC/vulcan-jwst-tool/deploy/Dockerfile" "$SRC"

# --- start ----------------------------------------------------------------
cd "$SRC/vulcan-jwst-tool/deploy"
if [ ! -f .env ]; then
    cp env.example .env
    echo
    echo "NOT started: edit $SRC/vulcan-jwst-tool/deploy/.env first"
    echo "(SITE_ADDRESS + basic-auth hash; see env.example), then run:"
    echo "  cd $SRC/vulcan-jwst-tool/deploy"
    echo "  sudo docker compose up -d"
    exit 0
fi
sudo docker compose up -d
echo
echo "Data-availability report (every row should be OK):"
sudo docker compose exec app jwst-tool data || true
