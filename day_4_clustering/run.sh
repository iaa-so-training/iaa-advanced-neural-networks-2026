#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The Day 4 workshop runner — one command, no Python needed on your machine.
#
# Prerequisites: Docker (section B of the School Software Installation Guide)
# and git. That is all: everything else lives inside the image.
#
#   ./run.sh download --all          # fetch the catalogue + embeddings (~2.5 GB)
#   ./run.sh run --fast              # the 2-minute warm-up run
#   ./run.sh run --spectral          # the same clustering on spectral embeddings
#   ./run.sh marimo                  # the interactive notebook, http://localhost:2718
#   ./run.sh shell                   # a shell inside the container
#
# Anything you pass is handed to the `cluster` command inside the container, so
# `./run.sh run --help`, `./run.sh download --list`, … all work the same as the
# README describes. Data and results stay on your machine in ./data and
# ./results.
# ---------------------------------------------------------------------------
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${DAY4_IMAGE:-ghcr.io/iaa-so-training/day4-clustering:latest}"
TAG="${DAY4_TAG:-day4-clustering}"

mkdir -p "$HERE/data" "$HERE/results" "$HERE/notebooks"

# Forward the tuning knobs documented in the README (CLUSTER_*, NUMBA_*, OMP_*)
# so `CLUSTER_USE_ELEMENT_WEIGHTS=1 ./run.sh run …` works from the host.
EXTRA_ENV=()
while IFS= read -r var_name; do
  case "$var_name" in
    CLUSTER_*|NUMBA_*|OMP_*) EXTRA_ENV=(${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} -e "$var_name") ;;
  esac
done < <(env | cut -d= -f1)

# Keep the files you create owned by *you*, not by root.
RUN_USER="$(id -u):$(id -g)"
TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-t)
fi
COMMON_ARGS=(
  --rm -i
  -u "$RUN_USER"
  -e HOME=/tmp/day4-home
  -e MPLCONFIGDIR=/tmp/day4-mpl
  -e MARIMO_HOME=/tmp/day4-marimo
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:///workspace/results/mlruns}"
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}
  -v "$HERE/data:/workspace/data"
  -v "$HERE/results:/workspace/results"
  -v "$HERE/notebooks:/workspace/notebooks"
  -w /workspace
)

ensure_image() {
  if docker image inspect "$TAG" >/dev/null 2>&1; then
    return 0
  fi
  echo "▸ No local '$TAG' image yet — trying the prebuilt one:"
  echo "    $IMAGE"
  if docker pull "$IMAGE" >/dev/null 2>&1; then
    docker tag "$IMAGE" "$TAG"
    echo "✓ pulled."
    return 0
  fi
  echo "  Not available (it is published from the workshop repository once merged)."
  echo "▸ Building it locally from this folder — this takes a few minutes once:"
  docker build -t "$TAG" "$HERE"
}

case "${1:-}" in
  ""|-h|--help|help)
    sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  shell|bash)
    ensure_image
    shift
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" bash "$@"
    ;;
  marimo|notebook)
    ensure_image
    shift
    NOTEBOOK="chemical_tagging.py"
    if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then   # a path, not a flag
      NOTEBOOK="$(basename "$1")"
      shift
    fi
    echo "▸ Notebook starting — open http://localhost:2718 (no password). Ctrl-C to stop."
    # The image's notebooks are read-only to your uid, and the notebook reads
    # `data/…` relative to the working directory. So: copy the shipped notebook
    # into the mounted folder (never overwriting your edits) and run it from
    # the workspace root.
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} -p 2718:2718 "$TAG" \
      sh -c '
        mkdir -p notebooks
        for f in /app/notebooks/*.py; do
          [ -e "notebooks/$(basename "$f")" ] || cp "$f" notebooks/
        done
        echo "▸ running notebooks/'"$NOTEBOOK"'"
        exec marimo edit "notebooks/'"$NOTEBOOK"'" --host 0.0.0.0 --no-token "$@"
      ' -- "$@"
    ;;
  python|python3|pytest|cluster|uv)
    # Any other command that exists inside the image, e.g.
    #   ./run.sh python scripts/red_clump.py --clusters "NGC 6819"
    ensure_image
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" "$@"
    ;;
  *)
    ensure_image
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" cluster "$@"
    ;;
esac
