#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# OPTIONAL shortcut for the exact commands in README.md / docs/docker.md.
#
# You do not need this file — it only saves typing the mount flags:
#
#   ./run.sh download --all    ==  docker run --rm -it \
#                                    -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#                                    ghcr.io/iaa-so-training/day4-clustering \
#                                    uv run cluster download --all
#
# Prerequisites: Docker (section B of the School Software Installation Guide)
# and git. uv and every dependency live inside the image.
#
#   ./run.sh download --all          # catalogue + embeddings (~2.2 GB, once)
#   ./run.sh run --fast              # the ~2 min warm-up run
#   ./run.sh run --spectral          # the same clustering on spectral embeddings
#   ./run.sh marimo                  # the notebook, http://localhost:2718
#   ./run.sh python scripts/red_clump.py --clusters "NGC 6819"
#   ./run.sh shell                   # a shell inside the image
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

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  TTY_ARGS=(-t)
fi

# The image's entrypoint drops to the owner of ./data, so files you create are
# yours and not root's; it also sets HOME and the cache dirs these mounts need.
COMMON_ARGS=(
  --rm -i
  -e PYTHONHASHSEED=42
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}
  -v "$HERE/data:/app/data"
  -v "$HERE/results:/app/results"
  -v "$HERE/notebooks:/app/notebooks"
  -w /app
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
  echo "▸ Building it locally from this folder — a few minutes, once:"
  docker build -t "$TAG" "$HERE"
}

case "${1:-}" in
  ""|-h|--help|help)
    sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} -p 2718:2718 "$TAG" \
      uv run marimo edit "notebooks/$NOTEBOOK" --host 0.0.0.0 --no-token "$@"
    ;;
  python|python3|pytest)
    ensure_image
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" uv run "$@"
    ;;
  uv)
    ensure_image
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" "$@"
    ;;
  *)
    ensure_image
    exec docker run "${COMMON_ARGS[@]}" ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} "$TAG" uv run cluster "$@"
    ;;
esac
