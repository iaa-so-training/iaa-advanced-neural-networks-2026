# syntax=docker/dockerfile:1
# Two stages: the build tools stay in `builder`, the published image is
# python:3.13-slim + the synced virtualenv. The 1.17 GB APOGEE catalogue is NOT
# baked in — students download it into a mounted volume on first run, and the
# embedding parquets/checkpoints come from the mounted Hugging Face bundle.

# --- builder: needs a C/C++ toolchain -------------------------------------
# opentsne (t-SNE) publishes wheels for linux x86_64, macOS and Windows but NONE
# for linux aarch64, so on Apple silicon / arm64 machines uv compiles it from
# source. Without g++ that build fails with
#   error: Failed to build `opentsne==1.0.4` … [Errno 2] No such file or directory: 'g++'
# and the whole arm64 image build dies. build-essential here costs nothing in the
# published image: nothing from this stage ships except /app.
# Pinned to a versioned tag rather than plain `slim`: a rebuild months later must
# not silently pick up a different interpreter. (A digest would be per-platform,
# so a version tag is the right pin for a multi-arch build.)
FROM python:3.13.15-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible, layer-cacheable installs
# uv pinned for the same reason (this is the binary copied into the runtime).
COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/day4-uv

WORKDIR /app

# deps + source in one layer; uv.lock pins everything
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY hf ./hf
# The default image is the workshop core. `--build-arg WITH_TORCH=1` adds the
# optional torch extra (CPU wheels, ~200 MB) for the re-embedding extension —
# it is left out by default so the pull 30 students do stays ~1.5 GB.
ARG WITH_TORCH=""
# Jupyter is on by default while the marimo-vs-jupyter comparison runs, so the
# published image can serve both front ends from one pull; build with
# `--build-arg WITH_JUPYTER=` for the lean marimo-only image.
ARG WITH_JUPYTER="1"
RUN uv sync --frozen --no-dev ${WITH_TORCH:+--extra torch} ${WITH_JUPYTER:+--extra jupyter} --no-cache

# --- runtime: the slim image students pull ---------------------------------
FROM python:3.13.15-slim

# numba (pulled in by umap-learn, hdbscan and — the real offender — EVoC's
# pynndescent KNN graph) needs a threadsafe threading layer. Without libgomp it
# silently falls back to the *workqueue* layer and the nested parallel regions in
# EVoC abort the process mid-benchmark (exit 139).
#
# libgomp1 gives numba the OpenMP layer. Do NOT set NUMBA_THREADING_LAYER=omp
# here: pynndescent has a workaround (numba#3341) that rewrites an explicit
# "omp" request to "tbb", and to "workqueue" when TBB is missing — putting the
# crash straight back. Leaving the layer at "default" makes numba pick omp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# GIT_PYTHON_REFRESH=quiet: there is no git binary in the image and mlflow's
# GitPython probe prints two "Failed to import Git" warnings per run without it.
# PYTHONHASHSEED is belt-and-braces next to cluster.seeding — the pipeline seeds
# its own RNGs, this only removes dict/set iteration order as a source of drift.
ENV UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/day4-uv \
    MLFLOW_TRACKING_URI=file:///app/results/mlruns \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    GIT_PYTHON_REFRESH=quiet

# Provenance stamped into the image: the commit it was built from and the
# reference it is published under. `cluster doctor` prints both; the digest can
# only be read from outside the container
# (`docker inspect --format '{{index .RepoDigests 0}}' <image>`).
ARG GIT_SHA=""
ARG IMAGE_REF=""
ENV DAY4_GIT_SHA=$GIT_SHA \
    DAY4_IMAGE=$IMAGE_REF
LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.source="https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026"

# the pinned environment + source, exactly as synced in the builder
COPY --from=builder /app /app

# `uv` itself, so the documented `docker run … $IMG uv run cluster …` works: the
# environment is synced and frozen, uv only resolves the command against it.
COPY --from=builder /bin/uv /bin/uvx /bin/

# Scratch dirs, all writable by whatever uid the entrypoint drops to: HOME and
# the caches matplotlib/numba/uv/marimo want, plus the mount points so the
# documented `docker run -v "$PWD/data:/app/data" …` works without `-w` flags.
RUN install -d -m 777 /tmp/day4-home /tmp/day4-mpl /tmp/day4-marimo /tmp/day4-uv \
        /app/data /app/results

# Runs the command as the owner of the mounted folders, so the files you get on
# your laptop belong to you and not to root.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Students work from /app: `data/` and `results/` resolve inside the mounts, and
# `uv run …` finds the pinned project environment without any extra flags.
WORKDIR /app
CMD ["uv", "run", "cluster", "--help"]
