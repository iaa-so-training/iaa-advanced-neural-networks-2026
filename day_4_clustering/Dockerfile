# syntax=docker/dockerfile:1
# Smallest practical image: python:3.13-slim + the workshop deps via uv.
# The 3.7 GB APOGEE allStar is NOT baked in — students download it into a
# mounted volume on first run. Embeddings parquets are also excluded.
FROM python:3.13-slim

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

# uv for fast, reproducible, layer-cacheable installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/day4-uv \
    MLFLOW_TRACKING_URI=file:///app/results/mlruns \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

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
RUN uv sync --frozen --no-dev ${WITH_TORCH:+--extra torch} --no-cache

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
