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
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# deps + source in one layer; uv.lock pins everything
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY notebooks ./notebooks
COPY scripts ./scripts
COPY hf ./hf
RUN uv sync --frozen --no-dev --no-cache

# run from a clean workspace where students mount data/ and results/
WORKDIR /workspace
CMD ["cluster", "--help"]
