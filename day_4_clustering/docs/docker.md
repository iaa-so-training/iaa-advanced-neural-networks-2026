# Run the workshop in Docker

Only requirement: **Docker**. No Python, no `uv`, no dependency pain.

## Pull (one time)

The image is published for **linux/amd64** and **linux/arm64** (Apple silicon) and is public —
no login needed:

```bash
docker pull ghcr.io/iaa-so-training/day4-clustering:latest
docker tag ghcr.io/iaa-so-training/day4-clustering:latest chemical-tagging
```

Or build it yourself:

```bash
docker build -t chemical-tagging .
```

## Run

The image contains the code and every dependency, but **not** the data (the 1.17 GB
catalogue and the ~1 GB embedding bundle live outside so the image stays small). Fetch
both once into a host folder, then mount that folder for every run:

```bash
mkdir -p data results

# 1. catalogue + embeddings/checkpoints (one time, ~2.2 GB total)
docker run -it --rm -v "$PWD/data:/workspace/data" chemical-tagging cluster download --all

# 1b. what is inside the bundle, and verify it afterwards
docker run --rm -v "$PWD/data:/workspace/data" chemical-tagging cluster download --assets --list
docker run --rm -v "$PWD/data:/workspace/data" chemical-tagging cluster download --assets --check

# 2. run the fast benchmark (M 67 demo, ~1 min)
docker run -it --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/results:/workspace/results" \
  chemical-tagging cluster run --fast

# 3. your cluster, scaled region
docker run -it --rm \
  -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  chemical-tagging cluster run --cluster "M 67" --region-scaled

# 4. the marimo notebook (open http://localhost:2718)
docker run -it --rm \
  -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  -p 2718:2718 \
  chemical-tagging marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token
```

The frontier scripts work the same way — any `cluster` command or script is
available inside the image:

```bash
docker run -it --rm -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  chemical-tagging python scripts/red_clump.py --clusters "NGC 6819"
```

## Performance

A container run is measurably slower than a native install on the same laptop —
measured here: `cluster run --fast` takes **1 m 51 s** natively vs **3 m 37 s** in
the container (16 CPUs, Docker CPU shares). The benchmark results are
**bit-identical** either way, so use Docker for convenience and `uv` when you
are timing something.

## What's inside / not inside

- **In**: the code (`src/`, `scripts/`, `notebooks/`, `hf/`), all runtime deps
  (astropy, scikit-learn, umap-learn, hdbscan, EVoC, marimo, asteca, emcee,
  astroquery, huggingface_hub…), pinned by `uv.lock`.
- **Not in** (keeps it small): the 1.17 GB catalogue, the ~1 GB embedding
  bundle, docs, tests, and dev tooling.

Track A needs `data/embeddings/attention_broad_merged.parquet`, which arrives with
`cluster download --assets` (or `--all` above) — see
`docs/spectral_embeddings_plan.md`.

## Troubleshooting

**The run dies with `exit 139` after "Running benchmark (t-SNE / UMAP / EVoC)"** — numba picked
its *workqueue* threading layer, which is not threadsafe; EVoC's nested parallel regions then
abort the process. The published image avoids this by shipping `libgomp1`. If you built your own:

```bash
apt-get install -y libgomp1        # gives numba the OpenMP layer
```

**Do not** "fix" it with `NUMBA_THREADING_LAYER=omp`: pynndescent rewrites an explicit `omp`
request to `tbb`, and to `workqueue` when TBB is missing, which puts the crash back. Either
leave the variable unset (numba then picks omp from libgomp1) or install TBB:

```bash
pip install tbb && export NUMBA_THREADING_LAYER=tbb
```

To check which layer is live inside the container:

```bash
docker run --rm chemical-tagging python -c "
import numba, pynndescent; from numba import njit, prange
@njit(parallel=True)
def f(n):
    s = 0.0
    for i in prange(n):
        s += i
    return s
f(100); print('layer:', numba.threading_layer())"   # want: omp (or tbb), never workqueue
```

- **"device or resource busy" / permission errors** — mount the folder with
  `:Z` (SELinux) or run from a fresh dir.
- **marimo doesn't open** — the `--host 0.0.0.0 --no-token` flags are
  required inside the container; browse to `http://localhost:2718`.
- **Out of disk** — the image (~1.5 GB) + the catalogue (1.17 GB) + embeddings
  (~0.3 GB) ≈ 5.5 GB total. Delete `data/` to reclaim the catalogue.
