# Docker: running the workshop in a container

Requirement: **Docker** — section B of the School Software Installation Guide.
No Python, no `uv`, no dependency installs on your machine, and nothing to keep
in sync with what we tested.

## The short version

The repo ships a wrapper that does the mount/flags/pull dance for you:

```bash
./run.sh download --all                 # 1.17 GB catalogue + ~1.0 GB embeddings (one time)
./run.sh download --assets --list       # what is in the bundle
./run.sh download --assets --check      # sha256-verify what you downloaded
./run.sh run --fast                     # M 67 demo, ~2 min
./run.sh run --cluster "M 67" --region-scaled
./run.sh marimo                         # notebook → http://localhost:2718
./run.sh shell                          # a bash prompt inside the image
./run.sh python scripts/red_clump.py --clusters "NGC 6819"
```

Windows: `.\run.ps1` with exactly the same arguments. If the prebuilt image is
not reachable it builds one from the folder you cloned — same result, a few
minutes once.

## What the wrapper does (the long form)

`run.sh` pulls `ghcr.io/iaa-so-training/day4-clustering:latest` on first use —
published for **linux/amd64** and **linux/arm64** (Apple silicon), public, no
login — then runs the container as *your* uid with this folder mounted:

```bash
docker pull ghcr.io/iaa-so-training/day4-clustering:latest      # once
docker tag  ghcr.io/iaa-so-training/day4-clustering:latest day4-clustering

# data, results and notebooks live on your machine, not in the image
mkdir -p data results notebooks

docker run -it --rm -u "$(id -u):$(id -g)" \
  -e HOME=/tmp/day4-home -e MPLCONFIGDIR=/tmp/day4-mpl -e MARIMO_HOME=/tmp/day4-marimo \
  -e MLFLOW_TRACKING_URI=file:///workspace/results/mlruns \
  -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  -v "$PWD/notebooks:/workspace/notebooks" \
  -w /workspace day4-clustering cluster download --all

docker run -it --rm -u "$(id -u):$(id -g)" \
  -e HOME=/tmp/day4-home -e MPLCONFIGDIR=/tmp/day4-mpl -e MARIMO_HOME=/tmp/day4-marimo \
  -e MLFLOW_TRACKING_URI=file:///workspace/results/mlruns \
  -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  -v "$PWD/notebooks:/workspace/notebooks" \
  -w /workspace day4-clustering cluster run --fast

# the marimo notebook (open http://localhost:2718), edits saved in your checkout
docker run -it --rm -p 2718:2718 -u "$(id -u):$(id -g)" \
  -e HOME=/tmp/day4-home -e MARIMO_HOME=/tmp/day4-marimo \
  -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  -v "$PWD/notebooks:/workspace/notebooks" \
  -w /workspace day4-clustering \
  sh -c 'for f in /app/notebooks/*.py; do [ -e "notebooks/$(basename "$f")" ] || cp "$f" notebooks/; done; \
         exec marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token'
```

MLflow records every run under `results/mlruns/` (that is what the wrapper sets
`MLFLOW_TRACKING_URI` to), so the tracking data is on your disk next to the plots
and survives the `--rm`.

Build it yourself instead (a few minutes), if you prefer — or to add the optional
torch extra for the re-embedding extension, which is left out of the published
image to keep the pull small:

```bash
docker build -t day4-clustering .
docker build -t day4-clustering --build-arg WITH_TORCH=1 .   # + ~200 MB, CPU wheels
```

Running as your own uid is what keeps `data/` and `results/` writable by you;
the image pre-creates the scratch HOME paths (`/tmp/day4-*`) that matplotlib,
numba and marimo want, so nothing needs `sudo` afterwards.

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
  bundle, docs, tests, dev tooling, and the optional `torch` extra for the
  re-embedding extension — add that with
  `docker build -t day4-clustering --build-arg WITH_TORCH=1 .` (~200 MB, CPU
  wheels; the default image stays lean for the pull 30 people do at once).

Track A needs `data/embeddings/attention_broad_merged.parquet`, which arrives with
`cluster download --assets` (or `--all` above) — see
`docs/spectral_embeddings_plan.md`.

## Troubleshooting

**Downloads** resume where they stopped (HTTP range requests inside the image —
no `wget`/`curl` needed, and none installed); re-run the same command after a
dropped connection. The catalogue is verified against its exact byte size, the
bundle against per-file sha256 (`./run.sh download --assets --check`).

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
./run.sh python -c "
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
- **Files in `data/`/`results/` belong to root** — the wrapper runs as your uid;
  only hand-written `docker run` commands without `-u` hit this. Fix once with
  `sudo chown -R "$USER" data results`.
- **marimo doesn't open** — the `--host 0.0.0.0 --no-token` flags are
  required inside the container; browse to `http://localhost:2718`.
- **`./run.sh` says "permission denied"** — `chmod +x run.sh` (git preserves the
  bit, but zip downloads may not), or run `bash run.sh …`.
- **Out of disk** — the image (~1.5 GB) + the catalogue (1.17 GB) + embeddings
  (~0.3 GB) ≈ 5.5 GB total. Delete `data/` to reclaim the catalogue.
