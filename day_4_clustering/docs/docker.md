# Docker: running the workshop in a container

Requirement: **Docker** — section B of the School Software Installation Guide.
Nothing else is installed on your machine: `uv`, Python and every dependency live
inside the image, and the commands below run them there.

## The short version

```bash
docker pull ghcr.io/iaa-so-training/day4-clustering:latest
mkdir -p data results notebooks

# the flags every command repeats, once per shell session
export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

docker run --rm -it $DAY4 $IMG uv run cluster download --all          # ~2.2 GB, once
docker run --rm -it $DAY4 $IMG uv run cluster download --assets --list    # what is in the bundle
docker run --rm -it $DAY4 $IMG uv run cluster download --assets --check   # sha256-verify it
docker run --rm -it $DAY4 $IMG uv run cluster run --fast              # M 67 demo, ~2 min
docker run --rm -it $DAY4 $IMG uv run cluster run --cluster "M 67" --region-scaled
docker run --rm -it $DAY4 $IMG uv run python scripts/red_clump.py --clusters "NGC 6819"
docker run --rm -it $DAY4 $IMG bash                                   # a shell inside the image

# the notebook → http://localhost:2718
docker run --rm -it -p 2718:2718 $DAY4 $IMG \
  uv run marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token
```

Windows (PowerShell) — same commands, array splatting instead of `$DAY4`:

```powershell
$Day4 = @("-v","$($PWD.Path)/data:/app/data","-v","$($PWD.Path)/results:/app/results","-v","$($PWD.Path)/notebooks:/app/notebooks")
docker run --rm -it @Day4 ghcr.io/iaa-so-training/day4-clustering uv run cluster download --all
```

`./run.sh` / `.\run.ps1` are optional shortcuts that only save typing the mount
flags — `./run.sh run --fast` is exactly the `docker run … uv run cluster run
--fast` above, and if the published image is not reachable they build it from the
folder you cloned.

The embedding/checkpoint bundle is public on the Hub —
[`RafaelDias/iaa-chemical-tagging-2026`](https://huggingface.co/datasets/RafaelDias/iaa-chemical-tagging-2026)
— so you can browse the files, sizes and provenance without an account; the
`download --assets` commands above fetch the same files into `./data` and verify
them against the sha256 recorded in the dataset's `MANIFEST.json`.

## Why `uv run` inside the container

The image is built with `uv` from the same `pyproject.toml` + `uv.lock` that the
native instructions use, and the environment is already synced at `/app/.venv`.
`uv run cluster …` therefore starts in **under a second** and executes the pinned
environment — no resolution, no downloads, no sync (`UV_NO_SYNC=1` is baked in).
Inside the container, `uv run cluster …`, a bare `cluster …`, and `/app/.venv/bin/python`
are all the same interpreter.

## What the mounts are for

| host folder | in the container | holds |
|---|---|---|
| `./data` | `/app/data` | the 1.17 GB SDSS catalogue, the ~1 GB embeddings/checkpoints bundle |
| `./results` | `/app/results` | the score tables, `benchmark_grid.png`, and `mlruns/` (MLflow tracking) |
| `./notebooks` | `/app/notebooks` | the marimo notebooks, so your edits are saved in your checkout |

The image starts as root, then the entrypoint **drops to the uid that owns
`./data`**, so every file it writes belongs to you and needs no `sudo` to delete.
It also sets `HOME`, `MPLCONFIGDIR`, `MARIMO_HOME` and `UV_CACHE_DIR` to scratch
directories inside the container, which is what makes the plain `docker run`
above work without `-u`/`-e` flags. `DAY4_KEEP_ROOT=1 docker run …` opts out.

Build it yourself instead (a few minutes), if you prefer — or to add the optional
torch extra for the re-embedding extension, which is left out of the published
image to keep the pull small:

```bash
docker build -t day4-clustering .
docker build -t day4-clustering --build-arg WITH_TORCH=1 .   # + ~200 MB, CPU wheels
```

## Performance

A container run is slower than a native install on the same laptop — measured
2026-09-24 on the reference machine: the all-sky fast run takes **≈2 minutes
natively vs ≈3 minutes in the container** (16 CPUs, Docker CPU shares).

The scores are **identical to the last printed digit** between native and
container on that machine — same code, same data, same thread setting. That is a
property of the *pinned* image: the earlier unpinned build gave 0.1974 where
native gives 0.2093, because `python:3.13-slim` had moved underneath it. Since
`day4-v5` the base images are pinned (`python:3.13.15-slim`, uv 0.12.18) and the
parity holds. Across *machines* the same command still lands a decimal or two
away — that tolerance, and the readings behind it, are in
`docs/reproducibility.md`.

Use Docker for convenience and `uv` when you are timing something.

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
bundle against per-file sha256
(`docker run --rm -it $DAY4 $IMG uv run cluster download --assets --check`).

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
docker run --rm -it $DAY4 $IMG uv run python -c "
import numba, pynndescent; from numba import njit, prange
@njit(parallel=True)
def f(n):
    s = 0.0
    for i in prange(n):
        s += i
    return s
f(100); print('layer:', numba.threading_layer())"   # want: omp (or tbb), never workqueue
```

- **`docker pull` says "unauthorized" or asks for a login** — either you have a
  stale (private) login cached for another Day 4 image, `docker logout ghcr.io`
  fixes it, or the published package is not public yet; you do not need it:
  build the image yourself, it is the same thing and takes a few minutes:
  `docker build -t day4-clustering .` then use `day4-clustering` as `$IMG`
  (add `--build-arg WITH_TORCH=1` for the torch extra; see below).
- **"device or resource busy" / permission errors** — mount the folder with
  `:Z` (SELinux) or run from a fresh dir.
- **Files in `data/`/`results/` belong to root** — only if you created the mount
  folder as root before the first run (e.g. `sudo mkdir data`), because the
  entrypoint derives your uid from that folder's owner. Fix once with
  `sudo chown -R "$USER" data results`, or run as root deliberately with
  `docker run -e DAY4_KEEP_ROOT=1 …`.
- **marimo doesn't open** — the `--host 0.0.0.0 --no-token` flags are
  required inside the container, and remember `-p 2718:2718`; browse to
  `http://localhost:2718`.
- **`./run.sh` says "permission denied"** — `chmod +x run.sh` (git preserves the
  bit, but zip downloads may not), or run `bash run.sh …`. You can always fall
  back to the plain `docker run` commands above.
- **Out of disk** — the image (~1.6 GB) + the catalogue (1.17 GB) + the asset
  bundle (1.0 GB) ≈ 3.9 GB, plus what Docker itself keeps. Delete `data/` to
  reclaim the downloads; `results/` holds only figures and MLflow runs.
