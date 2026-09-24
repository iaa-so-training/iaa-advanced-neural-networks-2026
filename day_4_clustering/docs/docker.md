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

## The notebooks: marimo (default) and Jupyter

The material ships in **two front ends that call the same library with the same
seeds**, so you can compare them on your own laptop:

| notebook | front end | open it with |
| --- | --- | --- |
| `notebooks/chemical_tagging.py` | **marimo** (reactive; the default) | `docker run --rm -it -p 2718:2718 $DAY4 $IMG uv run marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token` → http://localhost:2718 |
| `notebooks/chemical_tagging.ipynb` | **JupyterLab** | `docker run --rm -it -p 8888:8888 $DAY4 $IMG uv run --extra jupyter jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --IdentityProvider.token=""` → http://localhost:8888 |

The tuning lab is the same pair (`tuning_template.py` / `tuning_template.ipynb`).
In JupyterLab, open the `.ipynb` from the file browser (you start in `/app`, the
notebooks are under `notebooks/`).

What actually differs:

| | marimo | Jupyter |
| --- | --- | --- |
| execution | dependency graph: touching a widget re-runs that cell and everything that depends on it | you run cells; the widget cells re-run one render callback in place |
| memoisation | `mo.cache` per argument set | no built-in cell cache — the .ipynb memoises its expensive calls itself (and still uses the on-disk prepared-sample cache) |
| outputs | all cells shipped with the single-page app | outputs live in the notebook document; JupyterLab renders the ones you scroll to |
| extra | default image | `uv run --extra jupyter …` (the published image carries the extra) |

Both front ends cap the interactive figures identically — same
`CLUSTER_PLOT_MAX_POINTS`, same "every member is always drawn" rule (see
*A sluggish notebook is usually the page, not the CPU* below) — so the figures
themselves are not a variable in the comparison.

Measured on the workshop laptop (16 cores, warm data cache, both front ends
running the same cells end to end):

| | marimo | Jupyter |
| --- | --- | --- |
| full run, headless | `marimo export html`: **3 min 10 s** | executed `.ipynb`: **3 min 6 s** (main), **2 min 51 s** (tuning) |
| what the browser gets | 1.71 MB single page, heaviest figure 376 kB | 1.48 MB of cell outputs, heaviest figure 391 kB (2.13 MB file on disk) |
| interactivity | widget change re-runs the cell and its dependents through the dependency graph | widget change re-runs one render callback in place |
| run order | the graph heals a cell run out of order | cells must be run top to bottom, as usual |

Two Jupyter-specific notes: a *saved* notebook (one someone else executed, like
the copy in `results/`) shows the widgets as their plain-text repr until you
trust it (`File → Trust Notebook`; live output has no such gate), and `nbconvert`
only persists widget state when it drives the whole notebook — a hand-rolled
cell-by-cell driver has to call `client.set_widgets_metadata()`.

Prefer a lean image without the Jupyter stack? Build with
`--build-arg WITH_JUPYTER=` (measured: 1.75 GB → 1.61 GB without it).

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
| `./results` | `/app/results` | the score tables, `benchmark_grid.png`, the prepared-sample cache (`cache/prepared/`), and `mlruns/` — the image sets `MLFLOW_TRACKING_URI=file:///app/results/mlruns`, so container runs keep their MLflow record here (a *native* run writes to `./mlruns` instead) |
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

### Why it is slow even though the machine looks idle

The heavy stages of this pipeline are **serial by design**, so a 16-core laptop
shows a load average near 1 while the notebook feels stuck. Measured
2026-09-24 in this image (`scripts/phase_profile.py`, `cluster run --fast`
sampling, 25 000 stars × 16 elements):

| stage | wall | CPU time | cores used |
| --- | --- | --- | --- |
| `prepare` — read the 1.17 GB `.fits.gz`, cuts, membership | 20.6 s | 20.6 s | **1.00 / 16** |
| t-SNE (sklearn Barnes-Hut, 1000 iters) | 42.7 s | 259.1 s | 6.06 |
| UMAP (`random_state` ⇒ `n_jobs=1`) | 50.1 s | 80.9 s | **1.62** |
| HDBSCAN | 1.9 s | 0.3 s | 0.14 |
| EVoC | 75.5 s | 82.0 s | **1.09** |
| UMAP again with `n_jobs=-1, random_state=None` (not reproducible) | **3.2 s** | 32.5 s | 10.19 |

Three of the six stages cannot use more than one core:

- **gzip** decompression is serial and astropy cannot `memmap` a `.gz`, so the
  catalogue read is ~20 s on one core no matter how many cores you have.
- **UMAP forces `n_jobs=1` whenever `random_state` is set** — it prints
  `n_jobs value 1 overridden to 1 by setting random_state` on every run. This
  project seeds every stochastic step on purpose (see
  `docs/reproducibility.md`), so the parallel path is the *unseeded* one.
- **EVoC** builds its tree on one core; it has no threading option.

The two levers that keep iteration cheap, both on by default:

- the **prepared sample is cached on disk** (`results/cache/prepared/`, keyed by
  catalogue bytes + settings + seed + cache format): first call ≈20 s, later
  calls **≈0.04 s**, and the restored frame is bit-identical to a fresh read
  (`tests/test_data_cache.py`). `cluster doctor` reports the cache;
  `cluster run --no-cache` (or `CLUSTER_NO_CACHE=1`) forces a real read;
  `CLUSTER_CACHE_DIR=<dir>` moves it.
- the **notebooks memoise their expensive calls with `mo.cache`**, so re-running
  a cell whose arguments have not changed returns instantly instead of
  recomputing minutes.

Deliberately *not* cached: the benchmark itself. A new configuration means a
real computation — say so in your report, and remember that scores move ~±0.02
across machines anyway. `CLUSTER_TSNE_N_ITER=250` shortens t-SNE (43 s → 14 s on
the same sample) when you are exploring rather than quoting.

### A sluggish notebook is usually the page, not the CPU

Slow *interaction* is a different layer: marimo ships every trace — and every
hover string — to the browser, so a figure with 25 000 stars that each carry
hover text is megabytes of JSON. Measured on the shipped notebook (30° cone
around M 67): three cells carried 6.2 MB of the 7.1 MB page, and the mouse
stopped responding long before the machine was busy. With the cap in place the
same notebook exports to **1.71 MB** (from 7.11 MB) — those three figures fall
from 2.51, 2.46 and 1.25 MB to 0.38, 0.36 and 0.13 MB, with every member still
plotted. The plots cap the grey field at `CLUSTER_PLOT_MAX_POINTS` (default
**3000**) points per panel:

- **every member is always drawn** — only *unlabelled* field stars are thinned,
  the same budget `abundance_violins` already applies on the data side;
- hover text is built only for the stars that display it (field traces set
  `hoverinfo="skip"`);
- the sample is seeded, so the same figure is drawn twice for the same seed —
  nothing about the science changes, only what is *painted*.

`CLUSTER_PLOT_MAX_POINTS=0` draws members only; raise it and restart the kernel
for the full crowd.

## What's inside / not inside

- **In**: the code (`src/`, `scripts/`, `notebooks/`, `hf/`), all runtime deps
  (astropy, scikit-learn, umap-learn, hdbscan, EVoC, marimo, asteca, emcee,
  astroquery, huggingface_hub…), pinned by `uv.lock`. The Jupyter stack
  (`--extra jupyter`: jupyterlab, ipykernel, ipywidgets, ipympl) is in the
  published image too, so one pull serves both front ends.
- **Not in** (keeps it small): the 1.17 GB catalogue, the ~1 GB embedding
  bundle, docs, tests, dev tooling, and the optional `torch` extra for the
  re-embedding extension — add that with
  `docker build -t day4-clustering --build-arg WITH_TORCH=1 .` (~200 MB, CPU
  wheels; the default image stays lean for the pull 30 people do at once).

Track A needs `data/embeddings/attention_broad_merged.parquet`, which arrives with
`cluster download --assets` (or `--all` above) — see
`docs/spectral_embeddings_plan.md`.

## Troubleshooting

**A cell hangs with no error, and the CPU stays near zero** — an archive that
accepted the connection and then went quiet (hotel wifi, or VizieR/Gaia having a
bad day). Every archive call the pipeline makes is bounded: after
`CLUSTER_NET_TIMEOUT` seconds (default 30) it gives up with a message instead of
hanging, `literature_table` stops after the *first* timeout and fills the
remaining rows with a `note`, and §8 of the notebooks prints "Gaia archive
unreachable …" and carries on. Raise the budget on a slow-but-working link with
`CLUSTER_NET_TIMEOUT=120`, or work from the caches under `data/gaia/` and
`data/literature/`.

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
