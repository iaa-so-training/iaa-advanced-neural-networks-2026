# Chemical Tagging of Star Clusters — t-SNE vs UMAP vs EVoC

Hands-on module for the **IAA-SO School on AI/ML in Astronomy 2026**
(Unsupervised Learning pillar). Reproduces and extends
[Kos et al. (2017)](https://arxiv.org/abs/1709.00794) *"Chemical tagging of
star clusters and new members in the Pleiades"*.

The original paper tagged clusters with **t-SNE** on GALAH abundances. This
workshop benchmarks **t-SNE vs UMAP vs [EVoC](https://github.com/TutteInstitute/evoc)**
on **APOGEE SDSS-V DR19 + Gaia DR3**, over the **25 clusters** of
Garcia-Dias et al. (2019, A&A 629, A34) — 23 from the paper, plus the Pleiades
(Kos et al. 2017) and the two southern sweet-spots NGC 2243 and Collinder 261 —
and scores each method's cluster recovery against **kinematic ground truth**.

## Quick start (Docker — all you need is Docker and git)

Docker is section B of the School Software Installation Guide, so it is already
on every laptop in the room. Nothing is installed on your machine: `uv`, Python
and every dependency live inside the image, and we run them there.

```bash
git clone https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026/day_4_clustering
mkdir -p data results notebooks

docker pull ghcr.io/iaa-so-training/day4-clustering:latest

# the flags every command repeats, once per shell session
export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

docker run --rm -it $DAY4 $IMG uv run cluster download --all   # 1.17 GB catalogue + ~1.0 GB embeddings
docker run --rm -it $DAY4 $IMG uv run cluster run --fast       # smoke test: ~2 min, no GPU
docker run --rm -it -p 2718:2718 $DAY4 $IMG \
  uv run marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token   # http://localhost:2718
```

That is the whole workflow: `uv run cluster <command>` executes *inside* the
container, on the pinned environment we tested. Data, results and notebooks stay
on your machine in `./data`, `./results` and `./notebooks`; the image starts as
root but drops to your own uid, so the files it writes belong to you.

**Windows (PowerShell)** — same commands, different mount syntax:

```powershell
$Day4 = @("-v","$($PWD.Path)/data:/app/data","-v","$($PWD.Path)/results:/app/results","-v","$($PWD.Path)/notebooks:/app/notebooks")
docker run --rm -it @Day4 ghcr.io/iaa-so-training/day4-clustering uv run cluster download --all
docker run --rm -it @Day4 ghcr.io/iaa-so-training/day4-clustering uv run cluster run --fast
```

`docs/docker.md` has the full form of every command, what is inside the image,
and the alternatives (building it yourself, adding the optional torch extra).
`./run.sh` / `.\run.ps1` are optional shortcuts that only save typing the mounts.

**Reading the rest of this README:** every example is written as
`uv run cluster <flags>`. In Docker the same command is
`docker run --rm -it $DAY4 $IMG uv run cluster <flags>` — the flags never change.

## Alternative: native Python with uv

If you would rather not use Docker, you need Python ≥ 3.13 on your machine
(`uv` builds the virtualenv for you):

```bash
uv sync                     # add --extra torch for the re-embedding extension
uv run cluster download --all
uv run cluster run --fast
```

Identical commands and flags — the container simply removes the possibility of a
mismatched Python, library or JIT environment.

## How it works

```
astraAllStarASPCAP-0.6.0.fits.gz  (1.17 GB, one file)
        │  abundances (16 elements) + Gaia EDR3 astrometry + RV, all inside
        ▼
   quality cuts  (SNR, ASPCAPFLAG/STARFLAG)
        ▼
   kinematic membership labels  (position + parallax + proper motion + RV)
        ▼
   16-D abundance matrix  (standardised; missing abundances imputed by column median)
        ▼
   ┌──────────────┬──────────────┬──────────────┐
   │ t-SNE (2-D)  │ UMAP (2-D)   │  EVoC (16-D) │
   │   + HDBSCAN  │   + HDBSCAN  │  clusters    │
   │              │              │  internally  │
   └──────────────┴──────────────┴──────────────┘
        ▼
   precision / recall per cluster, per method
```

Key point: **EVoC is not a projector** — it fuses a UMAP-style graph embedding
with HDBSCAN/PLSCAN-style density clustering and returns labels directly.
t-SNE and UMAP only *embed*; a separate clustering step (HDBSCAN) is needed.

## Install

With Docker there is nothing to install — see **Quick start** above.
Natively:

```bash
uv sync            # or: python -m venv .venv && . .venv/bin/activate && pip install -e .
```

Python ≥ 3.13.

## Download the data

Commands from here on are shown in the `uv run cluster …` form; under Docker
(Quick start) each one is `docker run --rm -it $DAY4 $IMG uv run cluster …` — the
flags are identical, `$DAY4`/`$IMG` just carry the mounts and image name.

```bash
uv run cluster download            # the 1.17 GB DR19 catalogue (resumable)
uv run cluster download --assets   # embeddings + checkpoints, ~1.0 GB (Hugging Face)
uv run cluster download --all      # both — the one-liner for a fresh checkout
```

The catalogue comes from SDSS, the embeddings/checkpoints from the published
Hugging Face dataset
[`RafaelDias/iaa-chemical-tagging-2026`](https://huggingface.co/datasets/RafaelDias/iaa-chemical-tagging-2026)
(public — no account needed; the dataset card documents every file's provenance);
both land under `data/`. Useful flags:

```bash
uv run cluster download --assets --list    # what is inside, with sizes + consumers
uv run cluster download --assets --check   # sha256-verify what is already on disk
uv run cluster download --assets --only embeddings/attention_broad_merged.parquet
```

**Only have Docker?** `docs/docker.md` has the no-install path (public multi-arch
image; the same two downloads run inside the container).

The catalogue is SDSS-V DR19
`https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarASPCAP-0.6.0.fits.gz`
(the DR17 `allStar` successor — see `docs/data_releases.md`).

## The FAST tag

Everything is driven by `src/cluster/config.py` — a single file of flags.

- `FAST = True`  → subsample the **field** to `MAX_STARS` (default 25 000) while
  keeping **every** cluster member. **Measured 1 m 51 s** for the all-sky fast run
  on a 2026 laptop (`cluster run --fast`, 25 000 field stars).
- `FAST = False` → drop the cap; the DR19 quality cuts leave **358 058** stars
  (16 elements), not the ~183 000 of the DR17 era. Re-measure the runtime before
  quoting one — sklearn's Barnes-Hut t-SNE is single-threaded.

Flip the flag, or override on the command line:

```bash
uv run cluster run --fast            # fast run
uv run cluster run --full            # full all-sky run
uv run cluster run --fast --max-stars 5000

# paper-faithful: focus on a region around one cluster (Kos et al. ran
# t-SNE on 30-45 deg regions, NOT the whole sky)
uv run cluster run --fast --cluster "M 67" --region 30
uv run cluster run --fast --cluster Pleiades --region 40
```

Environment overrides work too: `CLUSTER_FAST=0`, `CLUSTER_MAX_STARS=10000`,
`CLUSTER_SNR_MIN=50`, …

## Run

```bash
uv run cluster run --fast
```

Prints per-cluster membership, the macro recall/precision table, the
per-cluster recall × method matrix, the **chemical-cohesion (kNN purity)**
score, and saves `results/benchmark_grid.png`.

## Notebooks

A [marimo](https://marimo.io) notebook (reactive Python, no Jupyter):

```bash
uv run marimo edit notebooks/chemical_tagging.py   # edit
uv run marimo run notebooks/chemical_tagging.py    # read-only app
```

In Docker the wrapper opens the port for you:

```bash
docker run --rm -it -p 2718:2718 $DAY4 $IMG \
  uv run marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token
# any notebook in notebooks/ works — swap the file name
```

Two ship with the day:

- `chemical_tagging.py` — the end-to-end demo. The abundance benchmark in §2–§3,
  the **published spectral latent vs the abundances on the same stars** in §0c
  (via `cluster head-to-head`), then the HR / isochrone / Gaia-age material.
- `tuning_template.py` — the knob-turning lab for the student activities.

`cluster download --all` fetches everything both notebooks read: the catalogue and
the embeddings/checkpoints bundle.

## Development

```bash
uv run pyrefly check            # strict type checking (currently reports errors — see below)
uv run pytest                   # 206 tests (5 need `--extra torch`, 2 need the PARSEC grid)
uv run coverage run -m pytest && uv run coverage report   # 90% (branch coverage)
uv run mlflow ui                # inspect experiment runs (mlruns/)
```

In Docker: `docker run --rm -it $DAY4 $IMG uv run pytest`,
`docker run --rm -it $DAY4 $IMG uv run pyrefly check`,
`docker run --rm -it $DAY4 $IMG uv run python scripts/…`.

Data contracts are enforced at runtime: pydantic `Settings`/`Cluster` models
+ pandera `ALLSTAR_SCHEMA` / `ABUNDANCE_SCHEMA` (see `src/cluster/schemas.py`).
MLflow logs params, per-method macro recall/precision, kNN purity and the
plot artifact for every `cluster run`.

## Configuration (main flags)

| flag | default | meaning |
|---|---|---|
| `FAST` | `True` | cap field stars for a fast run |
| `MAX_STARS` | `25000` (fast) / `None` (full) | field-star cap |
| `SNR_MIN` | `100` | minimum spectrum SNR |
| `DWARF_ONLY` | `False` | keep `logg ≥ 3.5` only |
| `ELEMENTS` | 16 `[X/Fe]` + `[Fe/H]` | the C-space |
| `STANDARDIZE` | `True` | zero-median / unit-std per element |
| `NORMALIZE_ROWS` | `True` | L2-normalise rows (Euclidean ≡ cosine) |
| `IMPUTE_MISSING` | `True` | fill NaN with column median |
| `MIN_FINITE_ELEMENTS` | `8` | keep stars with ≥8/16 finite (globulars) |
| `USE_ELEMENT_WEIGHTS` | `False` | weight by `1/σ` |
| `MEMBERSHIP_METHOD` | `kinematic` | ground-truth labels |
| `CLUSTER_NAMES` | `all` | which clusters to score (e.g. `["M 67", "Pleiades"]`) |
| `REGION_RADIUS_DEG` | `None` | per-cluster sky cut (paper's region method) |
| `TSNE_BACKEND` | `sklearn` | `sklearn` (fast BH) or `opentsne` (multi-core) |

Full list (with `TSNE`, `UMAP`, `EVOC`, `HDBSCAN` hyperparameters) is in
`src/cluster/config.py`.

## Results & interpretation

> ⚠️ **The two tables below are DR17-era numbers.** They were produced on
> `allStar-dr17-synspec_rev1.fits` and **do not reproduce on a current checkout**
> (the project moved to SDSS-V DR19 — `docs/data_releases.md`). A run of
> `cluster run --fast` today gives t-SNE 0.21/0.22, UMAP 0.22/0.13, EVoC 0.48/0.00
> (24 clusters, 829 members — the field-retrieval row of
> `docs/spectral_benchmark_results.md`). Current, reproducible numbers live in
> `docs/dr19_rerun_results.md` and `docs/spectral_benchmark_results.md`;
> regenerating the per-cluster tables is tracked in `docs/student_assets_plan.md §5`.

Measured with the defaults (`NORMALIZE_ROWS`, `IMPUTE_MISSING`).
Recall/precision are macro-averages over the clusters.

| mode | stars | t-SNE r / p | UMAP r / p | EVoC r / p |
|---|---|---|---|---|
| fast all-sky | 25k | 0.20 / **0.17** | 0.26 / 0.11 | 0.54 / 0.01 |
| region (M 67, 30°) | 14k | 0.05 / **0.47** | 0.61 / 0.05 | 0.49 / 0.05 |

Full per-cluster table: `docs/region_sweep_results.md` (**DR17, historical** —
see the warning at the top of that file). The current-region-mode table has to be
regenerated from the frozen student config; the task is item 8 in
`docs/student_assets_plan.md`.

### Paper baseline (cluster-only multiclass separation)

> ⚠️ Also DR17-era (see the warning in `docs/baseline_results.md`). The current
> numbers are in `docs/dr19_rerun_results.md`.

Before pulling clusters out of the field, re-create Garcia-Dias et al. 2019 —
cluster *only* the known members and ask whether they separate from each
other (`uv run cluster baseline`):

| features | homogeneity (t-SNE / UMAP / EVoC) |
|---|---|
| abundances | 0.26 / 0.32 / 0.50 |
| abundances + kinematics | 0.76 / 0.80 / 0.72 |

Full table + interpretation: `docs/baseline_results.md`. Abundances alone
separate the clusters only weakly (the paper's 0.85 used supervised LDA and
chemically pre-clipped membership); kinematics carry the separation.

What it teaches:

1. **Full-sky collapses.** The target paper never ran one giant all-sky
   t-SNE — it ran t-SNE on **30–45° regions** around each cluster
   (Kos et al. Fig. 3). The `--region` flag reproduces that: precision
   jumps to 0.47 for M 67 (vs 0.17 all-sky). Embedding 163k stars at once
   buries every cluster in field stars.
2. **L2-normalisation is the precision lever.** Without `NORMALIZE_ROWS`
   HDBSCAN merges the whole field into one blob (recall ≈ 1, precision ≈
   0.03 everywhere). Normalising makes Euclidean equal cosine — EVoC's
   native metric — and breaks the blob. Flip the flag and watch the
   precision/recall trade-off.
3. **Precision stays low for open clusters** even in region mode (≈0.1–0.5).
   The field is chemically similar to solar-metallicity open clusters; the
   globulars (M 5 / M 3 / M 15) are the clean showcase (purity ≈ 0.4–0.5).
   This is the paper's own caveat (47 Tuc untaggable). Kinematics *confirm*
   membership — exactly the paper's conclusion.
4. **Chemical cohesion (kNN purity)** is the cleaner, parameter-free score:
   it asks "after embedding, do known members sit together?" — the paper's
   visual-polygon test, automated.
5. **Imputation decides who you study.** Strict complete-case silently
   deletes metal-poor globulars (M 15 / M 92 have `NaN` abundances from
   undetected weak lines). `IMPUTE_MISSING` recovers them.
6. **Full-run t-SNE ≈ 8 min** (sklearn Barnes-Hut is single-threaded);
   set `TSNE_BACKEND=opentsne` for multi-core (pays off only at full size).

## Target clusters

**25 clusters**: 23 from Garcia-Dias et al. (2019, Table 1) + the Pleiades
(Kos et al. 2017) + the two southern sweet-spots NGC 2243 and Collinder 261 —
**18 open clusters** (Pleiades, King 7, Berkeley 71, IC 166, NGC 2158, NGC 1245,
King 5, NGC 7789, NGC 1798, NGC 2420, NGC 6819, M 67, Berkeley 66, NGC 188,
NGC 6791, Berkeley 17, NGC 2243, Collinder 261) and **7 globulars** (M 5, M 3,
M 13, M 15, M 71, M 107, M 92).

## Repo layout

```
src/cluster/
  config.py      # all flags
  clusters.py    # cluster catalogue (seeds)
  download.py    # catalogue + HF asset-bundle downloader
  data.py        # load → cuts → matrix
  membership.py  # kinematic labels
  catalog.py     # Simbad referee membership
  gaia.py        # Gaia DR3 cone queries (Track B)
  literature.py  # literature cluster parameters
  isochrone.py   # ASteCA/PARSEC fitting (Track B)
  spectral.py    # embedding feature source (Track A)
  headtohead.py  # same-population comparisons
  benchmark.py   # t-SNE / UMAP / EVoC + scoring
  baseline.py    # paper baseline (cluster-only separation)
  provenance.py  # DR17-vs-DR19 artifact check
  plots.py       # embedding scatter
  cli.py         # `cluster download` / `run` / `baseline` / `head-to-head` / …
notebooks/       # marimo notebooks
hf/              # asset-bundle manifest, dataset card, publisher
.github/         # (in the repo root) multi-arch docker image + test workflows
```

Reference papers used during development are **not** redistributed here — the
citations in each doc link to the published versions.
