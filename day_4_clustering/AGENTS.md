# AGENTS.md — day_4_clustering

## Project Overview

Day 4 of the IAA advanced neural networks school: **unsupervised chemical
tagging of open clusters**. Given stellar abundances from SDSS-V DR19 Astra
ASPCAP, can clustering recover the birth clusters without ever being told the
labels? The answer the material builds towards is *partly* — and the interesting
teaching is in *why* the honest number is lower than the published one.

The repository is three things at once, and a change to one usually implies the
others:

| Layer | Lives in | What it is |
|---|---|---|
| The pipeline | `src/cluster/` | The `cluster` CLI: download, prepare, embed, cluster, score |
| The workbook | `article/` | The LaTeX text students read, with 56 exercises |
| The exercises | `src/exercises/`, `notebooks/` | One module per exercise, rendered into Jupyter decks |

**The golden rule: every number asserted anywhere comes from a real run.** Not
from the paper, not from memory, not from a plausible estimate. The workbook's
own subject is that published numbers are hard to reproduce, so a fabricated
number here would be self-refuting. If you cannot run it, say so — do not
approximate.

## Repository Structure

```
day_4_clustering/
├── src/cluster/      # The pipeline package + `cluster` CLI entry point
├── src/exercises/    # One module per workbook exercise (56) + shared helpers
├── notebooks/        # Generated Jupyter decks (master + 16 per-chapter)
├── scripts/          # Research scripts, re-runs, generators (35 files)
├── tests/            # pytest suite (~812 tests) over both packages
├── article/          # The LaTeX workbook, figures, bibliography
├── docs/             # Result write-ups + recorded reference runs
├── hf/               # Hugging Face dataset publishing
├── data/             # Catalogue + embeddings (gitignored, downloaded)
└── results/          # Run outputs (gitignored)
```

Detail lives with the code it describes. This file keeps what is true across
day 4; each tree's own `AGENTS.md` carries the rest:

| File | Covers |
|---|---|
| `src/cluster/AGENTS.md` | Pipeline modules, the CLI, config, seeding, data contracts |
| `src/exercises/AGENTS.md` | The exercise-module contract, citations, figures |
| `notebooks/AGENTS.md` | Deck generation, the present-don't-compute rule, `--check` |
| `scripts/AGENTS.md` | What each script is for, and which produce shipped artifacts |
| `tests/AGENTS.md` | Suite layout, the data-dependent skip guard, what is actually guarded |
| `article/AGENTS.md` | Workbook build, chapters, figures, the bibliography contract |
| `docs/AGENTS.md` | Result documents and the recorded reference runs |
| `hf/AGENTS.md` | Dataset publishing to the Hub |

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package mgmt | **uv** (`uv sync`, `uv run`) — never bare `pip` |
| Data | astropy, pandas, numpy, pyarrow |
| ML | scikit-learn, UMAP, EVoC, HDBSCAN, PyTorch (optional extra) |
| Plotting | matplotlib (shipped figures), plotly (interactive, exploration only) |
| Astronomy | astropy, astroquery, ASteCA (isochrones), Gaia/Simbad/SkyView |
| Notebooks | JupyterLab, nbformat + nbclient (generation and execution) |
| Validation | pandera (runtime data schemas) |
| Tracking | MLflow (optional) |
| Testing | pytest |
| Types | pyrefly |
| Containers | Docker + Docker Compose |
| Text | LaTeX (the `ar-1col` Annual Reviews class) |

## Common Commands

All commands run from `day_4_clustering/`.

```bash
uv sync                          # Install (add --extra torch for spectral work)
uv run cluster --help            # The pipeline CLI
uv run pytest -q                 # The whole suite (~700 tests, ~5 min)
uv run pyrefly check src/        # Type check
```

The CLI's commands:

| Command | Purpose |
|---|---|
| `cluster download` | Fetch the DR19 catalogue (~1.17 GB) and/or the embedding bundle |
| `cluster run` | Prepare data, run the benchmark, print the score table |
| `cluster baseline` | The paper baseline: cluster-only multiclass separation |
| `cluster head-to-head` | Compare feature sets on one common population, with seed error bars |
| `cluster hr` | HR-diagram comparison across membership definitions |
| `cluster ablate` | Re-score with clusters removed (e.g. drop the globular M 3) |
| `cluster provenance` | Batch-effect check: does the latent encode DR17 vs DR19? |
| `cluster doctor` | Print the environment fingerprint a quoted number belongs to |

## The container is the supported path

Students run the decks in Docker, not in a bare venv:

```bash
mkdir -p data results notebooks
docker compose up               # JupyterLab on http://localhost:9999
docker compose run --rm jupyter uv run cluster download --all
JUPYTER_PORT=9998 docker compose up   # when 9999 is taken
```

`data/`, `results/` and `notebooks/` are bind-mounted, so work survives the
container and a download done inside it lands on the host.

**`article/references.bib` must be in the image.** Every exercise module builds
its reference block at *import* time, so without the bib every deck fails at
the first cell with `FileNotFoundError`. The `Dockerfile` copies it after
`uv sync`, so editing the bibliography does not invalidate the dependency layer.

## Data

The DR19 catalogue is ~1.17 GB and **not** in git. `CLUSTER_ASTRA_ASPCAP_PATH`
overrides its location; without the file, data-dependent tests skip rather than
fail (see `tests/AGENTS.md`), and exercise modules raise `DataNotAvailable`
naming the command that would fetch it.

### What a student can actually get

Three tiers, and they are not equal — this matters when an exercise fails:

| Tier | Command | What it gets |
|---|---|---|
| Catalogue | `cluster download` | The DR19 Astra ASPCAP file (~1.17 GB) |
| Assets | `cluster download --assets` | Embeddings, PCA baselines, model checkpoints (32 files, ~1.29 GB) |
| Optional | `cluster download --assets --with-optional` | The raw mwmStar spectra archive (~240 MB), **not fetched by default** |

The optional archive needs unpacking after download, which nothing does for you:

```bash
uv run cluster download --assets --with-optional
tar -xf data/mwmstar.tar -C data/        # yields data/mwmstar/ (736 spectra)
```

**Some results are not downloadable at all** — they are *computed* by a script
and land in `results/`. Exercise 13.4 needs
`results/casamiquela_comparison.csv`, which is not in the HF bundle and never
will be: it is a derived scoring table, regenerated with

```bash
uv run python scripts/casamiquela_comparison.py     # needs the catalogue; slow
```

Likewise `data/galah_apogee_Collinder_261.parquet` (exercise 15.1) is built by
`scripts/build_galah_apogee.py`. When you add an exercise that reads a file,
decide which tier it belongs to and make the `DataNotAvailable` message say so
exactly — a message naming the wrong command is worse than no message, because
the student cannot tell the difference.

Never commit anything under `data/` or `results/`, and never add the published
PDFs in `ref/` — they are copyright, and git history is permanent and public.

## Conventions

- **uv for everything.** `uv run <cmd>`, never a bare `python` against system site-packages.
- **Seed every stochastic step** through `cluster.seeding` — a number that moves between runs cannot be taught.
- **matplotlib for anything shipped.** Plotly is fine for exploration but renders as an empty div in a committed notebook, so it never reaches a deck or the workbook.
- **Cite by bibliography key**, never a hand-typed author-year string. An unknown key must raise, not silently render.
- **Real runs only.** Recording a number in `docs/` means it came out of a run whose environment fingerprint (`cluster doctor`) is recorded alongside it.
- Line length 100, ruff-formatted, type-annotated. `uv run pyrefly check src/` before you claim a change is clean.

## CI

`.github/workflows/day4-tests.yml` runs the suite, gated on `day_4_clustering/**`
and its own path, **without the catalogue** — so anything data-dependent must
carry the skip guard or CI goes red on a machine that can never have the data.
It also runs `scripts/make_exercise_notebooks.py --check`, which fails if the
shipped decks have drifted from the exercise modules.

`day4-docker-image.yml` builds the image with `context: day_4_clustering`.
