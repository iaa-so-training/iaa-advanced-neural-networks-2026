# AGENTS.md — src/cluster (the pipeline)

The chemical-tagging pipeline and the `cluster` CLI. Everything that touches
the catalogue, builds features, clusters them, or scores the result lives here.
The exercises (`../exercises/AGENTS.md`) import *from* this package and never
duplicate its logic.

## Modules

| File | Purpose |
|---|---|
| `cli.py` | Click entry point (`cluster = "cluster.cli:main"`). Every subcommand is defined here |
| `config.py` | Central configuration. **Every knob is an env var** — see below |
| `data.py` | Load the DR19 Astra ASPCAP catalogue, apply quality cuts, build the C-space matrix |
| `download.py` | Fetch the catalogue (~1.17 GB) and the published embedding bundle |
| `clusters.py` | The target cluster catalogue (which objects the workshop studies) |
| `membership.py` | Kinematic ground-truth membership — the labels the benchmark scores against |
| `catalog.py` | External membership catalogue (Simbad), used as an independent referee |
| `benchmark.py` | The benchmark proper: t-SNE vs UMAP vs EVoC for chemical tagging |
| `baseline.py` | The published baseline (Garcia-Dias et al. 2019): multiclass separation, plus `recovery_fraction` |
| `headtohead.py` | Same-population head-to-head between feature sets, with seed error bars |
| `spectral.py` | Spectral-embedding feature source (the masked-autoencoder latents) |
| `provenance.py` | Provenance tracking for spectral embeddings — the DR17-vs-DR19 batch-effect check |
| `isochrone.py` | Isochrone fitting via ASteCA — a quantitative membership-quality proxy |
| `_parsec.py` | Vendored PARSEC/Padova CMD query helper (third-party; excluded from coverage) |
| `gaia.py` | Gaia DR3 photometry for deep CMDs |
| `literature.py` | Published parameters for the target clusters, for fit comparison |
| `stability.py` | Seed-stability and degeneracy checks |
| `seeding.py` | One place to seed everything stochastic |
| `schemas.py` | Pandera schemas validating dataframes at runtime |
| `net.py` | Bounded waits around archive calls |
| `plots.py` | Visualisation helpers (matplotlib **and** plotly — mind which you use) |
| `tracking.py` | Thin MLflow wrapper |
| `doctor.py` | The environment fingerprint a quoted number belongs to |

## CLI

```bash
uv run cluster --help
uv run cluster download --all          # catalogue + embedding bundle
uv run cluster run                     # prepare, benchmark, print the score table
uv run cluster doctor                  # environment fingerprint
```

| Command | Purpose |
|---|---|
| `download` | Fetch the DR19 catalogue and/or the embedding bundle |
| `run` | Prepare the data, run the benchmark, print the score table |
| `baseline` | Paper baseline: cluster-only multiclass separation |
| `head-to-head` | Compare feature sets on ONE common population, with seed error bars |
| `hr` | HR-diagram comparison: Simbad vs kinematic vs combined membership |
| `ablate` | Re-score with clusters removed — e.g. drop the globular M 3 |
| `provenance` | Batch-effect check: does the latent encode DR17 vs DR19? |
| `doctor` | Print the environment fingerprint a quoted number belongs to |

## Configuration is environment variables

`config.py` defines every knob through one helper:

```python
def env(name, default):
    return _parse(os.environ[f"CLUSTER_{name}"]) if f"CLUSTER_{name}" in os.environ else default
```

So **every module-level name in `config.py` is settable as `CLUSTER_<NAME>`** —
`CLUSTER_FAST=0`, `CLUSTER_MAX_STARS=50000`, `CLUSTER_SNR_MIN=80`,
`CLUSTER_ASTRA_ASPCAP_PATH=/data/…`. Values are parsed, so `0`/`false` become
`False` and comma lists become lists. This is how the tests exercise
configuration without editing files, and how the container points at a mounted
catalogue.

The ones you will actually reach for:

| Variable | Meaning |
|---|---|
| `CLUSTER_ASTRA_ASPCAP_PATH` | Where the DR19 catalogue lives (default `data/astraAllStarASPCAP-0.6.0.fits.gz`) |
| `CLUSTER_FAST` | Subsample for speed (default `True` — **the default is not the full run**) |
| `CLUSTER_MAX_STARS` | Row cap; `25_000` under `FAST`, unset otherwise |
| `CLUSTER_SNR_MIN` | Spectral S/N floor (default 100) |
| `CLUSTER_MIN_FINITE_ELEMENTS` | How many abundances a star must have (default 8) |
| `CLUSTER_REQUIRE_ELEMENT_FLAG_CLEAN` | Strict per-element flags (default `False` — **turning it on deletes M 15 and M 92 entirely**) |
| `CLUSTER_NET_TIMEOUT` | Bound on archive calls |
| `CLUSTER_CACHE_DIR` / `CLUSTER_NO_CACHE` | Cache location / bypass |

`CLUSTER_FAST` defaulting to `True` is the single most common source of
"my number doesn't match the docs": a quoted result is only comparable to a run
with the same fingerprint. `cluster doctor` prints that fingerprint, and
`docs/reference_runs/*.json` records it for the runs the workbook quotes.

## Conventions

- **Seed through `seeding.py`.** Every stochastic step — embedding, clustering, subsampling — draws from there. A number that moves between runs cannot be taught, and the workbook makes claims about seed stability that only hold if this is respected.
- **Validate at the boundary.** Dataframes crossing a module boundary get a pandera schema from `schemas.py`; a silent column rename is otherwise found three modules later as a wrong number.
- **`plots.py` has both matplotlib and plotly helpers.** Anything that reaches a shipped notebook or the workbook must be matplotlib — a plotly figure renders as an empty div in a committed `.ipynb` and cannot be exported into LaTeX. The plotly helpers (`hr_interactive`, `embedding_interactive`) are for live exploration only.
- **`_parsec.py` is vendored third-party code.** It is omitted from coverage and should not be reformatted or refactored to match house style.
- Network calls go through `net.py` so they are bounded; nothing in this package may hang indefinitely on an archive.

## Data contract

`data.prepare()` is the funnel: catalogue → quality cuts → C-space matrix. Its
result feeds the benchmark, the exercises and the workbook alike. Two things
about it are load-bearing and easy to get wrong:

- The quality cuts are **configuration**, not constants. Changing
  `CLUSTER_MIN_FINITE_ELEMENTS` or the flag-clean switches changes the star
  count *and which clusters survive at all*, which changes every downstream
  number.
- Under the shipped defaults it yields **1002 members across 25 clusters**.
  If you get a different count, your configuration differs from the one the
  docs and exercises quote — find out why before trusting anything you compute.

## Tests

`../../tests/` has one `test_<module>.py` per module here. Tests that need the
catalogue carry the `needs_data` marker and skip without it, because CI runs
with no catalogue at all (see `../../tests/AGENTS.md`).
