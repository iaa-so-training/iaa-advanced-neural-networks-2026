# Reproducibility — what is guaranteed, and what is only recorded

`cluster run --fast` on a laptop takes about two minutes and prints a score
table. Running the same command on a different machine prints *nearly* the same
table, and the difference is not a broken setup — it is how the numerical stack
reduces floating point. This document splits the guarantee into three tiers,
says which tier each quoted number belongs to, and gives the recipe for
regenerating those numbers with their environment attached.

## Three tiers

| tier | promise | what is in it |
|---|---|---|
| **exact** | identical, and tested | catalogue bytes; asset bundle; sampled population; run-to-run determinism on one machine; native ↔ container parity on one machine |
| **tolerance** | stable to ~±0.02 | the macro recall/precision of t-SNE, UMAP and EVoC *across* machines |
| **illustrative** | not promised | exact per-cluster recall/precision on hardware other than the reference machine |

Why the middle tier exists: sklearn's Barnes-Hut t-SNE (OpenMP), numba (UMAP,
EVoC, HDBSCAN) and BLAS each reduce floating-point sums in an order that depends
on the CPU and its thread count, so identical code on identical data lands on
slightly different floats elsewhere. Pinning `OMP_NUM_THREADS=1` removes part of
that (measured below: it moves t-SNE recall from 0.209 to 0.195) but not all of
it, and it costs speed — which is why the workshop does **not** ask anyone to
pin threads.

## What is exact (tier 1)

- **Catalogue** — `data/astraAllStarASPCAP-0.6.0.fits.gz`, 1 171 102 556 bytes,
  sha256 `5324bf39baeede7553b0a3c8a50bea1760f47eb67a95a8a84e6db303aa2edb04`
  (written to a `.sha256` sidecar by `cluster download --all`; see
  `docs/data_releases.md`).
- **Asset bundle** — 31 files, sha256-verified against `hf/MANIFEST.json`
  (`cluster download --assets --check` reports 0 problems).
- **Population** — the `--fast` sample is 25 000 stars: 1002 kinematically
  selected member stars across 25 clusters, of which 829 member stars in 24
  clusters are scored (a cluster needs ≥ 5 members to enter the score), plus
  23 998 field stars.
- **Run-to-run determinism** — same machine, same thread setting, identical
  scores on every repeat. `tests/test_reproducibility.py` pins this in CI.
- **native ↔ container** — identical on the same machine and thread setting
  (verify with a new pair any time; it is two of the rows below).

## What moves (tier 2) — the readings behind the quoted ranges

Reference machine: AMD Ryzen 9 8945HS, 16 threads, Linux. All rows are
`cluster run --fast` on the same data, 2026-09-24; the image rows name the
release they were produced on.

| configuration | t-SNE recall / precision | UMAP recall / precision | EVoC recall / precision |
|---|---|---|---|
| native, default threads | 0.2093 / 0.2230 | 0.2227 / 0.1333 | 0.4828 / 0.0024 |
| container `day4-v7` (= `:latest`, `sha256:b1590f7d…`), default threads | 0.2093 / 0.2230 | 0.2227 / 0.1333 | 0.4828 / 0.0024 |
| container `day4-v4`, `day4-v5`, `day4-v6`, default threads | 0.2093 / 0.2230 | 0.2227 / 0.1333 | 0.4828 / 0.0024 |
| container, threads pinned to 1 | 0.1949 / 0.2287 | 0.2227 / 0.1333 | 0.4918 / 0.0026 |
| container `day4-v2` (*unpinned* base images), default threads | 0.1974 / 0.2241 | 0.1755 / 0.1499 | 0.5092 / 0.0024 |
| container `day4-v2`, threads pinned to 1 | 0.2068 / 0.2303 | 0.1755 / 0.1499 | 0.4543 / 0.0025 |

Two things to read from that table:

1. **The base image is part of the environment.** The `day4-v2` image was built
   from `python:3.13-slim` and `uv:latest`; by the time it was re-measured, the
   tag pointed at a different interpreter, and the container disagreed with
   native by 0.012 on t-SNE and 0.047 on UMAP — the same code, the same data.
   Since `day4-v4` the bases are pinned (`python:3.13.15-slim`, uv 0.12.18) and
   native and container agree to the last printed digit on three consecutive
   releases (`day4-v4` … `day4-v7`), each re-measured end to end.
2. **What remains is the tolerance.** Pinning threads (row 4) still moves t-SNE
   and EVoC; UMAP does not move at all here because it is pinned internally by
   `random_state`. Across the rows, recall spans roughly ±0.03.

So the workshop quotes two decimals and a range:

| method | recall | precision |
|---|---|---|
| t-SNE | ≈0.19–0.21 | ≈0.22–0.23 |
| UMAP | ≈0.18–0.22 | ≈0.13–0.15 |
| EVoC | ≈0.45–0.51 | ≈0.002–0.003 |

**The cross-machine spread is not measured yet.** Every row above is one laptop
plus its thread settings and two image builds. The school also supports Intel
and Apple-silicon laptops; those have not been read. If you run the recipe below
on one, add the row — the format is designed for it.

## The reference recipe

Regenerate a quoted number *with its environment attached*:

```bash
export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

# the score table plus a JSON carrying the environment that produced it
docker run --rm -it $DAY4 $IMG uv run python scripts/reference_run.py --fast \
  --out /app/results/ref_$(hostname).json

# the same fingerprint, on screen: seed, threads, versions, hashes, image, git
docker run --rm -it $DAY4 $IMG uv run cluster doctor
docker run --rm -it $DAY4 $IMG uv run cluster doctor --deep   # re-hash the data too
```

To pin threads the way row 4 does, add
`-e OMP_NUM_THREADS=1 -e NUMBA_NUM_THREADS=1`. To freeze the environment, pull by
digest rather than by tag:

```bash
docker pull ghcr.io/iaa-so-training/day4-clustering@sha256:b1590f7dc77815ba502f1a60d5cba2cf7a502f6e3e6b570a72b1af46cf96a36b
```

The tag moves with each release; the digest does not. The committed readings are
in `docs/reference_runs/*.json` — one file per configuration, each with
`environment`, `population`, `macro` and `per_cluster_recall`. That directory,
not the prose, is the source of truth for any exact number quoted in this
workshop.

## Knobs that change a number

| knob | effect | default |
|---|---|---|
| `--seed N` on `cluster run` / `cluster baseline`, or `CLUSTER_RANDOM_STATE` | the seed everything stochastic derives from; printed as `SEED=` in the settings line and logged to MLflow | 42 |
| `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | thread pinning — slower, changes the last digits | unset (all cores) |
| `PYTHONHASHSEED` | iteration order of hash-based containers; fixed in the image | 42 (image), unset natively |

`cluster doctor` prints all of them, so no number in an issue or a slide has to
be anonymous.

## Caching (and why it is tier 1)

The prepared sample — 1.17 GB of gzipped FITS read, quality cuts, membership,
subsample, standardisation — is cached on disk
(`results/cache/prepared/`) and restored on the next call. This belongs to the
**exact** tier, not to a hurry:

- the key is `sha256(cache format · catalogue identity · full settings · cluster
  list · seed arguments)`, where the catalogue identity is its size + mtime plus
  the sha256 recorded in the downloader's sidecar. Change the data, a knob or the
  seed and you get a different entry;
- the restored frame and matrix are **bit-identical** to a cold computation, and
  the suite pins that (`tests/test_data_cache.py`, 9 tests, including one that
  booby-traps the FITS reader and still gets a sample);
- `cluster doctor` reports what is in the cache; `cluster run --no-cache` /
  `cluster baseline --no-cache` / `CLUSTER_NO_CACHE=1` recompute regardless;
  `CLUSTER_CACHE_DIR=<dir>` moves the directory;
- the *benchmark* is not cached on disk — a new configuration is always a real
  computation. The notebooks memoise their own calls with `mo.cache` (keyed by
  argument values, in memory), which changes only how often work is done, never
  what a number is.

The same file mounted at a different path shares an entry: the identity uses the
file name and bytes, not the mount path, so `./data/...` on the host and
`/app/data/...` in the container hit one cache.

## Where the seeds live (audit)

- field subsample and data prep — `settings.random_state` (`src/cluster/data.py`;
  `scripts/build_dr19_star_list.py --seed`)
- t-SNE / UMAP / EVoC — `settings.random_state` (`benchmark.py`, `baseline.py`)
- stability, `ablate`, `head-to-head` — explicit `--seeds 42,0,1,2,7,13,99`
- ASteCA isochrone fits and emcee — `isochrone.py` (emcee draws from the global
  RNG, which `cluster.seeding.seed_everything` seeds)
- batch-effect probe and the violin-plot subsample — `settings.random_state`
  (hardcoded 0 and 42 before the 2026-09-24 review)

## Known gaps

- `run.ps1` (the optional PowerShell wrapper) has never been run on Windows or
  WSL — no Windows machine was available. `docker run …` is the verified path;
  the wrapper is a convenience only.
- The cross-machine spread (see the note above).
- `docs/region_sweep_results.md` carries a DR17-era table, annotated there. A
  fresh all-cluster 30° sweep is a multi-hour run on a laptop; the current
  single-cluster M 67 reading is in `docs/reference_runs/`, and the per-cluster
  table regenerates with `scripts/region_sweep.py` (documented in that file).
