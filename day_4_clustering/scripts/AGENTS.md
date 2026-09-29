# AGENTS.md — scripts

Research scripts, artifact generators, and re-run tooling. Unlike `src/`, this
tree is **not** a library: nothing here is imported by the pipeline or the
exercises, and scripts are free to be single-purpose and dated.

Three different kinds of thing live here, and the distinction matters:

## 1. Generators whose output is committed

These produce artifacts that ship. Changing one means regenerating and
committing the result.

| Script | Produces |
|---|---|
| `make_exercise_notebooks.py` | The 17 exercise decks in `notebooks/`. **CI runs its `--check`** |
| `make_results_figures.py` | The results figures for the deck/workbook |
| `make_teaching_assets.py` | Didactic plots + GIFs for the lecture deck |
| `make_masked_ae_figures.py` | Teaching figures for the masked spectral autoencoder |
| `make_sky_images.py` | DSS2 colour composites for cluster slides, via SkyView |
| `reference_run.py` | `docs/reference_runs/<name>_<date>.json` — the readings the docs quote |

```bash
uv run python scripts/make_exercise_notebooks.py --check   # what CI runs
```

`teaching_assets/` is the support package for `make_teaching_assets.py`
(`common.py`, `embeddings.py`, `iterative.py`, `steps.py`) — the one place here
that is importable code rather than a script.

## 2. Runners and diagnostics

| Script | Purpose |
|---|---|
| `run_notebooks.py` | Execute the decks end to end and time each one (works on **copies**) |
| `run_all_exercises.py` | Run every exercise module's `solve()` and report what breaks |
| `experiment.py` | One reproducible benchmark config, logged to MLflow |
| `phase_profile.py` | Where the wall clock goes, and how many cores are really used |
| `per_cluster_diagnostics.py` | Per-cluster diagnostics for the masked-AE latent |
| `diagnose_two_stage.py` | Is a two-stage pipeline viable? |
| `diagnose_product_mismatch.py` | Why the DR17 backfill is not comparable to DR19 embeddings |

## 3. The DR19 re-run chain

The workshop's numbers come from one **uniform** DR19 product. These scripts
build it, and their order is load-bearing:

```
build_dr19_*_list.py  →  download_apstar_dr19.py / parallel_download.py  →  embed_dr19_rerun.py
```

| Script | Step |
|---|---|
| `build_dr19_star_list.py` | The cluster-aware star list for the spectral re-embed |
| `build_dr19_rerun_list.py` | mwmStar list for members previously backfilled from DR17 |
| `build_dr19_all_members_list.py` | mwmStar list for *all* cluster members |
| `build_dr19_field_list.py` | The field-sample list |
| `download_apstar_dr19.py`, `parallel_download.py` | Bulk spectrum download |
| `embed_dr19_rerun.py` | Re-embed into one uniform product |
| `baseline_uniform.py`, `field_retrieval_uniform.py`, `headtohead_extras.py` | Re-score on the uniform common population |

`build_dr17_missing_list.py` is **deprecated and built on a false premise** —
its own docstring says so. Use `build_dr19_rerun_list.py`. Do not revive it
because it looks like it fits; mixing DR17 and DR19 products is precisely the
batch effect `cluster provenance` exists to detect.

## 4. Analysis one-offs

`region_sweep.py`, `spectral_region_sweep.py`, `spectral_dr19_analysis.py`,
`rerun_combined.py`, `build_galah_apogee.py`, `sweet_spot.py`, `red_clump.py`,
`isochrone_check.py`, `isochrone_grid.py`, `verify_isochrone_finale.py`,
`two_stage_pipeline.py`.

These back specific claims in `docs/` and the workbook. A script here that
disagrees with a published number is usually the *interesting* result — the
workbook's subject is reproducibility — so report the discrepancy rather than
tuning until it matches.

## Conventions

- Run through uv: `uv run python scripts/<name>.py`.
- **Heavy by design.** Several read the full catalogue or hit archives; check what a script does before running it casually, and prefer running it in the background.
- Scripts write to `results/` and `data/`, both gitignored. Only the generators above produce committed artifacts.
- A number that lands in `docs/` must come with the environment fingerprint that produced it — that is what `reference_run.py` is for.
