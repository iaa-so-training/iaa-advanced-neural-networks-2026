# DR19/DR4/DR3 full re-run results

>
> **Running these:** the commands below are written in the native form. In the
> Docker setup (the default — see `docs/docker.md`) prefix them with
> `docker run --rm -it $DAY4 $IMG`, e.g.
> `docker run --rm -it $DAY4 $IMG uv run cluster baseline --kinematics`; scripts
> become `… $IMG uv run python scripts/…`, and pass knobs with `-e`
> (`docker run --rm -it -e CLUSTER_MAX_STARS=10000 $DAY4 $IMG uv run cluster run`).

## Spectral embeddings (masked AE, self-supervised)

Added after the abundance re-run: a **masked spectral autoencoder** (no
abundance labels) gives the best cluster-only homogeneity. `cluster baseline
--spectral data/embeddings/masked_latent.parquet`:

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| abundances (16-d) | 0.292 | 0.547 | 0.409 |
| masked AE (256-d) | **0.789** | **0.874** | **0.770** |
| masked + abundance (272-d) | **0.879** | **0.874** | 0.712 |

Field retrieval (Simbad referee): masked+abundance UMAP precision 0.23 vs
abundances 0.12; kinematics cap the field at 0.30. Full detail in
`docs/spectral_benchmark_results.md`.

## Abundance re-run

Full 25-cluster benchmark re-run on the migrated data
(APOGEE SDSS-V DR19 `astraAllStarASPCAP`, GALAH DR4, Gaia DR3), `CLUSTER_FAST=0`
(all clean stars). Commands:

```bash
CLUSTER_FAST=0 uv run cluster baseline            # abundances only
CLUSTER_FAST=0 uv run cluster baseline --kinematics
CLUSTER_FAST=0 uv run cluster run                 # field retrieval, all-sky
```

Re-verified 2026-09-24 on the reference laptop (native, 16 threads): both
baseline tables below reproduce **to the last printed digit** (36 s and 35 s),
while the full all-sky `cluster run` takes **≈57 min** (3418 s) — budget an hour
for it, not a coffee break.

## Paper baseline — cluster-only separation (1002 member stars, 25 clusters)

| features | method | homogeneity | completeness | v-measure | accuracy |
|---|---|---|---|---|---|
| abundances (16-d) | t-SNE | 0.292 | 0.879 | 0.439 | 0.420 |
| abundances (16-d) | UMAP | 0.547 | 0.554 | 0.551 | 0.395 |
| abundances (16-d) | EVoC | 0.409 | 0.555 | 0.470 | 0.440 |
| abundances + kinematics | t-SNE | **0.767** | **0.832** | **0.798** | **0.724** |
| abundances + kinematics | UMAP | 0.748 | 0.666 | 0.705 | 0.506 |
| abundances + kinematics | EVoC | 0.680 | 0.683 | 0.681 | 0.520 |

## Field retrieval — all-sky (25 clusters scored, 1002 true members)

Two readings of the same command on the same data, different machines and thread
settings — the spread is the subject of `docs/reproducibility.md`, and both
readings keep the shape:

| features | method | recall (2026-08 → 2026-09-24) | precision (2026-08 → 2026-09-24) |
|---|---|---|---|
| abundances (16-d) | t-SNE | 0.091 → 0.081 | 0.120 → 0.105 |
| abundances (16-d) | UMAP | 0.079 → 0.060 | 0.112 → 0.088 |
| abundances (16-d) | EVoC | 0.314 → 0.279 | 0.001 → 0.0007 |

Consistent with the DR17 behaviour: kinematics dominate (the known ceiling),
abundance-only field retrieval is hard, and EVoC's high recall comes with
near-zero precision. Full per-cluster tables are in `results/dr19_*.txt`.

The second reading is the `cluster run` of the re-verification above (reference
laptop, native, 16 threads, `SEED=42`); the first is the run that produced this
table. Neither is portable to the last printed digit on other hardware — quote
them as ranges.