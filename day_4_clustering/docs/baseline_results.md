# Paper baseline — cluster-only multiclass separation

> **Historical (APOGEE DR17).** These numbers were produced on
> `allStar-dr17-synspec_rev1.fits`, which `cluster download` no longer
> fetches — the project moved to SDSS-V DR19 (`docs/data_releases.md`).
> The commands below need that DR17 file supplied manually; they will
> not reproduce against a fresh checkout. For current numbers see
> `docs/dr19_rerun_results.md` and `docs/spectral_benchmark_results.md`.

Re-creates Garcia-Dias et al. (2019, A&A 629, A34): take **only the known
cluster members** (no field stars), cluster them, and ask how well each star
is assigned back to its own cluster. Scored with the paper's own merit
functions — **homogeneity**, **completeness**, **v-measure**, and **accuracy**
(best label permutation via the Hungarian algorithm).

Reproduce with:

```bash
uv run cluster baseline                # abundances only (the 2019 setup)
uv run cluster baseline --kinematics   # abundances + parallax/PM/RV
```

> **Running these:** the commands below are written in the native form. In the
> Docker setup (the default — see `docs/docker.md`) drop `uv run cluster` and use
> `./run.sh` instead, e.g. `./run.sh baseline --kinematics`; scripts become
> `./run.sh python scripts/…` and `CLUSTER_*` variables pass straight through.

## Setup

APOGEE DR17 allStar · Gaia EDR3 astrometry · `SNR ≥ 100` ·
`ASPCAPFLAG == 0` & `STARFLAG == 0` · 16 elements (C, N, O, Na, Mg, Al, Si,
S, K, Ca, Ti, V, Cr, Mn, Ni + [Fe/H]) · kinematic membership (Gaia parallax
+ proper motion + RV, σ-clipped) · standardised (zero-median/unit-std), no
L2-normalisation, no element weights (the paper's standard-scaler setup) ·
`random_state=42` · clusters with ≥ 5 members (`min_members=5`).

`n_stars = 646`, `n_clusters = 20` (King 5, King 7, and M 92 drop out:
their kinematic membership is below 5).

## Results

### Abundances only (the 2019 question)

| method | homogeneity | completeness | v-measure | accuracy |
|---|---|---|---|---|
| t-SNE → HDBSCAN | 0.261 | **1.000** | 0.414 | 0.382 |
| UMAP → HDBSCAN | 0.320 | 0.895 | 0.471 | 0.415 |
| EVoC | **0.500** | 0.497 | 0.498 | 0.362 |

### Abundances + kinematics

| method | homogeneity | completeness | v-measure | accuracy |
|---|---|---|---|---|
| t-SNE → HDBSCAN | 0.763 | 0.829 | 0.794 | 0.720 |
| UMAP → HDBSCAN | **0.800** | 0.829 | **0.814** | **0.754** |
| EVoC | 0.716 | 0.690 | 0.703 | 0.546 |

## Interpretation

1. **Our abundances-only homogeneity (0.26–0.50) is far below the paper's
   0.85.** Two honest corrections explain the gap:

   - the paper's best result used **LDA**, a *supervised* projection that
     already knows the cluster labels; ours is fully unsupervised;
   - the paper's membership was **2σ-clipped in the same abundances** it
     then clustered — circular. Our kinematic (Gaia) membership has no
     such leak.

2. **t-SNE's completeness 1.0 vs homogeneity 0.26 is the blob.** HDBSCAN
   merges the similar-age, solar-metallicity open clusters into one predicted
   cluster — the paper's own "indistinguishable pairs" (NGC 2158–NGC 2420,
   NGC 2158–Pleiades, NGC 2420–Pleiades, M 15–M 92), now visible as a
   merged group rather than a score.

3. **Kinematics carry the separation chemistry can't.** Adding parallax,
   proper motion and radial velocity lifts homogeneity to 0.72–0.80 and
   accuracy to 0.55–0.75. Chemistry narrows; kinematics decide — the same
   conclusion as the field-retrieval benchmark, arrived at from the
   cluster-only direction.

This is the baseline against which the field-contaminated retrieval
benchmark (`docs/region_sweep_results.md`) is the follow-up: first "can we
tell the clusters apart?", then "can we pull one out of the field?".