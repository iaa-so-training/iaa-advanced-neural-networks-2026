# Region-mode sweep — benchmark results

> **Historical (APOGEE DR17).** These numbers were produced on
> `allStar-dr17-synspec_rev1.fits`, which `cluster download` no longer
> fetches — the project moved to SDSS-V DR19 (`docs/data_releases.md`).
> The commands below need that DR17 file supplied manually; they will
> not reproduce against a fresh checkout. For current numbers see
> `docs/dr19_rerun_results.md` and `docs/spectral_benchmark_results.md`.

Full per-cluster comparison of **t-SNE vs UMAP vs EVoC** under the target
paper's own setup: for each cluster, cut a 30° sky region, embed the 16-D
APOGEE abundance space, cluster, and score recovery against the kinematic
ground truth.

**Setup**: APOGEE DR17 allStar · Gaia EDR3 astrometry · `SNR ≥ 100` ·
`ASPCAPFLAG == 0` & `STARFLAG == 0` · 16 elements (C, N, O, Na, Mg, Al, Si,
S, K, Ca, Ti, V, Cr, Mn, Ni + [Fe/H]) · `IMPUTE_MISSING` (keep stars with
≥ 8 finite elements, fill the rest with column median) · zero-median/unit-std
standardisation · `NORMALIZE_ROWS` (L2-normalise so Euclidean ≡ cosine) ·
`random_state=42`.

**Metrics**:
- **recall** — fraction of a cluster's true members recovered by the best-overlapping predicted cluster (HDBSCAN for t-SNE/UMAP; EVoC labels).
- **precision** — purity of that predicted cluster.
- **kNN purity** — parameter-free chemical cohesion: fraction of a member's 10 nearest neighbours in the embedding that share its cluster (t-SNE/UMAP only; mirrors the paper's visual-polygon test).

## Macro (mean over clusters with ≥ 3 members)

| method | recall | precision | kNN-purity |
|---|---|---|---|
| t-SNE | 0.30 | **0.15** | **0.11** |
| UMAP | 0.55 | 0.06 | 0.09 |
| EVoC | 0.45 | 0.02 | — |

## Per cluster

| cluster | kind | n_members | n_field | t-SNE r / p / purity | UMAP r / p / purity | EVoC r / p |
|---|---|---|---|---|---|---|
| Pleiades | open | 10 | 15851 | 0.50 / 0.24 / 0.17 | 0.30 / 0.30 / 0.10 | 0.30 / 0.00 |
| King 7 | open | 3 | 19430 | 1.00 / 0.00 / 0.00 | 0.33 / 0.09 / 0.00 | 0.33 / 0.00 |
| Berkeley 71 | open | 9 | 23033 | 0.22 / 0.15 / 0.00 | 0.11 / 0.11 / 0.00 | 0.44 / 0.00 |
| IC 166 | open | 13 | 16290 | 0.15 / 0.01 / 0.00 | 0.85 / 0.00 / 0.00 | 0.23 / 0.00 |
| NGC 2158 | open | 6 | 26042 | 0.17 / 0.00 / 0.00 | 0.17 / 0.00 / 0.00 | 0.50 / 0.00 |
| NGC 1245 | open | 24 | 19073 | 0.21 / 0.12 / 0.09 | 0.12 / 0.16 / 0.05 | 0.88 / 0.00 |
| King 5 | open | 3 | 19859 | 0.33 / 0.14 / 0.00 | 1.00 / 0.00 / 0.00 | 0.33 / 0.00 |
| NGC 7789 | open | 48 | 15350 | 0.17 / 0.26 / 0.14 | 0.17 / 0.12 / 0.05 | 0.65 / 0.05 |
| NGC 1798 | open | 13 | 21437 | 0.15 / 0.29 / 0.03 | 1.00 / 0.00 / 0.02 | 0.54 / 0.00 |
| NGC 2420 | open | 13 | 22186 | 0.15 / 0.08 / 0.02 | 1.00 / 0.00 / 0.00 | 0.62 / 0.00 |
| NGC 6819 | open | 61 | 22721 | 0.08 / 0.23 / 0.08 | 1.00 / 0.00 / 0.02 | 0.25 / 0.03 |
| M 67 | open | 137 | 13636 | 0.05 / 0.47 / 0.12 | 0.61 / 0.05 / 0.08 | 0.49 / 0.05 |
| Berkeley 66 | open | 19 | 18210 | 0.05 / 0.03 / 0.00 | 0.05 / 0.03 / 0.00 | 0.11 / 0.00 |
| NGC 188 | open | 24 | 15458 | 0.25 / 0.08 / 0.16 | 1.00 / 0.00 / 0.08 | 0.75 / 0.01 |
| NGC 6791 | open | 14 | 22364 | 0.14 / 0.11 / 0.01 | 0.14 / 0.08 / 0.01 | 0.36 / 0.00 |
| Berkeley 17 | open | 8 | 21106 | 0.25 / 0.08 / 0.04 | 0.25 / 0.09 / 0.02 | 0.12 / 0.00 |
| M 5 | globular | 70 | 6937 | 0.57 / 0.26 / **0.44** | 0.41 / 0.07 / 0.45 | 0.97 / 0.12 |
| M 3 | globular | 110 | 9333 | 0.61 / 0.11 / **0.41** | 0.63 / 0.13 / 0.38 | 0.98 / 0.16 |
| M 13 | globular | 21 | 16585 | 0.48 / 0.02 / 0.11 | 0.90 / 0.00 / 0.08 | 0.00 / 0.00 |
| M 15 | globular | 11 | 7807 | 0.82 / 0.05 / **0.42** | 1.00 / 0.00 / 0.46 | 0.09 / 0.00 |
| M 71 | globular | 25 | 17716 | 0.12 / 0.50 / 0.12 | 0.12 / 0.07 / 0.08 | 0.12 / 0.00 |
| M 107 | globular | 10 | 17948 | 0.20 / 0.03 / 0.01 | 1.00 / 0.00 / 0.00 | 0.90 / 0.00 |
| M 92 | globular | 2 | 27278 | — | — | — |

## What it teaches

1. **Globulars tag cleanly, open clusters do not.** M 5 / M 3 / M 15 reach
   kNN-purity ≈ 0.4–0.5 — chemically distinct (metal-poor, homogeneous).
   Open clusters sit in a chemically similar solar-metallicity field, so
   purity stays ≲ 0.2. This mirrors the target paper (globulars recovered,
   open clusters harder; 47 Tuc untaggable).
2. **L2-normalisation is the precision lever.** Without `NORMALIZE_ROWS`,
   HDBSCAN merges the whole dense field into one blob (recall ≈ 1.0,
   precision ≈ 0.03 everywhere). Normalising makes Euclidean distance equal
   cosine (EVoC's native metric) and breaks the blob: precision rises ~3–5×
   (t-SNE 0.10 → 0.15; M 67 0.06 → 0.47) at a recall cost. A genuine
   precision/recall trade-off, and the reason the paper's t-SNE uses angular
   structure rather than raw Euclidean.
3. **t-SNE isolates best** (highest precision + purity), the reason the paper
   chose it. UMAP still shows residual recall-1.0 blobs on several open
   clusters — watch its precision column collapse to ≈ 0. EVoC is fast and
   competitive on recall, but its clusters are coarse at these sizes.
4. **M 15 / M 92 need imputation.** Their metal-poor members have several
   `NaN` abundances (weak, undetected lines). Strict complete-case drops them
   entirely (0 members); `IMPUTE_MISSING` recovers M 15 (11 members,
   t-SNE recall 0.82) and M 92 (2 members, still too few to score). A lesson
   in how data-cleaning choices silently delete the objects you came to
   study.

Reproduce with:

```bash
# needs the DR17 allStar (no longer downloaded by `cluster download`):
#   https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits
uv run python scripts/region_sweep.py --radius 30 \
    --allstar data/allStar-dr17-synspec_rev1.fits --out results/region_sweep.csv

# current (DR19) equivalent:
uv run python scripts/region_sweep.py --radius 30 \
    --allstar data/astraAllStarASPCAP-0.6.0.fits.gz --out results/region_sweep_dr19.csv
```
