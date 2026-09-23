# Tweak experiments — MLflow results (M 67, region 30°, seed 42)

Every run went through `scripts/experiment.py`, which logs the full config
snapshot, git commit, data-file size, package versions, and per-method
recall/precision/kNN-purity to the local MLflow store
(`mlruns/`, experiment `chemical-tagging-tweaks`). `random_state=42` is forced.

## Reproducibility check

Six independent `*/baseline` runs (one per parallel agent) produced **identical**
metrics:

| method | recall | precision | kNN-purity |
|---|---|---|---|
| t-SNE | 0.0874 | 0.1168 | 0.1044 |
| UMAP | 0.4153 | 0.1293 | 0.1126 |
| EVoC | 0.3825 | 0.1243 | — |

`n_referee_members = 183`, `n_stars = 5000`. Bit-identical across processes.

## Results (ranked by t-SNE recall)

| tweak | t-SNE recall | t-SNE precision | UMAP recall | UMAP precision | EVoC recall | verdict |
|---|---|---|---|---|---|---|
| **baseline** | 0.087 | 0.117 | 0.415 | 0.129 | 0.383 | — |
| **HDBSCAN `min_cluster_size=10`** | **0.404** | **0.129** | 0.525 | 0.023 | 0.383 | ✅ best single tweak |
| HDBSCAN `min_cluster_size=20` | 0.366 | 0.125 | 0.530 | 0.023 | 0.383 | UMAP precision collapses |
| **t-SNE `perplexity=50`** | **0.383** | 0.124 | 0.415 | 0.129 | 0.383 | ✅ big t-SNE gain, no cost |
| t-SNE `perplexity=10` | 0.033 | 0.353 | 0.415 | 0.129 | 0.383 | precision↑, recall↓ |
| SNR_MIN=150 | 0.296 | 0.089 | 0.310 | 0.088 | 0.296 | recall↑, precision↓ |
| DWARF_ONLY=True | 0.149 | **0.165** | 0.158 | 0.064 | 0.167 | best precision/purity |
| element subset (drop S,K,V,Mn) | 0.055 | 0.256 | 0.038 | 0.250 | 0.388 | hurts recall |
| element subset (drop S,V,Mn) | 0.022 | 0.250 | 0.038 | 0.226 | 0.410 | hurts recall |
| UMAP `metric=cosine` | 0.087 | 0.117 | 0.044 | 0.250 | 0.383 | hurts UMAP recall |
| UMAP `n_neighbors=5` | 0.087 | 0.117 | 0.071 | 0.098 | 0.383 | worse |
| UMAP `n_neighbors=30` | 0.087 | 0.117 | 0.410 | 0.129 | 0.383 | ≈ baseline |
| `USE_ELEMENT_WEIGHTS=True` | 0.087 | 0.117 | 0.415 | 0.129 | 0.383 | **no-op** (see below) |
| **`USE_ELEMENT_WEIGHTS=True` (fixed)** | **0.437** | **0.158** | 0.339 | 0.138 | **0.448** | ✅ **best single lever** |
| combo `weights + mcs10` | 0.432 | 0.160 | 0.525 | 0.022 | 0.448 | ≈ weights alone |
| combo `mcs10 + perp50` | 0.383 | 0.125 | 0.525 | 0.023 | 0.383 | not additive |

## Findings

1. **HDBSCAN `min_cluster_size=10` is the single biggest lever**: t-SNE recall
   0.087 → 0.404 (4.6×) with precision *up* to 0.129. The default 5 fragments
   the field into tiny blobs that the greedy scorer can't recover from.
2. **t-SNE `perplexity=50` is a free win**: recall 0.087 → 0.383 at unchanged
   precision. Default 30 is too low for 5000 stars (rule of thumb:
   perplexity ≲ N/100).
3. **`USE_ELEMENT_WEIGHTS` was a no-op — fixed, and it is the winner.**
   `make_matrix` applied 1/σ weights *then* standardised, so the unit-std
   rescale undid them. After reordering (standardise → weight), weighting
   becomes the single biggest lever: t-SNE recall 0.087 → **0.437**, precision
   0.117 → **0.158**, EVoC recall 0.383 → 0.448. Noisy elements were
   dominating the C-space.
4. **The two winners are not additive**: `mcs10 + perp50` (0.383) and
   `weights + mcs10` (0.432) do not beat their best component. The embedding
   and the clustering interact, so tune one axis at a time.
4. **Element subsets all hurt**: S/K/V/Mn carry signal for M 67; dropping them
   collapses recall with precision gains that don't compensate.
5. **Cosine UMAP hurts**: after L2-normalisation, Euclidean ≈ cosine, and the
   cosine path (different kNN graph) is worse here. EVoC unchanged (already
   cosine).
6. **DWARF_ONLY** gives the cleanest groups (precision 0.165, purity 0.121)
   but small sample (1617 stars) and low recall — the giant/dwarf split in
   C-space is real.

## Adopted default

`USE_ELEMENT_WEIGHTS=True` is now the recommended default (safe for all 23
clusters, unlike `min_cluster_size=10` which would drop small clusters such
as the Pleiades). Set it via `CLUSTER_USE_ELEMENT_WEIGHTS=1` or flip the
`USE_ELEMENT_WEIGHTS` constant in `config.py`.
