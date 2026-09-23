---
license: other
language:
- en
tags:
- astronomy
- spectroscopy
- chemical-tagging
- star-clusters
- sdss-apogee
- galah
- embeddings
pretty_name: Chemical tagging of star clusters — IAA-SO school 2026
size_categories:
- 1K<n<100K
---

# Chemical tagging of star clusters — embeddings & checkpoints

Data artifacts behind the **Unsupervised & Semi-Supervised Learning** session of the
[IAA-CSIC Severo Ochoa School on AI/ML in Astronomy 2026](https://www.granadacongresos.com/ai-ml)
(Granada, 28 Sep – 2 Oct 2026): latent embeddings, PCA baselines and model checkpoints for
benchmarking **t-SNE / UMAP / EVoC** on **chemical tagging** of star clusters
(APOGEE + Gaia + GALAH), reproducing and extending Kos et al. (2017).

The code lives in the workshop repository. These files have **no public upstream** — they were
produced on a GPU machine from public survey data and are published here so students can
reproduce the published tables without retraining.

## Download

**Recommended — one command inside the workshop repo:**

```bash
cluster download --assets            # ~1.0 GB, resumable, sha256-verified
cluster download --assets --list     # what is inside, with sizes + consumers
cluster download --assets --check    # verify what is already on disk
```

**Or straight from the Hub:**

```bash
hf download REPO_ID --repo-type dataset --local-dir data
```

Both routes land the files where the code expects them: parquets in `data/embeddings/`,
checkpoints in `data/embeddings/`, the optional archive in `data/`.

## Contents

`MANIFEST.json` is the source of truth: for every file it records the local path, byte size,
**sha256**, parquet shape, and which document/table/slide consumes it. Highlights:

| path | ~size | what it backs |
|---|---|---|
| `embeddings/attention_broad_merged.parquet` | 107 MB | **Track A** of the student activity: `cluster run --spectral` |
| `embeddings/masked_latent.parquet` | 64 MB | the headline masked-AE arm (head-to-head, `ablate`, `provenance`) |
| `embeddings/pca_64.parquet`, `pca_256.parquet` | 16 + 63 MB | the linear baselines the autoencoder is compared against |
| `embeddings/attention_dr19*.parquet` | 46 + 46 + 12 MB | supervised DR19 arms |
| `embeddings/masked_latent_dr17*.parquet` | 53 + 1 MB | DR17 arms used by the DR17-vs-DR19 provenance check |
| `embeddings/members_rawflux.parquet` | 40 MB | raw 8 575-px flux control arm |
| `models/masked_ae_rerun.pt`, `models/masked_ae.pt` | 35 MB each | **masked autoencoder** checkpoints — load with `cluster.models.MaskedSpectralAE`; `masked_ae_rerun.pt` is what `scripts/embed_dr19_rerun.py` uses |
| `models/model_dr19.pt`, `model.pt`, `model_long.pt` | 10 MB each | **supervised CNN-LSTM-attention** checkpoints — load with `cluster.models.CnnLstmAttention` |
| `models/model_convpool.pt` | 8 MB | supervised conv-pool checkpoint — `cluster.models.ConvPoolRegressor` |
| `optional/mwmstar.tar` | 237 MB | the 736 DR19 `mwmStar` FITS behind the uniform re-run |

Every parquet is keyed by `APOGEE_ID` plus one column per latent dimension (`z0…zN`).

## Loading the checkpoints

```python
# uv sync --extra torch
from cluster.models import MaskedSpectralAE, CnnLstmAttention, ConvPoolRegressor

ae = MaskedSpectralAE(n_features=8575, latent_dim=256)
ae.load_state_dict(torch.load("data/embeddings/masked_ae_rerun.pt", map_location="cpu"))

net = CnnLstmAttention(n_features=8575)                      # model*.pt
net.load_state_dict(torch.load("data/embeddings/model_dr19.pt", map_location="cpu"))
```

The architectures are vendored (MIT) in the workshop package as
`cluster.models`, so nothing here requires the upstream project. `tests/test_models.py`
loads every `.pt` in this bundle strictly, architecture-checked.

## Provenance

| artifact | built from |
|---|---|
| catalogue + labels | SDSS-V DR19 `astraAllStarASPCAP-0.6.0` (`[X/H]` → `[X/Fe]`), Gaia DR3 astrometry |
| DR19 spectra | `mwmStar-0.6.0` / `apStar-1.3-apo25m` (APO-North) |
| DR17 arms | SDSS-IV DR17 `allStar` + `aspcapStar` (kept for the provenance story) |
| GALAH (Track C) | GALAH DR4 (`galah_dr4_allstar_240705.fits` + Gaia DR3 VAC) |

## Caveats — read before quoting numbers

1. **Superseded arms.** The `*_v1` / `dr17` / `all_mixed_v1` files are historical. The
   comparable, uniform DR19 numbers come from `masked_latent.parquet` +
   `pca_{64,256}.parquet` on the 994 members / 39 945 field stars of one product.
2. **Product mismatch.** The old DR17-vs-DR19 split was a *data-product* artifact
   (raw `apStar` vs continuum-normalised `aspcapStar`), not a real batch effect — see the
   workshop `docs/spectral_benchmark_results.md`.
3. **PCA is competitive.** On the uniform sample the linear baseline is close to the
   autoencoder on t-SNE/UMAP; the gap is only clear on EVoC/UMAP. Quote the re-run tables,
   not the earlier ones.
4. **Licence**: these are derived products of public survey data (SDSS-V DR19, GALAH DR4,
   Gaia DR3) released for teaching. Confirm the final licence wording with the survey
   policies before any wider redistribution; attribution to SDSS-IV/V, GALAH and Gaia is required.

## Reproducing the artifacts

```bash
# masked-AE latents from raw spectra (needs the checkpoint + torch)
python scripts/embed_dr19_rerun.py --model models/model_dr19.pt --out data/embeddings/masked_latent_dr19_rerun.parquet

# everything else: the published tables
cluster head-to-head --out results/head_to_head/scores.csv
cluster baseline --spectral data/embeddings/masked_latent.parquet
```
