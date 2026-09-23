# Spectral-embedding runbook (GPU machine → benchmark)

Step-by-step to go from the lightsurf RNN to a spectral-embedding benchmark
in the workshop repo. Training needs a GPU; everything downstream is CPU.

> The training steps below run the **upstream** project (`lightsurf`), which is a
> separate, private repository — its `scripts/train_embedding_model.py`,
> `combine_fits.py` and friends are deliberately not shipped here. What this
> repository ships instead are the **artifacts** they produced: the checkpoints in
> the Hugging Face bundle, loadable with the vendored `cluster.models`
> (`uv sync --extra torch`), plus the CPU-side embedding/benchmark code.

## 0. Data (SDSS-V DR19 — updated)

DR19 replaced the DR17 `aspcapStar` + `mdwarfs_DR17.fits` pair:

* **Spectra** — `apStar` (APOGEE redux 1.3), HDU 1 = flux `(n_rows, 8575)`,
  **row 0 = pixel-weighted combined spectrum** (HDU 2 error, HDU 3 mask).
  Exact filename comes from `allStar-1.3-apo25m.fits` column `file`
  (e.g. `apStar-1.3-apo25m-2M00002479+6025311-59830.fits`).
* **Labels** — `astraAllStarASPCAP-0.6.0.fits.gz` (`[X/H]` abundances, converted
  to `[X/Fe]` by the workshop loader `cluster.data.load_allstar`).

Download apStar (the `allStar-1.3-apo25m.fits` `uri` column holds the exact
relative path — use it directly; the SAS serves it under `spectro/apogee/`):

```bash
# URL = https://dr19.sdss.org/sas/dr19/ + uri.replace("apogee/spectro", "spectro/apogee", 1)
# e.g. .../spectro/apogee/redux/1.3/stars/apo25m/12/12640/apStar-1.3-apo25m-2M00002479+6025311-59830.fits
# ready-made: scripts/build_dr19_star_list.py + scripts/download_apstar_dr19.py
```

Note: the field grouping is **not** `healpix//256` — the `uri` column is the
only reliable source (it also carries the `{field}/{healpix}` pair).

Build the star-list FITS (columns `APOGEE_ID`, `TEFF`, `LOGG`, `FE_H`,
`C_FE`, `CA_FE`, `K_FE`, `MG_FE`, `NI_FE`, `O_FE`, `SI_FE`, `TI_FE`,
`telescope`, `healpix`, `file`) from the workshop loader + allStar-1.3 join on
`sdss_id`. Then `lightsurf combine_fits.py` (`extract_flux` already takes
apStar row 0) produces `flux_abundances.csv` as before.

## 1. On the GPU machine — train + export

```bash
cd lightsurf
uv sync
uv run python scripts/train_embedding_model.py \
    --spectra data/raw_data/flux_abundances.csv \
    --layer attention \
    --epochs 100 \
    --model-out models/cnn_lstm_attention_model.keras \
    --embeddings-out data/embeddings/attention.parquet
```

- Trains `CnnLstmAttentionModel(output_dimension=9)` (all 9 APOGEE targets
  simultaneously → shared chemical latent).
- Saves the `.keras` model and the latent embeddings parquet (keyed by
  `APOGEE_ID`, one column per latent dimension).
- Other layers: `--layer dense_0` (20-d) or `--layer dense_1` (8-d).
- `train_embedding_model.py` now strips `apStar-1.3-apo25m-` + trailing
  `-MJD` (and still accepts `aspcapStar-dr17-`) when building `APOGEE_ID`.

## 2. Transfer to this machine

```bash
scp <gpu>:lightsurf/data/embeddings/attention.parquet \
    data/embeddings/attention.parquet
# (the .keras is only needed to re-extract embeddings; the parquet is enough
#  for the benchmark itself)
```

## 3. On this machine — benchmark abundances vs embeddings

Both benchmarks now take a `--spectral <parquet>` flag (loads the artifact,
inner-joins on `APOGEE_ID`, standardises + L2-normalises, and swaps the
abundance matrix):

```bash
# paper baseline (cluster-only separation):
uv run cluster baseline --spectral data/embeddings/attention.parquet
uv run cluster baseline --spectral data/embeddings/dense_1.parquet  # 8-d tap

# field retrieval (region mode):
uv run cluster run --spectral data/embeddings/attention.parquet --cluster "M 67" --region 30
```

Run each once with `--spectral` and once without (abundances) and compare the
tables. The module + `test_spectral.py` prove the path end-to-end on synthetic
data.

## 4. Compare the two feature sources

| metric | abundances (16-d) | embeddings (8/20/256-d) |
|---|---|---|
| paper baseline homogeneity/v-measure | `docs/baseline_results.md` | run `cluster baseline` on the spectral matrix |
| field retrieval recall/precision | `docs/region_sweep_results.md` | run `cluster run` on the spectral matrix |
| kNN purity | 0.11 t-SNE | to run |

Success = embeddings beat abundances on ≥ one metric, or tie at lower
dimension (8-d vs 16-d = a useful denoised compression).

## 5. Phase B (disentangled autoencoder)

Implemented in lightsurf
(`src/lightsurf/domain/models/disentangled_ae.py`, 2 smoke tests):
`DisentangledSpectralAE` — CNN+LSTM encoder → chemical latent `z`;
decoder `z + Teff/logg → spectrum`; gradient-reversal head makes `z`
uninformative about Teff/logg. `loss = MSE(spectrum) + λ · MSE(Teff/logg)`.

Train + export (GPU machine), the same interface as Phase A:

```bash
uv run python scripts/train_disentangled_ae.py \
    --spectra data/raw_data/flux_abundances.csv \
    --lambda 0.1 --epochs 100 \
    --model-out models/disentangled_ae.keras \
    --embeddings-out data/embeddings/disentangled.parquet
```

This is the design that goes *beyond* the 16 ASPCAP abundances (de Mijolla
et al. 2021, arXiv:2103.06377).

## Commits

| repo | commit | change |
|---|---|---|
| lightsurf | `e2e2a6a` | embedding taps + multi-task output head |
| lightsurf | `3524e8b` | export embeddings script |
| lightsurf | `71556e4` | export keyed by APOGEE_ID |
| lightsurf | `d35a270` | train + export single command |
| lightsurf | `dcd5767` | DR19 apStar pipeline (extract_flux row 0, download URL, prefix) |
| workshop | `de95802` | spectral feature source + plan doc |
