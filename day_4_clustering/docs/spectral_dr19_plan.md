# DR19 spectral re-analysis — plan & status

Re-run the DR17 spectral experiment matrix (docs/spectral_benchmark_results.md)
on the DR19 long re-train embeddings.

## Data

- Embeddings: `data/embeddings/attention_dr19_long.parquet` (256-d attention,
  200-epoch re-train, patience 25, lr 5e-4, batch 8, 31,506 stars).
- Catalogue: `data/astraAllStarASPCAP-0.6.0.fits.gz` (SDSS-V DR19).
- Cluster labels: kinematic σ-clip (the benchmark's own `label_clusters`).
- APO-only coverage: DR19 `allStar-1.3` is apo25m → ~11 of 25 clusters have
  spectral members (southern/LCO clusters missing — Collinder 261, NGC 2243,
  M 15, etc.).

## Experiment matrix (run via scripts/spectral_dr19_analysis.py)

| # | experiment | features | benchmark | status |
|---|---|---|---|---|
| 1 | baseline cluster-only | abundances (16-d) | t-SNE/UMAP/EVoC | ✅ script |
| 2 | baseline cluster-only | spectral (256-d) | " | ✅ script |
| 3 | baseline cluster-only | combined (272-d) | " | ✅ script |
| 4 | baseline cluster-only | kin-only (4-d) | " | ✅ script |
| 5 | baseline cluster-only | each of 1-3 + kin | " | ✅ script |
| 6 | field retrieval (scaled regions) | spectral | recall/precision | ✅ CLI `run --spectral --region-scaled` |
| 7 | field retrieval | spectral + kin | " | follow-up |
| 8 | field retrieval | kin-only | " | ✅ script |

## Follow-ups (from the DR17 matrix, not yet re-run)

- [ ] **Raw-spectra control** (8575-d + raw+abundance 8591-d) — cluster the
      raw flux directly; needs the flux CSV aligned to labels.
- [ ] **Phase B disentangled AE** — re-train `DisentangledSpectralAE` on DR19
      (64-d latent; DR17 result: below Phase A but the honest abundance-free arm).
- [ ] **Two-stage pipeline** (kin → spectral rejection) — re-run
      `scripts/two_stage_pipeline.py` with the DR19 spectral embeddings.
- [ ] **Per-cluster diagnostics** — subagents, one per cluster, for the
      confusion matrix + completeness + the globular/open split.

## Docs to update after the run

- `docs/spectral_benchmark_results.md` — append DR19 section.
- `docs/dr19_rerun_results.md` — add spectral table.
- `docs/narrative.md` — refresh Beats 2-6 numbers with DR19 spectral.
- Slides `IaaSoChemicalTaggingDeck.vue` §47c — swap the DR17 spectral numbers.

## Training status

- training log tail (GPU machine, upstream project): `grep -a Epoch <train>.log | tail`
  (output is block-buffered; expect a burst, or watch GPU utilisation).
