# Plan: spectral-RNN embeddings for chemical tagging

> Paths such as `scripts/export_embeddings.py`, `scripts/fetch_spectra.py` and
> `src/lightsurf/…` below belong to the **upstream** `lightsurf` repository (a
> separate, private checkout — see `docs/spectral_gpu_runbook.md`), not to this
> tree. This repository ships the CPU-side code and the artifacts those scripts
> produced; the names are kept so the plan can be read against that history.

## Goal

Train the **lightsurf** CNN-LSTM-Attention network to regress APOGEE chemical
abundances from the raw 8575-pixel H-band spectrum (on a GPU machine), export
the trained model, extract its **latent embeddings**, and use those embeddings
as clustering features for chemical tagging — benchmarked head-to-head against
the current 16-D ASPCAP abundance features in the workshop pipeline.

## 1. Motivation & hypothesis

- **Now**: clustering features = 16 ASPCAP abundances ([X/Fe] + [Fe/H]).
- **Proposal**: feed the full spectrum through a trained RNN; take the
  penultimate layer as the feature vector.
- **Hypothesis**: the embedding is a learned, denoised, compressed
  representation of the *full* spectrum — it sees continuum shape, line wings,
  and blends that 16 summary abundances discard. It should therefore be
  **≥** the abundance features for tagging, at lower dimensionality.
- **Risk (circularity)**: a supervised embedding trained only to regress the
  16 ASPCAP abundances *cannot* contain more chemical information than those
  abundances — it is at best a denoised compression. Real novelty requires
  either (a) multi-element targets beyond ASPCAP's 16, or (b) a
  disentangled / self-supervised objective (see §2). This is the honest
  benchmark question the pipeline must answer.

## 2. Literature findings

| work | method | relevance |
|---|---|---|
| Leung & Bovy 2019 (astroNN, arXiv:1808.04428) | supervised CNN regresses multi-element abundances from APOGEE spectra, per-element wavelength windows | validates spectrum→abundances; **multi-element** design |
| Ting et al. 2019 (The Payne) | NN spectral model, all labels fit **simultaneously** | multi-task beats single-task |
| Casey et al. 2020 (ApJ 889, 30) | latent nucleosynthetic factors, lower-dim latent for tagging | alternative latent structure |
| Disentangled repr. learning (arXiv:2103.06377; ApJ 2021) | learns a **chemical latent disentangled from Teff/logg**, no label catalogue | the SOTA direction; our supervised regression is its baseline |
| "Model-free abundances" (A&A 2025, aa55376-25) | VAE, per-element decoders → chemically meaningful latent | self-supervised path |
| Spina et al. 2025 (deep chemical tagging, GAT) | graph-attention autoencoder over chemistry + kinematics + age | the "informed" extension we already parallel |
| de Mijolla, Ness, Viti & Wheeler 2021 (arXiv:2103.06377) | **conditional autoencoder** `enc(spectrum, Teff/logg)→z`, `dec(z, Teff/logg)→spectrum`, loss = MSE + λ·disentanglement(z ⊥ Teff/logg). Two losses: FaderDis (adversarial critic) and FactorDis (scramble u,z pairs + WGAN critic) | the **abundance-free** chemical latent — the real target design |
| Price-Jones & Bovy 2019 | fit non-chemical params per wavelength bin, take residuals → PCA → cluster | the non-deep baseline the above beats |
| Ness et al. 2018 | "doppelganger rate": ~1% of field stars chemically identical to cluster stars using ~20 abundances | **why** the full-spectrum latent matters — abundances alone saturate |

**Advice distilled**: (1) train **multi-element** (shared latent), not 9
separate single-element nets; (2) tap the **bottleneck just before the output
head** as the embedding; (3) treat the supervised embedding as a **baseline**,
and the **disentangled conditional autoencoder** (de Mijolla 2021) as the
primary design — a supervised regression latent is *entangled* with Teff/logg,
so it mixes physical and chemical variation; (4) the doppelganger argument
(Ness 2018) is the scientific justification for going beyond 16 abundances.

## 3. Architecture decisions

### Phase A — supervised regression embedding (the user's baseline)

1. **Multi-task**: change the output head from `Dense(1)` to `Dense(9)`
   (all lightsurf targets: FE_H, C_FE, CA_FE, K_FE, MG_FE, NI_FE, O_FE, SI_FE,
   TI_FE) so the shared latent encodes common chemical structure.
   (Lightsurf currently trains one element at a time — `Dense(1)`.)
2. **Embedding taps** (name the layers, expose all three):
   - attention output — 256-d (rich)
   - `Dense(20)` — 20-d
   - `Dense(8)` — 8-d bottleneck (most compressed)
3. **Spectrum normalisation**: per-spectrum standard scaling, **save the scaler**.
4. **Single-task fallback**: keep per-element training as a comparison arm.

### Phase B — disentangled conditional autoencoder (recommended, de Mijolla 2021)

Reuse lightsurf's CNN-LSTM(-Attention) as the **encoder** of a conditional
autoencoder:

- `enc(x, u) → z`  (x = spectrum, u = Teff/logg, z = chemical latent)
- `dec(z, u) → x̂`  (reconstruct the spectrum)
- loss = `MSE(x, x̂) + λ · L_dis(z, u)`

`L_dis` enforces statistical independence between the latent and the known
physical parameters — FaderDis (adversarial critic) or FactorDis
(scramble-then-WGAN-critic). The latent `z` is then the **abundance-free**
chemical embedding, free of Teff/logg contamination. This is the design that
actually goes *beyond* the 16 ASPCAP abundances (blended lines, weak features,
no stellar-model labels).

**Recommendation**: implement Phase A first (it is cheap — the model exists),
use it as the benchmark baseline, then implement Phase B as the scientific
novelty. Both feed the same workshop benchmark.

## 4. lightsurf changes (CPU-testable)

1. `deep_models.py`:
   - name every layer in `_build_model` (`attention`, `dense_20`, `dense_8`,
     `output`),
   - add `output_dimension: int = 1` to `CnnLstmAttentionModel`,
   - add `embedding_model(layer_name)` → `tfk.Model(inputs, layer.output)`,
   - add `predict_embeddings(X, layer_name)` → (n, d) array.
2. `deep_controller.py` / new `scripts/export_embeddings.py`:
   - after training, `best_estimator_.model.save("model.keras")`,
   - save the spectrum scaler (`joblib`),
   - `extract` subcommand: load `.keras` + scaler, read spectra CSV, write
     embeddings parquet.
3. **CPU smoke test** (`tests/`): 16 synthetic spectra (8575) → 1-epoch fit →
   embeddings have correct shape; multi-task output shape (n, 9).

## 5. Workshop changes

1. New `src/cluster/spectral.py`:
   - `load_spectral_model(model_path, scaler_path)`,
   - `extract_spectral_embeddings(spectra, layer)` → matrix,
   - standardise + L2-normalise the embeddings the same way as abundances.
2. `data.py`: `make_matrix(..., feature_source="abundances" | "spectral")`.
3. **Spectra acquisition** (the new data dependency):
   - cluster members + field sample are keyed by `APOGEE_ID` (`2M...`),
   - lightsurf's `data/raw_data/flux_abundances.csv` already holds 33 756
     stars' fluxes (`FILE` = `aspcapStar-dr17-<APOGEE_ID>`), so cross-match
     there first,
   - fall back to `download_spectra.py` (SDSS aspcapStar FITS) for missing
     stars,
   - new `scripts/fetch_spectra.py` returns `{APOGEE_ID: 8575-flux}`.
4. Wire into `baseline.py` + `benchmark.py`: same t-SNE/UMAP/EVoC, same
   scoring, two feature sources side-by-side.
5. **CPU smoke test**: synthetic embeddings → t-SNE/UMAP/EVoC → score table.

## 6. GPU training (other machine)

1. lightsurf repo + `flux_abundances.csv` there.
2. Train multi-task `CnnLstmAttentionModel(output_dimension=9)` via the
   existing `RandomizedSearchCV` path (or a fixed sensible config).
3. Save `model.keras` + scaler; copy the two artifacts to this machine
   (CPU inference only — embedding extraction is ~ms/star).

## 7. Validation

Head-to-head on both benchmarks:

| metric | abundances (16-d) | embeddings (8/20/256-d) |
|---|---|---|
| paper baseline homogeneity/v-measure | §2 of baseline_results.md | to run |
| field retrieval recall/precision | region_sweep_results.md | to run |
| kNN purity | 0.11 t-SNE | to run |

Success = embeddings beat abundances on ≥ one metric, or tie at much lower
dimension (8-d vs 16-d). A tie at 8-d is already a useful result (denoised
compression).

## 8. Risks & mitigations

- **Circularity** → the head-to-head is the test; parity ⇒ compression is real
  but not novel ⇒ Phase B (disentangled autoencoder) is the answer.
- **Teff/logg contamination** → the supervised latent is entangled with stellar
  parameters; the disentanglement loss in Phase B removes it — and this is the
  point of de Mijolla 2021.
- **Spectra coverage** → cross-match `flux_abundances.csv` first; only download
  the missing subset.
- **No GPU here** → train remotely, CPU-only inference locally; keep a
  synthetic-data smoke test so the pipeline is verifiable without the artifact.
- **TensorFlow on CPU** → pin TF version; embeddings extraction needs only the
  forward pass (cheap).

## 9. Implementation status

| item | status | commit |
|---|---|---|
| lightsurf embedding taps (`embedding_model` / `predict_embeddings`, multi-task `output_dimension`) | ✅ done + 3 tests | lightsurf `e2e2a6a` |
| lightsurf export script (`export_embeddings.py`) | ✅ done | lightsurf `3524e8b`, `71556e4` |
| lightsurf train+export single command (`train_embedding_model.py`) | ✅ done, CPU round-trip tested | lightsurf `d35a270` |
| workshop spectral feature source (`spectral.py`: load/align/post-process) | ✅ done + 6 tests | workshop `de95802` |
| CPU smoke test (synthetic spectra → embeddings → t-SNE/UMAP/EVoC) | ✅ done (both repos) | — |
| GPU runbook | ✅ done | workshop `45819cd` |
| Phase B disentangled conditional autoencoder (skeleton) | ✅ done + 2 tests | lightsurf `403c6c2` |
| train real model (GPU machine) + export artifact | ⏳ deferred — needs GPU | — |
| head-to-head abundances vs embeddings benchmark | ⏳ blocked on trained model | — |

## 10. Open questions (resolve during implementation)

1. Which tap wins: attention 256-d vs bottleneck 8-d vs 20-d?
2. Multi-task vs single-task embedding — does the shared latent help?
3. Per-spectrum vs global flux normalisation.
4. Do embeddings recover the metal-poor globulars better (the spectrum may
   carry weak-line info the ASPCAP pipeline dropped)?

## 11. Data facts (verified)

- `flux_abundances.csv`: 33 756 rows × 8587 cols = `FILE` + 8575 flux + TEFF +
  LOGG + 9 abundances. Flux columns are wavelength-named strings
  (15096.68 … 16995.17 Å, from `APOGEE_WAVELENGTH_AIR_STR`).
- `APOGEE_SPECTRUM_LENGTH = 8576` (constants) vs 8575 stored (off-by-one:
  `APOGEE_WAVELENGTH_AIR = APOGEE_WAVELENGTH_AIR[:-1]`).
- `models/` is empty — no trained artifact exists yet.
- Cross-match key: `FILE` = `aspcapStar-dr17-<APOGEE_ID>`.
