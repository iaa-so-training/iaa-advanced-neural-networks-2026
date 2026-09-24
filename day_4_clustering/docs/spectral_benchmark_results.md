# Spectral-embedding benchmark results (paper baseline)

> **Read the caveats first.** Two systematics shape everything below:
> a data-PRODUCT mismatch in the 25-cluster union (§ Provenance — raw
> `apStar` vs continuum-normalised `aspcapStar`, fixable by re-downloading
> from DR19) and a comparison that was not like-for-like (§ Same-population
> head-to-head). Both are now measured, and the corrected numbers — not the
> originals — are the ones to quote.

>
> **Running these:** the commands below are written in the native form. In the
> Docker setup (the default — see `docs/docker.md`) prefix them with
> `docker run --rm -it $DAY4 $IMG`, e.g.
> `docker run --rm -it $DAY4 $IMG uv run cluster baseline --kinematics`; scripts
> become `… $IMG uv run python scripts/…`, and pass knobs with `-e`
> (`docker run --rm -it -e CLUSTER_MAX_STARS=10000 $DAY4 $IMG uv run cluster run`).

## Uniform DR19 re-run (the numbers to quote now)

The product mismatch below was fixed at the source. On the desktop RTX 5090,
the masked AE was **retrained from scratch on raw DR19 `mwmStar` spectra**
(39,945 field stars, 100 epochs / early stop), and *every* star — 994 members
plus 39,945 field — was embedded from that one product. PCA arms were rebuilt
from the same spectra. The old `masked_latent*.parquet` artifacts are backed
up as `*_v1.parquet`. All numbers here are from the uniform data.

**Same-population head-to-head — 982 stars, 25 clusters** (7 seeds):

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| PCA 256-d | 0.69 ± 0.00 | 0.69 ± 0.01 | 0.65 ± 0.02 |
| PCA 64-d | 0.74 ± 0.00 | 0.75 ± 0.01 | 0.73 ± 0.01 |
| abundances (16-d) | 0.56 ± 0.00 | 0.58 ± 0.01 | 0.42 ± 0.04 |
| **masked AE 256-d** | **0.74 ± 0.00** | **0.76 ± 0.01** | **0.69 ± 0.02** |

What survives the re-run, honestly:

1. **The masked AE still beats abundances.** UMAP 0.76 vs 0.58, EVoC 0.69 vs
   0.42 — a real, seed-stable gap on the same stars. Chemical tagging from
   spectra alone is not a fluke.
2. **But PCA-64d is now competitive** (0.73/0.75/0.74 vs 0.69/0.76/0.74).
   The old "3× the linear baseline" was an artefact of the product mismatch
   inflating the AE arm *and* the degenerate PCA arm. On the full sample the
   self-supervised AE's edge over a linear projection is modest, and only
   clear on EVoC (0.69 vs 0.73 reversed — see note below) and UMAP.
3. **M 3 is not the whole story.** Ablating it (884 stars, 24 clusters)
   leaves the AE at t-SNE 0.71 / UMAP 0.75 / EVoC 0.67 — essentially
   unchanged, so the separation is not a globular/open discriminator.

**Field retrieval** (24,171 field vs 829 members, 24 clusters — the number
that the product mismatch had forced us to withdraw):

| method | recall | precision |
|---|---|---|
| t-SNE | 0.209 | 0.223 |
| UMAP | 0.223 | 0.133 |
| EVoC | 0.483 | 0.002 |

Chance precision is ~3% (829 members in 25k stars), so t-SNE's 0.22 is a real
signal, but retrieval from spectra alone is weak and EVoC over-segments the
field (recall up, precision ~0). This is the honest baseline.

**Provenance resolved.** The old confound was a *product* signature — the same
star embedded twice landed 1.70× farther apart across products (AUC 0.9992,
6.57σ). After the re-run there is no product split to probe: every star went
through `mwmStar`. A linear probe can still separate the *was-DR17* members
from the *was-DR19* members at AUC 0.83 — but that is the genuine population
difference between the original APOGEE target list and the SDSS-V
observations, not a reduction artefact.

```bash
# reproduce (data/embeddings/masked_latent*.parquet are now the uniform re-run)
uv run cluster head-to-head --out results/head_to_head/scores.csv
uv run cluster ablate --spectral data/embeddings/masked_latent.parquet --exclude 'M 3'
uv run cluster run
```

> **Superseded (pre-rerun, product-mismatched):** the "DR19 masked-AE" table
> (0.789/0.874/0.770), the 55-star head-to-head (0.79/0.87), and the M 3
> ablation (0.756/0.761/0.714) below were computed on the mixed-product union.
> They are retained for the audit trail but **must not be quoted**.

## DR19 masked-AE (the headline)

A **masked spectral autoencoder** (MAE-style, `feat/masked-spectral-foundation`
in lightsurf): mask contiguous wavelength blocks, encode the visible pixels,
reconstruct the masked ones — **no abundance labels**. The 256-d latent beats
the supervised latent and the ASPCAP abundances on DR19.

Cluster-only (55 members, 5 APO clusters; `cluster baseline --spectral
masked_latent.parquet`):

| features | t-SNE homog | UMAP | EVoC | v-measure |
|---|---|---|---|---|
| PCA 64-d (linear) | 0.534 | 0.269 † | 0.462 | 0.580 |
| PCA 256-d (linear) | 0.269 † | 0.269 † | 0.532 | 0.373 |
| abundances (16-d) | 0.292 ‡ | 0.547 ‡ | 0.409 ‡ | — |
| supervised ConvPool (64-d) | 0.560 | 0.560 | 0.634 | 0.65 |
| **masked AE (256-d, no labels)** | **0.789** | **0.874** | **0.770** | **0.829** |
| masked + abundance (272-d) | **0.879** | **0.874** | 0.712 | **0.915** |

† **Degenerate.** The clusterer returned 2 groups with 64–67% of stars in one
of them — 0.269 is the collapse floor, not a measured baseline. Verify with
`cluster head-to-head`, which flags these rows automatically. **Do not quote
ratios against these numbers**; "3× better than PCA" is 3× better than a
crash.

‡ **Not the same population.** This abundance row was scored on all 25
clusters / 1215 stars, while every other row is 5 clusters / 55 stars. The
harder problem scores lower, and the gap reads as a win it has not earned.
See the corrected table below.

## Same-population head-to-head (corrected)

`cluster head-to-head` intersects the arms' `APOGEE_ID` sets before scoring,
so every row is the same stars and the same clusters. Homogeneity, mean ± std
over 7 seeds (42, 0, 1, 2, 7, 13, 99):

**55 stars, 5 clusters** (Berkeley 66, IC 166, M 3, M 67, NGC 188):

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| PCA 256-d | 0.29 ± 0.00 † | 0.27 ± 0.00 † | 0.37 ± 0.07 † |
| PCA 64-d | 0.54 ± 0.00 | 0.27 ± 0.00 † | 0.41 ± 0.10 |
| abundances (16-d) | 0.48 ± 0.00 | 0.48 ± 0.00 | 0.64 ± 0.08 |
| **masked AE 256-d** | **0.79 ± 0.00** | **0.87 ± 0.00** | **0.70 ± 0.08** |

**45 stars, 4 clusters** — adding the supervised CNN arm costs NGC 188,
which its embedding does not cover:

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| PCA 256-d | 0.29 ± 0.00 † | 0.29 ± 0.00 † | 0.34 ± 0.06 |
| PCA 64-d | 0.29 ± 0.00 † | 0.29 ± 0.00 † | 0.55 ± 0.04 |
| abundances (16-d) | 0.56 ± 0.00 | 0.56 ± 0.00 | 0.63 ± 0.14 |
| **masked AE 256-d** | **0.84 ± 0.00** | **0.84 ± 0.00** | 0.64 ± 0.08 |
| supervised CNN 64-d | 0.56 ± 0.00 | 0.56 ± 0.00 | **0.80 ± 0.03** |

What survives the correction, and what does not:

1. **t-SNE and UMAP: the masked AE wins clearly.** 0.79/0.87 vs 0.48 for
   abundances on the same 55 stars. This is the result.
2. **EVoC: the win does not survive.** 0.70 ± 0.08 vs 0.64 ± 0.08 — the
   error bars overlap. On the 4-cluster sample the *supervised* CNN takes
   EVoC outright (0.80 ± 0.03). Any claim that self-supervision beats
   supervision must be stated for t-SNE/UMAP only.
3. **PCA rows are mostly degenerate**, so "3× the linear baseline" is not a
   supportable phrasing. Say "PCA+UMAP collapses; the masked AE does not."

Reproduce:

```bash
uv run cluster head-to-head --out results/head_to_head/scores.csv
```

## Is it just "globular vs open"?

M 3 is 24 of the 55 stars (44%), so cluster separation could be the trivial
globular/open split. `cluster ablate --exclude 'M 3'` drops it and re-scores
the four remaining **open** clusters (31 stars), 7 seeds:

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| abundances (16-d) | 0.204 ± 0.000 | 0.184 ± 0.089 | 0.232 ± 0.100 |
| **masked AE (256-d)** | **0.756 ± 0.000** | **0.761 ± 0.000** | **0.714 ± 0.110** |

**This is the strongest result in the project.** With the globular removed —
all four clusters open, same metallicity regime, the hard case — the masked
latent still separates them ~3.7× better than ASPCAP abundances, and the gap
is far outside the seed spread. The win is real chemical tagging, not a
globular/open discriminator.

```bash
uv run cluster ablate --spectral data/embeddings/masked_latent.parquet --exclude 'M 3'
```

## Provenance: a data-PRODUCT mismatch (not a data-release one)

`masked_latent_all.parquet` is the union of two reductions, and the split is
not random (`cluster provenance`):

```
source  DR17  DR19    n  frac_DR19
role
field      0   894  894      1.000
member   726    69  795      0.087
```

Every field star is DR19; 91% of members are the DR17 backfill. The latent
encodes which pipeline produced the spectrum:

- latent → release AUC, **members only**: **0.9992 ± 0.0016**
- latent dims separating release: **223 / 256** (p < 1e-6)

### What actually went wrong

The first diagnosis called this a *data-release* effect. That was wrong, and
the distinction changes the fix.

**DR19 is not a disjoint sample from DR17 — it reanalyses and includes it.**
`astraAllStarASPCAP-0.6.0.fits.gz` carries **717,689 rows with
`release='dr17'`**, all reduced by one pipeline (`v_astra=0.6.0`). Every one
of the 738 backfilled members resolves to an `sdss_id` there, with finite
DR19 ASPCAP parameters, and DR19 serves a spectrum for **738/738** of them.
There was never a coverage gap to backfill.

The real confound is that the two arms used **different data products**:

| arm | product | flux | median |
|---|---|---|---|
| DR19 | `apStar` / `mwmStar` | raw | ~5.8e3 |
| backfill | `aspcapStar-dr17` | continuum-normalised | ~1.01 |

Three orders of magnitude apart, fed to one autoencoder. `apStar` files are
also 2-D (row 0 = combined spectrum) while `aspcapStar` is already 1-D, so the
two paths differ before the model sees a pixel.

**Paired control** (`scripts/diagnose_product_mismatch.py`). 253 stars were
embedded through *both* pipelines — same star, same physics, only the product
changes:

```
same star,  different product : 3.655 ± 0.463
different star, same product  : 2.150 ± 0.680   ->  ratio 1.70x
cosine(same star, two products): 0.389 ± 0.195
shared fraction of the offset : 74.9%
```

A star is **1.7× farther from itself** across products than from a random
different star within one product, and 75% of that displacement is a single
shared direction. That is a pipeline offset, not astrophysics — which is why
the member-only probe hits AUC 0.999.

### The fix: re-run, don't caveat

Because DR19 covers 100% of these stars, this does not need a caveat — it
needs a re-download. `scripts/build_dr19_rerun_list.py` writes the 738
`mwmStar` URLs (sharded on the **last four digits of `sdss_id`**, split 2+2):

```
https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star/<d1d2>/<d3d4>/mwmStar-0.6.0-<sdss_id>.fits
```

**Status (2026-09-20).** The 736 mwmStar spectra are downloaded and verified
(708 APO + 28 LCO, raw flux, 8575 px; two stars 404 on the SAS and are
dropped — 0.3%). `scripts/embed_dr19_rerun.py` is ready: it reproduces the
DR19 arm's preprocessing (nan→0, per-star standardisation) and embeds through
the masked AE with a `--verify` gate that checks the checkpoint reproduces
the published latents before anything is written.

**Checkpoint:** the masked-AE checkpoint that produced `masked_latent.parquet` is
published with the workshop asset bundle as `data/embeddings/masked_ae_rerun.pt`
(plus `masked_ae.pt`, same architecture), so nothing here depends on the machine
that trained it. Load it with the vendored `cluster.models.MaskedSpectralAE`
(`uv sync --extra torch`) — the supervised `model*.pt` checkpoints are a
different architecture (`CnnLstmAttention`), and `scripts/embed_dr19_rerun.py`
defaults to the masked autoencoder. Re-embedding needs those checkpoints, the
736 DR19 `mwmStar` spectra, and the torch extra:

```bash
uv sync --extra torch
uv run cluster download --assets --with-optional   # adds optional/mwmstar.tar
uv run python scripts/embed_dr19_rerun.py --verify # gate: reproduce before writing
uv run python scripts/embed_dr19_rerun.py          # full re-embed
```

The DR19 apStar **flux matrix** (2.9 GB) the checkpoint was pretrained on is not
part of the bundle — it is only needed to *retrain*, which is out of scope here.
The `data/raw_data/flux_abundances.csv` shipped locally is the DR17 arm
(33 756 aspcapStar rows).

### Interim: what survives on the current data

On the mixed population, "member vs field" is substantially "DR17 vs DR19":

| probe (member vs field) | AUC |
|---|---|
| masked-AE latent, all rows | 0.969 |
| **provenance flag alone (1 bit)** | **0.957** |
| abundances, all rows | 0.764 |
| masked-AE latent, DR19-only rows | 0.794 |
| abundances, DR19-only rows | **0.824** |

One bit of "which product" reproduces almost the entire latent score. On a
provenance-clean subsample the ordering **inverts** and abundances win.

**Consequence — the retracted claim.** The previously published line
*"spectral t-SNE field precision 0.435 vs abundances 0.199 — the masked
latent is ~2× purer"* is **withdrawn**. Re-run on DR19-only rows it does not
hold (spectral t-SNE precision collapses to 0.007 on 70 members). Field
retrieval on the mixed population is not interpretable and must not be
quoted until it is recomputed on a uniform-provenance sample.

**Cluster-only homogeneity is much less affected**, because it is computed
*within* members where the split is 91:9 rather than 0:100:

| population | spectral t-SNE / UMAP / EVoC | abundances |
|---|---|---|
| all members (mixed provenance) | 0.655 / 0.700 / 0.642 | 0.484 / 0.550 / 0.488 |
| DR17-only members (uniform) | 0.612 / 0.686 / 0.575 | 0.525 / 0.527 / 0.507 |

The cluster-only win survives on a provenance-clean sample, shrunk but
intact. This is why the headline is cluster-only separation and not field
retrieval.

```bash
uv run cluster provenance
```


Per-cluster (Simbad referee): M 3 (globular) precision 1.00; open clusters
0.05–0.94 — the globular/open split reproduces on the self-supervised latent.

Training bugs fixed along the way (see memory): (1) exploding LSTM gradients
→ `clip_grad_norm_(1.0)`; (2) LSTM+attention head vanishes on the 31k-star
sample → conv+global-pool head; (3) DR19 apStar flux is raw (~1e4) →
per-star standardization.

## Full 25-cluster coverage (DR17 + DR19 spectra)

> **⚠ This union mixes two data products.** See § Provenance above before
> quoting any member-vs-field number computed on `masked_latent_all.parquet`.
> Cluster-only scores are usable; field retrieval is not.
> **This is fixable**: DR19 serves a uniform `mwmStar` spectrum for 738/738
> of these members — run `scripts/build_dr19_rerun_list.py` and re-embed.

The DR19 apStar star list used here (`allStar-1.3-apo25m.fits`) was APO-North
only, so ~738 members had no spectrum in that particular list — **not**
because DR19 lacks them. They were instead embedded from their DR17
aspcapStar spectra (`synspec/<tel>/<field>/aspcapStar-dr17-<id>.fits`), which
are continuum-normalised while apStar is raw, giving
`masked_latent_all.parquet` (39,645 stars, 25/25 clusters) a product split
that aligns with the member/field label.

Cluster-only (24 clusters, 791 stars):

| features | t-SNE homog | UMAP | EVoC | v-measure |
|---|---|---|---|---|
| abundances (16-d) | 0.416 | 0.523 | 0.488 | 0.474 |
| **masked AE (256-d, no labels)** | **0.655** | **0.677** | **0.642** | **0.612** |
| masked + abundance (272-d) | 0.629 | 0.660 | 0.644 | 0.658 |
| kinematics only (4-d) | 0.946 | 0.943 | 0.892 | 0.943 |

Field retrieval (Simbad referee, 24 clusters / 716 members): spectral t-SNE
precision 0.435 vs abundances 0.199. **⚠ WITHDRAWN — do not quote.** This
comparison is confounded by the data-product mismatch (see § Provenance):
the members are 91% continuum-normalised `aspcapStar` while the field is 100%
raw `apStar`, and a 1-bit product flag alone scores AUC 0.957 on the same
member-vs-field task. Recomputed on single-product rows the ordering inverts.
Re-embedding all members from DR19 `mwmStar` should restore this experiment.
Kinematics still cap the field at ~0.60.

---

Trained on a single workstation with an RTX 5090: `CnnLstmAttentionModel` regressing 9
APOGEE abundances from the 8575-flux H-band spectrum, and
`DisentangledSpectralAE` (Phase B). Scored on the paper baseline
(cluster-only multiclass separation, Garcia-Dias et al. 2019 metrics).

## Setup (clean head-to-head)

Both feature sources scored on the **same relaxed population** —
`STARFLAG==0`, `SNR≥100`, no `ASPCAPFLAG==0` gate — 21 clusters with ≥ 5
members, 1029 stars. The ASPCAP gate is dropped because spectral embeddings
read the raw spectrum and are valid even when ASPCAP flags abundance issues
(this is itself an advantage: the spectral pipeline covers more stars).

| features | dim | t-SNE homog | UMAP homog | EVoC homog |
|---|---|---|---|---|
| **Spectral — Phase A (attention)** | 256 | **0.576** | **0.617** | **0.555** |
| Abundances (ASPCAP) | 16 | 0.254 | 0.563 | 0.537 |
| Spectral — Phase B (disentangled) | 16 | 0.314 | 0.277 | 0.362 |

v-measure / completeness:

| features | t-SNE v / c | UMAP v / c | EVoC v / c |
|---|---|---|---|
| Phase A | 0.519 / 0.472 | 0.485 / 0.400 | 0.481 / 0.424 |
| Abundances | 0.403 / 0.978 | 0.488 / 0.430 | 0.499 / 0.465 |
| Phase B | 0.274 / 0.243 | 0.249 / 0.227 | 0.302 / 0.258 |

## Finding

1. **The full-spectrum RNN latent beats ASPCAP abundances at separating
   clusters.** Phase A (256-d attention) gives the highest homogeneity for
   every method — the biggest win is t-SNE: 0.576 vs 0.254 (2.3×). The paper's
   method (t-SNE) no longer collapses the open clusters into one blob when
   fed the spectral latent.
2. **The gain is in purity, not completeness.** The spectral latent recovers
   fewer members (completeness 0.40–0.47 vs 0.43–0.98) but the groups it
   finds are much purer. Homogeneity (precision) is the harder, more
   publishable axis.
3. **Phase B (disentangled, 16-d) underperforms.** The abundance-free latent
   is too compressed (16 dims) and the disentanglement objective (Teff/logg
   removal) does not recover the cluster-separating structure. A larger
   latent and/or a lighter disentanglement weight is the natural next
   experiment.
4. **Caveat**: the two populations differ slightly (1030 relaxed vs 646
   ASPCAP-clean members), but the clean relaxed-abundance comparison above
   uses the same 1029 stars, so the Phase-A-vs-abundance gap is real.

## Reproduce

```bash
# train (GPU): scripts/train_embedding_model.py + export
# then locally:
uv run cluster baseline --spectral data/embeddings/attention_merged.parquet
uv run cluster baseline --spectral data/embeddings/disentangled_merged.parquet
# abundances (relaxed, same population):
uv run python - <<'EOF'
from cluster import config
from cluster.clusters import CLUSTERS
from cluster.data import prepare
from cluster.baseline import run_baseline
s = config.Settings(); s.require_aspcap_flag_clean = False
cl = [c for c in CLUSTERS if c.name in s.resolve_cluster_names([c.name for c in CLUSTERS])]
p = prepare("data/allStar-dr17-synspec_rev1.fits", s, cl,
    seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
    seed_parallax_frac=config.SEED_PARALLAX_FRAC, seed_pm_tol=config.SEED_PM_TOL,
    seed_rv_tol=config.SEED_RV_TOL, n_refine_passes=config.N_REFINE_PASSES,
    refine_sigma=config.REFINE_SIGMA)
print(run_baseline(p, s, min_members=5).round(3).to_string(index=False))
EOF
```

## Architecture experiments (what changes the result)

| change | t-SNE homog | verdict |
|---|---|---|
| multi-task head Dense(9) vs single Dense(1) | the Phase-A win | key lever |
| tap: 256-d attention / 20-d / 8-d | 0.549 / 0.501 / 0.457 | inconclusive |
| per-spectrum normalisation | 0.549 -> 0.467 | hurts (reverted) |
| Phase B latent 16 -> 64-d | 0.314 -> 0.430 (t-SNE), 0.277 -> 0.488 (UMAP) | improves, still < Phase A |

Phase B 64-d full table:

| method | homogeneity | completeness | v-measure |
|---|---|---|---|
| t-SNE | 0.430 | 0.315 | 0.364 |
| UMAP | 0.488 | 0.317 | 0.384 |
| EVoC | 0.430 | 0.312 | 0.362 |

Conclusion: the supervised multi-task regression (Phase A, 256-d attention)
is the best architecture tested. The disentangled AE (Phase B) improves with
a larger latent but stays below Phase A — the Teff/logg removal also strips
cluster-separating signal. Per-spectrum normalisation hurts (the continuum
carries information). The shared multi-task latent — not depth, tap, or
normalisation — is what makes the embedding competitive.

## Combined features: spectral + abundance (the best)

Concatenating the 256-d spectral latent with the 16-d ASPCAP abundances
(272-d total, both standardised) beats either feature set alone — no
re-training required:

| features | t-SNE homog | UMAP homog | t-SNE v-measure |
|---|---|---|---|
| abundances | 0.254 | 0.563 | 0.403 |
| spectral (Phase A) | 0.549 | 0.580 | 0.519 |
| combined | 0.611 | 0.631 | 0.596 |

The gain is synergistic (0.611 > max(0.549, 0.254)): the full-spectrum latent
and the ASPCAP elements are complementary. Completeness also recovers
(0.492 -> 0.582 for t-SNE), so the combined feature is better on both axes.

## Raw spectra control

Clustering the raw 8575-flux directly (standardised per wavelength), and
raw + abundance, as a control against the RNN embedding:

| features | t-SNE homog | UMAP homog | t-SNE v-measure |
|---|---|---|---|
| raw 8575-d | 0.301 | 0.662 | 0.460 |
| raw + abundance 8591-d | 0.300 | 0.698 | 0.459 |
| RNN latent 256-d | 0.549 | 0.580 | 0.519 |
| RNN + abundance 272-d | 0.611 | 0.631 | 0.596 |

Interpretation: the raw flux wins UMAP (its continuum carries global
Teff/metallicity structure) but collapses for t-SNE (0.30, the continuum
blob — t-SNE sees local structure and the continuum swamps the chemistry).
The RNN latent is a chemically-meaningful compression: it beats raw spectra
on t-SNE 0.549 vs 0.301, and the combined RNN+abundance is the best balanced
feature (t-SNE 0.611, UMAP 0.631, v-measure 0.596).

## RNN + abundance + kinematics (276-d)

| method | homogeneity | completeness | v-measure |
|---|---|---|---|
| t-SNE | 0.566 | 0.698 | 0.625 |
| UMAP | 0.703 | 0.541 | 0.611 |
| EVoC | 0.676 | 0.555 | 0.609 |

Kinematics help UMAP/EVoC but are diluted by the 272-d chemical block
(4 of 276 dims), so they don't dominate. And they are the *ground-truth*
signal — membership is defined kinematically — so adding them is circular:
the honest chemical-tagging comparison is RNN+abundance (272-d) without
kinematics.

## Consistent re-run (de-M-dwarfed model, scaled regions)

Same population (878 members, 21 clusters, relaxed ASPCAP), same scaled
regions (max(3 deg, 10 x diameter)) for the field benchmark.

### Cluster-only (homogeneity)

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| abundances (16-d) | 0.252 | 0.507 | 0.507 |
| spectral (256-d, broad) | 0.402 | 0.689 | 0.642 |

### Field retrieval (recall / precision)

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| abundances | 0.70 / 0.21 | 0.54 / 0.32 | 0.63 / 0.28 |
| spectral | 0.72 / 0.51 | 0.68 / 0.54 | 0.48 / 0.56 |

**Verdict**: the de-M-dwarfed spectral embedding beats abundances on both
experiments and both axes — homogeneity +0.13-0.18, precision ~2x at
comparable recall. Globulars near-perfect (M 15 0.97, M 71 0.98, M 92 1.00).

## Kinematics — how close to 1?

Cluster-only homogeneity, same 878 stars:

| features | t-SNE | UMAP | EVoC |
|---|---|---|---|
| abundances (16-d) | 0.252 | 0.507 | 0.507 |
| RNN latent (256-d) | 0.402 | 0.689 | 0.642 |
| abundances + kin (20-d) | 0.617 | 0.739 | 0.719 |
| RNN + kin (260-d) | 0.670 | 0.716 | 0.693 |
| kinematics only (4-d) | 0.968 | 0.958 | 0.918 |

Field-retrieval precision (RNN vs RNN+kin): 0.51/0.54/0.56 -> 0.61/0.66/0.65.

Kinematics alone nearly reach 1 (t-SNE 0.97, UMAP 0.96) but not exactly — a ~3%
residual from kinematic overlap + HDBSCAN noise. Combined with the 256-d
chemistry, the 4 kinematic dims are diluted and the score drops to 0.62-0.74.
The clustering methods never re-weight the dimensions to let kinematics
dominate.

## Kinematics-only field retrieval

| features | t-SNE r/p | UMAP r/p | EVoC r/p |
|---|---|---|---|
| RNN latent (256-d) | 0.72 / 0.51 | 0.68 / 0.54 | 0.48 / 0.56 |
| RNN + kin (260-d) | 0.80 / 0.61 | 0.63 / 0.66 | 0.60 / 0.65 |
| kinematics only (4-d) | 0.93 / 0.63 | 0.75 / 0.67 | 0.68 / 0.65 |

Kinematics alone give the best recall (0.93) — they recover almost every
member — but precision still plateaus at 0.63-0.67: the field contains stars
kinematically indistinguishable from the cluster. Recall approaches 1, purity
does not.

## Corrected same-population field retrieval (cross-matched)

| features | dim | t-SNE r/p | UMAP r/p | EVoC r/p |
|---|---|---|---|---|
| abundances | 16 | 0.73 / 0.47 | 0.75 / 0.46 | 0.61 / 0.53 |
| abundances + kin | 20 | 0.83 / 0.63 | 0.74 / 0.65 | 0.64 / 0.59 |
| RNN latent | 256 | 0.72 / 0.51 | 0.68 / 0.54 | 0.48 / 0.56 |
| RNN + kin | 260 | 0.80 / 0.61 | 0.63 / 0.66 | 0.60 / 0.65 |
| kinematics only | 4 | 0.93 / 0.63 | 0.75 / 0.67 | 0.68 / 0.65 |

Corrections: the RNN field advantage over abundances is ~8% precision (0.51
vs 0.47), not 2x — the earlier gap was a population artefact (full vs
cross-matched field). Abundances+kin (20-d) ties RNN+kin (260-d): field
precision is capped ~0.65 by kinematic doppelgangers, so dimensionality does
not matter here. The curse of dimensionality shows in cluster-only
homogeneity (kin-only 0.97 -> chem+kin 0.62), not in field precision.