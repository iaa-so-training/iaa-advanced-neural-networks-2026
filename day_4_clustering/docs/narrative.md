# Chemical Tagging: A Benchmark Narrative

> **Abundances suggest. Kinematics decide. Spectra refine. Giants give the distance; the main sequence gives the age.**

A 90-minute workshop story bridging two opposite verdicts on chemical tagging,
built from a benchmark of clustering methods (t-SNE, UMAP, EVoC) across stellar
signals (elemental abundances, RNN spectral latents, kinematics), scored on
membership determination in APOGEE DR19 + Gaia DR3.

---

## Beat 1 — The question: two papers, opposite verdicts

| | Kos+ 2017 (GALAH) | Garcia-Dias+ 2019 (APOGEE) |
|---|---|---|
| Question | Can chemistry alone tag clusters? | Can chemistry alone tag clusters? |
| Answer | **Yes** — t-SNE recovers 7/9 clusters, finds 2 Pleiades members 6° out | **No** — some clusters *chemically indistinguishable* (similar ages) |
| Verdict | optimistic (confirm with kinematics) | skeptical ("weak tagging" → families) |

Same method family, same ~13 elements, opposite conclusion. The difference is
the *task*: Kos tagged a **young** cluster and leaned on kinematics to confirm;
Garcia-Dias measured the **chemistry-only ceiling** across many clusters.

A third verdict belongs in this table and is easy to forget because it cuts
against the framing: **Hogg et al. 2016** ("Chemical tagging *can* work",
ApJ 833, 262) ran K-means on 15 *Cannon* abundances for ~10⁵ APOGEE stars,
with no positional information, and found that abundance-space overdensities
really are phase-space clusters. The reconciliation is measurement precision,
not principle — Cannon reaches ~0.04 dex, well inside the ASPCAP scatter this
work operates on. Any claim here must therefore be scoped as "on *these*
abundances, for *these* clusters", never as "abundances cannot tag".

The modern competitor is **Spina et al. 2025** (A&A 702, A267): a graph
attention autoencoder over chemistry + orbital similarity + age on ~47k
APOGEE thin-disk stars, yielding 282 groups and recovering 5 of 6 open
clusters. It is the "informed" counterpart to the approach here — they inject
kinematics and age into the representation; this work deliberately withholds
every label, including abundances. **We have not benchmarked against it.**
That is the honest next experiment, not a gap to paper over.

## Beat 2 — The benchmark grid

Method (t-SNE / UMAP / EVoC → HDBSCAN) × signal (abundances 16-d, RNN latent
256-d, kinematics 4-d), scored on two tasks:

- **cluster-only** homogeneity (separate clusters from each other)
- **field retrieval** precision/recall (separate members from background)

Same 878 members, scaled regions (`max(3°, 10 × diameter)`), no kinematics in
the ground truth (circular otherwise).

## Beat 3 — Curse of dimensionality

Concatenating high-d chemistry onto low-d kinematics *dilutes* the latter:

| signal (dims) | cluster homogeneity | field recall | field precision |
|---|---|---|---|
| abundances (16) | 0.25 / 0.51 / 0.51 | 0.73 | 0.47 |
| RNN latent (256) | 0.40 / 0.69 / 0.64 | 0.72 | 0.51 |
| + kinematics (20 / 260) | 0.62 / 0.74 / 0.72 ≈ 0.67 / 0.72 / 0.69 | 0.83 / 0.80 | 0.63 / 0.61 |
| **kinematics only (4)** | **0.97 / 0.96 / 0.92** | **0.93** | **0.63–0.67** |

Kinematics alone near-solve cluster separation (0.97) and recall (0.93), but
field precision caps at ~0.65 (kinematic doppelgängers). Stacking chemistry on
top never helps — it dilutes.

## Beat 4 — The two-stage pipeline (the smart combination)

- **Stage 1**: kinematics → t-SNE → HDBSCAN → candidates (recall 0.93).
- **Stage 2**: reject candidates whose RNN-latent distance to the candidate
  centroid is a robust outlier.

Precision 0.63 → 0.67 (mean), recall 0.93 → 0.89; globulars clean up to 1.00:

| cluster | precision |
|---|---|
| M15 / M92 / M3 (globulars) | 0.76 / 0.93 / 0.98 → **1.00** |
| NGC 188 (7.6 Gyr) | 0.76 → 0.83 |
| Pleiades (0.1 Gyr) | 0.83 → 0.84 (no help) |

## Beat 5 — Age is the through-line

Spectral-rejection power scales with cluster age:

- AUROC of spectral distance separating true members from false positives:
  **corr(age) = +0.67**.
- Final precision vs age: **corr = +0.51**.

Young clusters (Pleiades, Berkeley 71) are spectrally indistinguishable from
the field — they formed from the same recent ISM. Old/metal-poor clusters
(NGC 6791, globulars) stand out. Chemistry shows **no** age correlation
(r = −0.07): the RNN latent carries age/evolutionary signal that element
ratios do not.

## Beat 6 — The RNN edge: where abundances collapse

For metal-poor globulars, element ratios lose all discriminating power while
the full spectrum retains it:

| cluster | [Fe/H] | AUROC (abundances) | AUROC (RNN latent) |
|---|---|---|---|
| M15 | −2.2 | 0.23 | **1.00** |
| M92 | −2.3 | 0.52 | **1.00** |

## Beat 6b — The masked foundation model: self-supervised beats supervised

Train a **masked spectral autoencoder** (MAE-style, He et al. 2022): mask
contiguous wavelength blocks, encode the visible pixels, reconstruct the
masked ones — MSE on the hidden pixels only. **No abundance labels.** The
256-d latent is abundance-free, so it cannot be circular the way a supervised
embedding is (a supervised latent regresses the 16 ASPCAP elements and can
carry no more information than they do).

DR19 head-to-head (55 members, 5 APO clusters):

| features | t-SNE homog | UMAP | EVoC |
|---|---|---|---|
| abundances (16-d) | 0.29 | 0.55 | 0.41 |
| supervised ConvPool (64-d) | 0.56 | 0.56 | 0.63 |
| **masked AE (256-d, no labels)** | **0.79** | **0.87** | **0.77** |
| masked + abundance (272-d) | **0.88** | **0.87** | 0.71 |

The abundance-free latent beats the supervised latent (0.79 vs 0.56) and the
ASPCAP abundances (0.29). Field precision (Simbad referee): combined 0.23 vs
abundances 0.12 (UMAP) — kinematics still cap the field at 0.30. This is the
workshop's real novelty: the full spectrum, read by a model that never saw an
element ratio, separates clusters better than the ratios themselves.

## Beat 7 — The finale: better membership → better parameters

Fit a PARSEC isochrone (ASteCA + emcee) to three membership sources for M67
(literature: age 2.82 Gyr, dm 9.54, [Fe/H] 0.03):

| membership | n | age (Gyr) | age residual |
|---|---|---|---|
| stage1 (contaminated) | 259 | 1.89 | **0.93** |
| kinematic truth | 219 | 2.45 | 0.37 |
| **two-stage** | 237 | **2.65** | **0.17** |

The two-stage's spectral rejection removes 22 field contaminants and recovers
the age **closest to literature** — even beating the raw kinematic truth. The
young field doppelgängers were dragging the age estimate down.

---

## Beat 8 — The two-survey synthesis: giants for distance, MS for age

GALAH covers DEC ≲ +25° and sees the **main sequence + turnoff**; APOGEE
sees the **giants**. A combined sample (6 clusters, 14 common abundances,
Gaia parallax/PM/RV, 98% kinematics) closes the loop.

**Red-clump distance** (median K of the J−K clump slice, member giants):

| cluster | dm_clump | dm_lit | resid |
|---|---|---|---|
| NGC 6819 | 11.92 | 11.90 | **+0.01** |
| NGC 2243 | 13.08 | 13.25 | **−0.16** |
| NGC 7789 | 11.68 | 11.27 | +0.41 |

**The sweet spot** (NGC 2243): APOGEE member giants locate the cluster, the
GALAH main sequence is selected around that kinematic centroid (relaxed
parallax — at 4.7 kpc the parallax error rivals the parallax). The combined
isochrone fit with the clump dm as a tight prior:

| fit | age (lit 1.07 Gyr) | resid | std_dm |
|---|---|---|---|
| APOGEE giants only | 1.59 | 0.52 | 0.111 |
| **combined (MS + giants)** | **0.98** | **0.09** | **0.086** |

The main sequence sharpens the age MAP (resid 0.52 → 0.09) and the clump
narrows the distance (std_dm 0.111 → 0.086). The age *posterior* stays ~tied
(std_loga ≈ 0.97) — age–metallicity degeneracy is intrinsic to the CMD.

**The takeaway:** chemical tagging is a two-survey problem. Kos's GALAH sees
the turnoff (age); Garcia-Dias's APOGEE sees the giants (distance). Neither
alone gives both cluster parameters — together they do.

---

## Caveats (honest)

- **⚠ Data-product mismatch (the big one) — fixable, not fundamental.**
  `masked_latent_all.parquet` merges two *products* along a line that
  coincides with the labels: the field is 100% raw `apStar`, while 91% of
  members are continuum-normalised `aspcapStar-dr17` (median flux ~5.8e3 vs
  ~1.01). A linear probe recovers the product from the latent at **AUC 0.999
  on members alone** (223/256 dims, p < 1e-6), and a 1-bit flag scores AUC
  0.957 on member-vs-field — essentially the whole latent score. A paired
  control on 253 stars embedded through both pipelines is decisive: the same
  star sits **1.70× farther from itself** across products than from a random
  different star within one product (cosine 0.389, 75% of the offset in one
  shared direction). **Field retrieval on the mixed population is therefore
  not interpretable and the "~2× purer" precision claim is withdrawn.**
  This was originally mis-diagnosed as a DR17-vs-DR19 *release* effect. It is
  not: DR19 reanalyses and includes DR17 (717,689 rows carry
  `release='dr17'`), and DR19 serves a uniform `mwmStar` spectrum for
  **738/738** of the backfilled members. The correct fix is to re-download
  and re-embed everything from DR19 (`scripts/build_dr19_rerun_list.py`), not
  to caveat the result. Cluster-only homogeneity is computed within members
  and survives on a single-product subsample (0.612/0.686/0.575 vs
  0.484/0.550/0.488 for abundances). Run `cluster provenance` and
  `scripts/diagnose_product_mismatch.py` before quoting that artifact.
- **Small samples, real error bars.** The flagship head-to-head is 55 stars
  across 5 clusters. Over 7 seeds the t-SNE/UMAP numbers are deterministic
  but EVoC spreads ±0.08–0.11, which is wider than several of the gaps being
  claimed. Quote `mean ± std` from `cluster head-to-head`, never one seed.
- **Degenerate baselines.** Several PCA rows (0.269/0.29) are the clusterer
  collapsing to 2 groups with ~2/3 of stars in one — a floor artifact, not a
  measurement. Ratios against them ("3× PCA") are ratios against a crash.
- **The EVoC win does not hold.** On the corrected same-population table the
  masked AE scores 0.70 ± 0.08 vs 0.64 ± 0.08 for abundances (overlapping),
  and the *supervised* CNN wins EVoC outright at 0.80 ± 0.03 on the
  4-cluster sample. The self-supervision claim is a t-SNE/UMAP result.
- **Not benchmarked against Spina et al. 2025** (A&A 702, A267), the closest
  competitor — a graph-attention autoencoder doing deep chemical tagging on
  ~47k APOGEE stars. They add kinematics and age; we deliberately do not.
- **Hogg et al. 2016** ("Chemical tagging *can* work", ApJ 833, 262) is the
  standing counter-result: K-means on 15 Cannon abundances recovers
  phase-space structure. Our claim must stay scoped to *our* ASPCAP
  precision and *these* clusters, not "abundances fail".
- **Heavily-contaminated opens** (NGC 6791: 71 field vs 15 members) stay at low
  precision — Stage 1 leaks so many field stars the self-calibrating centroid is
  polluted. A Stage-1 limit, not a chemistry limit.
- Field coverage for non-M67 clusters is ~7% (only the 50k random all-sky sample
  has embeddings).
- The RNN field-retrieval edge over abundances is ~8% precision; its real win is
  cluster-only homogeneity (0.69 vs 0.51 UMAP) and the age signal.
- GALAH coverage is DEC ≲ +25°, so 18 of 25 clusters stay APOGEE-only; the
  southern sweet-spot (NGC 2243, Collinder 261) was added to bridge the gap.
- Collinder 261 has only 18 member giants (< 20) — too sparse for a red clump.
