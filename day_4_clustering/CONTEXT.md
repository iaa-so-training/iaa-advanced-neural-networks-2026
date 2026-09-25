# Chemical Tagging Workshop

Benchmark of clustering methods and stellar signals for **chemical tagging** — deciding which stars in a field belong to a cluster — for an IAA-SO School session.

## Language

**Chemical tagging**:
Recovering which stars in a field share a common origin (a cluster), via their measured stellar properties.
_Avoid_: classification, cluster search

**Membership determination**:
The core task — labelling each star as `member` or `field` for a given cluster.
_Avoid_: tagging, clustering

**Discovery**:
Blind search — finding members of a (possibly dispersed) cluster without knowing its location/kinematics in advance.
_Avoid_: unsupervised search

**Refinement**:
Cleaning an existing candidate list — pruning field contamination from a known cluster's member sample.
_Avoid_: cleaning, decontamination

**Signal**:
A feature set fed to a clustering method: abundances (16-d), spectroscopy latent (256-d), kinematics (4-d), colours (J, J−K).
_Avoid_: features, inputs

**Kinematics**:
Parallax, proper motion (PMRA, PMDEC), radial velocity. Defines the ground-truth labels (σ-clip), so it is the *ceiling*, not a fair discovery signal.
_Avoid_: astrometry, phase space

**Field retrieval**:
The benchmark task — separating `member` from `field` stars at a given angular scale, scored by precision/recall/purity.
_Avoid_: contamination test

**Curse of dimensionality**:
Naive concatenation of a high-dimensional signal (256-d latent) with a low-dimensional one (4-d kinematics) drowns the latter in the distance metric.
_Avoid_: dilution (alias)

**Isochrone fitting**:
Downstream use of membership — fitting a stellar-population model (ASteCA) to the member sample to recover cluster age, metallicity, distance.
_Avoid_: CMD fitting (narrower)

**Two-stage pipeline**:
Kinematics for recall (Stage 1), then a spectral-distance cut to reject field doppelgängers (Stage 2). The smart combination that concatenation is not.
_Avoid_: stacking, concatenation

**Spectral rejection**:
Rejecting kinematic doppelgängers via their RNN-latent distance to the cluster centroid.
_Avoid_: chemical cut (narrower — abundances only)

**Age-dependence**:
The empirical result that chemical tagging's difficulty scales with cluster age — young clusters blend into the field, old/metal-poor clusters stand out spectrally.
_Avoid_: age bias

## Relationships

- **Chemical tagging** reduces to **Membership determination** across a field.
- **Membership determination** splits into **Discovery** and **Refinement**.
- **Kinematics** defines the **Membership** ground truth.
- **Field retrieval** scores a **Signal**'s ability to reproduce **Membership**.
- Better **Membership** → tighter **Isochrone fitting** → better cluster parameters.
- **Two-stage pipeline** = **Kinematics** (Stage 1 recall) → **Spectral rejection** (Stage 2 precision).
- **Age-dependence** governs **Spectral rejection**'s success (young blend, old stand out).

## Example dialogue

> **Dev:** "When we score a **Signal** on **Field retrieval**, do we use **Kinematics** as one of the signals?"
> **Domain expert:** "Only as the *ceiling*. **Kinematics** define the labels, so scoring them is circular — the honest benchmark is how close **Discovery** signals (abundances, spectroscopy, colours) get to that ceiling."

## Flagged ambiguities

- "tagging" used for both the task (**chemical tagging**) and the labelling step — resolved: **Membership determination** is the task, **Chemical tagging** is the goal.
