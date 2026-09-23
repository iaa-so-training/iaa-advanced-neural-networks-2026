# Hands-on: take a cluster, beat our baseline

**IAA-SO School on AI/ML in Astronomy 2026 · Chemical tagging · ~2 hours · 25 clusters, ~30 of you**

The results in `docs/region_sweep_results.md` are a **baseline, not a ceiling**.
Your job: pick one cluster, reproduce our numbers, then extend the pipeline
and get better numbers for *your* cluster. Your cluster is assigned in
`docs/cluster_assignment.md`.

---

## 0 · Setup (do this first, before the session if you can)

You need **Docker** (section B of the School Software Installation Guide) and
`git`. Nothing else is installed on your machine.

```bash
git clone https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026/day_4_clustering

# both downloads at once: the 1.17 GB SDSS-V DR19 catalogue + the ~1.0 GB
# embeddings/checkpoint bundle from Hugging Face (hotel wifi tonight, not now)
./run.sh download --all
```

On Windows use `.\run.ps1 download --all` — every command in this document takes
the same arguments (`. \run.ps1 …` instead of `./run.sh …`).

The first `./run.sh` also pulls the workshop image (~1.5 GB) — or builds it from
this folder if the published one is not reachable. Downloads are resumable and
skip whatever is already on disk, so a dropped wifi connection costs nothing —
just re-run it. Check what you have:

```bash
./run.sh download --assets --check     # sha256-verify the bundle
```

Verify it runs (~2 min with the field cap):

```bash
./run.sh run --fast
```

**Environment knobs** (e.g. `CLUSTER_USE_ELEMENT_WEIGHTS=1`) are forwarded into
the container: `CLUSTER_USE_ELEMENT_WEIGHTS=1 ./run.sh run …` works as written.

**Prefer native Python?** Replace `./run.sh` with `uv run cluster` (or
`uv run python` for a script) after `uv sync` — same flags, same results. Python
≥ 3.13 required; see the README. Everything below is shown in the Docker form.

`--fast` caps the field at 25 000 stars (measured 1 m 51 s on a laptop).
`--full` drops the cap — at DR19 quality cuts that is 358 058 stars, so budget
well over 10 minutes; measure it on your own machine before believing any
number.

---

## 1 · Reproduce the baseline (0–20 min)

Run the sweep for **your cluster only**, in the paper's region mode:

```bash
./run.sh run --cluster "M 67" --region-scaled
```

Or use the tuning notebook — same loop, with widgets:

```bash
./run.sh marimo notebooks/tuning_template.py
```

Open `docs/region_sweep_results.md` and find your cluster's row. Your numbers
(recall / precision / kNN-purity) should match the table — same `random_state=42`.

**What you're looking at**: for each cluster, the pipeline cuts a sky region,
embeds the 16-D APOGEE abundance space (t-SNE / UMAP), clusters it (HDBSCAN;
EVoC fuses both), and scores the best-overlapping cluster against the
**kinematic ground truth** (Gaia parallax + proper motion + radial velocity).
Kinematics are the referee — never a feature.

---

## 2 · Tune a lever (20–60 min)

The biggest gains we already found, in order of payoff:

| lever | where | what it did |
|---|---|---|
| **element 1/σ weights** | `USE_ELEMENT_WEIGHTS=True` in `src/cluster/config.py` | t-SNE recall 0.09 → 0.44 on M 67 — the noisiest elements were owning every distance |
| **row-normalisation** | `NORMALIZE_ROWS` | breaks the blob — precision up, recall down |
| **SNR cut** | `SNR_MIN` | culls the noisy tail |
| **t-SNE perplexity** | `TSNE.perplexity` | neighbourhood size |
| **HDBSCAN min_cluster_size** | `HDBSCAN.min_cluster_size` | cluster size floor |
| **region size** | `--region 30` / `--region-scaled` | the paper's 30° vs the scaled radius |

Pick **one** lever, change it, re-run, and watch your row move. Log what you
tried in a scratch file. The full map is `docs/experiment_results.md`.

Every lever is also an env var (prefix `CLUSTER_`), e.g.
`CLUSTER_USE_ELEMENT_WEIGHTS=1 ./run.sh run --cluster "M 67" --region-scaled`.

**Rule of the game**: change one thing at a time, and know *why* it moved.
A recall jump with a precision collapse is a lesson, not a win.

---

## 3 · The frontier — pick one (60–100 min)

The baseline uses 16 abundances. Three extensions are waiting. Pick **one**:

### Track A — Spectral embeddings (the RNN latent)

A CNN-LSTM-Attention network trained on the raw 8575-pixel spectrum gives a
256-d latent that beats the 16 abundances on every benchmark.

```bash
./run.sh run --cluster "M 67" --region-scaled \
    --spectral data/embeddings/attention_broad_merged.parquet
```

Compare your cluster's precision/recall to the abundance baseline. Metal-poor
globulars (M 15, M 92) are where the latent wins hardest — abundances
collapse, spectra don't.

### Track B — Isochrone + red-clump distance

Cleaner membership → better cluster parameters. Fit a PARSEC isochrone to your
cluster's members and measure the red-clump distance:

```bash
./run.sh python scripts/red_clump.py --clusters "NGC 6819"
./run.sh python scripts/sweet_spot.py --clusters "NGC 2243"
```

Compare the recovered age + distance to `src/cluster/literature.py`. The red
clump pins the distance (NGC 6819 within 0.01 mag); the main sequence pins the
age. Open clusters with ≥ 20 member giants are the clean cases.

### Track C — Two surveys (GALAH + APOGEE)

APOGEE sees giants, GALAH sees the main sequence. Cross-match them:

```bash
./run.sh python scripts/build_galah_apogee.py --clusters "M 67"
./run.sh python scripts/rerun_combined.py
```

GALAH covers DEC ≲ +25°, so this works for the southern clusters (M 67,
Pleiades, M 15, M 107, NGC 2243, Collinder 261). The north stays APOGEE-only.

---

## 4 · Deliverable (last 20 min)

**One slide**:

1. your cluster + one-sentence context (age, distance, why it's interesting)
2. your best recall / precision / purity — baseline vs yours
3. **the one change that moved them** — and *why*

Beat the baseline — or explain honestly why your cluster resists. A null
result is still a result (that's the paper's own finding for 47 Tuc).

---

## Cluster assignment

Pair up on the crowded fields. Globulars tag cleanly already — the real job
is the open clusters that don't.

| | clusters |
|---|---|
| **Globulars (7, "easy")** | M 3, M 5, M 13, M 15, M 71, M 107, M 92 |
| **Open clusters (18, "the real job")** | Pleiades, King 7, Berkeley 71, IC 166, NGC 2158, NGC 1245, King 5, NGC 7789, NGC 1798, NGC 2420, NGC 6819, M 67, Berkeley 66, NGC 188, NGC 6791, Berkeley 17, **NGC 2243**, **Collinder 261** |

The last two (NGC 2243, Collinder 261) are the new southern sweet-spot
clusters — both surveys see them, so Tracks B and C are wide open there.
M 67 is the workhorse (most data, most results to beat).

---

## Rubric — what "good" looks like

- **Reproduce** — your baseline matches the scoreboard (you can run the pipeline).
- **Tune** — you moved a number *and* can say which knob did it and why.
- **Frontier** — you ran one of the three extensions and reported what it did to your cluster.
- **Honesty** — a precise explanation of a null result beats a lucky big number.