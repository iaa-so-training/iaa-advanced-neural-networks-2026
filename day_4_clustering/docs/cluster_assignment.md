# Cluster assignment — 25 clusters, ~30 of you

Pair up on the open clusters (they're the hard part). Globulars already tag
cleanly, so they're a faster, solo-friendly track. The **suggested track**
is advisory — you may pick any of A (spectral latent), B (isochrone +
red-clump), or C (GALAH + APOGEE cross-match).

| cluster | kind | note | suggested track |
|---|---|---|---|
| **M 3** | globular | metal-poor (−1.6) | A — spectral wins |
| **M 5** | globular | | A |
| **M 13** | globular | | A |
| **M 15** | globular | metal-poor (−2.2), southern | A or C |
| **M 71** | globular | | A |
| **M 107** | globular | southern (DEC −13°) | C |
| **M 92** | globular | metal-poor (−2.3) | A |
| **Pleiades** | open | young (0.1 Gyr) — Kos's cluster; chemistry *fails* here | A (expect a null result) |
| **King 7** | open | sparse, faint | A |
| **Berkeley 71** | open | sparse | A |
| **IC 166** | open | sparse, contaminated | A |
| **NGC 2158** | open | | A |
| **NGC 1245** | open | | A |
| **King 5** | open | | A |
| **NGC 7789** | open | giant-rich (1.4 Gyr) | B — clump |
| **NGC 1798** | open | | A |
| **NGC 2420** | open | | A |
| **NGC 6819** | open | giant-rich — clump resid +0.01 | B |
| **M 67** | open | workhorse — most data to beat | C (or A) |
| **Berkeley 66** | open | sparse | A |
| **NGC 188** | open | old (7.6 Gyr) | B |
| **NGC 6791** | open | old, metal-rich (+0.4) — 71 field contaminants | B (or A) |
| **Berkeley 17** | open | | A |
| **NGC 2243** | open | **sweet spot** — MS + clump, both surveys | B or C |
| **Collinder 261** | open | **sweet spot** — 214 GALAH MS members | C |

## The three tracks (recap)

- **A — Spectral embeddings.** `cluster run --spectral data/embeddings/attention_broad_merged.parquet`.
  The 256-d RNN latent vs the 16 abundances. Best on metal-poor globulars.
- **B — Isochrone + red-clump.** `python scripts/red_clump.py` +
  `python scripts/sweet_spot.py`. Recover age + distance; compare to
  `src/cluster/literature.py`. Best on giant-rich opens (NGC 6819, NGC 7789).
- **C — Two surveys.** `python scripts/build_galah_apogee.py` +
  `python scripts/rerun_combined.py`. GALAH main sequence + APOGEE giants.
  Southern clusters only (DEC ≲ +25°).

## Scoreboard

Your baseline numbers live in `docs/region_sweep_results.md`. The full
levers map is `docs/experiment_results.md`. The walkthrough is
`docs/student_activities.md`.
