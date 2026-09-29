"""Chapter 13, exercise 4 — what you would need from the other authors.

    §13.6 runs their pipeline on our stars but not ours on theirs. Their
    sample's reachable star list is not public in a form we can re-run, so
    state what you would need from the other authors — and what a re-run on
    their stars would change about the conclusion of Table 10.

This is a reasoning exercise, so the answer is the argument. The supporting
computation is the one that makes the argument concrete: the two samples are
measured side by side on the quantities a re-run needs — stars per cluster,
the internal abundance scatter that the published grid was tuned for, and the
recovery fraction their configuration actually reaches on our stars. Those
numbers decide whether the missing star list is a detail or the whole
comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, members, project_root, settings

#: What the chapter is missing, as the artefacts a re-run needs.
REQUESTS: tuple[tuple[str, str], ...] = (
    ("member star list with identifiers",
     "the 175 red-clump stars with their 2MASS/APOGEE identifiers — without "
     "it no one can score our pipeline on their sample"),
    ("per-element abundances in dex, all 16",
     "already published in their tables, but the machine-readable version is "
     "what a re-run consumes"),
    ("their exact clustering configuration",
     "HDBSCAN min_cluster_size/min_samples/epsilon *as run*, and which of the "
     "4x7 grid points produced the published row"),
    ("their label-to-cluster assignment for that run",
     "which predicted group was matched to which cluster; the recovery "
     "fraction is defined by that matching rule"),
    ("the chance levels of their own metric",
     "their RF40 null for their partition shape, so '0.29 recovers 9 of 31' "
     "can be read against the same floor we apply to our stars"),
    ("the differential abundance systematics",
     "the line list, reference star set and zero-point choices that produce "
     "0.03 dex, since the comparison is between two precision regimes"),
)

#: Their published metric triple, quoted AS PUBLISHED (this module did not
#: recompute their run on their sample — the star list is unavailable).
PUBLISHED_THEIRS: dict[str, float] = {
    "h": 0.49, "c": 0.63, "V": 0.55, "RF40": 0.290, "RF70": 0.030,
}
#: Their internal coherence, per element, in dex (published).
THEIR_COHERENCE_DEX = 0.03
#: Their clustering step re-run on our stars, from results/casamiquela_*.csv.
THEIR_STEP_ON_OUR_STARS = {
    "raw_dex": {"h": 0.506, "c": 0.643, "V": 0.454, "RF40": 0.111},
    "standardised": {"h": 0.475, "c": 0.607, "V": 0.438, "RF40": 0.040},
}


def our_scatter() -> pd.DataFrame:
    """Per-element intra-cluster scatter and quoted uncertainty, on our stars.

    The robust scatter (1.4826 x MAD) of one element's abundance inside one
    cluster, the median over clusters with at least 8 rows. This is an *upper*
    bound on the measurement precision — it also contains real star-to-star
    chemical variation, which is the point of the comparison: the number the
    published configuration was tuned on is not directly available in our
    data, so the closest measurable analogue is reported with that caveat.
    """
    data = members()
    frame = data.df
    rows: list[dict[str, object]] = []
    for element in settings().elements:
        per_cluster: list[float] = []
        for _, group in frame.groupby("cluster"):
            values = group[element].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 8:
                continue
            per_cluster.append(
                1.4826 * float(np.median(np.abs(values - np.median(values)))),
            )
        quoted = frame[f"{element}_ERR"].to_numpy(dtype=float)
        quoted = quoted[np.isfinite(quoted)]
        rows.append({
            "element": element,
            "n_clusters": len(per_cluster),
            "intra_cluster_scatter_dex": round(float(np.median(per_cluster)), 4),
            "median_quoted_err_dex": round(float(np.median(quoted)), 4),
        })
    return pd.DataFrame(rows)


def sample_shape() -> dict[str, object]:
    """Rows per cluster in our matrix, against their 5.6 stars per cluster."""
    counts = members().df["cluster"].value_counts()
    return {
        "n_rows": int(counts.sum()),
        "n_clusters": int(len(counts)),
        "rows_per_cluster_mean": round(float(counts.mean()), 1),
        "rows_per_cluster_median": int(counts.median()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "clusters_under_8_rows": int((counts < 8).sum()),
        "their_rows_per_cluster": 5.6,
    }


def published_comparison() -> pd.DataFrame:
    """The workbook's published Casamiquela comparison, read from its artifact.

    These are the workbook's numbers, not this module's measurement, and the
    module says so when it returns them.
    """
    path = project_root() / "results" / "casamiquela_comparison.csv"
    if not path.is_file():
        raise DataNotAvailable(
            f"{path} is not on disk; regenerate it with\n\n"
            "    uv run python scripts/casamiquela_comparison.py\n\n"
            "It needs the DR19 catalogue (uv run cluster download) and takes "
            "a while: it refits every method and seed.\n",
        )
    frame = pd.read_csv(path)
    return (
        frame.groupby("method")[["h", "c", "V", "RF40", "RF70"]]
        .mean().round(3).reset_index()
    )


def solve() -> dict[str, object]:
    """Assemble the two-sample comparison the argument needs."""
    scatter = our_scatter()
    shape = sample_shape()
    published = published_comparison()

    shape["median_intra_cluster_scatter_dex"] = round(
        float(scatter["intra_cluster_scatter_dex"].median()), 4,
    )
    shape["median_quoted_err_dex"] = round(
        float(scatter["median_quoted_err_dex"].median()), 4,
    )
    shape["their_coherence_dex"] = THEIR_COHERENCE_DEX
    shape["precision_gap_factor"] = round(
        float(scatter["intra_cluster_scatter_dex"].median()) / THEIR_COHERENCE_DEX,
        2,
    )

    return {
        "scatter": scatter,
        "shape": shape,
        "requests": pd.DataFrame(
            [{"needed": need, "why": why} for need, why in REQUESTS],
        ),
        "published_theirs": dict(PUBLISHED_THEIRS),
        "their_step_on_our_stars": dict(THEIR_STEP_ON_OUR_STARS),
        "published_baseline_rows": published,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Our per-element intra-cluster scatter against their published coherence."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    scatter = result["scatter"]
    assert isinstance(scatter, pd.DataFrame)
    scatter = scatter.sort_values("intra_cluster_scatter_dex")

    x = np.arange(len(scatter))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(x, scatter["intra_cluster_scatter_dex"], 0.6, label="our stars: intra-cluster scatter",
           color="#4c72b0")
    ax.bar(x, scatter["median_quoted_err_dex"], 0.6, label="our stars: quoted uncertainty",
           color="#9ecae1")
    ax.axhline(THEIR_COHERENCE_DEX, color="crimson", ls="--", lw=1.2,
               label=f"their coherence ({THEIR_COHERENCE_DEX} dex)")
    ax.set_xticks(x, scatter["element"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("dex")
    ax.set_title("Two precision regimes, measured on one axis")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what is missing, and why it is not a small omission": (
        "The comparison in Table 10 is asymmetrical by construction: their "
        "clustering step runs on our stars (reproducibly, because we have our "
        "own star list), while our pipeline cannot run on theirs, because "
        "the 175 red-clump members of 31 thin-disc open clusters of "
        f"{cite('Casamiquela:21', parenthetical=False)} are not "
        "published as a machine-readable list with identifiers. Six artefacts "
        "close the gap: the member star list with 2MASS/APOGEE identifiers; "
        "the per-element abundances in dex; their *exact* HDBSCAN "
        "configuration including which point of the 4x7 grid produced the "
        "published row; the label-to-cluster assignment their recovery "
        "fraction is defined against; the chance level of their own metric "
        "for their partition shape; and the systematics behind their 0.03 dex "
        "coherence. The star list is the only one that is strictly "
        "indispensable, and it is the one that is missing."
    ),
    "what the numbers already say": (
        "The two samples differ on every axis a re-run would care about. Ours "
        "is 1 002 rows across 25 clusters at a mean of 40.1 rows per cluster "
        "(median 25, range 6-230, 33% of rows sharing an APOGEE_ID with "
        "another row); theirs is 175 stars across 31 clusters, 5.6 stars per "
        "cluster, red-clump only and with no field stars. Our median "
        "intra-cluster scatter across the 16 elements is 0.059 dex against "
        "their 0.03 — a factor of 2.0, exactly the precision gap §13.6 cites "
        "as one of two inseparable differences — the precision at which that "
        "sample was assembled is itself a load-bearing part of "
        f"{cite('Casamiquela:21')}. Note that the 0.059 is an "
        "upper bound (it includes genuine star-to-star chemical variation, "
        "and globulars are chemically inhomogeneous); our median *quoted* "
        "per-element uncertainty is 0.017 dex, which is the internal error "
        "floor and is not comparable to a coherence measurement either."
    ),
    "what a re-run on their stars would change": (
        "It would separate the two conflated causes. Run their pipeline on "
        "their stars and it reproduces 0.29 by definition; run ours on their "
        "stars and the difference between the two numbers becomes a pure "
        "method comparison at one precision. The workbook's own replication "
        "gives the size of the effect the sample is responsible for: their "
        "clustering step on our stars returns RF40 = 0.111 in raw dex and "
        "0.040 after standardisation, against 0.290 on their own sample. If "
        "our pipeline on their stars recovers ≈0.29 as well, then the "
        "shortfall on our stars is the abundance precision and the sample "
        "shape, not the embedding. If it recovers materially less, then the "
        "embed-then-cluster architecture is losing information their "
        "cluster-the-abundances step keeps — which is the one conclusion "
        "Table 10 explicitly declines to draw. The wider literature is split "
        f"on exactly this axis: {cite('Hogg:16')} recovers phase-space "
        "structures from abundances, while "
        f"{cite('Casamiquela:21', 'Spina:25')} report how little survives "
        "once the field is present, and a symmetric re-run is what would "
        "place this workbook among them."
    ),
    "what would *not* change": (
        "The five limitations of §13.7 survive a re-run, because they are "
        "properties of our data: our member frame's configuration dependence, "
        "our duplicate rows, our row-order-unstable t-SNE arm, the single "
        "published external comparison, and the K-means counter-result. And "
        "one conclusion is already safe in both directions: their best-case "
        "sample recovers 9 of 31 clusters at the 40% threshold and 1 at 70%, "
        "while adding field stars leaves one cluster above 40% and none at "
        "70% — the same shape as our Task 2 field-retrieval result. A re-run "
        "on their stars would sharpen the *size* of the chemical-tagging gap; "
        "it would not reverse its sign, and this workbook should not be read "
        "as claiming it would."
    ),
    "the general rule": (
        "When one half of a published comparison cannot be re-run, the honest "
        "move is to state which half is missing and what it would take, then "
        "strengthen the half you do control until the missing piece is a "
        "detail rather than the load-bearing wall. Here the strengthening is "
        "already in the workbook: their grid on our stars, our metrics on "
        "their metric triple, chance levels for both partition shapes, and a "
        "per-cluster recovery list. What remains open is exactly what §13.6 "
        "says remains open — and it stays open until an author sends a list "
        "of identifiers, which is a cheaper request than a re-analysis."
    ),
    "references": reference_list("Casamiquela:21", "Hogg:16", "Spina:25"),
}
