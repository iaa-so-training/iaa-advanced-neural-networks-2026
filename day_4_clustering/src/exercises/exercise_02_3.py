"""Chapter 2, exercise 3 — how much of the summary is one cluster?

    M 67 contributes 230 of 1,002 members. Recompute a macro-average metric
    over clusters with and without M 67 in the pool. How much of the
    published summary depends on one cluster?

The point of the exercise is \\S 9.3's second failure mode, in miniature: a
macro average over clusters is only as stable as its most populous member,
and a sample that is 23% one open cluster can move a headline number on its
own. The companion check is the M 3 ablation of \\S 13 (``cluster ablate``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, members

#: The cluster the exercise removes.
TARGET = "M 67"


def population() -> pd.DataFrame:
    """Member counts per cluster, largest first."""
    data = members()
    counts = data.df["cluster"].value_counts()
    frame = pd.DataFrame({
        "cluster": [str(c) for c in counts.index],
        "n_members": [int(v) for v in counts.to_numpy()],
    })
    frame["share"] = (frame["n_members"] / frame["n_members"].sum()).round(4)
    return frame


def _macro_scores(
    X: np.ndarray, labels: np.ndarray, seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Homogeneity per method, mean ± std over ``seeds``."""
    from cluster.stability import stability
    from exercises.utils import settings

    return stability(X, labels, settings(), seeds=seeds)


def _knn_macro(X: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    """Macro-averaged kNN purity — the parameter-free half of the pair."""
    from exercises.utils import knn_purity_raw

    purity = knn_purity_raw(X, labels, k=k)
    return float(np.mean(list(purity.values()))) if purity else float("nan")


def solve(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    """Score the cluster-only population with and without M 67."""
    data = members()
    labels = data.labels
    keep = labels != TARGET

    with_target = _macro_scores(data.X, labels, seeds)
    without = _macro_scores(data.X[keep], labels[keep], seeds)

    merged = with_target.merge(
        without, on="method", suffixes=("_with", "_without"),
    )
    merged["delta"] = (merged["mean_without"] - merged["mean_with"]).round(4)
    table = merged[[
        "method", "mean_with", "std_with", "mean_without", "std_without",
        "delta",
    ]].round(4)

    # Is the shift larger than the seed noise it has to beat?
    noise = merged[["std_with", "std_without"]].to_numpy().max(axis=1)
    table["beats_seed_noise"] = np.abs(merged["delta"].to_numpy()) > noise

    counts = population()
    target_share = float(
        counts.loc[counts["cluster"] == TARGET, "share"].iloc[0],
    )

    return {
        "population": counts,
        "target": TARGET,
        "target_share": round(target_share, 4),
        "n_with": int(len(labels)),
        "n_without": int(keep.sum()),
        "homogeneity": table,
        "knn_purity_with": round(_knn_macro(data.X, labels), 4),
        "knn_purity_without": round(_knn_macro(data.X[keep], labels[keep]), 4),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Bar chart of homogeneity with and without the target cluster."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["homogeneity"]
    assert isinstance(table, pd.DataFrame)

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 0.2, table["mean_with"], 0.4, yerr=table["std_with"],
           label="all clusters", color="#4c72b0", capsize=3)
    ax.bar(x + 0.2, table["mean_without"], 0.4, yerr=table["std_without"],
           label=f"without {TARGET}", color="#dd8452", capsize=3)
    ax.set_xticks(x, table["method"])
    ax.set_ylabel("homogeneity")
    ax.set_title(f"Does the summary survive dropping {TARGET}?")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the share": (
        "The sample is the 23 open and globular clusters of "
        f"{cite('GarciaDias:19', parenthetical=False)} plus the Pleiades of "
        f"{cite('Kos:17', parenthetical=False)}, and it is badly skewed. "
        "M 67 is 230 of the 1 002 member rows — 23% of the sample, and the "
        "largest single contributor (M 3 is next at 154, 15%). Two clusters "
        "are therefore 38% of every macro average in the workbook. Note these "
        "are row counts, not star counts: section 9.3 shows M 67 also carries "
        "~90 duplicate rows, so its weight in the average is inflated twice "
        "over — once by being large, once by being repeated."
    ),
    "what happens when you drop it": (
        "Measured on the 7-seed protocol (solve() reproduces it), with t-SNE "
        f"{cite('vanderMaaten:08')}, UMAP {cite('McInnes:18')} and EVoC "
        f"{cite('EVoC')} run as in §13: t-SNE "
        "homogeneity falls 0.520 -> 0.322, UMAP 0.521 -> 0.500, EVoC rises "
        "0.454 -> 0.492. kNN purity barely moves (0.365 -> 0.362). So the "
        "answer depends entirely on which method you quote: one loses a fifth "
        "of its score to a single cluster, one is unaffected, one improves. "
        "Only the t-SNE and EVoC shifts clear the seed-to-seed noise "
        "('beats_seed_noise'); the UMAP move does not, and reporting it as an "
        "effect would be reporting noise."
    ),
    "why t-SNE moves most": (
        "M 67 is 230 rows of one chemically coherent, well-populated cluster — "
        "the easiest large blob in the sample, and the one a neighbourhood "
        "embedding most reliably keeps together. Remove it and t-SNE is left "
        "with smaller, sparser groups that its perplexity-30 neighbourhoods "
        "smear into each other. Note also that t-SNE's std is 0.000 here: "
        "with `init='pca'` the embedding is deterministic given the data, so "
        "its seed spread understates its true instability — §9.3 shows the "
        "same matrix moves between 0.218 and 0.560 under *row permutation*, "
        "which is the variation the seed loop cannot see."
    ),
    "the methodological answer": (
        "A macro average over clusters weights every cluster equally by "
        "construction, which is what makes it preferable to a micro average "
        "here — a micro average over *stars* would give M 67 and M 3 almost "
        "40% of the vote outright. But macro averaging does not protect the "
        "number from a cluster that is easy: drop one easy cluster and the "
        "mean moves even though nothing about the method changed. So: report "
        "the per-cluster table next to the macro number, always, and quote "
        "the leave-one-out spread when one cluster dominates the population. "
        "A 0.20 swing from deleting 23% of the rows is not a small print "
        "detail — it is larger than most of the method gaps the workbook "
        "compares."
    ),
    "the general rule": (
        "Any summary statistic over a skewed population needs a leave-one-out "
        "check before it is quoted. Section 13 does exactly this for M 3 "
        "(`uv run cluster ablate --exclude 'M 3'`), and the workbook reports "
        "the result even though it weakened the original claim. Doing the same "
        "for your own cluster is the cheapest robustness check available."
    ),
    "references": reference_list(
        "GarciaDias:19", "Kos:17", "vanderMaaten:08", "McInnes:18", "EVoC",
    ),
}
