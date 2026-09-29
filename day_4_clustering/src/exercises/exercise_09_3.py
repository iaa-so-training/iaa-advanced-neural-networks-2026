"""Chapter 9, exercise 3 — collapsing the duplicate rows.

    Collapse the duplicate rows of the member matrix (for example by taking
    the median abundance vector per APOGEE_ID) and re-run the cluster-only
    table. Which rows of Table 'honest' move, and does M 3's share of the
    sample change the conclusion of the ablation?

\\S 9.3's failure mode 3, second half: the 1 002 "members" of
\\textbf{Table~\\ref{tab:clusters}} are row counts, not star counts, and the
repeats are concentrated in the two clusters that already dominate every macro
average. This module verifies the audit's counts on the current frame,
collapses the matrix by median per identifier, re-runs the seven-seed
cluster-only table, and repeats the M 3 ablation on both populations — where
it turns up a degenerate partition that the score alone would have hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Column holding the star identifier. 34 member rows carry an empty string.
ID_COLUMN: str = "APOGEE_ID"

#: The cluster the workbook's published ablation removes (\\S 13).
ABLATED: str = "M 3"


def duplicate_audit(df: pd.DataFrame) -> dict[str, object]:
    """Reproduce the audit of \\S 9.3: how many rows are repeats, and whose."""
    ids = df[ID_COLUMN].astype(str)
    missing = int((ids == "").sum())
    named = pd.Series(ids[ids != ""])
    shared = int(ids.duplicated(keep=False).sum())

    per_cluster = (
        pd.DataFrame(df[ids.duplicated(keep=False).to_numpy(dtype=bool)])
        ["cluster"].value_counts()
        .rename("duplicate_rows").to_frame().reset_index()
    )
    per_cluster.columns = ["cluster", "duplicate_rows"]

    return {
        "n_rows": int(len(df)),
        "n_distinct_ids": int(named.nunique()),
        "n_without_id": missing,
        "n_rows_sharing_an_id": shared,
        "n_stars_after_collapse": int(named.nunique() + missing),
        "max_rows_for_one_star": int(named.value_counts().max()),
        "per_cluster": per_cluster,
    }


def abundance_spread(df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """Per-element median peak-to-peak spread between rows of the same star.

    If the copies were re-reductions of one spectrum this would be ~0; it is
    not, which is why collapsing them is a real change to the data and not a
    bookkeeping tidy-up.
    """
    ids = df[ID_COLUMN].astype(str)
    named = df[ids != ""].copy()
    named["_id"] = ids[ids != ""]
    grouped = named.groupby("_id")[elements]
    spread = grouped.max() - grouped.min()
    repeated = spread[(grouped.size() > 1).to_numpy(dtype=bool)]
    return pd.DataFrame({
        "element": elements,
        "median_spread_dex": [
            round(float(repeated[e].median()), 4) for e in elements
        ],
    })


def collapse(df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """One row per star: median abundance vector per ``APOGEE_ID``.

    Rows with no identifier cannot be matched to anything, so each is kept as
    its own star — the conservative choice, and the one that keeps the 34
    anonymous rows in the sample rather than silently merging them. The
    result is left in groupby order (sorted by star key); exercise 9.2's
    lesson applies, so the row order is re-randomised explicitly downstream
    rather than being inherited by accident.
    """
    ids = df[ID_COLUMN].astype(str).to_numpy()
    key = np.where(ids == "", [f"__row_{i}" for i in range(len(df))], ids)
    work = df.copy()
    work["_star"] = key
    aggregation: dict[str, str] = dict.fromkeys(elements, "median")
    aggregation["cluster"] = "first"
    return pd.DataFrame(work.groupby("_star", as_index=False).agg(aggregation))


def _matrix(frame: pd.DataFrame, min_members: int = 5) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
]:
    """C-space matrix, labels and star keys for a collapsed frame (≥5 rule)."""
    from exercises.utils import abundance_matrix

    counts = frame["cluster"].value_counts()
    keep = frame["cluster"].isin(
        [c for c in counts.index if int(counts[c]) >= min_members],
    ).to_numpy(dtype=bool)
    sub = pd.DataFrame(frame[keep]).reset_index(drop=True)
    return (
        abundance_matrix(sub, settings()),
        sub["cluster"].to_numpy(),
        sub["_star"].to_numpy(),
    )


def _reorder(keys: np.ndarray, order: str, seed: int = 7) -> np.ndarray:
    """Row index for one of the three orders the ablation is run under."""
    if order == "as built":
        return np.arange(len(keys))
    if order == "by identifier":
        return np.argsort(keys.astype(str), kind="stable")
    return np.random.default_rng(seed).permutation(len(keys))


def _table(
    X: np.ndarray, labels: np.ndarray, seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    from cluster.stability import stability

    frame = stability(X, labels, settings(), seeds=seeds)
    return pd.DataFrame(
        frame[["method", "mean", "std", "min", "max"]],
    ).round(4)


def solve(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    """Audit, collapse, re-score, and re-run the M 3 ablation on both."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import (
        cluster_embedding,
        fit_evoc,
        fit_tsne,
        fit_umap,
    )
    from cluster.stability import degeneracy
    from exercises.utils import knn_purity_raw, members

    data = members()
    audit = duplicate_audit(data.df)
    spread = abundance_spread(data.df, data.elements)

    collapsed = collapse(data.df, data.elements)
    Xc, yc, keys_c = _matrix(collapsed)
    keys_rows = data.df[ID_COLUMN].astype(str).to_numpy()

    before = _table(data.X, data.labels, seeds)
    after = _table(Xc, yc, seeds)
    merged = before.merge(after, on="method", suffixes=("_rows", "_stars"))
    merged["delta"] = (merged["mean_stars"] - merged["mean_rows"]).round(4)

    # The ablation, on both populations and under three row orders — because
    # \\S 9.3's other failure mode is still live here, and it shows up.
    cfg = settings()
    params = {k: v for k, v in cfg.tsne.items() if k != "method"}
    ablation_rows = []
    populations: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("rows (1 002)", data.X, data.labels, keys_rows),
        ("stars (collapsed)", Xc, yc, keys_c),
    ]
    for tag, X, y, keys in populations:
        for order in ("as built", "by identifier", "shuffled"):
            index = _reorder(keys, order)
            Xo, yo = X[index], y[index]
            mask = yo != ABLATED
            target = yo[mask]
            pred = cluster_embedding(
                fit_tsne(Xo[mask], params, seeds[0]), cfg.hdbscan,
            )
            collapse_flag = degeneracy(pred)
            umap_pred = cluster_embedding(
                fit_umap(Xo[mask], cfg.umap, seeds[0]), cfg.hdbscan,
            )
            evoc_pred = fit_evoc(Xo[mask], cfg.evoc, seeds[0])
            ablation_rows.append({
                "population": tag,
                "row_order": order,
                "n": int(mask.sum()),
                "tsne_homogeneity": round(
                    separation_scores(target, pred)["homogeneity"], 4),
                "umap_homogeneity": round(
                    separation_scores(target, umap_pred)["homogeneity"], 4),
                "evoc_homogeneity": round(
                    separation_scores(target, evoc_pred)["homogeneity"], 4),
                "n_groups": collapse_flag["n_clusters"],
                "largest_fraction": collapse_flag["largest_fraction"],
                "degenerate": collapse_flag["degenerate"],
            })

    shares = {
        "M 67_rows": round(float((data.labels == "M 67").mean()), 4),
        "M 67_stars": round(float((yc == "M 67").mean()), 4),
        "M 3_rows": round(float((data.labels == ABLATED).mean()), 4),
        "M 3_stars": round(float((yc == ABLATED).mean()), 4),
    }

    return {
        "audit": audit,
        "per_element_spread": spread,
        "median_spread_all_elements": round(
            float(spread["median_spread_dex"].median()), 4),
        "n_rows": int(len(data.X)),
        "n_stars": int(len(Xc)),
        "shares": shares,
        "homogeneity": merged,
        "knn_purity_rows": round(
            float(np.mean(list(knn_purity_raw(data.X, data.labels, k=15).values()))), 4),
        "knn_purity_stars": round(
            float(np.mean(list(knn_purity_raw(Xc, yc, k=15).values()))), 4),
        "ablation": pd.DataFrame(ablation_rows),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Homogeneity before and after collapsing rows to stars."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["homogeneity"]
    assert isinstance(table, pd.DataFrame)

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - 0.2, table["mean_rows"], 0.4, yerr=table["std_rows"],
           label="1 002 rows", color="#4c72b0", capsize=3)
    ax.bar(x + 0.2, table["mean_stars"], 0.4, yerr=table["std_stars"],
           label="840 stars (median-collapsed)", color="#dd8452", capsize=3)
    ax.set_xticks(x, table["method"])
    ax.set_ylabel("homogeneity (7 seeds)")
    ax.set_title("Does the cluster-only table survive de-duplication?")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the audit reproduces": (
        "Every count in §9.3 checks out on the current frame. Of 1 002 member "
        "rows, 34 carry no APOGEE_ID — the per-star identifier APOGEE "
        f"{cite('Majewski:17')} assigns — and the remaining 968 resolve to "
        "806 "
        "distinct stars (the workbook's 807 counts the empty identifier as one "
        "more value), so exactly 332 rows share an identifier with another row "
        "— and the concentration is as reported: M 3 contributes 105 of those "
        "rows and M 67 ninety, with NGC 6819 (32), IC 166 (23) and NGC 2243 "
        "(23) next. One star appears four times. The copies are not identical: "
        "the median peak-to-peak spread between two rows of the same star is "
        "0.036 dex across elements, ranging from 0.015 dex in FE_H to 0.22 dex "
        "in V_FE — comparable to the abundance differences the whole exercise "
        "is trying to resolve."
    ),
    "what collapsing does to the population": (
        "Median per identifier leaves 840 stars from 1 002 rows, still 25 "
        "clusters, all still above the 5-member floor. M 67 goes 230 -> 182 "
        "rows (share 0.230 -> 0.217) and M 3 goes 154 -> 95 (share 0.154 -> "
        "0.113). So de-duplication does what it should: it removes about a "
        "quarter of M 3's weight in every macro average, and rather less of "
        "M 67's."
    ),
    "which rows of the table move": (
        "Seven seeds, same settings. t-SNE "
        f"{cite('vanderMaaten:08')} 0.5199 -> 0.4897 (-0.030); UMAP "
        f"{cite('McInnes:18')} "
        "0.5214 +- 0.018 -> 0.5217 +- 0.011 (unchanged, and its seed spread "
        f"narrows); EVoC {cite('EVoC')} 0.4536 +- 0.025 -> 0.4697 +- 0.027 "
        "(+0.016, inside its "
        "own noise). kNN purity, the parameter-free check, barely moves: 0.365 "
        "-> 0.355. So only the t-SNE row shifts by more than its neighbours' "
        "noise, and it shifts *down* — the duplicates were helping it, by "
        "giving its two easiest clusters extra near-identical points to build "
        "dense neighbourhoods from. Nothing in the table is overturned; the "
        "ranking of the three methods is unchanged."
    ),
    "the ablation, and the surprise": (
        "This is where the exercise earns its place. Dropping M 3 from the "
        "*row* matrix (848 rows) leaves t-SNE at homogeneity 0.512, a mild "
        "fall from 0.520, while UMAP rises to 0.527-0.538 and EVoC to "
        "0.458-0.528 — the range is over the three row orders of exercise 9.2. "
        "Dropping M 3 from the *collapsed* matrix (745 stars) collapses t-SNE "
        "to 0.029 in two of the three row orders, and the degeneracy check "
        "explains why: it returns 3 groups with 94% of the stars in one of "
        "them, flagged degenerate. That is not a measurement of chemical "
        "tagging without M 3, it is §9.3's failure mode 1, and anyone reading "
        "the homogeneity alone would have concluded that M 3 carried the "
        "entire result. UMAP (0.49-0.54) and EVoC (0.48-0.51) are untouched "
        "on the same collapsed data, which is the tell: the collapse is a "
        "t-SNE optimisation failure on a smaller, less-redundant matrix, not "
        "a property of the sample. Note the third row order, where t-SNE "
        "recovers to 0.444 — the collapse is not even stable in its own "
        "failure, which is exercise 9.2's lesson arriving from a third "
        "direction."
    ),
    "does M 3 change the conclusion": (
        "No — and the reason it does not is the degeneracy column, not the "
        "score. On the row matrix M 3's removal moves t-SNE by 0.008 and "
        "*raises* UMAP (0.521 -> 0.527-0.538) and EVoC (0.454 -> 0.458-0.528); "
        "on the collapsed matrix it raises UMAP and EVoC by similar amounts "
        "and "
        "crashes t-SNE into a flagged degenerate partition that must be "
        "discarded rather than quoted. Either way the workbook's published "
        "conclusion — that abundances alone separate these clusters poorly, "
        "around h ≈ 0.5, far below the kinematic ceiling — survives both "
        "de-duplication and the ablation."
    ),
    "the lesson": (
        "Quote star counts, not row counts, and say which you mean. The "
        "duplicates inflated Table 'clusters', double-weighted the two largest "
        "clusters in every macro average, and — as §9.3 suspected — propped up "
        "the abundances-only t-SNE row. Collapsing them costs three lines of "
        "pandas and changes one number by 0.03, which is small; but the same "
        "three lines turn a quiet 0.51 in the ablation into a loud degenerate "
        "flag, which is exactly the kind of thing a validation chapter exists "
        "to surface. Run the de-duplication *before* the science, not after a "
        "reviewer asks."
    ),
    "references": reference_list(
        "Majewski:17", "vanderMaaten:08", "McInnes:18", "EVoC",
    ),
}
