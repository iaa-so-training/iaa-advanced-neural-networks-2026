"""Same-population head-to-head between feature sets.

The trap this module exists to avoid: scoring abundances on all 25 clusters
(1215 stars) and the spectral latent on the 5 clusters its embedding happens
to cover (55 stars), then printing the two numbers in adjacent table rows.
The harder problem scores lower, and the gap gets read as a win for the
feature set that was handed the easier task.

:func:`head_to_head` intersects the ``APOGEE_ID`` sets of every arm first,
then scores all of them on exactly those stars and those clusters. Rows are
only comparable when ``n_stars`` and ``n_clusters`` match across them — the
returned frame always carries both so the reader can check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import baseline_matrix, cluster_only, separation_scores
from .config import Settings
from .data import PreparedData
from .spectral import ID_COLUMN, spectral_prepared
from .stability import DEFAULT_SEEDS, degeneracy, stability


@dataclass
class Arm:
    """One feature set in the comparison."""

    label: str
    #: ``None`` scores the ASPCAP abundance columns from ``Settings``.
    embedding_path: str | Path | None = None
    notes: str = ""


@dataclass
class HeadToHead:
    """Result of a common-population comparison."""

    scores: pd.DataFrame
    stability: pd.DataFrame
    degeneracy: pd.DataFrame
    n_stars: int
    n_clusters: int
    clusters: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)


def _arm_frame(
    prepared: PreparedData, arm: Arm, settings: Settings,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(dataframe, feature_columns)`` for one arm."""
    if arm.embedding_path is None:
        df = prepared.df.drop_duplicates(subset=[ID_COLUMN], keep="first")
        return df, list(settings.elements)
    spec = spectral_prepared(prepared, arm.embedding_path, settings)
    return spec.df, list(spec.elements)


def common_population(
    prepared: PreparedData,
    arms: Sequence[Arm],
    settings: Settings,
    *,
    min_members: int = 5,
) -> tuple[set[str], list[str], dict[str, int]]:
    """Intersect the arms, then apply the >=5-members rule to the result.

    Order matters: the minimum-members cut has to come *after* the
    intersection, otherwise a cluster can pass the cut in one arm and fail it
    in another and the arms end up scoring different label sets.
    """
    per_arm: dict[str, set[str]] = {}
    for arm in arms:
        df, _ = _arm_frame(prepared, arm, settings)
        members = cluster_only(df)
        per_arm[arm.label] = set(members[ID_COLUMN].astype(str))

    shared: set[str] = (
        set.intersection(*per_arm.values()) if per_arm else set()
    )
    dropped = {label: len(ids - shared) for label, ids in per_arm.items()}

    members = cluster_only(
        prepared.df.drop_duplicates(subset=[ID_COLUMN], keep="first"),
    )
    members = members[members[ID_COLUMN].astype(str).isin(sorted(shared))]
    counts = members["cluster"].value_counts()
    keep: list[str] = sorted(
        str(c) for c in counts.index if int(counts[c]) >= min_members
    )

    kept = members[pd.Series(members["cluster"]).isin(keep).to_numpy()]
    shared &= set(kept[ID_COLUMN].astype(str))
    return shared, keep, dropped


def head_to_head(
    prepared: PreparedData,
    arms: Sequence[Arm],
    settings: Settings,
    *,
    min_members: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    metric: str = "homogeneity",
) -> HeadToHead:
    """Score every arm on the same stars, same clusters, same clusterers."""
    shared, keep, dropped = common_population(
        prepared, arms, settings, min_members=min_members,
    )
    if not shared:
        raise ValueError(
            "the arms share no cluster members; nothing to compare. "
            f"per-arm losses: {dropped}",
        )

    score_rows: list[dict[str, object]] = []
    stab_rows: list[pd.DataFrame] = []
    degen_rows: list[dict[str, object]] = []
    n_stars = n_clusters = 0

    for arm in arms:
        df, columns = _arm_frame(prepared, arm, settings)
        sub = cluster_only(df)
        sub = sub[sub[ID_COLUMN].astype(str).isin(shared)]
        sub = sub.sort_values(ID_COLUMN).drop_duplicates(
            subset=[ID_COLUMN], keep="first",
        )

        true_labels = sub["cluster"].to_numpy()
        X = baseline_matrix(sub, settings, False, elements=columns)
        n_stars, n_clusters = len(true_labels), int(np.unique(true_labels).size)

        stab = stability(
            X, true_labels, settings, seeds=seeds, metric=metric,
        )
        stab.insert(0, "features", arm.label)
        stab_rows.append(stab)

        from .baseline import _fit_all

        for name, pred in _fit_all(X, settings).items():
            scores = separation_scores(true_labels, pred)
            score_rows.append({
                "features": arm.label,
                "dim": X.shape[1],
                "method": name,
                "n_stars": n_stars,
                "n_clusters": n_clusters,
                **{k: round(v, 4) for k, v in scores.items()},
            })
            degen_rows.append({
                "features": arm.label, "method": name, **degeneracy(pred),
            })

    return HeadToHead(
        scores=pd.DataFrame(score_rows),
        stability=pd.concat(stab_rows, ignore_index=True),
        degeneracy=pd.DataFrame(degen_rows),
        n_stars=n_stars,
        n_clusters=n_clusters,
        clusters=keep,
        dropped=dropped,
    )


def pivot_scores(
    result: HeadToHead, metric: str = "homogeneity",
) -> pd.DataFrame:
    """Slide-shaped table: one row per feature set, one column per method."""
    return result.scores.pivot_table(
        index=["features", "dim"], columns="method", values=metric,
    ).round(3)


def _first(values: pd.Series) -> object:
    """``aggfunc`` that keeps the first value ("first" is absent from the stubs)."""
    return values.iloc[0]


def pivot_with_errors(result: HeadToHead) -> pd.DataFrame:
    """Same shape as :func:`pivot_scores` but cells read ``0.79 ± 0.02``."""
    cells = result.stability.copy()
    cells["cell"] = [
        f"{m:.2f} ± {s:.2f}"
        for m, s in zip(cells["mean"], cells["std"], strict=True)
    ]
    return cells.pivot_table(
        index="features", columns="method", values="cell", aggfunc=_first,
    )
