"""Seed-stability and degeneracy checks for clustering scores.

A single homogeneity number from one random seed is not a measurement. EVoC
is stochastic, UMAP is stochastic unless pinned, and on samples of a few
dozen stars the spread across seeds can exceed the gap between the feature
sets being compared. :func:`stability` reports mean ± std over seeds so a
claimed win can be checked against its own noise floor.

:func:`degeneracy` catches the other failure mode: a clusterer that gives up
and returns one blob still produces a finite homogeneity, and quoting a ratio
against that number compares against a crash, not against a baseline.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .baseline import _fit_all, separation_scores
from .config import Settings

DEFAULT_SEEDS: tuple[int, ...] = (42, 0, 1, 2, 7, 13, 99)

#: A partition this concentrated means the clusterer collapsed rather than
#: found structure — the score is a floor artifact, not a result.
DEGENERACY_FRACTION = 0.6


def stability(
    X: np.ndarray,
    true_labels: np.ndarray,
    settings: Settings,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    metric: str = "homogeneity",
) -> pd.DataFrame:
    """Score every method over several seeds.

    Returns one row per method with ``mean``, ``std``, ``min``, ``max`` and
    the raw per-seed values, so a quoted number can always be paired with
    its uncertainty.
    """
    runs: dict[str, list[float]] = {}
    for seed in seeds:
        local = copy.deepcopy(settings)
        local.random_state = seed
        for name, pred in _fit_all(X, local).items():
            runs.setdefault(name, []).append(
                separation_scores(true_labels, pred)[metric],
            )

    rows = []
    for name, values in runs.items():
        arr = np.asarray(values, dtype=float)
        rows.append({
            "method": name,
            "metric": metric,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_seeds": len(arr),
            "values": [round(float(v), 4) for v in arr],
        })
    return pd.DataFrame(rows)


def degeneracy(pred_labels: np.ndarray) -> dict[str, object]:
    """Flag a collapsed partition (one giant cluster, or everything noise)."""
    labels = np.asarray(pred_labels)
    n = labels.size
    if n == 0:
        return {"n_clusters": 0, "largest_fraction": 0.0, "degenerate": True}

    real = labels[labels != -1]
    values, counts = np.unique(real, return_counts=True)
    largest = int(counts.max()) if counts.size else 0
    n_clusters = int(values.size)
    frac = largest / n

    return {
        "n_clusters": n_clusters,
        "largest_cluster": largest,
        "largest_fraction": round(frac, 3),
        "n_noise": int((labels == -1).sum()),
        "degenerate": bool(n_clusters <= 1 or frac >= DEGENERACY_FRACTION),
    }


def degeneracy_table(
    X: np.ndarray, settings: Settings, true_labels: np.ndarray,
) -> pd.DataFrame:
    """Per-method degeneracy + score, to expose collapsed baselines."""
    rows = []
    for name, pred in _fit_all(X, settings).items():
        row: dict[str, object] = {"method": name}
        row.update(degeneracy(pred))
        row["homogeneity"] = round(
            separation_scores(true_labels, pred)["homogeneity"], 4,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def format_stability(frame: pd.DataFrame) -> str:
    """Render a stability table as ``0.761 ± 0.016`` strings."""
    lines = []
    for _, row in frame.iterrows():
        lines.append(
            f"  {row['method']:<7} {row['mean']:.3f} ± {row['std']:.3f}"
            f"   [{row['min']:.2f}, {row['max']:.2f}]"
            f"   n_seeds={row['n_seeds']}",
        )
    return "\n".join(lines)
