"""Paper baseline (Garcia-Dias et al. 2019): multiclass separation of
cluster-only stars.

Re-creates the 2019 setup — take *only* the known cluster members (no field
stars), cluster them in abundance space (optionally extended with
kinematics), and score how well each star is assigned back to its own
cluster. Scoring uses the paper's own merit functions: homogeneity,
completeness, v-measure, and accuracy (best label permutation).

This is the "can we separate the 23 clusters from each other?" question,
complementary to the field-contaminated retrieval benchmark in
``benchmark.py`` ("can we pull one cluster out of the field?").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import cluster_embedding, fit_evoc, fit_tsne, fit_umap
from .config import Settings
from .data import PreparedData

# Kinematic feature block: distance + space motion (position is deliberately
# excluded — RA/DEC is not kinematics and would trivially separate clusters).
KINEMATIC_COLUMNS = [
    "GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG",
]

_SCORE_COLUMNS = [
    "method", "n_stars", "n_clusters",
    "homogeneity", "completeness", "v_measure", "accuracy",
]


def cluster_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the field, keep only stars labelled as cluster members."""
    return df[df["cluster"] != "field"]


def baseline_matrix(
    df: pd.DataFrame, settings: Settings, use_kinematics: bool,
    elements: list[str] | None = None,
) -> np.ndarray:
    """Cluster-only feature matrix.

    Builds the paper's standard-scaler matrix directly (impute → standardise,
    no L2-normalisation, no weights) for the given feature columns, so it
    works identically for abundances (``settings.elements``) and spectral
    embeddings (``prepared.elements``). When ``use_kinematics`` is set, a
    standardised kinematic block (parallax, pmra, pmdec, rv) is appended.
    """
    columns = list(elements) if elements is not None else list(settings.elements)
    X = df[columns].to_numpy(dtype=float)

    if settings.impute_missing:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med[np.newaxis, :])

    med = np.nanmedian(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    X = (X - med) / scale

    if not use_kinematics:
        return X

    kin = df[KINEMATIC_COLUMNS].to_numpy(dtype=float)
    med = np.nanmedian(kin, axis=0)
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    kin = (kin - med) / scale
    return np.hstack([X, kin])


def _codes(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to 0..k-1 integer codes (sklearn needs one type)."""
    return np.unique(labels, return_inverse=True)[1]


def _accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Fraction correctly classified under the best label permutation.

    Clustering labels are arbitrary, so the confusion matrix is matched to
    the true labels with the Hungarian algorithm (same idea as the paper's
    cross-matched accuracy score).
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    if len(true_labels) == 0:
        return 0.0
    cm = confusion_matrix(_codes(true_labels), _codes(pred_labels))
    row_ind, col_ind = linear_sum_assignment(-cm)
    return float(cm[row_ind, col_ind].sum() / len(true_labels))


def separation_scores(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """The paper's merit functions for a multiclass cluster assignment."""
    from sklearn.metrics import (
        completeness_score,
        homogeneity_score,
        v_measure_score,
    )

    true = _codes(true_labels)
    pred = _codes(pred_labels)
    return {
        "homogeneity": float(homogeneity_score(true, pred)),
        "completeness": float(completeness_score(true, pred)),
        "v_measure": float(v_measure_score(true, pred)),
        "accuracy": _accuracy(true_labels, pred_labels),
    }


def _fit_all(X: np.ndarray, settings: Settings) -> dict[str, np.ndarray]:
    """Run t-SNE/UMAP -> HDBSCAN and EVoC; return ``{method: labels}``."""
    out: dict[str, np.ndarray] = {}
    tsne_params = {k: v for k, v in settings.tsne.items() if k != "method"}
    z_tsne = fit_tsne(X, tsne_params, settings.random_state)
    out["t-SNE"] = cluster_embedding(z_tsne, settings.hdbscan)

    z_umap = fit_umap(X, settings.umap, settings.random_state)
    out["UMAP"] = cluster_embedding(z_umap, settings.hdbscan)

    try:
        out["EVoC"] = fit_evoc(X, settings.evoc, settings.random_state)
    except ImportError as exc:
        print(f"⚠ EVoC unavailable ({exc}); skipping.")

    return out


def baseline_labels(
    prepared: PreparedData,
    settings: Settings,
    *,
    use_kinematics: bool = False,
    min_members: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cluster-only fit: ``(true_labels, {method: predicted_labels})``.

    ``min_members`` drops clusters with fewer than that many members (the
    paper required ≥ 5), mirroring the 2019 sample selection.
    """
    sub = cluster_only(prepared.df)
    counts = sub["cluster"].value_counts()
    keep = [c for c in counts.index if counts[c] >= min_members]
    sub = sub[sub["cluster"].isin(keep)]

    if sub.empty:
        return np.array([], dtype=object), {}

    true_labels = sub["cluster"].to_numpy()
    X = baseline_matrix(
        sub, settings, use_kinematics, elements=list(prepared.elements),
    )
    return true_labels, _fit_all(X, settings)


def run_baseline(
    prepared: PreparedData,
    settings: Settings,
    *,
    use_kinematics: bool = False,
    min_members: int = 5,
) -> pd.DataFrame:
    """Cluster-only multiclass benchmark: t-SNE / UMAP / EVoC scored with
    the paper's homogeneity / completeness / v-measure / accuracy.
    """
    true_labels, labels = baseline_labels(
        prepared, settings, use_kinematics=use_kinematics, min_members=min_members,
    )
    if len(true_labels) == 0:
        return pd.DataFrame(columns=_SCORE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for name, pred in labels.items():
        scores = separation_scores(true_labels, pred)
        rows.append({
            "method": name,
            "n_stars": int(true_labels.size),
            "n_clusters": int(np.unique(true_labels).size),
            **scores,
        })
    return pd.DataFrame(rows)


def confusion_matrix_frame(
    true_labels: np.ndarray, pred_labels: np.ndarray,
) -> pd.DataFrame:
    """Row-normalised confusion matrix.

    Rows = true clusters, columns = predicted clusters (renamed ``c0, c1, …,
    noise``). Each cell is the fraction of that true cluster assigned to that
    predicted cluster, so every row sums to 1.
    """
    df = pd.DataFrame({"true": true_labels, "pred": pred_labels})
    cm = pd.crosstab(df["true"], df["pred"], dropna=False)
    cols = [c for c in cm.columns if c != -1] + [c for c in cm.columns if c == -1]
    cm = cm[cols]
    cm = cm.div(cm.sum(axis=1), axis=0).fillna(0.0)
    cm.columns = [f"c{int(c)}" if c != -1 else "noise" for c in cm.columns]
    return cm


def plot_confusion(
    cm: pd.DataFrame, out_path: str | Path | None, title: str = "",
) -> Any:
    """Heatmap of a row-normalised confusion matrix."""
    import matplotlib.pyplot as plt

    n_rows, n_cols = cm.shape
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.55 * n_cols + 3.0), max(5.0, 0.32 * n_rows + 2.0)),
    )
    im = ax.imshow(cm.to_numpy(), cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_cols), cm.columns.astype(str), rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows), cm.index.astype(str), fontsize=8)
    ax.set_xlabel("predicted cluster")
    ax.set_ylabel("true cluster")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="fraction of true cluster")
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
    return fig
