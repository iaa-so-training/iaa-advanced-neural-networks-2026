"""Benchmark: t-SNE vs UMAP vs EVoC for chemical tagging.

- t-SNE and UMAP reduce the 16-D abundance space to 2-D; HDBSCAN clusters
  the 2-D map.
- EVoC clusters the high-D abundance vectors directly (it fuses a UMAP-style
  graph embedding with HDBSCAN/PLSCAN-style density clustering internally).

Each method's cluster labels are scored against the kinematic ground-truth
membership (precision / recall per cluster).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .config import Settings
from .data import PreparedData


def fit_tsne(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
    """t-SNE 2-D embedding.

    ``params["backend"]`` selects the implementation:

    - ``"opentsne"`` — openTSNE (same BH algorithm, multi-core ``n_jobs=-1``)
    - ``"sklearn"`` — scikit-learn (single-threaded BH)
    - ``"auto"`` — openTSNE if importable, else scikit-learn

    ``params["method"]`` (a sklearn legacy key) is ignored.
    """
    params = dict(params)
    backend = str(params.pop("backend", "auto"))
    params.pop("method", None)

    if backend in ("auto", "opentsne"):
        try:
            from openTSNE import TSNE as OpenTSNE
        except ImportError:
            if backend == "opentsne":
                raise
        else:
            # translate sklearn-style keys to openTSNE's
            if "max_iter" in params:
                params["n_iter"] = params.pop("max_iter")
            if "init" in params:
                params["initialization"] = params.pop("init")
            params.setdefault("n_jobs", -1)
            model = OpenTSNE(n_components=2, random_state=random_state, **params)
            return np.asarray(model.fit(X))

    model = TSNE(n_components=2, random_state=random_state, **params)
    return model.fit_transform(X)


def fit_umap(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
    import umap

    model = umap.UMAP(n_components=2, random_state=random_state, **params)
    return model.fit_transform(X)


def fit_evoc(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
    from evoc import EVoC

    model = EVoC(random_state=random_state, **params)
    return model.fit_predict(X)


def cluster_embedding(Z: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    import hdbscan

    model = hdbscan.HDBSCAN(**params)
    return model.fit_predict(Z)


def knn_purity(embedding: np.ndarray, true_labels: np.ndarray, k: int = 10, min_members: int = 5) -> dict[str, float]:
    """Parameter-free chemical-cohesion score.

    For each true cluster, the mean fraction of a member's ``k`` nearest
    neighbours (in the embedding) that belong to the same cluster. Mirrors
    the target paper's visual test — "do members sit together?" — without
    any clustering hyperparameters. ``k`` is capped at the cluster size.
    """
    from sklearn.neighbors import NearestNeighbors

    clusters = [c for c in np.unique(true_labels) if c != "field"]
    n = len(embedding)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(embedding)
    neighbors = nn.kneighbors(embedding, return_distance=False)
    idx = np.asarray(neighbors)[:, 1:]  # drop self

    out: dict[str, float] = {}
    for c in clusters:
        m = true_labels == c
        n_members = int(m.sum())
        if n_members < min_members:
            continue
        k_c = min(k, n_members - 1)
        neigh = true_labels[idx[m][:, :k_c]]  # (n_members, k_c)
        out[c] = float((neigh == c).mean())
    return out


def _score_one(true_labels: np.ndarray, pred_labels: np.ndarray) -> pd.DataFrame:
    """Greedy match each true cluster to its best-overlapping predicted cluster."""
    rows = []
    true_clusters = [c for c in np.unique(true_labels) if c != "field"]
    for tc in true_clusters:
        tmask = true_labels == tc
        n_true = int(tmask.sum())
        best_overlap, best = 0, None
        for pc in np.unique(pred_labels):
            if pc == -1:
                continue
            overlap = int(((pred_labels == pc) & tmask).sum())
            if overlap > best_overlap:
                best_overlap, best = overlap, pc
        if best is None or n_true == 0:
            precision = recall = 0.0
            n_pred = 0
        else:
            n_pred = int((pred_labels == best).sum())
            precision = best_overlap / n_pred if n_pred else 0.0
            recall = best_overlap / n_true
        rows.append({
            "cluster": tc, "n_true": n_true, "n_pred": n_pred,
            "overlap": best_overlap, "precision": precision, "recall": recall,
        })
    return pd.DataFrame(rows)


@dataclass
class MethodResult:
    name: str
    embedding: np.ndarray | None      # 2-D coords (None for EVoC, which clusters internally)
    labels: np.ndarray
    scores: pd.DataFrame


@dataclass
class BenchmarkResult:
    results: dict[str, MethodResult] = field(default_factory=dict)
    df: pd.DataFrame | None = None
    X: np.ndarray | None = None

    def summary(self) -> pd.DataFrame:
        """Per-cluster recall/precision for every method, wide format."""
        frames = []
        for name, r in self.results.items():
            s = r.scores.copy()
            s["method"] = name
            frames.append(s)
        if not frames:
            return pd.DataFrame()
        long = pd.concat(frames, ignore_index=True)
        return long.pivot_table(
            index="cluster", columns="method",
            values=["recall", "precision", "n_true", "n_pred"],
        )

    def macro(self) -> pd.DataFrame:
        """Macro-averaged recall/precision per method (clusters with >=1 member)."""
        rows = []
        for name, r in self.results.items():
            s = r.scores[r.scores["n_true"] >= 1]
            if len(s) == 0:
                continue
            rows.append({
                "method": name,
                "recall": s["recall"].mean(),
                "precision": s["precision"].mean(),
                "n_clusters_scored": len(s),
                "n_true_members": s["n_true"].sum(),
            })
        return pd.DataFrame(rows)


def _log_benchmark_metrics(result: BenchmarkResult, true_labels: np.ndarray) -> None:
    """Best-effort MLflow logging of macro scores; never raises."""
    try:
        from . import tracking

        macro = result.macro()
        for method, recall, precision in zip(
            macro["method"], macro["recall"], macro["precision"]
        ):
            tracking.log_metrics({
                f"{method}_macro_recall": float(recall),
                f"{method}_macro_precision": float(precision),
            })
        for name in ("t-SNE", "UMAP"):
            r = result.results.get(name)
            if r is not None and r.embedding is not None:
                purity = knn_purity(r.embedding, true_labels)
                if purity:
                    tracking.log_metrics({
                        f"{name}_knn_purity": sum(purity.values()) / len(purity),
                    })
    except Exception:
        pass


def run_benchmark(prepared: PreparedData, settings: Settings) -> BenchmarkResult:
    """Run all three methods against the prepared abundance matrix.

    Scores against the ``referee`` column (the external Simbad catalogue)
    when present, falling back to the in-pipeline ``cluster`` labels.
    """
    X = prepared.X
    df = prepared.df
    true_labels = (
        df["referee"].to_numpy() if "referee" in df.columns
        else df["cluster"].to_numpy()
    )
    result = BenchmarkResult(df=df, X=X)

    def _add(name: str, embedding: np.ndarray | None, labels: np.ndarray) -> None:
        result.results[name] = MethodResult(
            name=name, embedding=embedding, labels=labels,
            scores=_score_one(true_labels, labels),
        )

    # t-SNE -> HDBSCAN
    tsne_params = {k: v for k, v in settings.tsne.items() if k != "method"}
    z_tsne = fit_tsne(X, tsne_params, settings.random_state)
    _add("t-SNE", z_tsne, cluster_embedding(z_tsne, settings.hdbscan))

    # UMAP -> HDBSCAN
    z_umap = fit_umap(X, settings.umap, settings.random_state)
    _add("UMAP", z_umap, cluster_embedding(z_umap, settings.hdbscan))

    # EVoC (clusters the high-D vectors directly)
    try:
        _add("EVoC", None, fit_evoc(X, settings.evoc, settings.random_state))
    except ImportError as exc:
        print(f"⚠ EVoC unavailable ({exc}); skipping.")

    _log_benchmark_metrics(result, true_labels)

    return result
