"""Tests for benchmark scoring and the benchmark driver."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster import benchmark, tracking
from cluster.benchmark import (
    BenchmarkResult,
    MethodResult,
    _score_one,
    cluster_embedding,
    fit_evoc,
    fit_tsne,
    fit_umap,
    knn_purity,
    run_benchmark,
)
from cluster.config import Settings
from cluster.data import PreparedData


def test_score_one_perfect_match() -> None:
    true = np.array(["A", "A", "B", "B", "field", "field"], dtype=object)
    pred = np.array([0, 0, 1, 1, -1, -1], dtype=int)
    scores = _score_one(true, pred)
    assert list(scores["cluster"]) == ["A", "B"]
    assert scores.loc[0, "precision"] == 1.0
    assert scores.loc[0, "recall"] == 1.0
    assert scores.loc[1, "precision"] == 1.0
    assert scores.loc[1, "recall"] == 1.0


def test_score_one_merged_predicted_cluster() -> None:
    true = np.array(["A", "A", "B", "B", "field"], dtype=object)
    pred = np.array([0, 0, 0, 0, -1], dtype=int)
    scores = _score_one(true, pred)
    assert scores["n_pred"].tolist() == [4, 4]
    assert scores["precision"].tolist() == [0.5, 0.5]
    assert scores["recall"].tolist() == [1.0, 1.0]


def test_score_one_no_predicted_cluster_is_zero() -> None:
    true = np.array(["A", "A", "B", "B"], dtype=object)
    pred = np.array([-1, -1, -1, -1], dtype=int)
    scores = _score_one(true, pred)
    assert scores["precision"].tolist() == [0.0, 0.0]
    assert scores["recall"].tolist() == [0.0, 0.0]


def test_knn_purity_perfectly_separated_clusters(prepared_data: PreparedData) -> None:
    labels = prepared_data.df["cluster"].to_numpy()
    purity = knn_purity(prepared_data.X, labels, k=3, min_members=2)
    assert set(purity) == {"A", "B"}
    assert purity["A"] == pytest.approx(1.0)
    assert purity["B"] == pytest.approx(1.0)


def test_knn_purity_respects_min_members(prepared_data: PreparedData) -> None:
    labels = prepared_data.df["cluster"].to_numpy()
    purity = knn_purity(prepared_data.X, labels, k=3, min_members=99)
    assert purity == {}


def test_knn_purity_mixed_labels_below_one() -> None:
    n = 20
    labels = np.array(["A"] * 10 + ["B"] * 10, dtype=object)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)])
    purity = knn_purity(X, labels, k=5, min_members=2)
    assert 0.0 < purity["A"] < 1.0
    assert 0.0 < purity["B"] < 1.0


def _scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": ["A", "B"],
            "n_true": [4, 4],
            "n_pred": [4, 4],
            "overlap": [4, 4],
            "precision": [1.0, 0.5],
            "recall": [1.0, 1.0],
        }
    )


def _benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
        results={
            "m1": MethodResult(
                name="m1",
                embedding=np.zeros((8, 2)),
                labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                scores=_scores_frame(),
            ),
            "m2": MethodResult(
                name="m2",
                embedding=None,
                labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                scores=_scores_frame(),
            ),
        },
        df=pd.DataFrame({"cluster": ["A"] * 4 + ["B"] * 4}),
        X=np.zeros((8, 2)),
    )


def test_benchmark_result_summary_wide_format() -> None:
    summary = _benchmark_result().summary()
    assert summary.shape == (2, 8)
    assert summary.loc["A", ("recall", "m1")] == 1.0
    assert summary.loc["B", ("precision", "m2")] == 0.5


def test_benchmark_result_macro_averages() -> None:
    macro = _benchmark_result().macro()
    assert macro.shape == (2, 5)
    assert macro.loc[0, "method"] == "m1"
    assert macro.loc[0, "recall"] == 1.0
    assert macro.loc[0, "precision"] == 0.75
    assert macro.loc[0, "n_clusters_scored"] == 2


def test_benchmark_result_empty_summary_and_macro() -> None:
    result = BenchmarkResult()
    assert result.summary().empty
    assert result.macro().empty


class _FakeTSNE:
    def __init__(self, n_components: int, random_state: int, **params: Any) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.params = params

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, : self.n_components]


def test_fit_tsne_uses_configured_model(monkeypatch: Any) -> None:
    monkeypatch.setattr(benchmark, "TSNE", _FakeTSNE)
    X = np.arange(12, dtype=float).reshape(4, 3)
    Z = fit_tsne(X, {"perplexity": 2, "max_iter": 10, "backend": "sklearn"}, random_state=7)
    assert Z.shape == (4, 2)
    assert np.array_equal(Z, X[:, :2])


class _FakeOpenTSNE:
    def __init__(self, n_components: int, random_state: int, **params: Any) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.params = params

    def fit(self, X: np.ndarray) -> np.ndarray:
        self.X_ = np.asarray(X)
        return self.X_[:, : self.n_components]


def test_fit_tsne_opentsne_backend_translates_keys(monkeypatch: Any) -> None:
    fake_module = types.SimpleNamespace(TSNE=_FakeOpenTSNE)
    monkeypatch.setitem(sys.modules, "openTSNE", fake_module)
    X = np.arange(12, dtype=float).reshape(4, 3)
    Z = fit_tsne(
        X,
        {"perplexity": 2, "max_iter": 10, "init": "pca", "backend": "opentsne"},
        random_state=7,
    )
    assert Z.shape == (4, 2)
    assert np.array_equal(Z, X[:, :2])


def test_fit_tsne_auto_falls_back_to_sklearn_when_opentsne_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(benchmark, "TSNE", _FakeTSNE)
    # None in sys.modules makes `from openTSNE import TSNE` raise ImportError
    monkeypatch.setitem(sys.modules, "openTSNE", None)
    X = np.arange(12, dtype=float).reshape(4, 3)
    Z = fit_tsne(X, {"perplexity": 2, "max_iter": 10, "backend": "auto"}, random_state=7)
    assert Z.shape == (4, 2)
    assert np.array_equal(Z, X[:, :2])


class _FakeUMAP:
    def __init__(self, n_components: int, random_state: int, **params: Any) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.params = params

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, : self.n_components] + 1.0


def test_fit_umap_uses_configured_model(monkeypatch: Any) -> None:
    fake_module = types.SimpleNamespace(UMAP=_FakeUMAP)
    monkeypatch.setitem(sys.modules, "umap", fake_module)
    X = np.arange(12, dtype=float).reshape(4, 3)
    Z = fit_umap(X, {"n_neighbors": 2}, random_state=7)
    assert Z.shape == (4, 2)
    assert np.array_equal(Z, X[:, :2] + 1.0)


class _FakeEVoC:
    def __init__(self, random_state: int, **params: Any) -> None:
        self.random_state = random_state
        self.params = params

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)


def test_fit_evoc_uses_configured_model(monkeypatch: Any) -> None:
    fake_module = types.SimpleNamespace(EVoC=_FakeEVoC)
    monkeypatch.setitem(sys.modules, "evoc", fake_module)
    X = np.arange(12, dtype=float).reshape(4, 3)
    labels = fit_evoc(X, {"noise_level": 0.5}, random_state=7)
    assert np.array_equal(labels, np.zeros(4, dtype=int))


class _FakeHDBSCAN:
    def __init__(self, **params: Any) -> None:
        self.params = params

    def fit_predict(self, Z: np.ndarray) -> np.ndarray:
        return np.arange(len(Z), dtype=int) % 2


def test_cluster_embedding_uses_configured_model(monkeypatch: Any) -> None:
    fake_module = types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN)
    monkeypatch.setitem(sys.modules, "hdbscan", fake_module)
    Z = np.zeros((6, 2))
    labels = cluster_embedding(Z, {"min_cluster_size": 2})
    assert np.array_equal(labels, np.arange(6) % 2)


def _benchmark_prepared() -> PreparedData:
    df = pd.DataFrame({"cluster": ["A"] * 4 + ["B"] * 4})
    X = np.column_stack([np.arange(8, dtype=float), np.zeros(8)])
    return PreparedData(df=df, X=X, elements=["x", "y"])


def test_run_benchmark_returns_three_results_and_logs(
    monkeypatch: Any,
) -> None:
    calls: dict[str, int] = {"tsne": 0, "umap": 0, "evoc": 0, "hdbscan": 0}
    logged: list[dict[str, float]] = []

    def fake_tsne(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        calls["tsne"] += 1
        return np.asarray(X)[:, :2]

    def fake_umap(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        calls["umap"] += 1
        return np.asarray(X)[:, :2] + 0.5

    def fake_evoc(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        calls["evoc"] += 1
        return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    def fake_hdbscan(Z: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        calls["hdbscan"] += 1
        return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    def fake_purity(embedding: np.ndarray, true_labels: np.ndarray, k: int = 10, min_members: int = 5) -> dict[str, float]:
        return {"A": 1.0, "B": 1.0}

    def fake_log_metrics(metrics: dict[str, float]) -> None:
        logged.append(metrics)

    monkeypatch.setattr(benchmark, "fit_tsne", fake_tsne)
    monkeypatch.setattr(benchmark, "fit_umap", fake_umap)
    monkeypatch.setattr(benchmark, "fit_evoc", fake_evoc)
    monkeypatch.setattr(benchmark, "cluster_embedding", fake_hdbscan)
    monkeypatch.setattr(benchmark, "knn_purity", fake_purity)
    monkeypatch.setattr(tracking, "log_metrics", fake_log_metrics)

    settings = Settings(
        tsne={"perplexity": 2, "method": "barnes_hut"},
        umap={"n_neighbors": 2},
        evoc={"noise_level": 0.1},
        hdbscan={"min_cluster_size": 2},
    )
    result = run_benchmark(_benchmark_prepared(), settings)

    assert set(result.results) == {"t-SNE", "UMAP", "EVoC"}
    assert calls == {"tsne": 1, "umap": 1, "evoc": 1, "hdbscan": 2}
    assert result.results["EVoC"].embedding is None
    assert any("t-SNE_macro_recall" in metrics for metrics in logged)
    assert any("t-SNE_knn_purity" in metrics for metrics in logged)


def test_run_benchmark_skips_evoc_on_import_error(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    def fake_tsne(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        return np.asarray(X)[:, :2]

    def fake_umap(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        return np.asarray(X)[:, :2]

    def fake_evoc(X: np.ndarray, params: dict[str, Any], random_state: int) -> np.ndarray:
        raise ImportError("no evoc")

    def fake_hdbscan(Z: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        return np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    def fake_purity(embedding: np.ndarray, true_labels: np.ndarray, k: int = 10, min_members: int = 5) -> dict[str, float]:
        return {"A": 1.0, "B": 1.0}

    monkeypatch.setattr(benchmark, "fit_tsne", fake_tsne)
    monkeypatch.setattr(benchmark, "fit_umap", fake_umap)
    monkeypatch.setattr(benchmark, "fit_evoc", fake_evoc)
    monkeypatch.setattr(benchmark, "cluster_embedding", fake_hdbscan)
    monkeypatch.setattr(benchmark, "knn_purity", fake_purity)

    result = run_benchmark(
        _benchmark_prepared(),
        Settings(tsne={}, umap={}, evoc={}, hdbscan={}),
    )
    assert set(result.results) == {"t-SNE", "UMAP"}
    assert "EVoC unavailable" in capsys.readouterr().out
