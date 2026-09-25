"""Tests for the paper-baseline module (cluster-only multiclass separation)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster import baseline
from cluster.config import Settings
from cluster.data import PreparedData


def test_cluster_only_drops_field(prepared_data: PreparedData) -> None:
    sub = baseline.cluster_only(prepared_data.df)
    assert set(sub["cluster"]) == {"A", "B"}
    assert len(sub) == 12


def test_baseline_matrix_abundance_only(allstar_frame: pd.DataFrame) -> None:
    X = baseline.baseline_matrix(allstar_frame, Settings(), use_kinematics=False)
    assert X.shape == (len(allstar_frame), 16)
    # standardised but NOT L2-normalised (paper's standard scaler)
    norms = np.linalg.norm(X, axis=1)
    assert not np.allclose(norms, 1.0)


def test_baseline_matrix_with_kinematics(allstar_frame: pd.DataFrame) -> None:
    X = baseline.baseline_matrix(allstar_frame, Settings(), use_kinematics=True)
    assert X.shape == (len(allstar_frame), 16 + 4)


def test_accuracy_permutation_invariant() -> None:
    true = np.array(["A", "A", "B", "B"])
    pred = np.array([1, 1, 0, 0])  # labels permuted
    assert baseline._accuracy(true, pred) == 1.0


def test_separation_scores_perfect_and_blob() -> None:
    true = np.array(["A", "A", "B", "B"])
    perfect = baseline.separation_scores(true, np.array([0, 0, 1, 1]))
    assert perfect["homogeneity"] == 1.0
    assert perfect["completeness"] == 1.0
    assert perfect["v_measure"] == 1.0
    assert perfect["accuracy"] == 1.0

    blob = baseline.separation_scores(true, np.array([0, 0, 0, 0]))
    assert blob["homogeneity"] < 1.0
    assert blob["accuracy"] == 0.5


def _fake_matrix(
    df: pd.DataFrame, settings: Any, use_kinematics: bool,
    elements: list[str] | None = None,
) -> np.ndarray:
    return np.zeros((len(df), 2))


def _fake_tsne(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.zeros((len(X), 2))


def _fake_umap(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.zeros((len(X), 2))


def _fake_evoc(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.array([0] * 6 + [1] * 6)


def _fake_cluster_embedding(Z: np.ndarray, params: Any) -> np.ndarray:
    return np.array([0] * 6 + [1] * 6)


def test_run_baseline_returns_three_methods(
    monkeypatch: Any, prepared_data: PreparedData,
) -> None:
    monkeypatch.setattr(baseline, "baseline_matrix", _fake_matrix)
    monkeypatch.setattr(baseline, "fit_tsne", _fake_tsne)
    monkeypatch.setattr(baseline, "fit_umap", _fake_umap)
    monkeypatch.setattr(baseline, "fit_evoc", _fake_evoc)
    monkeypatch.setattr(baseline, "cluster_embedding", _fake_cluster_embedding)

    table = baseline.run_baseline(prepared_data, Settings(), min_members=5)
    assert list(table["method"]) == ["t-SNE", "UMAP", "EVoC"]
    assert (table["n_stars"] == 12).all()
    assert (table["n_clusters"] == 2).all()
    assert (table["homogeneity"] == 1.0).all()
    assert (table["accuracy"] == 1.0).all()


def test_run_baseline_min_members_filters_all(
    monkeypatch: Any, prepared_data: PreparedData,
) -> None:
    monkeypatch.setattr(baseline, "baseline_matrix", _fake_matrix)
    table = baseline.run_baseline(prepared_data, Settings(), min_members=7)
    assert table.empty
    assert list(table.columns) == [
        "method", "n_stars", "n_clusters",
        "homogeneity", "completeness", "v_measure", "accuracy",
    ]


def test_run_baseline_skips_evoc_on_import_error(
    monkeypatch: Any, prepared_data: PreparedData,
) -> None:
    monkeypatch.setattr(baseline, "baseline_matrix", _fake_matrix)
    monkeypatch.setattr(baseline, "fit_tsne", _fake_tsne)
    monkeypatch.setattr(baseline, "fit_umap", _fake_umap)
    monkeypatch.setattr(baseline, "cluster_embedding", _fake_cluster_embedding)

    def _raise_evoc(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
        raise ImportError("no evoc")

    monkeypatch.setattr(baseline, "fit_evoc", _raise_evoc)
    table = baseline.run_baseline(prepared_data, Settings(), min_members=5)
    assert list(table["method"]) == ["t-SNE", "UMAP"]


def test_confusion_matrix_frame_row_normalised() -> None:
    true = np.array(["A", "A", "B", "B", "B"])
    pred = np.array([0, 0, 1, -1, -1])
    cm = baseline.confusion_matrix_frame(true, pred)
    assert list(cm.columns) == ["c0", "c1", "noise"]
    assert np.allclose(cm.sum(axis=1), 1.0)
    # row A: both stars in c0; row B: one in c1, two in noise
    assert cm.loc["A", "c0"] == 1.0
    assert cm.loc["B", "c1"] == pytest.approx(1.0 / 3.0)
    assert cm.loc["B", "noise"] == pytest.approx(2.0 / 3.0)


def test_plot_confusion_saves_file(tmp_path: Any) -> None:
    cm = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["A", "B"], columns=["c0", "c1"],
    )
    out = tmp_path / "cm.png"
    fig = baseline.plot_confusion(cm, out, title="test")
    assert out.exists()
    assert fig is not None


def test_baseline_labels_returns_true_and_predictions(
    monkeypatch: Any, prepared_data: PreparedData,
) -> None:
    monkeypatch.setattr(baseline, "baseline_matrix", _fake_matrix)
    monkeypatch.setattr(baseline, "fit_tsne", _fake_tsne)
    monkeypatch.setattr(baseline, "fit_umap", _fake_umap)
    monkeypatch.setattr(baseline, "fit_evoc", _fake_evoc)
    monkeypatch.setattr(baseline, "cluster_embedding", _fake_cluster_embedding)

    true, labels = baseline.baseline_labels(prepared_data, Settings(), min_members=5)
    assert list(labels) == ["t-SNE", "UMAP", "EVoC"]
    assert true.size == 12
    assert (labels["t-SNE"] == np.array([0] * 6 + [1] * 6)).all()
