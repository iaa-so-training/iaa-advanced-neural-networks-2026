"""Tests for the spectral-embedding feature source."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster import spectral
from cluster.benchmark import run_benchmark
from cluster.config import Settings
from cluster.data import PreparedData


def _embedding_frame(n: int = 18, dim: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({spectral.ID_COLUMN: [f"2M{i:08d}" for i in range(n)]})
    for d in range(dim):
        frame[f"z{d}"] = rng.standard_normal(n)
    return frame


def test_load_embedding_frame_csv(tmp_path: Any) -> None:
    frame = _embedding_frame()
    path = tmp_path / "emb.csv"
    frame.to_csv(path, index=False)
    loaded = spectral.load_embedding_frame(path)
    assert list(loaded.columns) == list(frame.columns)
    assert len(loaded) == 18


def test_embedding_columns() -> None:
    frame = _embedding_frame()
    cols = spectral.embedding_columns(frame)
    assert cols == ["z0", "z1", "z2", "z3"]


def test_align_embeddings_inner_join() -> None:
    df = pd.DataFrame({
        spectral.ID_COLUMN: ["2M00000000", "2M00000001", "2M00000002"],
        "cluster": ["A", "B", "field"],
    })
    frame = _embedding_frame(18)  # ids 2M00000000..2M00000017
    merged = spectral.align_embeddings(df, frame)
    assert len(merged) == 3  # all three present in both
    assert "z0" in merged.columns


def test_align_embeddings_missing_id_column() -> None:
    with pytest.raises(ValueError, match="no column"):
        spectral.align_embeddings(pd.DataFrame({"x": [1]}), _embedding_frame())


def test_embedding_matrix_standardised_and_normalised() -> None:
    frame = _embedding_frame()
    cols = spectral.embedding_columns(frame)
    X = spectral.embedding_matrix(frame, cols, Settings())
    assert X.shape == (18, 4)
    # L2-normalised rows -> unit norm
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0)


def _fake_tsne(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.zeros((len(X), 2))


def _fake_umap(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.zeros((len(X), 2))


def _fake_evoc(X: np.ndarray, params: Any, random_state: int) -> np.ndarray:
    return np.array([0] * 6 + [1] * 6 + [-1] * 6)


def _fake_cluster(Z: np.ndarray, params: Any) -> np.ndarray:
    return np.array([0] * 6 + [1] * 6 + [-1] * 6)


def test_spectral_prepared_swaps_matrix(tmp_path: Any) -> None:
    n = 12
    df = pd.DataFrame({
        spectral.ID_COLUMN: [f"2M{i:08d}" for i in range(n)],
        "cluster": ["A"] * 6 + ["B"] * 6,
        "is_member": [True] * 12,
    })
    prepared = PreparedData(df=df, X=np.zeros((n, 16)), elements=[f"e{i}" for i in range(16)])
    frame = _embedding_frame(10)  # ids 2M00000000..2M00000009
    path = tmp_path / "emb.parquet"
    frame.to_parquet(path, index=False)

    out = spectral.spectral_prepared(prepared, path, Settings())
    assert out.X.shape == (10, 4)  # 10 aligned stars, 4 latent dims
    assert len(out.df) == 10
    assert np.allclose(np.linalg.norm(out.X, axis=1), 1.0)


def test_baseline_labels_with_spectral_elements(monkeypatch: Any) -> None:
    """Regression: baseline_labels must build from latent columns, not abundances."""
    from cluster.baseline import baseline_labels
    import cluster.baseline as baseline_mod

    n = 12
    rng = np.random.default_rng(3)
    latent = pd.DataFrame({
        f"z{d}": rng.standard_normal(n) for d in range(4)
    })
    df = pd.concat([
        pd.DataFrame({
            "cluster": ["A"] * 6 + ["B"] * 6,
            "is_member": [True] * 12,
        }),
        latent,
    ], axis=1)
    prepared = PreparedData(
        df=df, X=latent.to_numpy(), elements=[f"z{d}" for d in range(4)],
    )

    monkeypatch.setattr(baseline_mod, "fit_tsne", _fake_tsne)
    monkeypatch.setattr(baseline_mod, "fit_umap", _fake_umap)
    monkeypatch.setattr(baseline_mod, "fit_evoc", _fake_evoc)
    monkeypatch.setattr(baseline_mod, "cluster_embedding", _fake_cluster)

    true, labels = baseline_labels(prepared, Settings(), min_members=5)
    assert true.size == 12
    assert set(labels) == {"t-SNE", "UMAP", "EVoC"}


def test_embeddings_flow_into_benchmark(monkeypatch: Any) -> None:
    """Prove a spectral-embedding matrix runs through the same benchmark."""
    import cluster.benchmark as benchmark

    n = 18
    df = pd.DataFrame({
        spectral.ID_COLUMN: [f"2M{i:08d}" for i in range(n)],
        "cluster": ["A"] * 6 + ["B"] * 6 + ["field"] * 6,
        "is_member": [True] * 12 + [False] * 6,
    })
    frame = _embedding_frame(n)
    merged = spectral.align_embeddings(df, frame)
    cols = spectral.embedding_columns(frame)
    X = spectral.embedding_matrix(merged, cols, Settings())
    prepared = PreparedData(df=merged, X=X, elements=cols)

    monkeypatch.setattr(benchmark, "fit_tsne", _fake_tsne)
    monkeypatch.setattr(benchmark, "fit_umap", _fake_umap)
    monkeypatch.setattr(benchmark, "fit_evoc", _fake_evoc)
    monkeypatch.setattr(benchmark, "cluster_embedding", _fake_cluster)

    result = run_benchmark(prepared, Settings())
    assert set(result.results) == {"t-SNE", "UMAP", "EVoC"}
    assert not result.macro().empty
