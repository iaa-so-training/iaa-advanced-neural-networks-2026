"""Tests for seed-stability reporting and degeneracy detection."""

from __future__ import annotations

import numpy as np
import pytest

from cluster.config import Settings
from cluster.stability import (
    DEGENERACY_FRACTION,
    degeneracy,
    degeneracy_table,
    format_stability,
    stability,
)


@pytest.fixture
def small_settings() -> Settings:
    """Perplexity/neighbourhoods sized for the tiny synthetic fixtures."""
    return Settings(
        tsne={"perplexity": 2, "method": "barnes_hut"},
        umap={"n_neighbors": 4, "min_dist": 0.0},
    )


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated blobs — every clusterer should find them."""
    rng = np.random.default_rng(0)
    # 8 dims: EVoC runs an internal PCA that needs n_features > n_components.
    X = np.vstack([
        rng.normal(0.0, 0.1, (20, 8)),
        rng.normal(10.0, 0.1, (20, 8)),
        rng.normal(-10.0, 0.1, (20, 8)),
    ])
    labels = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20, dtype=object)
    return X, labels


def test_stability_reports_one_row_per_method(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = stability(X, labels, small_settings, seeds=(0, 1))
    assert set(frame["method"]) <= {"t-SNE", "UMAP", "EVoC"}
    assert (frame["n_seeds"] == 2).all()
    assert {"mean", "std", "min", "max", "values"} <= set(frame.columns)


def test_stability_records_every_seed(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = stability(X, labels, small_settings, seeds=(0, 1, 2))
    for values in frame["values"]:
        assert len(values) == 3


def test_stability_bounds_bracket_the_mean(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = stability(X, labels, small_settings, seeds=(0, 1, 2))
    # tolerance: the mean of several identical floats can land one ULP
    # outside them, so an exact <= is not a safe invariant here.
    eps = 1e-9
    assert (frame["min"] <= frame["mean"] + eps).all()
    assert (frame["mean"] <= frame["max"] + eps).all()
    assert (frame["std"] >= 0).all()


def test_stability_scores_well_separated_blobs_highly(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = stability(X, labels, small_settings, seeds=(0,))
    assert frame["mean"].max() > 0.8


def test_stability_supports_other_metrics(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = stability(X, labels, small_settings, seeds=(0,), metric="completeness")
    assert (frame["metric"] == "completeness").all()


def test_degeneracy_flags_a_single_blob() -> None:
    report = degeneracy(np.zeros(50, dtype=int))
    assert report["degenerate"] is True
    assert report["n_clusters"] == 1
    assert report["largest_fraction"] == 1.0


def test_degeneracy_flags_a_dominant_cluster() -> None:
    labels = np.array([0] * 40 + [1] * 10)
    report = degeneracy(labels)
    assert report["largest_fraction"] == 0.8
    assert report["degenerate"] is True


def test_degeneracy_accepts_a_balanced_partition() -> None:
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    report = degeneracy(labels)
    assert report["n_clusters"] == 3
    assert report["degenerate"] is False
    assert float(report["largest_fraction"]) < DEGENERACY_FRACTION  # type: ignore[arg-type]


def test_degeneracy_counts_noise_separately() -> None:
    labels = np.array([0] * 10 + [1] * 10 + [-1] * 5)
    report = degeneracy(labels)
    assert report["n_noise"] == 5
    assert report["n_clusters"] == 2


def test_degeneracy_handles_empty_input() -> None:
    assert degeneracy(np.array([], dtype=int))["degenerate"] is True


def test_degeneracy_table_scores_and_flags(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    frame = degeneracy_table(X, small_settings, labels)
    assert {"method", "n_clusters", "degenerate", "homogeneity"} <= set(frame.columns)
    assert len(frame) >= 2


def test_format_stability_renders_plus_minus(
    separable: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, labels = separable
    text = format_stability(stability(X, labels, small_settings, seeds=(0, 1)))
    assert "±" in text
    assert "n_seeds=2" in text
