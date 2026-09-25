"""Tests for :mod:`cluster.clusters`."""

from __future__ import annotations

import pydantic
import pytest

from cluster.clusters import CLUSTER_BY_NAME, CLUSTERS, Cluster


def test_catalogue_length() -> None:
    assert len(CLUSTERS) == 25
    assert len(CLUSTER_BY_NAME) == 25


def test_parallax_mas_inverse_distance() -> None:
    pleiades = CLUSTER_BY_NAME["Pleiades"]
    assert pleiades.parallax_mas == pytest.approx(1000.0 / 136.0)
    assert pleiades.parallax_mas == pytest.approx(7.352941176, rel=1e-8)


def test_cluster_is_frozen() -> None:
    cluster = CLUSTER_BY_NAME["M 67"]
    with pytest.raises(pydantic.ValidationError):
        cluster.name = "changed"  # type: ignore[misc]


def test_kind_values_are_supported() -> None:
    assert {c.kind for c in CLUSTERS} == {"open", "globular"}
    assert CLUSTER_BY_NAME["M 5"].kind == "globular"
    assert CLUSTER_BY_NAME["NGC 7789"].kind == "open"
