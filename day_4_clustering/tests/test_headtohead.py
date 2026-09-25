"""Tests for the common-population head-to-head harness.

The property under test is the one the deck got wrong: every arm must be
scored on the same stars and the same clusters, even when one arm's
embedding covers far fewer stars than another's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cluster.config import Settings
from cluster.data import PreparedData
from cluster.headtohead import (
    Arm,
    common_population,
    head_to_head,
    pivot_scores,
    pivot_with_errors,
)

N_PER_CLUSTER = 8
CLUSTERS = ("A", "B", "C")


@pytest.fixture
def settings() -> Settings:
    """Perplexity/neighbourhoods sized for the tiny synthetic fixtures."""
    return Settings(
        tsne={"perplexity": 2, "method": "barnes_hut"},
        umap={"n_neighbors": 4, "min_dist": 0.0},
    )


@pytest.fixture
def prepared(settings: Settings) -> PreparedData:
    """Members of three clusters plus field, with abundance columns."""
    rng = np.random.default_rng(0)
    rows: list[str] = []
    ids: list[str] = []
    for k, name in enumerate(CLUSTERS):
        for i in range(N_PER_CLUSTER):
            rows.append(name)
            ids.append(f"{name}{i:03d}")
    for i in range(10):
        rows.append("field")
        ids.append(f"F{i:03d}")

    df = pd.DataFrame({"APOGEE_ID": ids, "cluster": rows})
    centres = {name: (k + 1) * 5.0 for k, name in enumerate(CLUSTERS)}
    for j, element in enumerate(settings.elements):
        centre = np.array(
            [centres.get(c, 0.0) for c in df["cluster"]], dtype=float,
        )
        df[element] = centre + rng.normal(0.0, 0.1, len(df)) + j * 0.01

    X = df[list(settings.elements)].to_numpy(dtype=float)
    return PreparedData(df=df, X=X, elements=list(settings.elements))


def _write_embedding(
    path: Path, ids: list[str], n_dims: int = 4, seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"APOGEE_ID": list(ids)})
    offsets = {c: (i + 1) * 5.0 for i, c in enumerate(CLUSTERS)}
    for d in range(n_dims):
        frame[f"l{d}"] = [
            offsets.get(str(i)[0], 0.0) + rng.normal(0.0, 0.1) for i in ids
        ]
    frame.to_parquet(path)
    return path


def test_common_population_intersects_arms(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    """An arm covering only cluster A must shrink the shared population."""
    wide = _write_embedding(
        tmp_path / "wide.parquet",
        [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")],
    )
    narrow = _write_embedding(
        tmp_path / "narrow.parquet",
        [i for i in prepared.df["APOGEE_ID"] if i.startswith("A")],
    )

    shared, keep, dropped = common_population(
        prepared,
        [Arm("wide", wide), Arm("narrow", narrow)],
        settings,
    )
    assert keep == ["A"]
    assert len(shared) == N_PER_CLUSTER
    assert dropped["wide"] == 2 * N_PER_CLUSTER
    assert dropped["narrow"] == 0


def test_common_population_includes_the_abundance_arm(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    narrow = _write_embedding(
        tmp_path / "narrow.parquet",
        [i for i in prepared.df["APOGEE_ID"] if i.startswith("A")],
    )
    shared, keep, _ = common_population(
        prepared, [Arm("abundances", None), Arm("narrow", narrow)], settings,
    )
    assert keep == ["A"]
    assert len(shared) == N_PER_CLUSTER


def test_min_members_applies_after_intersection(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    """A cluster reduced below the threshold by the intersection is dropped."""
    ids = [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")]
    thin = [i for i in ids if not (i.startswith("B") and int(i[1:]) >= 2)]
    emb = _write_embedding(tmp_path / "thin.parquet", thin)

    _, keep, _ = common_population(
        prepared, [Arm("abundances", None), Arm("thin", emb)],
        settings, min_members=5,
    )
    assert "B" not in keep
    assert {"A", "C"} <= set(keep)


def test_head_to_head_scores_all_arms_on_equal_populations(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    members = [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")]
    a = _write_embedding(tmp_path / "a.parquet", members, seed=1)
    b = _write_embedding(tmp_path / "b.parquet", members[:-4], seed=2)

    result = head_to_head(
        prepared,
        [Arm("abundances", None), Arm("a", a), Arm("b", b)],
        settings,
        seeds=(0, 1),
    )
    # the whole point: identical n_stars / n_clusters on every row
    assert result.scores["n_stars"].nunique() == 1
    assert result.scores["n_clusters"].nunique() == 1
    # arm "b" drops the last 4 stars, which pushes cluster C below
    # min_members=5, so C leaves entirely: 2 clusters x 8 members remain.
    assert result.n_clusters == 2
    assert result.n_stars == 2 * N_PER_CLUSTER


def test_head_to_head_carries_sample_size_in_the_table(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    members = [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")]
    emb = _write_embedding(tmp_path / "e.parquet", members)
    result = head_to_head(
        prepared, [Arm("abundances", None), Arm("e", emb)], settings, seeds=(0,),
    )
    assert {"n_stars", "n_clusters", "dim"} <= set(result.scores.columns)


def test_head_to_head_reports_stability_and_degeneracy(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    members = [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")]
    emb = _write_embedding(tmp_path / "e.parquet", members)
    result = head_to_head(
        prepared, [Arm("abundances", None), Arm("e", emb)], settings, seeds=(0, 1),
    )
    assert {"mean", "std"} <= set(result.stability.columns)
    assert "degenerate" in result.degeneracy.columns
    assert set(result.stability["features"]) == {"abundances", "e"}


def test_head_to_head_rejects_disjoint_arms(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    a = _write_embedding(tmp_path / "a.parquet", ["A000", "A001"])
    b = _write_embedding(tmp_path / "b.parquet", ["B000", "B001"])
    with pytest.raises(ValueError, match="share no cluster members"):
        head_to_head(prepared, [Arm("a", a), Arm("b", b)], settings, seeds=(0,))


def test_pivot_helpers_shape_the_table(
    tmp_path: Path, prepared: PreparedData, settings: Settings,
) -> None:
    members = [i for i in prepared.df["APOGEE_ID"] if not i.startswith("F")]
    emb = _write_embedding(tmp_path / "e.parquet", members)
    result = head_to_head(
        prepared, [Arm("abundances", None), Arm("e", emb)], settings, seeds=(0, 1),
    )
    assert len(pivot_scores(result)) == 2
    errors = pivot_with_errors(result)
    assert len(errors) == 2
    assert "±" in errors.to_string()
