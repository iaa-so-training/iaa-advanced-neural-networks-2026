"""Tests for the data-release provenance / batch-effect diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cluster.provenance import (
    PROVENANCE_AUC_CEILING,
    attach_source,
    format_report,
    load_source_map,
    provenance_auc,
    provenance_crosstab,
    provenance_report,
    separating_dimensions,
    uniform_provenance_mask,
)


def _write_embeddings(
    tmp_path: Path, dr19_ids: list[str], dr17_ids: list[str],
) -> tuple[Path, Path]:
    dr19 = tmp_path / "dr19.parquet"
    dr17 = tmp_path / "dr17.parquet"
    pd.DataFrame({"APOGEE_ID": dr19_ids, "l0": 0.0}).to_parquet(dr19)
    pd.DataFrame({"APOGEE_ID": dr17_ids, "l0": 0.0}).to_parquet(dr17)
    return dr19, dr17


def test_load_source_map_labels_each_release(tmp_path: Path) -> None:
    dr19, dr17 = _write_embeddings(tmp_path, ["a", "b"], ["c"])
    mapping = load_source_map(dr19, dr17)
    assert mapping["a"] == "DR19"
    assert mapping["c"] == "DR17"
    assert len(mapping) == 3


def test_load_source_map_rejects_overlapping_ids(tmp_path: Path) -> None:
    dr19, dr17 = _write_embeddings(tmp_path, ["a", "b"], ["b"])
    with pytest.raises(ValueError, match="both"):
        load_source_map(dr19, dr17)


def test_load_source_map_requires_both_artifacts(tmp_path: Path) -> None:
    dr19, dr17 = _write_embeddings(tmp_path, ["a"], ["b"])
    dr17.unlink()
    with pytest.raises(FileNotFoundError):
        load_source_map(dr19, dr17)


def test_attach_source_marks_unknown_ids(tmp_path: Path) -> None:
    dr19, dr17 = _write_embeddings(tmp_path, ["a"], ["b"])
    df = pd.DataFrame({"APOGEE_ID": ["a", "b", "zzz"], "cluster": ["X", "X", "field"]})
    out = attach_source(df, dr19_path=dr19, dr17_path=dr17)
    assert list(out["source"]) == ["DR19", "DR17", "unknown"]


def test_attach_source_does_not_mutate_input(tmp_path: Path) -> None:
    dr19, dr17 = _write_embeddings(tmp_path, ["a"], ["b"])
    df = pd.DataFrame({"APOGEE_ID": ["a"], "cluster": ["X"]})
    attach_source(df, dr19_path=dr19, dr17_path=dr17)
    assert "source" not in df.columns


def test_provenance_crosstab_reports_member_field_imbalance() -> None:
    df = pd.DataFrame({
        "cluster": ["A"] * 8 + ["field"] * 4,
        "source": ["DR17"] * 8 + ["DR19"] * 4,
    })
    table = provenance_crosstab(df)
    assert table.loc["member", "DR17"] == 8
    assert table.loc["field", "DR19"] == 4
    # the pathological case: members and field come from different releases
    assert table.loc["member", "frac_DR19"] == 0.0
    assert table.loc["field", "frac_DR19"] == 1.0


def test_provenance_crosstab_requires_source_column() -> None:
    with pytest.raises(ValueError, match="attach_source"):
        provenance_crosstab(pd.DataFrame({"cluster": ["A"]}))


def test_provenance_auc_detects_a_planted_batch_effect() -> None:
    rng = np.random.default_rng(0)
    n = 120
    source = np.array(["DR17"] * (n // 2) + ["DR19"] * (n // 2))
    X = rng.normal(size=(n, 4))
    X[source == "DR19", 0] += 8.0  # a hard offset between releases
    mean, _ = provenance_auc(X, source)
    assert mean > 0.95


def test_provenance_auc_is_chance_when_releases_are_identical() -> None:
    rng = np.random.default_rng(1)
    n = 200
    source = np.array(["DR17", "DR19"] * (n // 2))
    X = rng.normal(size=(n, 4))
    mean, _ = provenance_auc(X, source)
    assert 0.3 < mean < 0.7


def test_provenance_auc_handles_single_release() -> None:
    X = np.zeros((10, 3))
    mean, std = provenance_auc(X, np.array(["DR19"] * 10))
    assert np.isnan(mean) and np.isnan(std)


def test_separating_dimensions_counts_only_offset_dims() -> None:
    rng = np.random.default_rng(2)
    n = 400
    source = np.array(["DR17"] * (n // 2) + ["DR19"] * (n // 2))
    X = rng.normal(size=(n, 5))
    X[source == "DR19", :2] += 5.0
    assert separating_dimensions(X, source) == 2


def test_uniform_provenance_mask_selects_one_release() -> None:
    df = pd.DataFrame({"source": ["DR17", "DR19", "DR19"]})
    assert uniform_provenance_mask(df, "DR19").tolist() == [False, True, True]


def test_provenance_report_flags_the_confound() -> None:
    rng = np.random.default_rng(3)
    n = 160
    # members mostly DR17, field entirely DR19 — the real repo's situation
    cluster = np.array(["A"] * 80 + ["field"] * 80, dtype=object)
    source = np.array(["DR17"] * 72 + ["DR19"] * 8 + ["DR19"] * 80)
    X = rng.normal(size=(n, 6))
    X[source == "DR19", 0] += 10.0

    df = pd.DataFrame({"cluster": cluster, "source": source})
    report = provenance_report(df, X)

    assert report["confounded"] is True
    assert float(report["auc_members_only"]) > PROVENANCE_AUC_CEILING  # type: ignore[arg-type]
    assert int(report["n_separating_dims"]) >= 1  # type: ignore[call-overload]
    text = format_report(report)
    assert "CONFOUNDED" in text


def test_provenance_report_passes_a_clean_population() -> None:
    rng = np.random.default_rng(4)
    n = 200
    cluster = np.array(["A"] * 100 + ["field"] * 100, dtype=object)
    source = np.array(["DR17", "DR19"] * 100)
    X = rng.normal(size=(n, 6))

    report = provenance_report(pd.DataFrame({"cluster": cluster, "source": source}), X)
    assert report["confounded"] is False
    assert "no material release signature" in format_report(report)
