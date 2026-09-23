"""Tests for the click command-line interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cluster import cli, download as download_module, tracking
from cluster.data import PreparedData


class _FakeRunResult:
    def __init__(self) -> None:
        self.results = {
            "t-SNE": SimpleNamespace(embedding=np.zeros((2, 2))),
            "UMAP": SimpleNamespace(embedding=np.zeros((2, 2))),
        }

    def macro(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"method": "t-SNE", "recall": 1.0, "precision": 1.0}]
        )

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"recall": [1.0]},
            index=pd.Index(["Pleiades"], name="cluster"),
        )


def test_cli_download_delegates(monkeypatch: Any) -> None:
    seen: list[str] = []

    def fake_download(destination: str) -> None:
        seen.append(destination)

    monkeypatch.setattr(download_module, "download_allstar", fake_download)
    result = CliRunner().invoke(cli.main, ["download", "--destination", "x.fits"])
    assert result.exit_code == 0
    assert seen == ["x.fits"]


def test_cli_run_missing_allstar_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.fits"
    result = CliRunner().invoke(cli.main, ["run", "--allstar", str(missing)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_run_end_to_end_with_fake_pipeline(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    allstar = tmp_path / "fake.fits"
    allstar.write_text("fake")
    outdir = tmp_path / "results"

    prepared = PreparedData(
        df=pd.DataFrame({"cluster": ["Pleiades", "field"]}),
        X=np.zeros((2, 2)),
        elements=["x", "y"],
    )
    calls: dict[str, Any] = {
        "prepare": [],
        "benchmark": [],
        "plot": [],
        "params": None,
        "artifact": None,
    }

    def fake_prepare(
        allstar_path: str | Path,
        settings: Any,
        clusters: Any,
        **kwargs: Any,
    ) -> PreparedData:
        calls["prepare"].append((allstar_path, settings, clusters, kwargs))
        return prepared

    def fake_run_benchmark(prepared_data: PreparedData, settings: Any) -> _FakeRunResult:
        calls["benchmark"].append((prepared_data, settings))
        return _FakeRunResult()

    def fake_plot(result: _FakeRunResult, path: str | Path) -> None:
        calls["plot"].append((result, path))

    def fake_knn_purity(embedding: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
        return {"Pleiades": 0.9}

    def fake_start(run_name: str) -> None:
        calls["start"] = run_name

    def fake_log_params(params: dict[str, Any]) -> None:
        calls["params"] = params

    def fake_log_artifact(path: str | Path) -> None:
        calls["artifact"] = path

    def fake_end() -> None:
        calls["end"] = True

    def fake_attach_referee(df: Any, clusters: Any, settings: Any, **kwargs: Any) -> Any:
        out = df.copy()
        out["referee"] = out["cluster"]
        return out

    monkeypatch.setattr("cluster.data.prepare", fake_prepare)
    monkeypatch.setattr("cluster.benchmark.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("cluster.benchmark.knn_purity", fake_knn_purity)
    monkeypatch.setattr("cluster.plots.plot_method_grid", fake_plot)
    monkeypatch.setattr("cluster.catalog.attach_referee", fake_attach_referee)
    monkeypatch.setattr(tracking, "start_run", fake_start)
    monkeypatch.setattr(tracking, "log_params", fake_log_params)
    monkeypatch.setattr(tracking, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(tracking, "end_run", fake_end)

    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "--allstar", str(allstar),
            "--outdir", str(outdir),
            "--cluster", "Pleiades",
            "--fast",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls["prepare"]) == 1
    assert len(calls["benchmark"]) == 1
    assert len(calls["plot"]) == 1
    assert calls["start"] == "cluster-benchmark"
    assert calls["params"] is not None
    assert calls["params"]["FAST"] == "True"
    assert calls["artifact"] == outdir / "benchmark_grid.png"
    assert calls["end"] is True
    assert "macro recall/precision" in result.output
    assert "Pleiades" in result.output


def test_cli_run_respects_max_stars_and_full_toggle(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    allstar = tmp_path / "fake.fits"
    allstar.write_text("fake")

    prepared = PreparedData(
        df=pd.DataFrame({"cluster": ["field", "field"]}),
        X=np.zeros((2, 2)),
        elements=["x", "y"],
    )

    seen_settings: list[Any] = []

    def fake_prepare(
        allstar_path: str | Path,
        settings: Any,
        clusters: Any,
        **kwargs: Any,
    ) -> PreparedData:
        seen_settings.append(settings)
        return prepared

    def fake_run_benchmark(prepared_data: PreparedData, settings: Any) -> _FakeRunResult:
        return _FakeRunResult()

    def fake_plot(result: _FakeRunResult, path: str | Path) -> None:
        return None

    def fake_knn_purity(embedding: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
        return {"Pleiades": 1.0}

    def fake_start(name: str) -> None:
        return None

    def fake_log_params(params: dict[str, Any]) -> None:
        return None

    def fake_log_artifact(path: str | Path) -> None:
        return None

    def fake_end() -> None:
        return None

    def fake_attach_referee(df: Any, clusters: Any, settings: Any, **kwargs: Any) -> Any:
        out = df.copy()
        out["referee"] = out["cluster"]
        return out

    monkeypatch.setattr("cluster.data.prepare", fake_prepare)
    monkeypatch.setattr("cluster.benchmark.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr("cluster.benchmark.knn_purity", fake_knn_purity)
    monkeypatch.setattr("cluster.plots.plot_method_grid", fake_plot)
    monkeypatch.setattr("cluster.catalog.attach_referee", fake_attach_referee)
    monkeypatch.setattr(tracking, "start_run", fake_start)
    monkeypatch.setattr(tracking, "log_params", fake_log_params)
    monkeypatch.setattr(tracking, "log_artifact", fake_log_artifact)
    monkeypatch.setattr(tracking, "end_run", fake_end)

    result = CliRunner().invoke(
        cli.main,
        ["run", "--allstar", str(allstar), "--full", "--max-stars", "123"],
    )
    assert result.exit_code == 0, result.output
    assert seen_settings[0].fast is False
    assert seen_settings[0].max_stars == 123


def test_cli_hr_command(monkeypatch: Any, tmp_path: Path) -> None:
    allstar = tmp_path / "fake.fits"
    allstar.write_text("x")
    outdir = tmp_path / "out"

    fake_df = pd.DataFrame({
        "RA": [132.8, 132.9], "DEC": [11.8, 11.9],
        "GAIAEDR3_PHOT_G_MEAN_MAG": [10.0, 11.0],
        "GAIAEDR3_PHOT_BP_MEAN_MAG": [10.8, 11.9],
        "GAIAEDR3_PHOT_RP_MEAN_MAG": [9.2, 10.1],
        "TEFF": [4800.0, 4700.0], "LOGG": [4.2, 4.3],
    })

    def _fake_load(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return fake_df

    def _fake_cuts(df: pd.DataFrame, settings: Any) -> pd.DataFrame:
        return df

    def _fake_complete(df: pd.DataFrame, settings: Any) -> pd.DataFrame:
        return df

    def _fake_matrix(df: pd.DataFrame, settings: Any) -> np.ndarray:
        return np.zeros((len(df), 2))

    def _fake_sep(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
        return np.zeros(len(ra))

    def _fake_masks(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        n = len(fake_df)
        return {
            "catalog": np.array([True, False]),
            "kinematic": np.array([True, False]),
            "combined": np.array([True, False]),
        }

    def _fake_hr_comparison(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr("cluster.data.load_allstar", _fake_load)
    monkeypatch.setattr("cluster.data.apply_quality_cuts", _fake_cuts)
    monkeypatch.setattr("cluster.data.complete_case", _fake_complete)
    monkeypatch.setattr("cluster.data.make_matrix", _fake_matrix)
    monkeypatch.setattr("cluster.membership.angular_separation", _fake_sep)
    monkeypatch.setattr("cluster.catalog.membership_masks_for", _fake_masks)
    monkeypatch.setattr("cluster.plots.hr_comparison", _fake_hr_comparison)

    result = CliRunner().invoke(
        cli.main,
        ["hr", "--cluster", "M 67", "--allstar", str(allstar), "--outdir", str(outdir)],
    )
    assert result.exit_code == 0, result.output
    assert "catalog" in result.output


def test_cli_baseline_spectral_swaps_feature_source(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    import cluster.baseline as baseline_mod
    from cluster import spectral as spectral_mod

    allstar = tmp_path / "fake.fits"
    allstar.write_text("fake")
    parquet = tmp_path / "emb.parquet"
    parquet.write_bytes(b"fake")

    prepared = PreparedData(
        df=pd.DataFrame({"cluster": ["A", "B"], "APOGEE_ID": ["2M1", "2M2"]}),
        X=np.zeros((2, 2)),
        elements=["x", "y"],
    )
    seen: dict[str, Any] = {}

    def fake_prepare(*args: Any, **kwargs: Any) -> PreparedData:
        return prepared

    def fake_spectral_prepared(p: PreparedData, path: str | Path, settings: Any) -> PreparedData:
        seen["spectral_path"] = str(path)
        return p

    def fake_baseline_labels(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return np.array(["A", "B"]), {"t-SNE": np.array([0, 1])}

    def fake_separation(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"homogeneity": 1.0, "completeness": 1.0, "v_measure": 1.0, "accuracy": 1.0}

    monkeypatch.setattr("cluster.data.prepare", fake_prepare)
    monkeypatch.setattr(spectral_mod, "spectral_prepared", fake_spectral_prepared)
    monkeypatch.setattr(baseline_mod, "baseline_labels", fake_baseline_labels)
    monkeypatch.setattr(baseline_mod, "separation_scores", fake_separation)

    result = CliRunner().invoke(
        cli.main,
        ["baseline", "--allstar", str(allstar), "--spectral", str(parquet)],
    )
    assert result.exit_code == 0, result.output
    assert seen["spectral_path"] == str(parquet)
    assert "spectral" in result.output  # FEATURES=spectral banner
