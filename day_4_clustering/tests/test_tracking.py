"""Tests for the optional MLflow tracking wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cluster import tracking


class _FakeExperiment:
    experiment_id = "exp-123"


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.run_started = False
        self.run_name: str | None = None
        self.run_active = False
        self.params: list[dict[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.artifacts: list[str] = []
        self.run_ended = False

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> _FakeExperiment:
        self.experiment_name = name
        return _FakeExperiment()

    def start_run(self, run_name: str) -> None:
        self.run_started = True
        self.run_name = run_name
        self.run_active = True

    def active_run(self) -> object | None:
        return object() if self.run_active else None

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.append(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.append(metrics)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def end_run(self) -> None:
        self.run_ended = True
        self.run_active = False


@pytest.fixture
def fake_mlflow(monkeypatch: Any) -> _FakeMlflow:
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "_get_mlflow", lambda: fake)
    return fake


def test_tracking_uri_default_and_override(monkeypatch: Any) -> None:
    assert tracking._tracking_uri() == "mlruns/"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "s3://bucket/experiments")
    assert tracking._tracking_uri() == "s3://bucket/experiments"


def test_get_experiment_id_creates_experiment(fake_mlflow: _FakeMlflow) -> None:
    assert tracking.get_experiment_id() == "exp-123"
    assert fake_mlflow.tracking_uri == "mlruns/"
    assert fake_mlflow.experiment_name == tracking.EXPERIMENT_NAME


def test_start_run_creates_experiment_then_starts(fake_mlflow: _FakeMlflow) -> None:
    tracking.start_run("cluster-benchmark")
    assert fake_mlflow.run_started is True
    assert fake_mlflow.run_name == "cluster-benchmark"
    assert fake_mlflow.experiment_name == tracking.EXPERIMENT_NAME


def test_logging_helpers_delegate_to_fake_mlflow(fake_mlflow: _FakeMlflow) -> None:
    tracking.start_run("run")
    tracking.log_params({"a": 1})
    tracking.log_metrics({"score": 0.9})
    tracking.log_artifact(Path("results/plot.png"))
    tracking.end_run()

    assert fake_mlflow.params == [{"a": 1}]
    assert fake_mlflow.metrics == [{"score": 0.9}]
    assert fake_mlflow.artifacts == ["results/plot.png"]
    assert fake_mlflow.run_ended is True


def test_all_calls_noop_when_mlflow_missing(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(tracking, "_get_mlflow", lambda: None)
    monkeypatch.setattr(tracking, "_warned", False)

    assert tracking.get_experiment_id() is None
    tracking.start_run("run")
    tracking.log_params({"a": 1})
    tracking.log_metrics({"score": 0.9})
    tracking.log_artifact("plot.png")
    tracking.end_run()

    assert "mlflow not installed" in capsys.readouterr().out


def test_missing_warning_is_emitted_once(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setattr(tracking, "_get_mlflow", lambda: None)
    monkeypatch.setattr(tracking, "_warned", False)

    tracking.get_experiment_id()
    tracking.get_experiment_id()
    assert capsys.readouterr().out.count("mlflow not installed") == 1
