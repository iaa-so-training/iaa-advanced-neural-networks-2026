"""Thin MLflow wrapper for the chemical-tagging benchmark.

Tracking is optional: every call degrades to a no-op (after one warning) when
``mlflow`` is not installed. Local runs are recorded under ``mlruns/`` relative
to the current working directory unless ``MLFLOW_TRACKING_URI`` is set.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "chemical-tagging")
_LOCAL_TRACKING_URI = "mlruns/"

# MLflow 3.x gates the local filesystem store behind this opt-out.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

_mlflow: Any = None
_import_attempted = False
_warned = False


def _get_mlflow() -> Any:
    """Lazily import mlflow and cache the module (or ``None``)."""
    global _mlflow, _import_attempted
    if not _import_attempted:
        _import_attempted = True
        try:
            _mlflow = importlib.import_module("mlflow")
        except ImportError:
            _mlflow = None
    return _mlflow


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", _LOCAL_TRACKING_URI)


def _warn_missing(mlflow: Any) -> None:
    global _warned
    if mlflow is None and not _warned:
        print("⚠ mlflow not installed; experiment tracking disabled.")
        _warned = True


def get_experiment_id() -> str | None:
    """Return the experiment id for ``chemical-tagging``, creating it if needed."""
    mlflow = _get_mlflow()
    if mlflow is None:
        _warn_missing(mlflow)
        return None
    mlflow.set_tracking_uri(_tracking_uri())
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment.experiment_id


def start_run(run_name: str) -> None:
    """Start an MLflow run under the ``chemical-tagging`` experiment."""
    mlflow = _get_mlflow()
    if mlflow is None:
        _warn_missing(mlflow)
        return
    get_experiment_id()
    mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the active run, if any."""
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    """Log metrics to the active run, if any."""
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_metrics(metrics)


def log_artifact(path: str | Path) -> None:
    """Log a local file as an artifact of the active run, if any."""
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_artifact(str(path))


def end_run() -> None:
    """End the active MLflow run, if any."""
    mlflow = _get_mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.end_run()
