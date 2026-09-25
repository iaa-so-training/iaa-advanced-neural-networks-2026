"""The reproducibility contract, pinned where CI can pin it.

``docs/reproducibility.md`` splits the promise into three tiers. These tests
cover the tier that must be exact and *can* be checked on a tiny fixture:

* the same seed gives the same partition and the same scores, every method;
* :func:`cluster.seeding.seed_everything` really does seed the global RNGs;
* the environment the tolerance tier refers to is recorded — ``thread_report``
  and ``cluster doctor`` end to end;
* the CLI exposes the seed and prints it, so a number quoted in a bug report can
  be traced back to the run that produced it.

What cannot be tested here is cross-machine identity: that is a tolerance, not a
promise (see the doc for the measured spread).
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
from click.testing import CliRunner

from cluster.baseline import _fit_all, separation_scores
from cluster.cli import main
from cluster.config import Settings
from cluster.seeding import seed_everything, thread_report


@pytest.fixture
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated blobs (8 dims: EVoC's internal PCA needs >2)."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(0.0, 0.1, (20, 8)),
        rng.normal(10.0, 0.1, (20, 8)),
        rng.normal(-10.0, 0.1, (20, 8)),
    ])
    labels = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 20, dtype=object)
    return X, labels


@pytest.fixture
def small_settings() -> Settings:
    """Perplexity/neighbourhoods sized for the tiny synthetic fixtures."""
    return Settings(
        tsne={"perplexity": 2, "method": "barnes_hut"},
        umap={"n_neighbors": 4, "min_dist": 0.0},
    )


def test_same_seed_gives_identical_labels(
    blobs: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, _ = blobs
    small_settings.random_state = 42
    first = _fit_all(X, small_settings)
    second = _fit_all(X, small_settings)
    assert first, "no method produced labels"
    for method, labels in first.items():
        assert np.array_equal(labels, second[method]), method


def test_same_seed_gives_identical_scores(
    blobs: tuple[np.ndarray, np.ndarray], small_settings: Settings,
) -> None:
    X, true = blobs
    small_settings.random_state = 42
    a = _fit_all(X, small_settings)
    b = _fit_all(X, small_settings)
    for method in a:
        assert separation_scores(true, a[method]) == separation_scores(true, b[method])


def test_seed_everything_seeds_the_global_rngs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seed_everything(7)
    first = (random.random(), float(np.random.rand()))
    seed_everything(7)
    assert (random.random(), float(np.random.rand())) == first
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_mlflow_params_carry_the_fingerprint() -> None:
    """The bundle `cluster run` logs must exist and be flattenable.

    This is the guard for a real bug: the module `cluster.doctor` and the CLI
    command `doctor` share a name inside `cli.py`, so a module-level reference
    to the module resolved to the click Command at import time and `cluster run`
    died with AttributeError before it reached the benchmark.
    """
    from cluster.doctor import mlflow_params

    params = mlflow_params(run="unit-test")
    assert {"git_sha", "image", "python", "platform", "run"} <= set(params)
    assert any(key.startswith("pkg_") for key in params)
    assert all(isinstance(value, str) for value in params.values())


def test_thread_report_names_the_knobs() -> None:
    report = thread_report()
    assert {"OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "cpu_count", "numba_effective"} <= set(report)
    assert isinstance(report["cpu_count"], int)


def test_cli_run_and_baseline_expose_seed() -> None:
    runner = CliRunner()
    for command in ("run", "baseline"):
        output = runner.invoke(main, [command, "--help"]).output
        assert "--seed" in output, command


def test_cli_doctor_reports_a_fingerprint() -> None:
    result = CliRunner().invoke(main, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout or result.output)
    assert payload["seed"] == 42
    assert {"threads", "versions", "git_sha", "image", "data"} <= set(payload)
    assert "path" in payload["data"]["catalogue"]
    assert payload["versions"]["numpy"] != ""
