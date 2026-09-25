"""The prepared-sample cache: same sample, less wall clock, no surprises.

``prepare`` is ~20 s of single-core gzip decompression per call, and both the
notebook and ``cluster run`` call it on every re-run, so the sample is cached on
disk keyed by (catalogue bytes, settings, clusters, seed arguments). These tests
pin the two properties that make that safe: a cached sample is bit-identical to
a freshly computed one, and everything that changes the sample moves the key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cluster import config, data
from cluster.clusters import CLUSTER_BY_NAME
from cluster.config import ELEMENTS, Settings
from cluster.data import PreparedData, cache_dir, prepare, prepare_cache_key


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "snr_min": 0.0,
        "require_clean_flags": False,
        "dwarf_only": False,
        "elements": list(ELEMENTS),
        "standardize": True,
        "use_element_weights": False,
        "require_element_flag_clean": False,
        "max_stars": None,
        "region_radius_deg": None,
    }
    return Settings(**{**base, **overrides})


def _seed_kwargs() -> dict[str, Any]:
    return {
        "seed_position_radius_deg": config.SEED_POSITION_RADIUS_DEG,
        "seed_parallax_frac": config.SEED_PARALLAX_FRAC,
        "seed_pm_tol": config.SEED_PM_TOL,
        "seed_rv_tol": config.SEED_RV_TOL,
        "n_refine_passes": config.N_REFINE_PASSES,
        "refine_sigma": config.REFINE_SIGMA,
    }


@pytest.fixture
def cache_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway cache directory, with the suite-wide opt-out lifted."""
    target = tmp_path / "prepared-cache"
    monkeypatch.delenv("CLUSTER_NO_CACHE", raising=False)
    monkeypatch.setenv("CLUSTER_CACHE_DIR", str(target))
    return target


def test_cache_returns_a_bit_identical_sample(astra_fits_path: Path, cache_on: Path) -> None:
    settings = _settings()
    clusters = [CLUSTER_BY_NAME["Pleiades"]]

    first = prepare(astra_fits_path, settings, clusters, **_seed_kwargs())
    assert not first.df.empty
    assert len(list(cache_on.glob("*.parquet"))) == 1

    second = prepare(astra_fits_path, settings, clusters, **_seed_kwargs())
    assert np.array_equal(first.X, second.X), "cached matrix must be bit-identical"
    assert first.df.equals(second.df)
    assert second.elements == first.elements
    assert len(list(cache_on.glob("*.parquet"))) == 1, "same key must not write a second entry"


def test_cached_call_does_not_touch_the_catalogue(
    astra_fits_path: Path, cache_on: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the second call must not read the catalogue again."""

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("load_allstar ran although the cache held the sample")

    settings = _settings()
    clusters = [CLUSTER_BY_NAME["Pleiades"]]
    prepare(astra_fits_path, settings, clusters, **_seed_kwargs())

    monkeypatch.setattr(data, "load_allstar", _explode)
    cached = prepare(astra_fits_path, settings, clusters, **_seed_kwargs())
    assert isinstance(cached, PreparedData)
    assert cached.X.shape[0] > 0


def test_no_cache_switch_computes_again_and_writes_nothing(
    astra_fits_path: Path, cache_on: Path
) -> None:
    settings, clusters = _settings(), [CLUSTER_BY_NAME["Pleiades"]]
    prepare(astra_fits_path, settings, clusters, **_seed_kwargs(), no_cache=True)
    prepare(astra_fits_path, settings, clusters, **_seed_kwargs(), no_cache=True)
    assert list(cache_on.glob("*")) == []
    assert cache_dir(enabled=False) is None


def test_cluster_no_cache_env_disables_it(astra_fits_path: Path, cache_on: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUSTER_NO_CACHE", "1")
    assert cache_dir() is None
    prepare(astra_fits_path, _settings(), [CLUSTER_BY_NAME["Pleiades"]], **_seed_kwargs())
    assert list(cache_on.glob("*")) == []


def test_key_moves_with_everything_that_changes_the_sample(astra_fits_path: Path) -> None:
    clusters = [CLUSTER_BY_NAME["Pleiades"]]
    seeds = _seed_kwargs()
    base = prepare_cache_key(astra_fits_path, _settings(), clusters, seeds)

    assert prepare_cache_key(astra_fits_path, _settings(snr_min=50.0), clusters, seeds) != base
    assert prepare_cache_key(astra_fits_path, _settings(random_state=7), clusters, seeds) != base
    assert prepare_cache_key(astra_fits_path, _settings(max_stars=500), clusters, seeds) != base
    assert prepare_cache_key(astra_fits_path, _settings(), [CLUSTER_BY_NAME["M 67"]], seeds) != base
    assert prepare_cache_key(astra_fits_path, _settings(), clusters, {**seeds, "seed_pm_tol": 99.0}) != base
    assert prepare_cache_key(astra_fits_path, _settings(), clusters, seeds) == base


def test_key_moves_when_the_catalogue_changes(astra_fits_path: Path) -> None:
    clusters = [CLUSTER_BY_NAME["Pleiades"]]
    seeds = _seed_kwargs()
    settings = _settings()
    before = prepare_cache_key(astra_fits_path, settings, clusters, seeds)

    stat = astra_fits_path.stat()
    os.utime(astra_fits_path, (stat.st_atime + 5, stat.st_mtime + 5))
    assert prepare_cache_key(astra_fits_path, settings, clusters, seeds) != before, "mtime is part of the identity"

    astra_fits_path.with_name(astra_fits_path.name + ".sha256").write_text("0" * 64 + "  x\n")
    with_sidecar = prepare_cache_key(astra_fits_path, settings, clusters, seeds)
    assert with_sidecar not in {before}, "the sidecar hash is part of the identity"


def test_same_bytes_elsewhere_share_the_entry(astra_fits_path: Path, tmp_path: Path) -> None:
    """Host ./data/x and container /app/data/x must hit the same entry."""
    import shutil

    elsewhere = tmp_path / "mnt" / astra_fits_path.name
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(astra_fits_path, elsewhere)  # copy2 keeps the mtime

    settings, clusters, seeds = _settings(), [CLUSTER_BY_NAME["Pleiades"]], _seed_kwargs()
    assert prepare_cache_key(elsewhere, settings, clusters, seeds) == prepare_cache_key(
        astra_fits_path, settings, clusters, seeds
    )


def test_corrupt_entry_falls_back_to_a_fresh_read(astra_fits_path: Path, cache_on: Path) -> None:
    settings, clusters = _settings(), [CLUSTER_BY_NAME["Pleiades"]]
    prepare(astra_fits_path, settings, clusters, **_seed_kwargs())

    next(cache_on.glob("*.parquet")).write_bytes(b"not a parquet file")
    rebuilt = prepare(astra_fits_path, settings, clusters, **_seed_kwargs())
    assert rebuilt.X.shape[0] > 0


def test_cli_exposes_no_cache() -> None:
    from click.testing import CliRunner

    from cluster import cli

    for command in ("run", "baseline"):
        result = CliRunner().invoke(cli.main, [command, "--help"])
        assert result.exit_code == 0
        assert "--no-cache" in result.output
