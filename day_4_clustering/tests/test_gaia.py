"""Tests for Gaia DR3 photometry + Gaia-only membership."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster.clusters import Cluster
from cluster.gaia import gaia_members, query_gaia_region


def _cluster() -> Cluster:
    return Cluster(
        name="T", kind="globular", ra_deg=0.0, dec_deg=0.0,
        dist_pc=7500.0, pmra=4.0, pmdec=-6.0, rv=0.0,
    )


def test_gaia_members_proper_motion_cut() -> None:
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "pmra": rng.normal(4.0, 0.3, n),
        "pmdec": rng.normal(-6.0, 0.3, n),
    })
    # some field stars far away in PM
    df.loc[80:, "pmra"] += 10.0
    df.loc[80:, "pmdec"] += 12.0
    mask = gaia_members(df, _cluster(), pm_tol=2.0)
    assert mask.sum() == 80
    assert not mask.iloc[80:].any()


def test_gaia_members_anchors_to_cluster_not_field_median() -> None:
    # a field-dominated sample where the median PM is NOT the cluster's
    rng = np.random.default_rng(1)
    field_pmra = rng.normal(5.0, 2.0, 500)
    field_pmdec = rng.normal(-3.0, 2.0, 500)
    cluster_pmra = rng.normal(4.0, 0.1, 30)
    cluster_pmdec = rng.normal(-6.0, 0.1, 30)
    df = pd.DataFrame({
        "pmra": np.concatenate([field_pmra, cluster_pmra]),
        "pmdec": np.concatenate([field_pmdec, cluster_pmdec]),
    })
    mask = gaia_members(df, _cluster(), pm_tol=2.0)
    # the 30 cluster stars (last 30) must all be selected
    assert mask.iloc[-30:].all()


def test_query_gaia_region_mock_and_cache(monkeypatch: Any, tmp_path: Any) -> None:
    import sys
    import types

    class _FakeTable:
        def to_pandas(self) -> pd.DataFrame:
            return pd.DataFrame({
                "source_id": [1, 2], "ra": [229.0, 230.0], "dec": [2.0, 2.5],
                "phot_g_mean_mag": [10.0, 18.0],
                "phot_bp_mean_mag": [10.8, 18.9],
                "phot_rp_mean_mag": [9.2, 17.1],
                "parallax": [0.13, 0.14], "pmra": [4.0, 4.1], "pmdec": [-6.0, -6.1],
            })

    class _FakeJob:
        def get_results(self) -> _FakeTable:
            return _FakeTable()

    class _FakeGaia:
        @staticmethod
        def launch_job(query: str) -> _FakeJob:
            return _FakeJob()

    monkeypatch.setitem(sys.modules, "astroquery.gaia", types.SimpleNamespace(Gaia=_FakeGaia))
    df = query_gaia_region(_cluster(), 0.5, cache_dir=tmp_path)
    assert len(df) == 2
    assert "phot_g_mean_mag" in df.columns
    # second call hits the cache (no network needed)
    monkeypatch.setitem(sys.modules, "astroquery.gaia", types.SimpleNamespace())
    df2 = query_gaia_region(_cluster(), 0.5, cache_dir=tmp_path)
    assert len(df2) == 2


@pytest.mark.skipif(
    not (Path("data/isochrones/parsec/parsec_solar_2color.dat")).exists(),
    reason="PARSEC grid not downloaded",
)
def test_fit_isochrone_gaia_column_names() -> None:
    """fit_isochrone accepts Gaia DR3 column names (no J/K -> single colour)."""
    from cluster.isochrone import fit_isochrone

    rng = np.random.default_rng(2)
    n = 60
    color = np.linspace(0.6, 1.6, n)
    g = 3.0 + 4.0 * color + rng.normal(0, 0.03, n)
    df = pd.DataFrame({
        "phot_g_mean_mag": g,
        "phot_bp_mean_mag": g + color / 2 + 0.4,
        "phot_rp_mean_mag": g - color / 2 - 0.4,
        "parallax": np.full(n, 1.0),
    })
    fit = fit_isochrone(df, seed=1, n_walkers=8, n_steps=40, grid="solar")
    assert fit.n_stars == n
    assert np.isfinite(fit.max_lkl)
