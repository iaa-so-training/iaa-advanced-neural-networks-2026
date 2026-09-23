"""Tests for the Simbad external-catalogue module (network mocked out)."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster import catalog
from cluster.clusters import Cluster


def test_cache_path_sanitizes_simbad_id() -> None:
    path = catalog._cache_path("data/simbad", "NGC 2682")
    assert path.name == "simbad_NGC_2682.csv"


def test_query_members_reads_cache_without_network(tmp_path: Any) -> None:
    cached = pd.DataFrame(
        {"main_id": ["a"], "ra": [1.0], "dec": [2.0], "membership": [100]}
    )
    cached.to_csv(tmp_path / "simbad_X.csv", index=False)
    out = catalog.query_members("X", 90, tmp_path)
    assert len(out) == 1
    assert out["ra"].iloc[0] == 1.0


class _FakeSearch:
    def to_table(self) -> Any:
        import astropy.table

        return astropy.table.Table(
            {"main_id": ["a"], "ra": [56.601], "dec": [24.114], "membership": [100]}
        )


class _FakeTAPService:
    def __init__(self, url: str) -> None:
        self.url = url

    def search(self, query: str) -> _FakeSearch:
        return _FakeSearch()


def test_query_members_live_writes_cache(monkeypatch: Any, tmp_path: Any) -> None:
    fake_dal = types.SimpleNamespace(TAPService=_FakeTAPService)
    fake_pyvo = types.SimpleNamespace(dal=fake_dal)
    monkeypatch.setitem(sys.modules, "pyvo", fake_pyvo)
    monkeypatch.setitem(sys.modules, "pyvo.dal", fake_dal)
    out = catalog.query_members("X", 90, tmp_path, force=True)
    assert len(out) == 1
    assert (tmp_path / "simbad_X.csv").exists()


def test_catalog_members_crossmatches_by_position(
    monkeypatch: Any, allstar_frame: pd.DataFrame, tmp_path: Any,
) -> None:
    # one member at row 0's sky position, one far away
    members = pd.DataFrame(
        {
            "main_id": ["m1", "m2"],
            "ra": [allstar_frame["RA"].iloc[0], 300.0],
            "dec": [allstar_frame["DEC"].iloc[0], -80.0],
            "membership": [100, 100],
        }
    )
    def _fake_resolve(name: str) -> str:
        return "X"

    def _fake_query(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return members

    monkeypatch.setattr(catalog, "resolve_simbad_id", _fake_resolve)
    monkeypatch.setattr(catalog, "query_members", _fake_query)

    mask = catalog.catalog_members(
        allstar_frame,
        Cluster(name="T", kind="open", ra_deg=0, dec_deg=0, dist_pc=1000,
                pmra=0, pmdec=0, rv=0),
        min_membership=90, crossmatch_arcsec=2.0, cache_dir=tmp_path,
    )
    assert bool(mask.iloc[0]) is True
    assert not mask.iloc[1:].any()


def test_membership_masks_returns_three_arrays(
    monkeypatch: Any, allstar_frame: pd.DataFrame, tmp_path: Any,
) -> None:
    # stub out the network part only; kinematic/combined run on the frame
    def _fake_catalog_members(*args: Any, **kwargs: Any) -> pd.Series:
        return pd.Series(
            np.zeros(len(allstar_frame), dtype=bool), index=allstar_frame.index,
        )

    monkeypatch.setattr(catalog, "catalog_members", _fake_catalog_members)
    X = np.zeros((len(allstar_frame), 2), dtype=float)
    masks = catalog.membership_masks(
        allstar_frame,
        Cluster(name="Pleiades", kind="open", ra_deg=56.6, dec_deg=24.1,
                dist_pc=136, pmra=20, pmdec=-45, rv=6),
        X,
        min_membership=90, crossmatch_arcsec=2.0, cache_dir=tmp_path,
        seed_position_radius_deg=0.6, seed_parallax_frac=0.30,
        seed_pm_tol=5.0, seed_rv_tol=25.0,
        n_refine_passes=3, refine_sigma=2.5,
        combined_chem_sigma=3.0, combined_kin_sigma=3.0,
    )
    assert set(masks) == {"catalog", "kinematic", "combined"}
    for arr in masks.values():
        assert arr.shape == (len(allstar_frame),)
        assert arr.dtype == bool


def test_hr_cell_returns_figure(monkeypatch: Any, allstar_frame: pd.DataFrame) -> None:
    from cluster.config import Settings

    def _fake_catalog_members(*args: Any, **kwargs: Any) -> pd.Series:
        return pd.Series(
            np.zeros(len(allstar_frame), dtype=bool), index=allstar_frame.index,
        )

    monkeypatch.setattr(catalog, "catalog_members", _fake_catalog_members)
    fig = catalog.hr_cell(allstar_frame, "Pleiades", "combined", Settings())
    # CMD + Kiel panels x (field + members)
    assert len(fig.data) == 4


def test_resolve_simbad_id(monkeypatch: Any) -> None:
    class _FakeResult:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, key: int) -> dict[str, str]:
            return {"main_id": "NGC 2682"}

    class _FakeSimbad:
        @staticmethod
        def query_object(name: str) -> _FakeResult:
            return _FakeResult()

    monkeypatch.setitem(sys.modules, "astroquery.simbad", types.SimpleNamespace(Simbad=_FakeSimbad))
    assert catalog.resolve_simbad_id("M 67") == "NGC 2682"


def test_attach_referee_adds_column(monkeypatch: Any, allstar_frame: pd.DataFrame) -> None:
    from cluster.config import Settings

    def _fake_catalog_members(*args: Any, **kwargs: Any) -> pd.Series:
        return pd.Series(
            np.array([True, False] + [False] * (len(allstar_frame) - 2)),
            index=allstar_frame.index,
        )

    monkeypatch.setattr(catalog, "catalog_members", _fake_catalog_members)
    cluster = Cluster(name="T", kind="open", ra_deg=0, dec_deg=0, dist_pc=1000, pmra=0, pmdec=0, rv=0)
    out = catalog.attach_referee(allstar_frame, [cluster], Settings())
    assert "referee" in out.columns
    assert (out["referee"] == "T").sum() == 1


def test_cluster_panels_cell(monkeypatch: Any, allstar_frame: pd.DataFrame) -> None:
    import pandas as _pd

    from cluster.config import Settings

    def _fake_catalog_members(*args: Any, **kwargs: Any) -> pd.Series:
        return pd.Series(np.zeros(len(allstar_frame), dtype=bool), index=allstar_frame.index)

    def _fake_gaia(cluster: Any, radius_deg: float, cache_dir: Any = "data/gaia") -> _pd.DataFrame:
        return _pd.DataFrame({
            "phot_g_mean_mag": [10.0, 18.0], "phot_bp_mean_mag": [10.8, 18.9],
            "phot_rp_mean_mag": [9.2, 17.1], "pmra": [4.0, 4.1], "pmdec": [-6.0, -6.1],
        })

    monkeypatch.setattr(catalog, "catalog_members", _fake_catalog_members)
    monkeypatch.setattr("cluster.gaia.query_gaia_region", _fake_gaia)
    fig = catalog.cluster_panels_cell(allstar_frame, "Pleiades", "combined", Settings())
    assert len(fig.data) == 6  # 3 panels x 2 traces
