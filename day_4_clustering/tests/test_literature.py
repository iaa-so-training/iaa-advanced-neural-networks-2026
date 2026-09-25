"""Tests for the literature module (no network: fallback + conversions)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cluster import literature
from cluster.clusters import CLUSTER_BY_NAME


def test_dm_from_distance() -> None:
    assert literature._dm(10.0) == 0.0
    assert literature._dm(100.0) == 5.0
    assert literature._dm(1000.0) == 10.0


def test_feh_from_z() -> None:
    assert literature._feh(0.0152) == 0.0
    assert literature._feh(0.00152) == -1.0
    assert np.isnan(literature._feh(0.0))


def test_fetch_literature_fallback_no_network() -> None:
    lit = literature.fetch_literature(CLUSTER_BY_NAME["Pleiades"])
    assert lit["age_Gyr"] == 0.10
    assert lit["dist_pc"] == 136.0
    assert abs(lit["dm"] - 5.67) < 0.01


def test_compare_fit_residuals() -> None:
    lit = {"age_Gyr": 10.0, "dm": 14.0, "feh": -1.0, "dist_pc": 6300.0}
    out = literature.compare_fit(fit_age_gyr=11.0, fit_dm=14.5, fit_met_z=0.00152, lit=lit)
    assert out["age_resid"] == 1.0
    assert out["dm_resid"] == 0.5
    assert out["feh_resid"] == 0.0


def _mock_vizier(monkeypatch: Any, open_row: dict[str, float] | None, glob_row: dict[str, float] | None) -> None:
    import sys
    import types

    class _FakeTable:
        def __init__(self, data: dict[str, float]) -> None:
            self.colnames = list(data)
            self._data = {k: [v] for k, v in data.items()}

        def __getitem__(self, key: str) -> list[float]:
            return self._data[key]

    class _FakeVizier:
        ROW_LIMIT = -1

        @staticmethod
        def query_object(name: str, catalog: list[str] | None = None) -> list[_FakeTable]:
            if catalog == ["VII/202"]:
                assert glob_row is not None
                return [_FakeTable(glob_row)]
            assert open_row is not None
            return [_FakeTable(open_row)]

    monkeypatch.setitem(sys.modules, "astroquery.vizier", types.SimpleNamespace(Vizier=_FakeVizier))


def test_fetch_literature_open_cluster(monkeypatch: Any, tmp_path: Any) -> None:
    _mock_vizier(monkeypatch, {"Age": 9.45, "Dist": 808.0, "__Fe_H_": 0.03}, None)
    lit = literature.fetch_literature(CLUSTER_BY_NAME["M 67"], cache_dir=tmp_path)
    assert abs(lit["age_Gyr"] - 2.82) < 0.1
    assert lit["dist_pc"] == 808.0
    assert abs(lit["feh"] - 0.03) < 0.01


def test_fetch_literature_globular(monkeypatch: Any, tmp_path: Any) -> None:
    _mock_vizier(monkeypatch, None, {"[Fe/H]": -1.29, "Rsun": 7.3})
    lit = literature.fetch_literature(CLUSTER_BY_NAME["M 5"], cache_dir=tmp_path)
    assert abs(lit["feh"] - (-1.29)) < 0.01
    assert lit["dist_pc"] == 7300.0
    assert abs(lit["age_Gyr"] - 10.5) < 0.1


def test_fetch_literature_reads_cache(monkeypatch: Any, tmp_path: Any) -> None:
    # write a cache row, then fetch without any network mock -> reads cache
    pd.DataFrame([{"age_Gyr": 1.5, "dist_pc": 500.0, "dm": 8.49, "feh": 0.1}]).to_csv(
        tmp_path / "lit_X.csv", index=False,
    )
    import sys
    import types

    monkeypatch.setitem(sys.modules, "astroquery.vizier", types.SimpleNamespace())
    lit = literature.fetch_literature(
        literature.Cluster(name="X", kind="open", ra_deg=0, dec_deg=0, dist_pc=500, pmra=0, pmdec=0, rv=0),
        cache_dir=tmp_path,
    )
    assert lit["age_Gyr"] == 1.5
    assert lit["dist_pc"] == 500.0


def test_literature_table(monkeypatch: Any) -> None:
    def _fake_fetch(cluster: literature.Cluster, cache_dir: Any = "data/literature") -> dict[str, float]:
        return {"age_Gyr": 1.0, "dist_pc": 1000.0, "dm": 10.0, "feh": 0.0}

    monkeypatch.setattr(literature, "fetch_literature", _fake_fetch)
    table = literature.literature_table([CLUSTER_BY_NAME["M 67"], CLUSTER_BY_NAME["M 5"]])
    assert len(table) == 2
    assert list(table["cluster"]) == ["M 67", "M 5"]
    assert table["lit_dm"].tolist() == [10.0, 10.0]
