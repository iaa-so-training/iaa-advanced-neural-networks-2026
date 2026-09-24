"""The bounded waits around archive calls (hotel-wifi insurance)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pytest

from cluster.net import NETWORK_TIMEOUT_S, NetworkTimeout, call_with_timeout


def test_fast_call_passes_through() -> None:
    assert call_with_timeout(lambda x: x * 2, 21, timeout_s=5) == 42
    assert call_with_timeout(lambda a, b=1: a + b, 1, b=2, timeout_s=5) == 3


def test_stuck_call_gives_up_instead_of_waiting() -> None:
    started = time.perf_counter()
    with pytest.raises(NetworkTimeout) as err:
        call_with_timeout(time.sleep, 30, what="a sleepy archive", timeout_s=0.3)
    elapsed = time.perf_counter() - started
    assert elapsed < 5, f"waited {elapsed:.1f}s instead of giving up"
    assert "sleepy archive" in str(err.value)
    assert "CLUSTER_NET_TIMEOUT" in str(err.value)   # tells you how to raise the budget


def test_default_budget_is_a_workshop_sized_number() -> None:
    assert 5 <= NETWORK_TIMEOUT_S <= 120


def test_network_timeout_is_a_runtime_error() -> None:
    assert issubclass(NetworkTimeout, RuntimeError)


def test_literature_table_stops_after_the_first_timeout(monkeypatch: Any) -> None:
    """23 clusters must not mean 23 waits when the archive is unreachable."""
    import cluster.literature as lit_mod
    from cluster.clusters import CLUSTERS

    calls = {"n": 0}

    def _always_timeout(cluster: Any, *a: Any, **k: Any) -> dict[str, float]:
        calls["n"] += 1
        raise NetworkTimeout("VizieR did not answer within 30 s")

    monkeypatch.setattr(lit_mod, "fetch_literature", _always_timeout)
    table = lit_mod.literature_table(list(CLUSTERS))

    assert calls["n"] == 1, "should only try once"
    assert len(table) == len(CLUSTERS)
    assert table["lit_age_Gyr"].isna().all()
    assert "unreachable" in table["note"].iloc[0]
    assert table["note"].iloc[1:].str.contains("skipped").all()


def test_literature_table_fills_notes_on_other_errors(monkeypatch: Any) -> None:
    import cluster.literature as lit_mod
    from cluster.clusters import CLUSTERS

    def _boom(cluster: Any, *a: Any, **k: Any) -> dict[str, float]:
        raise ValueError("catalogue gave a table I do not understand")

    monkeypatch.setattr(lit_mod, "fetch_literature", _boom)
    table = lit_mod.literature_table(list(CLUSTERS)[:3])
    assert len(table) == 3
    assert table["note"].str.contains("ValueError").all()


def test_gaia_age_cell_degrades_when_the_archive_is_quiet(monkeypatch: Any) -> None:
    import cluster.gaia as gaia_mod
    import cluster.isochrone as iso
    from cluster.config import Settings

    def _timeout(cluster: Any, *a: Any, **k: Any) -> pd.DataFrame:
        raise NetworkTimeout("the Gaia archive did not answer within 30 s")

    monkeypatch.setattr(gaia_mod, "query_gaia_region", _timeout)
    note = iso.gaia_age_cell("M 67", Settings())
    assert isinstance(note, str)
    assert "unreachable" in note and "M 67" in note
