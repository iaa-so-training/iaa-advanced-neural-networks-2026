"""Tests for the ASteCA isochrone-fitting module (heavy fit skipped offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster.clusters import Cluster
from cluster.isochrone import (
    ISOCHRONE_DIR,
    ISOCHRONE_FILE,
    IsochroneFit,
    _distance_modulus,
    fit_isochrone,
    plot_isochrone_fit,
)


def test_distance_modulus_from_parallax() -> None:
    # parallax 10 mas -> 100 pc -> dm = 5*log10(100/10) = 5
    assert _distance_modulus(np.array([10.0, 11.0, 9.0])) == pytest.approx(5.0)
    # empty / all bad -> fallback 9.5
    assert _distance_modulus(np.array([-1.0, np.nan])) == pytest.approx(9.5)


def test_fit_isochrone_rejects_few_members() -> None:
    df = pd.DataFrame({
        "GAIAEDR3_PHOT_G_MEAN_MAG": np.full(10, 10.0),
        "GAIAEDR3_PHOT_BP_MEAN_MAG": np.full(10, 10.8),
        "GAIAEDR3_PHOT_RP_MEAN_MAG": np.full(10, 9.2),
        "GAIAEDR3_PARALLAX": np.full(10, 1.0),
    })
    with pytest.raises(ValueError, match="need >= 25 members"):
        fit_isochrone(df)


def test_plot_isochrone_fit_has_three_traces() -> None:
    df = pd.DataFrame({
        "GAIAEDR3_PHOT_G_MEAN_MAG": np.linspace(9.0, 12.0, 30),
        "GAIAEDR3_PHOT_BP_MEAN_MAG": np.linspace(9.8, 13.0, 30),
        "GAIAEDR3_PHOT_RP_MEAN_MAG": np.linspace(8.2, 11.0, 30),
        "GAIAEDR3_PARALLAX": np.full(30, 1.0),
    })
    masks = {
        "catalog": np.array([True] * 15 + [False] * 15),
        "kinematic": np.array([True] * 10 + [False] * 20),
        "combined": np.array([True] * 12 + [False] * 18),
    }
    fit = IsochroneFit(
        best={"met": 0.015, "loga": 9.0, "dm": 5.0, "Av": 0.1},
        std={"met": 0.001, "loga": 0.1, "dm": 0.2, "Av": 0.05},
        max_lkl=-0.05,
        n_stars=30,
        curve_mag=np.linspace(9.0, 12.0, 50),
        curve_color=np.linspace(1.0, 2.0, 50),
    )
    fig = plot_isochrone_fit(df, masks, "T", fit, highlight="combined")
    # field + members + isochrone line
    assert len(fig.data) == 3


def test_plot_gaia_cmd() -> None:
    from cluster.isochrone import plot_gaia_cmd

    members = pd.DataFrame({
        "phot_g_mean_mag": [10.0, 11.0, 12.0],
        "phot_bp_mean_mag": [10.8, 11.9, 13.0],
        "phot_rp_mean_mag": [9.2, 10.1, 11.0],
    })
    fit = IsochroneFit(
        best={"met": 0.01, "loga": 9.0, "dm": 5.0, "Av": 0.1},
        std={"met": 0.001, "loga": 0.1, "dm": 0.2, "Av": 0.05},
        max_lkl=-0.05, n_stars=3,
        curve_mag=np.array([10.0, 11.0]), curve_color=np.array([1.0, 2.0]),
    )
    fig = plot_gaia_cmd(members, fit, "T")
    # members + isochrone line
    assert len(fig.data) == 2


def test_isochrone_cell_under25_members(monkeypatch: Any, allstar_frame: pd.DataFrame) -> None:
    import types

    import cluster.catalog as cat
    import cluster.isochrone as iso
    from cluster.config import Settings

    n = len(allstar_frame)

    def _fake_masks(df: pd.DataFrame, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        n = len(df)
        return {
            "catalog": np.zeros(n, dtype=bool),
            "kinematic": np.zeros(n, dtype=bool),
            "combined": np.ones(n, dtype=bool),
        }

    def _fake_md(s: str) -> str:
        return f"md:{s}"

    fake_mo = types.SimpleNamespace(md=_fake_md)
    monkeypatch.setattr(cat, "membership_masks_for", _fake_masks)
    out = iso.isochrone_cell(allstar_frame, "Pleiades", "combined", Settings(), fake_mo)
    assert str(out).startswith("md:")  # <25 members -> warning


def test_fit_isochrone_gaia_mock(monkeypatch: Any) -> None:
    import cluster.gaia as gaia_mod
    import cluster.isochrone as iso

    n = 60
    df_gaia = pd.DataFrame({
        "phot_g_mean_mag": np.linspace(12, 20, n),
        "phot_bp_mean_mag": np.linspace(12.8, 21.0, n),
        "phot_rp_mean_mag": np.linspace(11.2, 19.0, n),
        "parallax": np.full(n, 1.0),
        "pmra": np.full(n, 4.0),
        "pmdec": np.full(n, -6.0),
    })

    def _fake_query(cluster: Any, radius_deg: float, cache_dir: Any = "data/gaia") -> pd.DataFrame:
        return df_gaia

    def _fake_members(df: pd.DataFrame, cluster: Any, **k: Any) -> pd.Series:
        return pd.Series(np.ones(len(df), dtype=bool))

    monkeypatch.setattr(gaia_mod, "query_gaia_region", _fake_query)
    monkeypatch.setattr(gaia_mod, "gaia_members", _fake_members)

    fake_fit = IsochroneFit(
        best={"met": 0.001, "loga": 10.0, "dm": 14.3, "Av": 0.1},
        std={"met": 0.0001, "loga": 0.2, "dm": 0.3, "Av": 0.05},
        max_lkl=-0.05, n_stars=n,
    )

    def _fake_fit(*args: Any, **kwargs: Any) -> IsochroneFit:
        return fake_fit

    monkeypatch.setattr(iso, "fit_isochrone", _fake_fit)

    fit = iso.fit_isochrone_gaia(
        Cluster(name="M 5", kind="globular", ra_deg=0, dec_deg=0, dist_pc=7500, pmra=4, pmdec=-6, rv=0),
        radius_deg=0.5,
    )
    assert fit.best["loga"] == 10.0
    assert fit.n_stars == n


def test_gaia_age_cell(monkeypatch: Any) -> None:
    import cluster.gaia as gaia_mod
    import cluster.isochrone as iso
    from cluster.config import Settings

    n = 60
    df_gaia = pd.DataFrame({
        "phot_g_mean_mag": np.linspace(16, 21, n),
        "phot_bp_mean_mag": np.linspace(16.8, 22.0, n),
        "phot_rp_mean_mag": np.linspace(15.2, 20.0, n),
        "parallax": np.full(n, 0.13),
        "pmra": np.full(n, 4.0),
        "pmdec": np.full(n, -6.0),
    })

    def _fake_query(cluster: Any, radius_deg: float, cache_dir: Any = "data/gaia") -> pd.DataFrame:
        return df_gaia

    def _fake_members(df: pd.DataFrame, cluster: Any, **k: Any) -> pd.Series:
        return pd.Series(np.ones(len(df), dtype=bool))

    monkeypatch.setattr(gaia_mod, "query_gaia_region", _fake_query)
    monkeypatch.setattr(gaia_mod, "gaia_members", _fake_members)
    fake_fit = IsochroneFit(
        best={"met": 0.001, "loga": 10.0, "dm": 14.3, "Av": 0.1},
        std={"met": 0.0001, "loga": 0.2, "dm": 0.3, "Av": 0.05},
        max_lkl=-0.05, n_stars=n,
        curve_mag=np.array([10.0, 11.0]), curve_color=np.array([1.0, 2.0]),
    )

    def _fake_fit(*args: Any, **kwargs: Any) -> IsochroneFit:
        return fake_fit

    monkeypatch.setattr(iso, "fit_isochrone", _fake_fit)
    fig = iso.gaia_age_cell("M 5", Settings())
    assert len(fig.data) == 2  # members + isochrone


def test_download_grid_mock(monkeypatch: Any, tmp_path: Any) -> None:
    import cluster._parsec as par
    import cluster.isochrone as iso

    class _FakeQuery:
        def query_isochrones(self, **kwargs: Any) -> str:
            return "# header\n1.0 2.0 3.0\n"

    monkeypatch.setattr(par, "ParsecQuery", _FakeQuery)
    p1 = iso.ensure_isochrones(tmp_path)
    p2 = iso.ensure_isochrones_metal_poor(tmp_path)
    assert p1.exists() and p2.exists()
    assert "1.0 2.0 3.0" in p1.read_text()


def test_isochrone_cell_full_path(monkeypatch: Any) -> None:
    import types

    import cluster.catalog as cat
    import cluster.data as data_mod
    import cluster.isochrone as iso
    import cluster.membership as mem
    from cluster.config import Settings

    n = 30
    df_hr = pd.DataFrame({"RA": np.full(n, 132.8), "DEC": np.full(n, 11.8)})

    def _fake_complete(df: pd.DataFrame, settings: Any) -> pd.DataFrame:
        return df

    def _fake_matrix(df: pd.DataFrame, settings: Any) -> np.ndarray:
        return np.zeros((len(df), 2))

    def _fake_sep(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
        return np.zeros(len(ra))

    def _fake_masks(df: pd.DataFrame, cluster: Any, X: np.ndarray, settings: Any) -> dict[str, np.ndarray]:
        return {
            "catalog": np.ones(len(df), dtype=bool),
            "kinematic": np.ones(len(df), dtype=bool),
            "combined": np.ones(len(df), dtype=bool),
        }

    fake_fit = IsochroneFit(
        best={"met": 0.01, "loga": 9.0, "dm": 5.0, "Av": 0.1},
        std={"met": 0.001, "loga": 0.1, "dm": 0.2, "Av": 0.05},
        max_lkl=-0.05, n_stars=n,
    )

    def _fake_fit(*args: Any, **kwargs: Any) -> IsochroneFit:
        return fake_fit

    def _fake_plot(*args: Any, **kwargs: Any) -> str:
        return "fig"

    monkeypatch.setattr(data_mod, "complete_case", _fake_complete)
    monkeypatch.setattr(data_mod, "make_matrix", _fake_matrix)
    monkeypatch.setattr(mem, "angular_separation", _fake_sep)
    monkeypatch.setattr(cat, "membership_masks_for", _fake_masks)
    monkeypatch.setattr(iso, "fit_isochrone", _fake_fit)
    monkeypatch.setattr(iso, "plot_isochrone_fit", _fake_plot)

    def _fake_md(s: str) -> str:
        return f"md:{s}"

    fake_mo = types.SimpleNamespace(md=_fake_md)
    out = iso.isochrone_cell(df_hr, "M 67", "combined", Settings(), fake_mo)
    assert out == "fig"


@pytest.mark.skipif(
    not (Path(ISOCHRONE_DIR) / ISOCHRONE_FILE).exists(),
    reason="PARSEC isochrone grid not downloaded",
)
def test_fit_isochrone_smoke_on_synthetic_main_sequence() -> None:
    """A clean synthetic main sequence should fit quickly (no emcee blow-up)."""
    rng = np.random.default_rng(0)
    n = 60
    color = np.linspace(0.8, 1.8, n) + rng.normal(0, 0.02, n)
    g = 4.0 + 3.0 * color + rng.normal(0, 0.03, n)  # tight main sequence
    plx = np.full(n, 1.0)  # 1000 pc, dm = 5
    df = pd.DataFrame({
        "GAIAEDR3_PHOT_G_MEAN_MAG": g,
        "GAIAEDR3_PHOT_BP_MEAN_MAG": g + color / 2 + 0.5,
        "GAIAEDR3_PHOT_RP_MEAN_MAG": g - color / 2 - 0.5,
        "J": g - color * 0.4,
        "K": g - color * 0.6,
        "GAIAEDR3_PARALLAX": plx,
    })
    fit = fit_isochrone(df, seed=1, n_walkers=8, n_steps=40)
    assert fit.n_stars == n
    assert np.isfinite(fit.max_lkl)
    assert all(np.isfinite(v) for v in fit.best.values())
    assert all(v > 0 for v in fit.std.values())
