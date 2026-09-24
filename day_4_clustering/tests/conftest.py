"""Shared fixtures for the cluster test-suite.

All fixtures are synthetic: no real allStar FITS file is downloaded or read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from cluster.config import ELEMENTS, astra_h_col


@pytest.fixture(autouse=True)
def _prepared_cache_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic: tests that exercise the cache opt back in.

    The prepared-sample cache is on for real runs (it saves a ~20 s single-core
    catalogue read); in tests it would otherwise scatter entries next to the
    checkout and quietly speed up the "did prepare run?" assertions.
    """
    monkeypatch.setenv("CLUSTER_NO_CACHE", "1")


def make_allstar_frame(n: int = 8) -> pd.DataFrame:
    """Build a minimal, schema-valid allStar frame.

    ``n`` must be at least 4 so a couple of rows can be reserved for a
    Pleiades member and quality-cut variations.
    """
    if n < 4:
        raise ValueError("make_allstar_frame needs at least 4 rows")

    df = pd.DataFrame(
        {
            "APOGEE_ID": [f"2M{i:08d}" for i in range(n)],
            "RA": [56.601, 56.602, 120.0, 200.0] + [10.0 + i for i in range(n - 4)],
            "DEC": [24.114, 24.115, 30.0, -50.0] + [5.0 + i for i in range(n - 4)],
            "GLON": np.linspace(160.0, 170.0, n),
            "GLAT": np.linspace(-20.0, 10.0, n),
            "SNR": np.full(n, 150.0, dtype=float),
            "ASPCAPFLAG": np.zeros(n, dtype=np.int64),
            "STARFLAG": np.zeros(n, dtype=np.int64),
            "TEFF": np.full(n, 4800.0, dtype=float),
            "LOGG": np.full(n, 4.2, dtype=float),
            "VHELIO_AVG": np.full(n, 6.0, dtype=float),
            "GAIAEDR3_PARALLAX": np.full(n, 7.35, dtype=float),
            "GAIAEDR3_PMRA": np.full(n, 20.0, dtype=float),
            "GAIAEDR3_PMDEC": np.full(n, -45.5, dtype=float),
            # photometry for HR diagrams
            "GAIAEDR3_PHOT_G_MEAN_MAG": np.full(n, 10.0, dtype=float),
            "GAIAEDR3_PHOT_BP_MEAN_MAG": np.full(n, 10.8, dtype=float),
            "GAIAEDR3_PHOT_RP_MEAN_MAG": np.full(n, 9.2, dtype=float),
            "J": np.full(n, 8.5, dtype=float),
            "H": np.full(n, 8.0, dtype=float),
            "K": np.full(n, 7.8, dtype=float),
        }
    )

    for j, element in enumerate(ELEMENTS):
        df[element] = np.linspace(-0.5 + j * 0.05, 0.5 + j * 0.05, n)
        df[f"{element}_ERR"] = np.full(n, 0.05 + j * 0.001, dtype=float)
        df[f"{element}_FLAG"] = np.zeros(n, dtype=np.int64)

    # Rows 0 and 1 satisfy the target-paper Pleiades box exactly enough.
    # Later rows are outside the Pleiades position/astrometry region.
    df.loc[2:, "GAIAEDR3_PARALLAX"] = 1.0
    df.loc[2:, "GAIAEDR3_PMRA"] = 0.0
    df.loc[2:, "GAIAEDR3_PMDEC"] = 0.0
    df.loc[2:, "VHELIO_AVG"] = 0.0
    return df


def make_astra_frame(n: int = 8) -> pd.DataFrame:
    """Build a minimal astra-format frame (SDSS-V DR19 schema).

    After ``cluster.data._convert_astra`` it yields the same canonical values
    as :func:`make_allstar_frame` (Pleiades members in rows 0-1, field after).
    """
    if n < 4:
        raise ValueError("make_astra_frame needs at least 4 rows")

    fe_h = np.zeros(n, dtype=float)
    df = pd.DataFrame(
        {
            "sdss_id": np.arange(n, dtype=np.int64),
            "sdss4_apogee_id": [f"2M{i:08d}" for i in range(n)],
            "gaia_dr3_source_id": np.arange(10**9, 10**9 + n, dtype=np.int64),
            "ra": [56.601, 56.602, 120.0, 200.0] + [10.0 + i for i in range(n - 4)],
            "dec": [24.114, 24.115, 30.0, -50.0] + [5.0 + i for i in range(n - 4)],
            "snr": np.full(n, 150.0, dtype=float),
            "flag_bad": np.zeros(n, dtype=np.int64),
            "spectrum_flags": np.zeros(n, dtype=np.int64),
            "teff": np.full(n, 4800.0, dtype=float),
            "logg": np.full(n, 4.2, dtype=float),
            "v_rad": np.full(n, 6.0, dtype=float),
            "plx": np.full(n, 7.35, dtype=float),
            "pmra": np.full(n, 20.0, dtype=float),
            "pmde": np.full(n, -45.5, dtype=float),
            "g_mag": np.full(n, 10.0, dtype=float),
            "bp_mag": np.full(n, 10.8, dtype=float),
            "rp_mag": np.full(n, 9.2, dtype=float),
            "j_mag": np.full(n, 8.5, dtype=float),
            "h_mag": np.full(n, 8.0, dtype=float),
            "k_mag": np.full(n, 7.8, dtype=float),
            "fe_h": fe_h,
            "e_fe_h": np.full(n, 0.05, dtype=float),
            "fe_h_flags": np.zeros(n, dtype=np.int64),
        }
    )
    for j, element in enumerate(ELEMENTS):
        if element == "FE_H":
            continue
        h = astra_h_col(element)
        df[h] = np.linspace(-0.5 + j * 0.05, 0.5 + j * 0.05, n) + fe_h
        df[f"e_{h}"] = np.full(n, 0.05 + j * 0.001, dtype=float)
        df[f"{h}_flags"] = np.zeros(n, dtype=np.int64)

    # Rows 2+ sit outside the Pleiades position/astrometry box.
    df.loc[2:, "plx"] = 1.0
    df.loc[2:, "pmra"] = 0.0
    df.loc[2:, "pmde"] = 0.0
    df.loc[2:, "v_rad"] = 0.0
    return df


def write_allstar_fits(path: str | Path, df: pd.DataFrame) -> Path:
    """Write ``df`` as a FITS binary table in HDU 1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns: list[fits.Column] = []
    for name in df.columns:
        values = df[name].to_numpy()
        if values.dtype.kind == "f":
            columns.append(fits.Column(name=name, array=values.astype(np.float64), format="D"))
        elif values.dtype.kind in "iu":
            columns.append(fits.Column(name=name, array=values.astype(np.int64), format="K"))
        elif values.dtype.kind == "b":
            columns.append(fits.Column(name=name, array=values.astype(np.int64), format="K"))
        else:
            columns.append(
                fits.Column(name=name, array=values.astype(str).astype("S40"), format="40A")
            )

    hdu = fits.BinTableHDU.from_columns(columns)
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def allstar_frame() -> pd.DataFrame:
    return make_allstar_frame()


@pytest.fixture
def astra_frame() -> pd.DataFrame:
    return make_astra_frame()


@pytest.fixture
def astra_fits_path(tmp_path: Path, astra_frame: pd.DataFrame) -> Path:
    return write_allstar_fits(tmp_path / "astra.fits", astra_frame)


@pytest.fixture
def allstar_fits_path(tmp_path: Path, allstar_frame: pd.DataFrame) -> Path:
    return write_allstar_fits(tmp_path / "allStar.fits", allstar_frame)


@pytest.fixture
def element_list() -> list[str]:
    return list(ELEMENTS)


@pytest.fixture
def prepared_data() -> Any:
    """Small in-memory PreparedData used by benchmark/plot tests."""
    from cluster.data import PreparedData

    rng = np.random.default_rng(7)
    n = 18
    labels = np.array(
        ["A"] * 6 + ["B"] * 6 + ["field"] * 6, dtype=object
    )
    X = np.zeros((n, 2), dtype=float)
    X[:6, 0] = rng.normal(0.0, 0.05, 6)
    X[6:12, 0] = rng.normal(10.0, 0.05, 6)
    X[12:, 0] = rng.normal(-10.0, 0.05, 6)
    X[:, 1] = rng.normal(0.0, 0.05, n)
    df = pd.DataFrame({"cluster": labels, "is_member": labels != "field"})
    return PreparedData(df=df, X=X, elements=["x", "y"])
