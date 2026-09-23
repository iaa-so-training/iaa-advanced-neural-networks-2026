"""Pandera schemas for runtime data validation.

The APOGEE allStar file is ~4 GB and arrives with a mix of native-endian
columns, sentinel values and NaNs, so validation happens in two places:

* ``ALLSTAR_SCHEMA`` checks the raw frame produced by ``data.load_allstar``
  before any quality cuts are applied.
* ``ABUNDANCE_SCHEMA`` checks the 16-element abundance block immediately
  before ``data.make_matrix`` assembles the numeric matrix (after the
  complete-case filter has removed non-finite values).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .config import ELEMENTS


def _all_finite(series: pd.Series) -> bool:
    """True when every value in the column is finite (NaN-safe)."""
    return bool(np.isfinite(series.to_numpy(dtype=float)).all())


ALLSTAR_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        # RA/DEC have a handful of NaNs (no astrometric match); the range
        # checks ignore NaN so they still pass.
        "RA": pa.Column(float, nullable=True, checks=pa.Check.in_range(0.0, 360.0, ignore_na=True)),
        "DEC": pa.Column(float, nullable=True, checks=pa.Check.in_range(-90.0, 90.0, ignore_na=True)),
        "SNR": pa.Column(float, nullable=False, checks=pa.Check.greater_than_or_equal_to(0.0)),
        "ASPCAPFLAG": pa.Column(int, nullable=False, checks=pa.Check.greater_than_or_equal_to(0)),
        "STARFLAG": pa.Column(int, nullable=False, checks=pa.Check.greater_than_or_equal_to(0)),
        # stellar params and astrometry are frequently missing for faint stars
        "TEFF": pa.Column(float, nullable=True),
        "LOGG": pa.Column(float, nullable=True),
        "VHELIO_AVG": pa.Column(float, nullable=True),
        "GAIAEDR3_PARALLAX": pa.Column(float, nullable=True),
        "GAIAEDR3_PMRA": pa.Column(float, nullable=True),
        "GAIAEDR3_PMDEC": pa.Column(float, nullable=True),
    }
)


ABUNDANCE_SCHEMA: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        element: pa.Column(
            float,
            nullable=False,
            checks=pa.Check(_all_finite, error=f"{element} contains non-finite values"),
        )
        for element in ELEMENTS
    }
)
