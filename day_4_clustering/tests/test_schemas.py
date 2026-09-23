"""Direct pandera-schema validation tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from cluster.config import ELEMENTS
from cluster.schemas import ABUNDANCE_SCHEMA, ALLSTAR_SCHEMA


def test_allstar_schema_accepts_clean_frame(allstar_frame: pd.DataFrame) -> None:
    validated = ALLSTAR_SCHEMA.validate(allstar_frame, lazy=True)
    assert len(validated) == len(allstar_frame)


def test_allstar_schema_rejects_ra_out_of_range(allstar_frame: pd.DataFrame) -> None:
    frame = allstar_frame.copy()
    frame.loc[0, "RA"] = 361.0
    with pytest.raises(SchemaErrors):
        ALLSTAR_SCHEMA.validate(frame, lazy=True)


def test_allstar_schema_rejects_null_snr(allstar_frame: pd.DataFrame) -> None:
    frame = allstar_frame.copy()
    frame.loc[0, "SNR"] = np.nan
    with pytest.raises(SchemaErrors):
        ALLSTAR_SCHEMA.validate(frame, lazy=True)


def test_allstar_schema_allows_nullable_astrometry(allstar_frame: pd.DataFrame) -> None:
    frame = allstar_frame.copy()
    frame.loc[0, "TEFF"] = np.nan
    frame.loc[0, "GAIAEDR3_PARALLAX"] = np.nan
    validated = ALLSTAR_SCHEMA.validate(frame, lazy=True)
    assert pd.isna(validated.loc[0, "TEFF"])


def test_abundance_schema_accepts_finite_elements(allstar_frame: pd.DataFrame) -> None:
    abundance = allstar_frame[ELEMENTS]
    validated = ABUNDANCE_SCHEMA.validate(abundance, lazy=True)
    assert validated.shape[1] == len(ELEMENTS)


def test_abundance_schema_rejects_nonfinite_values(allstar_frame: pd.DataFrame) -> None:
    abundance = allstar_frame[ELEMENTS].copy()
    abundance.loc[0, "C_FE"] = np.nan
    with pytest.raises(SchemaErrors):
        ABUNDANCE_SCHEMA.validate(abundance, lazy=True)
