"""Tests for :mod:`cluster.data` using small synthetic allStar frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cluster import config
from cluster.clusters import CLUSTER_BY_NAME
from cluster.config import ELEMENTS, Settings
from cluster.data import (
    PreparedData,
    _to_native,
    apply_quality_cuts,
    complete_case,
    load_allstar,
    make_matrix,
    prepare,
    stratified_field_sample,
)
def _write_fits(path: Path, df: pd.DataFrame) -> Path:
    from astropy.io import fits

    columns = []
    for name in df.columns:
        values = df[name].to_numpy()
        if values.dtype.kind == "f":
            columns.append(fits.Column(name=name, array=values.astype(np.float64), format="D"))
        elif values.dtype.kind in "iu":
            columns.append(fits.Column(name=name, array=values.astype(np.int64), format="K"))
        else:
            columns.append(
                fits.Column(name=name, array=values.astype(str).astype("S40"), format="40A")
            )
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)])
    hdul.writeto(path, overwrite=True)
    return path


def test_to_native_converts_dtypes() -> None:
    assert _to_native(np.array([1.0], dtype=">f4")).dtype == np.float64
    assert _to_native(np.array([1], dtype=">i4")).dtype == np.int64
    result = _to_native(np.array([b"abc"], dtype="S3"))
    assert result.dtype.kind in "SU"
    assert result[0] == "abc"


def test_load_allstar_reads_synthetic_fits(
    astra_fits_path: Path, element_list: list[str]
) -> None:
    df = load_allstar(astra_fits_path, element_list)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8
    assert "C_FE" in df.columns
    assert "GAIAEDR3_PARALLAX" in df.columns


def test_load_allstar_rejects_bad_ra(tmp_path: Path, astra_frame: pd.DataFrame) -> None:
    frame = astra_frame.copy()
    frame.loc[0, "ra"] = 361.0
    path = _write_fits(tmp_path / "bad.fits", frame)
    with pytest.raises(ValueError, match="failed schema validation"):
        load_allstar(path, list(ELEMENTS))


def _quality_settings() -> Settings:
    return Settings(
        snr_min=100.0,
        require_clean_flags=True,
        dwarf_only=True,
        dwarf_logg_min=3.5,
        elements=list(ELEMENTS),
    )


def test_apply_quality_cuts_filters_snr_flags_and_dwarfs(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df.loc[0, "SNR"] = 50.0
    df.loc[1, "ASPCAPFLAG"] = 1
    df.loc[2, "STARFLAG"] = 1
    df.loc[3, "LOGG"] = 3.0
    out = apply_quality_cuts(df, _quality_settings())
    assert list(out.index) == [4, 5, 6, 7]


def test_apply_quality_cuts_can_disable_flag_and_dwarf_filters(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df.loc[0, "SNR"] = 50.0
    df.loc[1, "ASPCAPFLAG"] = 1
    df.loc[2, "LOGG"] = 3.0
    settings = Settings(
        snr_min=100.0,
        require_clean_flags=False,
        require_aspcap_flag_clean=False,
        dwarf_only=False,
        elements=list(ELEMENTS),
    )
    out = apply_quality_cuts(df, settings)
    assert 1 in out.index
    assert 2 in out.index
    assert 0 not in out.index


def test_complete_case_drops_nonfinite_and_optional_bad_flags(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df.loc[0, "C_FE"] = np.nan
    df.loc[1, "C_FE_FLAG"] = 7

    # strict complete-case (impute disabled) drops any NaN row
    settings = Settings(
        require_element_flag_clean=False,
        elements=list(ELEMENTS),
        impute_missing=False,
    )
    out = complete_case(df, settings)
    assert 0 not in out.index
    assert 1 in out.index

    settings_strict = Settings(
        require_element_flag_clean=True,
        elements=list(ELEMENTS),
        impute_missing=False,
    )
    out_strict = complete_case(df, settings_strict)
    assert 0 not in out_strict.index
    assert 1 not in out_strict.index


def test_complete_case_impute_keeps_rows_with_enough_finite_values(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df.loc[0, "C_FE"] = np.nan  # 15/16 finite -> kept
    # blank all but 2 elements -> dropped under min_finite_elements=8
    for e in ELEMENTS[2:]:
        df.loc[1, e] = np.nan

    settings = Settings(
        elements=list(ELEMENTS),
        impute_missing=True,
        min_finite_elements=8,
    )
    out = complete_case(df, settings)
    assert 0 in out.index
    assert 1 not in out.index


def test_stratified_field_sample_keeps_all_members_and_caps_field(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df["is_member"] = df.index.isin([0, 1])

    unchanged = stratified_field_sample(df, None, 42)
    assert len(unchanged) == len(df)

    capped = stratified_field_sample(df, 4, 42)
    assert len(capped) == 4
    assert capped["is_member"].sum() == 2
    assert (~capped["is_member"]).sum() == 2

    members_only = stratified_field_sample(df, 1, 42)
    assert members_only["is_member"].all()
    assert len(members_only) == 2


def test_make_matrix_standardizes_columns_to_zero_median_unit_std(
    allstar_frame: pd.DataFrame,
) -> None:
    settings = Settings(
        elements=list(ELEMENTS),
        standardize=True,
        use_element_weights=False,
        normalize_rows=False,
    )
    X = make_matrix(allstar_frame, settings)
    assert X.shape == (len(allstar_frame), len(ELEMENTS))
    assert np.allclose(np.median(X, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(X, axis=0), 1.0, atol=1e-12)


def test_make_matrix_normalizes_rows_to_unit_norm(
    allstar_frame: pd.DataFrame,
) -> None:
    settings = Settings(
        elements=list(ELEMENTS),
        standardize=True,
        use_element_weights=False,
        normalize_rows=True,
    )
    X = make_matrix(allstar_frame, settings)
    assert X.shape == (len(allstar_frame), len(ELEMENTS))
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-12)


def test_make_matrix_imputes_nan_with_column_median(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    df.loc[0, "C_FE"] = np.nan
    settings = Settings(
        elements=list(ELEMENTS),
        standardize=False,
        use_element_weights=False,
        normalize_rows=False,
        impute_missing=True,
    )
    X = make_matrix(df, settings)  # no ValueError
    assert np.isfinite(X).all()
    assert X[0, 0] == pytest.approx(np.nanmedian(df["C_FE"].to_numpy(dtype=float)))


def test_make_matrix_weights_survive_standardization(
    allstar_frame: pd.DataFrame,
) -> None:
    """Regression: weights must still matter after standardisation."""
    df = allstar_frame.copy()
    df["C_FE_ERR"] = 10.0  # make C_FE noisy
    s_unw = Settings(
        elements=list(ELEMENTS),
        standardize=True,
        use_element_weights=False,
        normalize_rows=False,
    )
    s_w = Settings(
        elements=list(ELEMENTS),
        standardize=True,
        use_element_weights=True,
        normalize_rows=False,
    )
    X_unw = make_matrix(df, s_unw)
    X_w = make_matrix(df, s_w)
    assert not np.allclose(X_unw, X_w)
    # unweighted: every column unit std
    assert np.allclose(np.std(X_unw, axis=0), 1.0)
    # weighted: noisy C_FE (col 0) down-weighted relative to N_FE (col 1)
    assert np.std(X_w[:, 0]) < np.std(X_w[:, 1])


def test_make_matrix_applies_element_weights_without_standardization(
    allstar_frame: pd.DataFrame,
) -> None:
    df = allstar_frame.copy()
    for j, element in enumerate(ELEMENTS):
        df[element] = np.arange(1, 9, dtype=float) + j
        df[f"{element}_ERR"] = np.full(8, 2.0, dtype=float)
    settings = Settings(
        elements=list(ELEMENTS),
        standardize=False,
        use_element_weights=True,
        normalize_rows=False,
    )
    X = make_matrix(df, settings)
    expected_first = (np.arange(1, 9, dtype=float)) / 2.0
    assert np.allclose(X[:, 0], expected_first)
    assert X.shape[1] == len(ELEMENTS)


def test_make_matrix_rejects_nonfinite_abundances(allstar_frame: pd.DataFrame) -> None:
    df = allstar_frame.copy()
    df.loc[0, "C_FE"] = np.nan
    settings = Settings(
        elements=list(ELEMENTS),
        standardize=False,
        impute_missing=False,
    )
    with pytest.raises(ValueError, match="abundance matrix failed schema validation"):
        make_matrix(df, settings)


def _seed_kwargs() -> dict[str, Any]:
    return {
        "seed_position_radius_deg": config.SEED_POSITION_RADIUS_DEG,
        "seed_parallax_frac": config.SEED_PARALLAX_FRAC,
        "seed_pm_tol": config.SEED_PM_TOL,
        "seed_rv_tol": config.SEED_RV_TOL,
        "n_refine_passes": config.N_REFINE_PASSES,
        "refine_sigma": config.REFINE_SIGMA,
    }


def test_prepare_end_to_end_synthetic_fits(
    astra_fits_path: Path,
) -> None:
    settings = Settings(
        snr_min=0.0,
        require_clean_flags=False,
        dwarf_only=False,
        elements=list(ELEMENTS),
        standardize=True,
        use_element_weights=False,
        require_element_flag_clean=False,
        max_stars=None,
        region_radius_deg=None,
    )
    prepared = prepare(
        astra_fits_path,
        settings,
        [CLUSTER_BY_NAME["Pleiades"]],
        **_seed_kwargs(),
    )
    assert isinstance(prepared, PreparedData)
    assert prepared.X.shape == (8, len(ELEMENTS))
    assert len(prepared.df) == 8
    assert (prepared.df["cluster"] == "Pleiades").sum() == 2
    assert (prepared.df["cluster"] == "field").sum() == 6


def test_prepare_applies_region_restriction(
    astra_fits_path: Path,
) -> None:
    settings = Settings(
        snr_min=0.0,
        require_clean_flags=False,
        dwarf_only=False,
        elements=list(ELEMENTS),
        standardize=False,
        use_element_weights=False,
        require_element_flag_clean=False,
        max_stars=None,
        region_radius_deg=1.0,
    )
    prepared = prepare(
        astra_fits_path,
        settings,
        [CLUSTER_BY_NAME["Pleiades"]],
        **_seed_kwargs(),
    )
    assert len(prepared.df) == 2
    assert (prepared.df["cluster"] == "Pleiades").all()


def test_prepared_data_dataclass_fields(allstar_frame: pd.DataFrame) -> None:
    X = np.ones((len(allstar_frame), 2))
    prepared = PreparedData(df=allstar_frame, X=X, elements=["a", "b"])
    assert prepared.df is allstar_frame
    assert prepared.X is X
    assert prepared.elements == ["a", "b"]
