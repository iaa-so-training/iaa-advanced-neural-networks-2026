"""Load the SDSS-V DR19 Astra ASPCAP catalog, apply quality cuts, build the C-space matrix.

The single DR17 allStar (ASPCAP synspec) is replaced by
``astraAllStarASPCAP-0.6.0.fits.gz``: stellar params (``teff``/``logg``/
``fe_h``), ``[X/H]`` abundances (lowercase ``_h`` columns), Gaia/2MASS
photometry, ``ra``/``dec``/``plx``/``pmra``/``pmde``/``v_rad``, and quality
flags (``calibrated_flags`` + ``spectrum_flags``).

The loader converts ``[X/H]`` to ``[X/Fe]`` (``x_h - fe_h``) and emits the
canonical internal schema, so the rest of the pipeline is release-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pandera.errors import SchemaErrors

from .clusters import Cluster
from .config import Settings, astra_h_col
from .schemas import ABUNDANCE_SCHEMA, ALLSTAR_SCHEMA


_ASTRA_BASE = [
    "sdss_id", "sdss4_apogee_id", "gaia_dr3_source_id", "ra", "dec", "snr",
    "flag_bad", "spectrum_flags",
    "teff", "logg", "v_rad", "plx", "pmra", "pmde",
    "g_mag", "bp_mag", "rp_mag", "j_mag", "h_mag", "k_mag",
    "fe_h", "e_fe_h", "fe_h_flags",
]


def _astra_cols(elements: list[str]) -> list[str]:
    cols = list(_ASTRA_BASE)
    for e in elements:
        if e == "FE_H":
            continue
        h = astra_h_col(e)
        cols += [h, f"e_{h}", f"{h}_flags"]
    return cols


def _table_hdu(hdul: fits.HDUList) -> fits.FITS_rec:
    """Return the largest non-empty binary-table HDU's data.

    SDSS-V catalogs ship an empty summary table in HDU 1 with the real data
    in a later HDU, so the first BinTableHDU is not always the right one.
    """
    best: fits.FITS_rec | None = None
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
            if best is None or len(hdu.data) > len(best):
                best = hdu.data
    if best is None:
        raise ValueError("FITS file has no non-empty binary table HDU")
    return best


def _to_native(arr: np.ndarray) -> np.ndarray:
    """FITS columns are big-endian; pandas boolean indexing needs native order."""
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return arr.astype(np.float64)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind in "SU":
        return arr.astype(str)
    return arr


def _flag_col(series: pd.Series) -> np.ndarray:
    """Cast a bitmask flag column to int64, NaN-safe (NaN -> 1e9, non-clean)."""
    arr = series.to_numpy()
    if arr.dtype.kind == "f":
        return np.nan_to_num(arr.astype(float), nan=1e9).astype(np.int64)
    return arr.astype(np.int64)


def _convert_astra(astra_df: pd.DataFrame, elements: list[str]) -> pd.DataFrame:
    """Convert the Astra frame to [X/Fe] and the canonical internal schema."""
    df = astra_df
    fe_h = df["fe_h"].to_numpy(dtype=float)
    e_fe_h = df["e_fe_h"].to_numpy(dtype=float)

    out: dict[str, np.ndarray] = {
        "APOGEE_ID": _to_native(df["sdss4_apogee_id"].to_numpy()),
        "GAIAEDR3_SOURCE_ID": _to_native(df["gaia_dr3_source_id"].to_numpy()),
        "RA": df["ra"].to_numpy(dtype=float),
        "DEC": df["dec"].to_numpy(dtype=float),
        "SNR": df["snr"].to_numpy(dtype=float),
        "ASPCAPFLAG": _flag_col(df["flag_bad"]),
        "STARFLAG": _flag_col(df["spectrum_flags"]),
        "TEFF": df["teff"].to_numpy(dtype=float),
        "LOGG": df["logg"].to_numpy(dtype=float),
        "VHELIO_AVG": df["v_rad"].to_numpy(dtype=float),
        "GAIAEDR3_PARALLAX": df["plx"].to_numpy(dtype=float),
        "GAIAEDR3_PMRA": df["pmra"].to_numpy(dtype=float),
        "GAIAEDR3_PMDEC": df["pmde"].to_numpy(dtype=float),
        "GAIAEDR3_PHOT_G_MEAN_MAG": df["g_mag"].to_numpy(dtype=float),
        "GAIAEDR3_PHOT_BP_MEAN_MAG": df["bp_mag"].to_numpy(dtype=float),
        "GAIAEDR3_PHOT_RP_MEAN_MAG": df["rp_mag"].to_numpy(dtype=float),
        "J": df["j_mag"].to_numpy(dtype=float),
        "H": df["h_mag"].to_numpy(dtype=float),
        "K": df["k_mag"].to_numpy(dtype=float),
        "FE_H": fe_h,
        "FE_H_ERR": e_fe_h,
        "FE_H_FLAG": _flag_col(df["fe_h_flags"]),
    }
    for e in elements:
        if e == "FE_H":
            continue
        h = astra_h_col(e)
        xh = df[h].to_numpy(dtype=float)
        out[e] = xh - fe_h
        ex = df[f"e_{h}"].to_numpy(dtype=float)
        out[f"{e}_ERR"] = np.sqrt(ex ** 2 + e_fe_h ** 2)
        out[f"{e}_FLAG"] = _flag_col(df[f"{h}_flags"])

    # Galactic coordinates (Astra ships no GLON/GLAT).
    c = SkyCoord(ra=out["RA"], dec=out["DEC"], unit="deg")
    gal = c.galactic
    lon = gal.l if gal is not None else None
    lat = gal.b if gal is not None else None
    if lon is None or lat is None:  # pragma: no cover — never for finite ICRS
        raise ValueError("galactic coordinate transform failed")
    out["GLON"] = np.asarray(lon.deg, dtype=float)
    out["GLAT"] = np.asarray(lat.deg, dtype=float)

    return pd.DataFrame(out)


def load_allstar(path: str | Path, elements: list[str]) -> pd.DataFrame:
    """Read the Astra ASPCAP catalog into the canonical schema.

    Abundances are converted from [X/H] to [X/Fe] via ``x_h - fe_h``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `cluster download` first.")

    with fits.open(path, memmap=True) as hdul:
        table = _table_hdu(hdul)
        astra_df = pd.DataFrame({
            c: _to_native(table[c]) for c in _astra_cols(elements)
        })

    df = _convert_astra(astra_df, elements)
    try:
        return ALLSTAR_SCHEMA.validate(df, lazy=True)
    except SchemaErrors as exc:
        raise ValueError(f"astra catalog failed schema validation:\n{exc}") from exc


def apply_quality_cuts(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """SNR / flag / dwarf cuts. Returns a filtered copy (index preserved)."""
    mask = df["SNR"].to_numpy(dtype=float) >= settings.snr_min
    if settings.require_clean_flags:
        mask &= (df["STARFLAG"].to_numpy(dtype=np.int64) == 0)
    if settings.require_aspcap_flag_clean:
        mask &= (df["ASPCAPFLAG"].to_numpy(dtype=np.int64) == 0)
    if settings.dwarf_only:
        mask &= (df["LOGG"].to_numpy(dtype=float) >= settings.dwarf_logg_min)
    return df[mask]


def complete_case(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Drop rows with non-finite abundances (and bad element flags).

    With ``settings.impute_missing``, rows keep at least
    ``settings.min_finite_elements`` finite abundances and the remaining NaN
    are imputed later in :func:`make_matrix` — this keeps metal-poor
    globulars (M 15 / M 92) whose weak lines go undetected.
    """
    if settings.impute_missing:
        n_finite = np.zeros(len(df), dtype=int)
        for e in settings.elements:
            n_finite += np.isfinite(df[e].to_numpy(dtype=float)).astype(int)
        keep = n_finite >= settings.min_finite_elements
    else:
        keep = np.ones(len(df), dtype=bool)
        for e in settings.elements:
            keep &= np.isfinite(df[e].to_numpy(dtype=float))
    if settings.require_element_flag_clean:
        for e in settings.elements:
            keep &= (df[f"{e}_FLAG"].to_numpy(dtype=np.int64) == 0)
    return df[keep]


def stratified_field_sample(
    df: pd.DataFrame, max_stars: int | None, random_state: int
) -> pd.DataFrame:
    """Cap the *field* at ``max_stars`` while keeping every cluster member."""
    if max_stars is None or len(df) <= max_stars:
        return df
    members = df[df["is_member"]]
    field = df[~df["is_member"]]
    n_field = max_stars - len(members)
    if n_field <= 0:  # members alone exceed the cap; keep them all
        return members
    field = field.sample(n=n_field, random_state=random_state)
    return pd.concat([members, field], ignore_index=True)


def make_matrix(df: pd.DataFrame, settings: Settings) -> np.ndarray:
    """Assemble the (imputed / standardised / weighted / normalised) C-space.

    Order: impute NaN (column median) -> validate -> standardise (zero
    median, unit std) -> weight by 1/sigma -> L2-normalise rows (optional).

    Weights are applied *after* standardisation so they are not undone by
    the per-column rescale.
    """
    elements = settings.elements
    X = df[elements].to_numpy(dtype=float)

    if settings.impute_missing:
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isfinite(X), X, med[np.newaxis, :])

    try:
        validated = ABUNDANCE_SCHEMA.validate(pd.DataFrame(X, columns=elements), lazy=True)
    except SchemaErrors as exc:
        raise ValueError(f"abundance matrix failed schema validation:\n{exc}") from exc
    X = validated.to_numpy(dtype=float)

    if settings.standardize:
        med = np.nanmedian(X, axis=0)
        scale = np.nanstd(X, axis=0)
        scale[scale == 0] = 1.0
        X = (X - med) / scale

    if settings.use_element_weights:
        for i, e in enumerate(elements):
            sigma = np.nanmedian(df[f"{e}_ERR"].to_numpy(dtype=float))
            if np.isfinite(sigma) and sigma > 0:
                X[:, i] /= sigma

    if settings.normalize_rows:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    return X


@dataclass
class PreparedData:
    """Clean abundance matrix + the frame it was built from (same row order)."""

    df: pd.DataFrame
    X: np.ndarray
    elements: list[str]


def prepare(
    allstar_path: str | Path,
    settings: Settings,
    clusters: list[Cluster],
    *,
    seed_position_radius_deg: float,
    seed_parallax_frac: float,
    seed_pm_tol: float,
    seed_rv_tol: float,
    n_refine_passes: int,
    refine_sigma: float,
) -> PreparedData:
    """load -> cuts -> region -> complete-case -> label -> sample -> matrix.

    The abundance matrix is built before labelling so that the "combined"
    membership method (kinematics + chemistry) can use it, then rebuilt on
    the sampled frame so rows stay aligned.
    """
    from .membership import angular_separation, label_clusters

    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)

    if clusters:
        if settings.region_scaled:
            # per-cluster region, scaled to the cluster angular diameter
            in_region = np.zeros(len(df), dtype=bool)
            for c in clusters:
                sep = angular_separation(
                    df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
                    c.ra_deg, c.dec_deg,
                )
                in_region |= sep <= c.region_deg
            df = df[in_region]
        elif settings.region_radius_deg is not None:
            in_region = np.zeros(len(df), dtype=bool)
            for c in clusters:
                sep = angular_separation(
                    df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
                    c.ra_deg, c.dec_deg,
                )
                in_region |= sep <= settings.region_radius_deg
            df = df[in_region]

    df = complete_case(df, settings)
    X_full = make_matrix(df, settings)  # for the combined membership method
    df = label_clusters(
        df, clusters,
        seed_position_radius_deg=seed_position_radius_deg,
        seed_parallax_frac=seed_parallax_frac,
        seed_pm_tol=seed_pm_tol,
        seed_rv_tol=seed_rv_tol,
        n_refine_passes=n_refine_passes,
        refine_sigma=refine_sigma,
        membership_method=settings.membership_method,
        X=X_full if settings.membership_method == "combined" else None,
        combined_chem_sigma=settings.combined_chem_sigma,
        combined_kin_sigma=settings.combined_kin_sigma,
    )

    df = stratified_field_sample(df, settings.max_stars, settings.random_state)
    X = make_matrix(df, settings)
    return PreparedData(df=df, X=X, elements=list(settings.elements))
