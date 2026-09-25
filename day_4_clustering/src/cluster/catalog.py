"""External membership catalogue (Simbad) used as the referee for metrics.

Queries Simbad live via ``astroquery`` + ``pyvo`` (TAP) and cross-matches the
returned member stars against the allStar frame by sky position. Results are
cached to CSV so repeat runs (and the workshop) do not re-hit the network.

The Simbad ``h_link`` table ("hierarchy of membership measure") links each
star to its parent cluster with a ``membership`` probability (0-100) from the
literature. We keep distinct stars whose *maximum* probability across all
citing papers reaches ``SIMBAD_MEMBERSHIP_MIN``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .clusters import Cluster
from .config import Settings


def resolve_simbad_id(name: str) -> str:
    """Resolve a cluster name to its canonical Simbad ``main_id``."""
    from astroquery.simbad import Simbad

    result = Simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"Simbad could not resolve cluster name {name!r}")
    return str(result[0]["main_id"])


def _cache_path(cache_dir: str | Path, simbad_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in simbad_id)
    return Path(cache_dir) / f"simbad_{safe}.csv"


def query_members(
    simbad_id: str,
    min_membership: int,
    cache_dir: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    """Distinct Simbad member stars (RA, Dec, membership) for a cluster.

    Columns: ``ra``, ``dec`` (deg), ``membership`` (max across papers).
    Cached to ``cache_dir``; pass ``force=True`` to re-query.
    """
    cache = _cache_path(cache_dir, simbad_id)
    if cache.exists() and not force:
        return pd.read_csv(cache)

    from pyvo.dal import TAPService

    svc = TAPService("https://simbad.u-strasbg.fr/simbad/sim-tap")
    query = f"""
    SELECT b.main_id AS main_id, b.ra AS ra, b.dec AS dec,
           max(l.membership) AS membership
    FROM basic b
    JOIN h_link l ON b.oid = l.child
    JOIN basic p ON l.parent = p.oid
    WHERE p.main_id = '{simbad_id}'
    GROUP BY b.oid, b.main_id, b.ra, b.dec
    HAVING max(l.membership) >= {int(min_membership)}
    """
    table = svc.search(query).to_table()
    members = table.to_pandas()
    cache.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(cache, index=False)
    return members


def catalog_members(
    df: pd.DataFrame,
    cluster: Cluster,
    *,
    min_membership: int,
    crossmatch_arcsec: float,
    cache_dir: str | Path,
    force: bool = False,
) -> pd.Series:
    """Boolean Series: True where an allStar row matches a Simbad member.

    Cross-match is positional (nearest sky neighbour within
    ``crossmatch_arcsec``), independent of the kinematic labels.
    """
    simbad_id = resolve_simbad_id(cluster.name)
    members = query_members(simbad_id, min_membership, cache_dir, force)
    # the Simbad TAP result can carry null coordinates for a few rows
    members = members[
        np.isfinite(members["ra"].to_numpy(dtype=float))
        & np.isfinite(members["dec"].to_numpy(dtype=float))
    ]

    c_df = SkyCoord(
        np.asarray(df["RA"]) * u.deg, np.asarray(df["DEC"]) * u.deg,
    )
    c_mem = SkyCoord(
        members["ra"].to_numpy() * u.deg, members["dec"].to_numpy() * u.deg,
    )
    idx, sep2d, _ = c_mem.match_to_catalog_sky(c_df)
    mask = np.zeros(len(df), dtype=bool)
    good = sep2d.arcsec <= crossmatch_arcsec
    if good.any():
        mask[np.asarray(idx)[good]] = True
    return pd.Series(mask, index=df.index, name=f"simbad_{cluster.name}")


def attach_referee(
    df: pd.DataFrame,
    clusters: list[Cluster],
    settings: Settings,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Add a ``referee`` column (Simbad catalogue membership) to ``df``.

    The referee is the external ground truth the benchmark scores against.
    Falls back to the in-pipeline ``cluster`` column if Simbad is
    unreachable (so the benchmark still runs offline).
    """
    referee = np.full(len(df), "field", dtype=object)
    try:
        for cluster in clusters:
            m = catalog_members(
                df, cluster,
                min_membership=settings.simbad_membership_min,
                crossmatch_arcsec=settings.simbad_crossmatch_arcsec,
                cache_dir=settings.simbad_cache_dir,
                force=force,
            )
            referee[m.to_numpy()] = cluster.name
    except Exception as exc:  # network / astroquery unavailable
        print(f"⚠ Simbad referee unavailable ({exc}); using in-pipeline labels")
        referee = df["cluster"].to_numpy(dtype=object)
    out = df.copy()
    out["referee"] = referee
    return out


def membership_masks_for(
    df: pd.DataFrame,
    cluster: Cluster,
    X: np.ndarray,
    settings: Settings,
    *,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """The three membership sources for one cluster, parameters from settings."""
    from . import config

    return membership_masks(
        df, cluster, X,
        min_membership=settings.simbad_membership_min,
        crossmatch_arcsec=settings.simbad_crossmatch_arcsec,
        cache_dir=settings.simbad_cache_dir,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
        combined_chem_sigma=settings.combined_chem_sigma,
        combined_kin_sigma=settings.combined_kin_sigma,
        force=force,
    )


def membership_masks(
    df: pd.DataFrame,
    cluster: Cluster,
    X: np.ndarray,
    *,
    min_membership: int,
    crossmatch_arcsec: float,
    cache_dir: str | Path,
    seed_position_radius_deg: float,
    seed_parallax_frac: float,
    seed_pm_tol: float,
    seed_rv_tol: float,
    n_refine_passes: int,
    refine_sigma: float,
    combined_chem_sigma: float,
    combined_kin_sigma: float,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """All three membership sources for one cluster, as boolean arrays.

    Returns ``{"catalog": ..., "kinematic": ..., "combined": ...}``, each a
    boolean array aligned to ``df``. "catalog" is the external Simbad
    referee; "kinematic" and "combined" are the in-pipeline methods.
    """
    from .membership import combined_members, kinematic_members

    kin = kinematic_members(
        df, cluster,
        seed_position_radius_deg, seed_parallax_frac,
        seed_pm_tol, seed_rv_tol, n_refine_passes, refine_sigma,
    ).to_numpy()
    comb = combined_members(
        df, cluster, X,
        seed_position_radius_deg, seed_parallax_frac,
        seed_pm_tol, seed_rv_tol, n_refine_passes, refine_sigma,
        combined_chem_sigma, combined_kin_sigma,
    ).to_numpy()
    cat = catalog_members(
        df, cluster,
        min_membership=min_membership,
        crossmatch_arcsec=crossmatch_arcsec,
        cache_dir=cache_dir,
        force=force,
    ).to_numpy()
    return {"catalog": cat, "kinematic": kin, "combined": comb}


def hr_cell(df_hr: pd.DataFrame, cluster_name: str, method: str, settings: Settings) -> Any:
    """Full interactive-HR panel for the notebook (region cut + masks + plot).

    Keeps the heavy computation out of the notebook cell so the cell itself
    stays a one-liner (no nested functions, no local-variable collisions).
    """
    from .clusters import CLUSTER_BY_NAME
    from .data import complete_case, make_matrix
    from .membership import angular_separation
    from .plots import hr_interactive

    cluster = CLUSTER_BY_NAME[cluster_name]
    sep = angular_separation(
        df_hr["RA"].to_numpy(dtype=float), df_hr["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    df = df_hr[sep <= 30.0].copy()
    df = complete_case(df, settings)
    X = make_matrix(df, settings)
    masks = membership_masks_for(df, cluster, X, settings)
    return hr_interactive(df, masks, cluster.name, highlight=method)


def cluster_panels_cell(df_hr: pd.DataFrame, cluster_name: str, method: str, settings: Settings) -> Any:
    """Three-panel cluster diagnostic (Kiel / 2MASS / Gaia) for the notebook."""
    from .clusters import CLUSTER_BY_NAME
    from .data import complete_case, make_matrix
    from .gaia import gaia_members, query_gaia_region
    from .membership import angular_separation
    from .plots import cluster_panels

    cluster = CLUSTER_BY_NAME[cluster_name]
    sep = angular_separation(
        df_hr["RA"].to_numpy(dtype=float), df_hr["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    region = df_hr[sep <= 30.0].copy()
    region = complete_case(region, settings)
    X = make_matrix(region, settings)
    masks = membership_masks_for(region, cluster, X, settings)
    apogee_mask = masks[method]

    gaia = query_gaia_region(cluster, 0.5)
    gaia_mask = gaia_members(gaia, cluster).to_numpy()

    return cluster_panels(region, gaia, cluster.name, apogee_mask, gaia_mask)
