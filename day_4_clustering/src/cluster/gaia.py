"""Gaia DR3 photometry — deep CMDs for isochrone age fitting.

APOGEE's SNR cut only reaches bright giants, so globular ages are
unconstrained (the main sequence / turnoff is too faint). This module queries
Gaia DR3 directly (via ``astroquery``) for full-depth photometry around a
cluster, so the isochrone fit sees the main sequence and the age becomes
constrained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clusters import Cluster

_GAIA_COLUMNS = [
    "source_id", "ra", "dec",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "parallax", "pmra", "pmdec",
]


def query_gaia_region(
    cluster: Cluster,
    radius_deg: float,
    cache_dir: str | Path = "data/gaia",
    max_rows: int = 20_000,
) -> pd.DataFrame:
    """Query Gaia DR3 around ``cluster`` (G/BP/RP + parallax + PM), cached.

    Columns follow Gaia names: ``phot_g_mean_mag``, ``phot_bp_mean_mag``,
    ``phot_rp_mean_mag``, ``parallax``, ``pmra``, ``pmdec``.
    """
    cache_dir = Path(cache_dir)
    cache = cache_dir / f"gaia_{cluster.name.replace(' ', '_')}_{radius_deg}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    from astroquery.gaia import Gaia

    cols = ", ".join(_GAIA_COLUMNS)
    query = f"""
    SELECT TOP {int(max_rows)} {cols}
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {cluster.ra_deg}, {cluster.dec_deg}, {radius_deg})
    )
    AND phot_g_mean_mag IS NOT NULL
    AND phot_bp_mean_mag IS NOT NULL
    AND phot_rp_mean_mag IS NOT NULL
    AND parallax IS NOT NULL AND parallax > 0
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"↓ querying Gaia DR3 ({radius_deg}° around {cluster.name}) ...")
    table = Gaia.launch_job(query).get_results()
    assert table is not None
    out = table.to_pandas()
    out.to_csv(cache, index=False)
    return out


def gaia_members(
    df: pd.DataFrame,
    cluster: Cluster,
    *,
    pm_tol: float = 2.0,
) -> pd.Series:
    """Gaia-only kinematic membership: a fixed proper-motion cut.

    Globular proper-motion dispersions are < 0.5 mas/yr, so a fixed cut of
    ``pm_tol`` around the cluster's PM selects members cleanly. A sigma-clip
    anchored to the data median drifts toward the field's PM, so we anchor
    to the cluster's published PM instead.
    """
    pmra = df["pmra"].to_numpy(dtype=float)
    pmdec = df["pmdec"].to_numpy(dtype=float)
    mask = (
        np.isfinite(pmra) & np.isfinite(pmdec)
        & (np.hypot(pmra - cluster.pmra, pmdec - cluster.pmdec) <= pm_tol)
    )
    return pd.Series(mask, index=df.index, name=f"gaia_{cluster.name}")
