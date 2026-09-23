"""Kinematic ground-truth membership (labels for the benchmark).

Follows the target paper's Appendix C: cluster members are selected with
cuts in position, radial velocity and proper motion, then refined with
iterative robust sigma clipping (same idea as Spina et al. 2025).

Kinematics come from the Gaia EDR3 cross-match already embedded in the
APOGEE allStar file, so no external catalogue is needed. Kinematics are
independent of the chemical abundances — the labels never leak into the
embedding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clusters import Cluster


def angular_separation(ra1: np.ndarray, dec1: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    """Great-circle angular distance (degrees) from a fixed point."""
    dra = np.radians(ra1 - ra0)
    d1 = np.radians(dec1)
    d2 = np.radians(dec0)
    cos_c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(dra)
    return np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))


def _robust_scale(values: np.ndarray, floor: float) -> float:
    """MAD-based robust sigma, floored so a tight cluster never collapses."""
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return max(1.4826 * mad, floor)


def _pleiades_members(df: pd.DataFrame) -> np.ndarray:
    """Exact target-paper (Appendix C) box, upgraded to Gaia parallax.

    RV 5-8 km/s, PM amplitude 45-55 mas/yr, PM angle 2.70-2.80 rad
    (atan2(pmra, pmdec)), parallax 7.0-7.7 mas. Position kept loose
    (<= 10 deg): the astrometric cuts already guarantee membership.
    """
    pmra = df["GAIAEDR3_PMRA"].to_numpy(dtype=float)
    pmdec = df["GAIAEDR3_PMDEC"].to_numpy(dtype=float)
    plx = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    rv = df["VHELIO_AVG"].to_numpy(dtype=float)
    pmamp = np.hypot(pmra, pmdec)
    phi = np.arctan2(pmra, pmdec)
    return (
        (plx > 7.0) & (plx < 7.7)
        & (pmamp > 45.0) & (pmamp < 55.0)
        & (phi > 2.70) & (phi < 2.80)
        & (rv > 5.0) & (rv < 8.0)
    )


def kinematic_members(
    df: pd.DataFrame,
    cluster: Cluster,
    seed_position_radius_deg: float,
    seed_parallax_frac: float,
    seed_pm_tol: float,
    seed_rv_tol: float,
    n_refine_passes: int,
    refine_sigma: float,
) -> pd.Series:
    """Boolean Series marking members of ``cluster``.

    Pleiades uses the paper's exact Appendix C box. Every other cluster
    uses a seed box (position + RV + PM + loose parallax) followed by
    ``n_refine_passes`` rounds of robust sigma clipping on
    (parallax, pmra, pmdec, rv).
    """
    if cluster.name == "Pleiades":
        mask = _pleiades_members(df)
        return pd.Series(mask, index=df.index, name=f"member_{cluster.name}")

    plx = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    pmra = df["GAIAEDR3_PMRA"].to_numpy(dtype=float)
    pmdec = df["GAIAEDR3_PMDEC"].to_numpy(dtype=float)
    rv = df["VHELIO_AVG"].to_numpy(dtype=float)

    sep = angular_separation(
        df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    plx0 = cluster.parallax_mas

    mask = (
        (sep <= seed_position_radius_deg)
        & np.isfinite(plx) & np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(rv)
        & (np.abs(plx - plx0) <= max(0.3, seed_parallax_frac * plx0))
        & (np.hypot(pmra - cluster.pmra, pmdec - cluster.pmdec) <= seed_pm_tol)
        & (np.abs(rv - cluster.rv) <= seed_rv_tol)
    )

    if mask.sum() < 2:
        return pd.Series(mask, index=df.index, name=f"member_{cluster.name}")

    for _ in range(n_refine_passes):
        for col, floor in (
            (plx, 0.10), (pmra, 0.5), (pmdec, 0.5), (rv, 2.0),
        ):
            med = np.nanmedian(col[mask])
            scale = _robust_scale(col[mask], floor)
            mask &= np.abs(col - med) <= refine_sigma * scale

    return pd.Series(mask, index=df.index, name=f"member_{cluster.name}")


def combined_members(
    df: pd.DataFrame,
    cluster: Cluster,
    X: np.ndarray,
    seed_position_radius_deg: float,
    seed_parallax_frac: float,
    seed_pm_tol: float,
    seed_rv_tol: float,
    n_refine_passes: int,
    refine_sigma: float,
    combined_chem_sigma: float,
    combined_kin_sigma: float,
) -> pd.Series:
    """Kinematic core expanded by chemical similarity, gated by kinematics.

    The strict kinematic members anchor the group. Every other star joins if
    it is (a) chemically close to the core centroid in the abundance matrix
    ``X`` (within ``combined_chem_sigma`` robust std of the core's internal
    spread) AND (b) loosely kinematic (parallax / pmra / pmdec / rv within
    ``combined_kin_sigma`` robust std of the core).

    This is the "kinematics + abundances" combination: two independent
    signals must both agree before a star is added.
    """
    kin = kinematic_members(
        df, cluster,
        seed_position_radius_deg, seed_parallax_frac,
        seed_pm_tol, seed_rv_tol, n_refine_passes, refine_sigma,
    ).to_numpy()
    if kin.sum() < 2:
        return pd.Series(kin, index=df.index, name=f"combined_{cluster.name}")

    plx = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    pmra = df["GAIAEDR3_PMRA"].to_numpy(dtype=float)
    pmdec = df["GAIAEDR3_PMDEC"].to_numpy(dtype=float)
    rv = df["VHELIO_AVG"].to_numpy(dtype=float)

    # loose kinematic gate: within N robust std of the core in all four
    loose = np.ones(len(df), dtype=bool)
    for col, floor in ((plx, 0.10), (pmra, 0.5), (pmdec, 0.5), (rv, 2.0)):
        med = np.nanmedian(col[kin])
        scale = _robust_scale(col[kin], floor)
        loose &= np.abs(col - med) <= combined_kin_sigma * scale

    # chemical gate: distance to the core centroid in C-space
    centroid = np.nanmedian(X[kin], axis=0)
    dist = np.linalg.norm(X - centroid, axis=1)
    core_dist = dist[kin]
    d_med = np.nanmedian(core_dist)
    d_scale = _robust_scale(core_dist, 1e-6)
    chem = dist <= d_med + combined_chem_sigma * d_scale

    combined = loose & chem
    combined[kin] = True  # the core is a member by construction
    return pd.Series(combined, index=df.index, name=f"combined_{cluster.name}")


def label_clusters(
    df: pd.DataFrame,
    clusters: list[Cluster],
    *,
    seed_position_radius_deg: float,
    seed_parallax_frac: float,
    seed_pm_tol: float,
    seed_rv_tol: float,
    n_refine_passes: int,
    refine_sigma: float,
    membership_method: str = "kinematic",
    X: np.ndarray | None = None,
    combined_chem_sigma: float = 3.0,
    combined_kin_sigma: float = 3.0,
) -> pd.DataFrame:
    """Add ``cluster`` (name or 'field') and ``is_member`` columns to ``df``.

    ``membership_method`` selects the labelling strategy:

    - ``"kinematic"`` — the strict Gaia + RV box / sigma-clip members.
    - ``"combined"`` — kinematic core expanded by abundance agreement
      (requires ``X``, the abundance matrix aligned to ``df``).
    """
    out = df.copy()
    names = np.full(len(df), "field", dtype=object)
    for cluster in clusters:
        if membership_method == "combined":
            assert X is not None, "combined membership requires the abundance matrix X"
            m = combined_members(
                out, cluster, X,
                seed_position_radius_deg, seed_parallax_frac,
                seed_pm_tol, seed_rv_tol, n_refine_passes, refine_sigma,
                combined_chem_sigma, combined_kin_sigma,
            )
        else:
            m = kinematic_members(
                out, cluster,
                seed_position_radius_deg, seed_parallax_frac,
                seed_pm_tol, seed_rv_tol, n_refine_passes, refine_sigma,
            )
        names[m.to_numpy()] = cluster.name
    out["cluster"] = names
    out["is_member"] = names != "field"
    return out
