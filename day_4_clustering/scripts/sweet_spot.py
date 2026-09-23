"""Sweet-spot demonstration: MS age (GALAH) + red-clump distance (APOGEE).

For a southern intermediate-age cluster:
  1. APOGEE allStar -> kinematic members -> member giants -> red-clump dm (J-K)
  2. GALAH (combined parquet) -> kinematic members -> main-sequence stars
  3. combined isochrone fit (APOGEE members + GALAH members, BP-RP only)
     with the clump dm as a tight prior (+-0.2)
  4. compare age uncertainty (std_loga) against APOGEE-only

If the hypothesis holds, the combined fit narrows the age because the GALAH
main-sequence/turnoff breaks the age-metallicity degeneracy.

Usage:
    uv run python scripts/sweet_spot.py [--clusters "NGC 2243,Collinder 261"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.clusters import CLUSTERS
from cluster.data import apply_quality_cuts, load_allstar
from cluster.isochrone import fit_isochrone
from cluster.literature import fetch_literature
from cluster.membership import angular_separation, kinematic_members


def _clump_dm(giants: pd.DataFrame, feh: float) -> float:
    giants = giants[np.isfinite(giants["J"]) & np.isfinite(giants["K"])]
    if len(giants) < 20:
        return float("nan")
    jk = (giants["J"] - giants["K"]).to_numpy(dtype=float)
    clump = giants[(jk >= 0.5) & (jk <= 0.85)]
    if len(clump) < 10:
        return float("nan")
    k_clump = float(np.median(clump["K"].to_numpy(dtype=float)))
    m_k = -1.6 + 0.25 * feh if np.isfinite(feh) else -1.6
    return k_clump - m_k


def _robust(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, max(1.4826 * mad, 1e-6)


def _ms_members(galah: pd.DataFrame, apogee_members: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
    """GALAH main-sequence members around the APOGEE kinematic centroid.

    Parallax is poorly constrained at distance, so it gets a loose cut
    (max(3 sigma, 0.5 mas)); RV and PM use the APOGEE centroid with a
    floored robust scale.
    """
    rv0, srv = _robust(apogee_members["VHELIO_AVG"].to_numpy(dtype=float))
    pmra0, spmra = _robust(apogee_members["GAIAEDR3_PMRA"].to_numpy(dtype=float))
    pmdec0, spmdec = _robust(apogee_members["GAIAEDR3_PMDEC"].to_numpy(dtype=float))
    plx0, splx = _robust(apogee_members["GAIAEDR3_PARALLAX"].to_numpy(dtype=float))
    grv = galah["VHELIO_AVG"].to_numpy(dtype=float)
    gpmra = galah["GAIAEDR3_PMRA"].to_numpy(dtype=float)
    gpmdec = galah["GAIAEDR3_PMDEC"].to_numpy(dtype=float)
    gplx = galah["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    mask = (
        (np.abs(grv - rv0) <= sigma * max(srv, 2.0))
        & (np.hypot(gpmra - pmra0, gpmdec - pmdec0) <= sigma * max(np.hypot(spmra, spmdec), 0.5))
        & (gplx > 0)
        & (np.abs(gplx - plx0) <= max(sigma * splx, 0.5))
    )
    return galah[mask]


def _fit(members: pd.DataFrame, dm_center: float | None, dm_width: float,
         grid: str) -> dict[str, float]:
    fit = fit_isochrone(members, n_walkers=8, n_steps=400, grid=grid,
                        dm_center=dm_center, dm_width=dm_width)
    return {"age": 10.0 ** (fit.best["loga"] - 9.0), "std_loga": fit.std["loga"],
            "dm": fit.best["dm"], "std_dm": fit.std["dm"], "n": len(members)}


def sweet_spot(allstar_path: Path, settings: config.Settings, names: list[str]) -> None:
    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)

    for c in CLUSTERS:
        if names and c.name not in names:
            continue
        parquet = Path(f"data/galah_apogee_{c.name.replace(' ', '_')}.parquet")
        if not parquet.exists():
            continue
        lit = fetch_literature(c)

        # APOGEE members + clump
        sep = angular_separation(df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
                                 c.ra_deg, c.dec_deg)
        reg = df[sep <= c.region_deg].copy()
        m = kinematic_members(reg, c, config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
                              config.SEED_PM_TOL, config.SEED_RV_TOL,
                              config.N_REFINE_PASSES, config.REFINE_SIGMA).to_numpy()
        apogee_members = reg[m]
        giants = apogee_members[(apogee_members["LOGG"] >= 2.0) & (apogee_members["LOGG"] <= 3.5)]
        dm_clump = _clump_dm(giants, lit["feh"])
        grid = "metal_poor" if c.kind == "globular" else "solar"

        # GALAH members (from the combined parquet)
        comb = pd.read_parquet(parquet).rename(columns={
            "PARALLAX": "GAIAEDR3_PARALLAX", "PMRA": "GAIAEDR3_PMRA",
            "PMDEC": "GAIAEDR3_PMDEC", "RV": "VHELIO_AVG",
        })
        galah = comb[comb["survey"] == "GALAH"].copy()
        galah_members = _ms_members(galah, apogee_members)

        # combined (drop J/K so the fit uses BP-RP only; GALAH lacks 2MASS)
        ap_cols = [x for x in ["GAIAEDR3_PHOT_G_MEAN_MAG", "GAIAEDR3_PHOT_BP_MEAN_MAG",
                               "GAIAEDR3_PHOT_RP_MEAN_MAG", "GAIAEDR3_PARALLAX"] if x in apogee_members.columns]
        gal_cols = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "GAIAEDR3_PARALLAX"]
        ap = apogee_members[ap_cols].rename(columns={
            "GAIAEDR3_PHOT_G_MEAN_MAG": "phot_g_mean_mag",
            "GAIAEDR3_PHOT_BP_MEAN_MAG": "phot_bp_mean_mag",
            "GAIAEDR3_PHOT_RP_MEAN_MAG": "phot_rp_mean_mag",
        })
        gl = galah_members[gal_cols]
        combined = pd.concat([ap, gl], ignore_index=True)

        print(f"\n=== {c.name}: APOGEE members={len(apogee_members)} ({len(giants)} giants)  "
              f"GALAH members={len(galah_members)}  lit age={lit['age_Gyr']:.2f} dm={lit['dm']:.2f}", flush=True)
        if not np.isfinite(dm_clump):
            print("  no clump (< 20 member giants), skip", flush=True)
            continue
        print(f"  clump dm = {dm_clump:.2f} (lit {lit['dm']:.2f}, resid {dm_clump - lit['dm']:+.2f})", flush=True)

        for label, mem in [("APOGEE-only", ap), ("combined (APOGEE+MS)", combined)]:
            if len(mem) < 25:
                print(f"  {label}: only {len(mem)} stars, skip", flush=True)
                continue
            r = _fit(mem, dm_clump, 0.2, grid)
            print(f"  {label:20} n={r['n']:4d}  age={r['age']:5.2f} (resid {abs(r['age']-lit['age_Gyr']):.2f})  "
                  f"std_loga={r['std_loga']:.3f}  dm={r['dm']:.2f} std_dm={r['std_dm']:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all covered")
    args = ap.parse_args()
    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    sweet_spot(Path(args.allstar), settings, names)


if __name__ == "__main__":
    main()
