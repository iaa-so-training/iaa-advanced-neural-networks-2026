"""Re-check the isochrone slide's numbers on the current DR19 data.

The deck quotes two kinds of PARSEC fit (ASteCA + emcee):

* distance modulus from the APOGEE kinematic members of M 67 and M 15,
  fitted in two colours (Gaia BP-RP + 2MASS J-Ks);
* age from a Gaia DR3 CMD down to the main sequence for M 5 and M 15
  (globulars keep only G > 16, as the notebook's Gaia fallback does).

Both are compared with the cached literature values. Fits run in parallel,
one process per fit.

Usage (repo root):
    .venv/bin/python scripts/isochrone_check.py --out results/isochrone_check_dr19.csv
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.clusters import CLUSTER_BY_NAME
from cluster.data import apply_quality_cuts, complete_case, load_allstar
from cluster.isochrone import fit_isochrone
from cluster.literature import fetch_literature
from cluster.membership import angular_separation, kinematic_members

APOGEE_CLUSTERS = ("M 67", "M 15")
GAIA_CLUSTERS = ("M 5", "M 15")


def _grid(name: str) -> str:
    return "metal_poor" if CLUSTER_BY_NAME[name].kind == "globular" else "solar"


def _apogee_members(df: pd.DataFrame, name: str, settings: config.Settings) -> pd.DataFrame:
    """Kinematic members inside 30 deg, as the notebook's isochrone cell."""
    c = CLUSTER_BY_NAME[name]
    sep = angular_separation(
        df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
        c.ra_deg, c.dec_deg,
    )
    reg = complete_case(df[sep <= 30.0].copy(), settings)
    m = kinematic_members(
        reg, c,
        config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
        config.SEED_PM_TOL, config.SEED_RV_TOL,
        config.N_REFINE_PASSES, config.REFINE_SIGMA,
    ).to_numpy()
    return reg[m]


def _gaia_members(name: str) -> pd.DataFrame:
    from cluster.gaia import gaia_members, query_gaia_region

    c = CLUSTER_BY_NAME[name]
    df = query_gaia_region(c, 0.5)
    members = df[gaia_members(df, c)]
    if c.kind == "globular":
        members = members[members["phot_g_mean_mag"] > 16.0]
    return members


def _fit(job: tuple[str, str, pd.DataFrame, int, int, int]) -> dict[str, object]:
    kind, name, members, seed, n_walkers, n_steps = job
    lit = fetch_literature(CLUSTER_BY_NAME[name])
    try:
        fit = fit_isochrone(
            members, seed=seed, n_walkers=n_walkers, n_steps=n_steps, grid=_grid(name),
        )
    except ValueError as exc:  # too few members with full photometry
        return {"cluster": name, "fit": kind, "n_stars": len(members), "error": str(exc)}
    age = 10.0 ** (fit.best["loga"] - 9.0)
    return {
        "cluster": name, "fit": kind, "n_stars": fit.n_stars,
        "dm_fit": fit.best["dm"], "std_dm": fit.std["dm"], "dm_lit": lit["dm"],
        "dm_resid": fit.best["dm"] - lit["dm"],
        "age_fit_gyr": age, "std_loga": fit.std["loga"], "age_lit_gyr": lit["age_Gyr"],
        "met_fit": fit.best["met"], "feh_lit": lit["feh"], "error": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/isochrone_check_dr19.csv")
    args = ap.parse_args()

    settings = config.Settings()
    df = apply_quality_cuts(load_allstar(Path(args.allstar), settings.elements), settings)
    jobs: list[tuple[str, str, pd.DataFrame, int, int, int]] = []
    for name in APOGEE_CLUSTERS:
        mem = _apogee_members(df, name, settings)
        has_jk = bool(np.isfinite(mem[["J", "K"]].to_numpy(dtype=float)).all(axis=1).sum())
        print(f"APOGEE {name}: {len(mem)} kinematic members (J/K present: {has_jk})", flush=True)
        jobs.append(("APOGEE 2-colour", name, mem, settings.isofit_seed,
                     settings.isofit_n_walkers, settings.isofit_n_steps))
    del df
    for name in GAIA_CLUSTERS:
        mem = _gaia_members(name)
        print(f"Gaia {name}: {len(mem)} members", flush=True)
        jobs.append(("Gaia CMD", name, mem, settings.isofit_seed,
                     settings.isofit_n_walkers, settings.isofit_n_steps))

    with Pool(len(jobs)) as pool:
        rows = pool.map(_fit, jobs)
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
