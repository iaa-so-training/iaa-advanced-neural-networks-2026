"""Build the DR19 star list for the spectral re-embed — cluster-aware.

Includes ALL cluster members (kinematic σ-clip, same labels the benchmark
uses) plus a stratified field sample, so the spectral embeddings cover the
benchmark's stars. Mirrors lightsurf's mdwarfs_DR17.fits columns.

Usage:
    uv run python scripts/build_dr19_star_list.py --n-stars 40000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from cluster import config
from cluster.clusters import CLUSTERS
from cluster.data import _table_hdu, _to_native
from cluster.membership import label_clusters

ASTRA = Path("data/astraAllStarASPCAP-0.6.0.fits.gz")
REDUX = Path("data/allStar-1.3-apo25m.fits")

# lightsurf.constants.APOGEE_ABUNDANCE_TARGETS -> astra [X/H] column
TARGETS = {
    "FE_H": "fe_h",
    "C_FE": "c_h",
    "CA_FE": "ca_h",
    "K_FE": "k_h",
    "MG_FE": "mg_h",
    "NI_FE": "ni_h",
    "O_FE": "o_h",
    "SI_FE": "si_h",
    "TI_FE": "ti_h",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stars", type=int, default=40_000)
    ap.add_argument("--out", default="data/dr19_star_list.fits")
    args = ap.parse_args()

    with fits.open(ASTRA, memmap=True) as hdul:
        t = _table_hdu(hdul)
        astra_cols = ["sdss_id", "ra", "dec", "plx", "pmra", "pmde", "v_rad",
                      "snr", "flag_bad", "spectrum_flags", "teff", "logg", "fe_h"] + [
            c for c in TARGETS.values() if c != "fe_h"
        ]
        astra = pd.DataFrame({c: _to_native(t[c]) for c in astra_cols})

    with fits.open(REDUX, memmap=True) as hdul:
        t = _table_hdu(hdul)
        redux = pd.DataFrame({
            c: _to_native(t[c])
            for c in ["sdss_id", "apogee_id", "file", "uri", "telescope", "healpix"]
        })

    df = astra.merge(redux, on="sdss_id", how="inner")
    df = df[df["file"].astype(str) != ""].copy()

    # canonical columns for the membership labelling (same names the loader emits)
    fe_h = df["fe_h"].to_numpy(dtype=float)
    df["RA"] = df["ra"].to_numpy(dtype=float)
    df["DEC"] = df["dec"].to_numpy(dtype=float)
    df["GAIAEDR3_PARALLAX"] = df["plx"].to_numpy(dtype=float)
    df["GAIAEDR3_PMRA"] = df["pmra"].to_numpy(dtype=float)
    df["GAIAEDR3_PMDEC"] = df["pmde"].to_numpy(dtype=float)
    df["VHELIO_AVG"] = df["v_rad"].to_numpy(dtype=float)

    # quality: SNR + flags + finite Teff/logg (the membership σ-clip drops
    # NaN kinematics itself; NaN abundances are dropped at training time).
    keep = df["snr"].to_numpy(dtype=float) >= 100.0
    keep &= (df["flag_bad"].to_numpy(dtype=np.int64) == 0)
    keep &= (df["spectrum_flags"].to_numpy(dtype=np.int64) == 0)
    keep &= np.isfinite(df["teff"].to_numpy(dtype=float))
    keep &= np.isfinite(df["logg"].to_numpy(dtype=float))
    df = df[keep].reset_index(drop=True)

    # label cluster members (kinematic σ-clip — the benchmark's own labels)
    df = label_clusters(
        df, CLUSTERS,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
        membership_method="kinematic",
    )
    members = df[df["is_member"]].copy()
    field = df[~df["is_member"]].copy()
    print(f"members: {len(members)}  field: {len(field)}")

    n_field = max(0, args.n_stars - len(members))
    if len(field) > n_field:
        field = field.sample(n=n_field, random_state=42)
    df = pd.concat([members, field], ignore_index=True)

    # [X/H] -> [X/Fe]
    out = {"APOGEE_ID": df["apogee_id"].astype(str)}
    for target, xh in TARGETS.items():
        out[target] = df[xh].to_numpy(dtype=float) - fe_h[df.index] if target != "FE_H" else df["fe_h"].to_numpy(dtype=float)
    out["TEFF"] = df["teff"].to_numpy(dtype=float)
    out["LOGG"] = df["logg"].to_numpy(dtype=float)
    out["telescope"] = df["telescope"].astype(str)
    out["healpix"] = df["healpix"].to_numpy(dtype=np.int64)
    out["file"] = df["file"].astype(str)
    out["uri"] = df["uri"].astype(str)
    out = pd.DataFrame(out)

    cols = []
    for name in out.columns:
        v = out[name].to_numpy()
        if v.dtype.kind == "f":
            cols.append(fits.Column(name=name, array=v.astype(np.float64), format="D"))
        elif v.dtype.kind in "iu":
            cols.append(fits.Column(name=name, array=v.astype(np.int64), format="K"))
        else:
            cols.append(fits.Column(name=name, array=v.astype(str).astype("S200"), format="200A"))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(args.out, overwrite=True)
    print(f"✓ {len(out)} stars ({len(members)} members + {len(field)} field) -> {args.out}")


if __name__ == "__main__":
    main()
