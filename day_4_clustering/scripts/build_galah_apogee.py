"""Build a combined GALAH + APOGEE sample for clusters with GALAH coverage.

GALAH DR4 (galah_dr4_allstar_240705.fits, Data Central) ships stellar
parameters + ``[X/Fe]`` abundances and a Gaia DR3 crossmatch VAC
(``galah_dr4_vac_wise_tmass_gaiadr3_240705.fits``) with parallax/PM/RV, so
no astroquery cone crossmatch is needed — just two local FITS files joined on
``sobject_id``.

For each covered cluster:
  1. load GALAH DR4 main + VAC (local FITS), join, quality-cut, region-cut
  2. load APOGEE (SDSS-V DR19 Astra) region + quality cuts
  3. merge GALAH with APOGEE by Gaia DR3 source_id
  4. build a common abundance set (14 elements) + Gaia kinematics

Saves data/galah_apogee_<cluster>.parquet and prints the sample size.

Usage:
    uv run python scripts/build_galah_apogee.py --clusters "M 67"  # pilot
    uv run python scripts/build_galah_apogee.py                      # all covered
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from cluster import config
from cluster.clusters import CLUSTERS, Cluster
from cluster.data import apply_quality_cuts, load_allstar
from cluster.membership import angular_separation

GALAH_MAIN_URL = (
    "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/"
    "catalogs/galah_dr4_allstar_240705.fits"
)
GALAH_VAC_URL = (
    "https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/"
    "catalogs/galah_dr4_vac_wise_tmass_gaiadr3_240705.fits"
)
GALAH_MAIN_PATH = Path("data/galah_dr4_allstar_240705.fits")
GALAH_VAC_PATH = Path("data/galah_dr4_vac_wise_tmass_gaiadr3_240705.fits")

# common abundance set: element -> (APOGEE column, GALAH DR4 column)
COMMON: dict[str, tuple[str, str]] = {
    "C": ("C_FE", "c_fe"),
    "O": ("O_FE", "o_fe"),
    "Na": ("NA_FE", "na_fe"),
    "Mg": ("MG_FE", "mg_fe"),
    "Al": ("AL_FE", "al_fe"),
    "Si": ("SI_FE", "si_fe"),
    "K": ("K_FE", "k_fe"),
    "Ca": ("CA_FE", "ca_fe"),
    "Ti": ("TI_FE", "ti_fe"),
    "V": ("V_FE", "v_fe"),
    "Cr": ("CR_FE", "cr_fe"),
    "Mn": ("MN_FE", "mn_fe"),
    "Ni": ("NI_FE", "ni_fe"),
    "Fe": ("FE_H", "fe_h"),
}
ABUND_COLS = [a for a, _ in COMMON.values()]
GALAH_COLS = [g for _, g in COMMON.values()]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"  downloading {url} -> {dest}", flush=True)
    subprocess.run(["wget", "-c", "--no-check-certificate", "-O", str(dest), url], check=True)


def _table(hdul: fits.HDUList) -> fits.FITS_rec:
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
            return hdu.data
    raise ValueError("no non-empty binary table HDU")


def _to_native(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        return arr.astype(np.float64)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    return arr.astype(str)


def load_galah_dr4(c: Cluster, region_deg: float) -> pd.DataFrame:
    """GALAH DR4 main + VAC joined on sobject_id, quality- and region-cut."""
    _download(GALAH_MAIN_URL, GALAH_MAIN_PATH)
    _download(GALAH_VAC_URL, GALAH_VAC_PATH)

    main_cols = [
        "sobject_id", "gaiadr3_source_id", "ra", "dec", "teff", "logg",
        "fe_h", "rv_gaia_dr3", "parallax", "phot_g_mean_mag",
        "flag_sp", "flag_fe_h", "snr_px_ccd3",
    ] + [g for g in GALAH_COLS if g != "fe_h"]
    with fits.open(GALAH_MAIN_PATH, memmap=True) as hdul:
        t = _table(hdul)
        main = pd.DataFrame({c: _to_native(t[c]) for c in main_cols})

    with fits.open(GALAH_VAC_PATH, memmap=True) as hdul:
        t = _table(hdul)
        vac = pd.DataFrame({
            c: _to_native(t[c])
            for c in ["sobject_id", "pmra", "pmdec", "phot_bp_mean_mag", "phot_rp_mean_mag"]
        })

    df = main.merge(vac, on="sobject_id", how="left")

    # quality cuts (GALAH DR4 best practices)
    mask = (df["flag_sp"].to_numpy(dtype=np.int64) == 0)
    mask &= (df["flag_fe_h"].to_numpy(dtype=np.int64) == 0)
    mask &= (df["snr_px_ccd3"].to_numpy(dtype=float) > 30.0)
    df = df[mask]

    # region cut
    sep = angular_separation(
        df["ra"].to_numpy(dtype=float), df["dec"].to_numpy(dtype=float),
        c.ra_deg, c.dec_deg,
    )
    df = df[sep <= region_deg].copy()

    rename = {g: a for a, g in COMMON.values()}
    df = df.rename(columns=rename)
    df["survey"] = "GALAH"
    df["RV"] = df["rv_gaia_dr3"]
    df["Teff"] = df["teff"]
    df["logg"] = df["logg"]
    df["RA"] = df["ra"]
    df["DEC"] = df["dec"]
    df["PARALLAX"] = df["parallax"]
    df["PMRA"] = df["pmra"]
    df["PMDEC"] = df["pmdec"]
    df["source_id"] = df["gaiadr3_source_id"]
    keep = ["source_id", "RA", "DEC", "RV", "Teff", "logg", *ABUND_COLS,
            "survey", "PARALLAX", "PMRA", "PMDEC",
            "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
    return df[[k for k in keep if k in df.columns]]


def load_apogee_region(allstar_path: Path, settings: config.Settings, c: Cluster, region_deg: float) -> pd.DataFrame:
    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)
    sep = angular_separation(
        df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
        c.ra_deg, c.dec_deg,
    )
    reg = df[sep <= region_deg].copy()
    reg["survey"] = "APOGEE"
    reg["RV"] = reg["VHELIO_AVG"]
    reg["Teff"] = reg["TEFF"]
    reg["logg"] = reg["LOGG"]
    reg["PARALLAX"] = reg["GAIAEDR3_PARALLAX"]
    reg["PMRA"] = reg["GAIAEDR3_PMRA"]
    reg["PMDEC"] = reg["GAIAEDR3_PMDEC"]
    reg["source_id"] = reg["GAIAEDR3_SOURCE_ID"]
    reg["phot_g_mean_mag"] = reg["GAIAEDR3_PHOT_G_MEAN_MAG"]
    reg["phot_bp_mean_mag"] = reg["GAIAEDR3_PHOT_BP_MEAN_MAG"]
    reg["phot_rp_mean_mag"] = reg["GAIAEDR3_PHOT_RP_MEAN_MAG"]
    keep = ["source_id", "RA", "DEC", "RV", "Teff", "logg", *ABUND_COLS,
            "survey", "PARALLAX", "PMRA", "PMDEC",
            "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "J", "K"]
    return reg[[k for k in keep if k in reg.columns]]


def build(allstar_path: Path, settings: config.Settings, names: list[str]) -> None:
    for c in CLUSTERS:
        if names and c.name not in names:
            continue
        print(f"\n=== {c.name} (region {c.region_deg:.1f} deg) ===", flush=True)
        try:
            galah = load_galah_dr4(c, c.region_deg)
        except Exception as exc:
            print(f"  GALAH load failed: {exc}", flush=True)
            continue
        if len(galah) < 20:
            print(f"  only {len(galah)} GALAH stars, skip", flush=True)
            continue
        apogee = load_apogee_region(allstar_path, settings, c, c.region_deg)

        apogee["source_id"] = apogee["source_id"].astype("Int64")
        galah["source_id"] = galah["source_id"].astype("Int64")
        merged = pd.concat([galah, apogee], ignore_index=True)
        # keep the APOGEE row for overlapping stars so J/K (2MASS) survives
        merged = merged.drop_duplicates(subset=["source_id"], keep="last")

        out = Path(f"data/galah_apogee_{c.name.replace(' ', '_')}.parquet")
        merged.to_parquet(out)
        n_galah = int((merged["survey"] == "GALAH").sum())
        n_apogee = int((merged["survey"] == "APOGEE").sum())
        print(f"  {c.name}: GALAH={n_galah}  APOGEE={n_apogee}  total={len(merged)}  -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all covered")
    args = ap.parse_args()
    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    build(Path(args.allstar), settings, names)


if __name__ == "__main__":
    main()
