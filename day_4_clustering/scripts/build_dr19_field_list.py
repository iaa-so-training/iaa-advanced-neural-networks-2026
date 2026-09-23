"""Build the DR19 mwmStar download list for the field sample (uniform re-run).

The masked-AE checkpoint that produced ``masked_latent.parquet`` is gone, and
the raw apStar arm it trained on is gone from both machines. The only clean
path is a fresh self-consistent re-run: retrain the masked AE on raw DR19
``mwmStar`` spectra, then embed *every* star (members + field) from that one
product.

The field sample is drawn straight from the DR19 Astra catalogue (which carries
``sdss_id`` directly), so no fragile 2M->sdss resolution is needed:

  * telescope == apo25m  (matches the original APO-North arm)
  * release   == sdss5   (the new DR19 observations, not the re-analysed DR17)
  * excludes every backfilled member (by sdss_id)
  * fixed-seed random sample of N stars

Writes ``data/dr19_rerun_field.csv`` with columns (APOGEE_ID, url), where
APOGEE_ID is the catalogue's ``sdss4_apogee_id`` (SDSS-style for sdss5 stars;
uniquely identifies each star and never collides with the 2M member ids).
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd
from astropy.io import fits

ASTRA = "data/astraAllStarASPCAP-0.6.0.fits.gz"
MEMBERS = "data/dr19_rerun_members.csv"
BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star"
V_ASTRA = "0.6.0"
OUT = "data/dr19_rerun_field.csv"


def mwmstar_url(sdss_id: int) -> str:
    last4 = f"{sdss_id}"[-4:].zfill(4)
    return f"{BASE}/{last4[:2]}/{last4[2:]}/mwmStar-{V_ASTRA}-{sdss_id}.fits"


def _n_rows(hdu: fits.BinTableHDU) -> int:
    return int(str(hdu.header["NAXIS2"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    members = pd.read_csv(MEMBERS, dtype=str)
    member_sdss = {
        int(m.group(1))
        for u in members["url"]
        if (m := re.search(r"mwmStar-\d+\.\d+\.\d+-(\d+)\.fits", u))
    }
    print(f"member sdss_ids to exclude: {len(member_sdss)}")

    with fits.open(ASTRA, memmap=True) as hdul:
        table = max((h for h in hdul if isinstance(h, fits.BinTableHDU)), key=_n_rows)
        d = table.data
        telescope = np.asarray(d["telescope"]).astype(str)
        release = np.char.strip(np.asarray(d["release"]).astype(str))
        apogee_id = np.char.strip(np.asarray(d["sdss4_apogee_id"]).astype(str))
        sdss_id = np.asarray(d["sdss_id"])

    sel = (telescope == "apo25m") & (release == "sdss5")
    print(f"apo25m + sdss5 stars: {sel.sum()}")

    ids: list[str] = []
    sids: list[int] = []
    for aid, sid, keep in zip(apogee_id, sdss_id, sel, strict=True):
        if keep and aid != "" and int(sid) not in member_sdss:
            ids.append(aid)
            sids.append(int(sid))

    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)
    rows = [
        {"APOGEE_ID": ids[i], "url": mwmstar_url(sids[i])} for i in sorted(take)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print(f"-> {OUT}: {len(frame)} field stars")


if __name__ == "__main__":
    main()
