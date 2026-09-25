"""Build the mwmStar URL list for ALL cluster members (completes the re-run).

The head-to-head member set comes from the benchmark's own kinematic
labelling of the DR19 Astra catalogue (~986 unique APOGEE_IDs). The DR17
backfill (``dr19_rerun_members.csv``) only covered 724 of them; the other
~262 were already in the DR19 apStar arm and were never re-downloaded.

Writes ``data/dr19_rerun_all_members.csv`` — every member's APOGEE_ID, sdss_id
and mwmStar URL, so the field arm + member arm together span the full sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

ASTRA = "data/astraAllStarASPCAP-0.6.0.fits.gz"
BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star"
V_ASTRA = "0.6.0"
OUT = "data/dr19_rerun_all_members.csv"


def mwmstar_url(sdss_id: int) -> str:
    last4 = f"{sdss_id}"[-4:].zfill(4)
    return f"{BASE}/{last4[:2]}/{last4[2:]}/mwmStar-{V_ASTRA}-{sdss_id}.fits"


def _n_rows(hdu: fits.BinTableHDU) -> int:
    return int(str(hdu.header["NAXIS2"]))


def main() -> None:
    from cluster import config
    from cluster.cli import _prepared_for

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    prep = _prepared_for(ASTRA, settings)
    member_ids = sorted(set(prep.df[prep.df["is_member"]]["APOGEE_ID"].astype(str)))
    print(f"unique members: {len(member_ids)}")

    with fits.open(ASTRA, memmap=True) as hdul:
        table = max((h for h in hdul if isinstance(h, fits.BinTableHDU)), key=_n_rows)
        d = table.data
        apogee_id = np.char.strip(np.asarray(d["sdss4_apogee_id"]).astype(str))
        sdss_id = np.asarray(d["sdss_id"])

    seen: dict[str, int] = {}
    for aid, sid in zip(apogee_id.tolist(), sdss_id.tolist(), strict=True):
        if aid and aid not in seen:
            seen[aid] = int(sid)

    missing = [m for m in member_ids if m not in seen]
    rows = [
        {"APOGEE_ID": m, "url": mwmstar_url(seen[m])}
        for m in member_ids if m in seen
    ]
    print(f"resolved to sdss_id: {len(rows)}; unresolved: {len(missing)}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
