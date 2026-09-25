"""Build the DR19 mwmStar download list for the members we backfilled from DR17.

Supersedes ``build_dr17_missing_list.py``, which was based on a wrong premise.

That script assumed DR19 did not contain these stars. It does. The DR19 Astra
summary file ``astraAllStarASPCAP-0.6.0.fits.gz`` carries 717,689 rows with
``release='dr17'`` — DR19 reanalyses the DR17 sample and ships it under one
pipeline (``v_astra=0.6.0``). Every backfilled member resolves to an
``sdss_id`` there and 100% of a sampled check returned HTTP 200 on the SAS.

What actually went wrong was the *product*, not the release:

    DR19 arm : apStar / mwmStar  -> RAW flux, median ~5.8e3
    backfill : aspcapStar-dr17   -> CONTINUUM-NORMALISED flux, median ~1.01

Feeding both to one autoencoder makes the latent encode the product. With 253
stars embedded through both pipelines, the same star lands 1.70x farther apart
across products than two different stars within one product (cosine 0.389).

SAS layout (mwmStar files are sharded on the LAST FOUR digits of sdss_id,
zero-padded, split 2+2 — not on healpix):

    .../spectro/astra/0.6.0/spectra/star/<d1d2>/<d3d4>/mwmStar-0.6.0-<sdss_id>.fits

Writes a two-column CSV (APOGEE_ID, url) for a parallel downloader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

ASTRA = "data/astraAllStarASPCAP-0.6.0.fits.gz"
BASE = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star"
V_ASTRA = "0.6.0"
OUT = "data/dr19_rerun_members.csv"


def mwmstar_url(sdss_id: int) -> str:
    """SAS URL for a star's mwmStar file (sharded on the last 4 id digits)."""
    last4 = f"{sdss_id}"[-4:].zfill(4)
    return f"{BASE}/{last4[:2]}/{last4[2:]}/mwmStar-{V_ASTRA}-{sdss_id}.fits"


def _n_rows(hdu: fits.BinTableHDU) -> int:
    """Row count from the header (avoids touching ``.data``)."""
    return int(str(hdu.header["NAXIS2"]))


def main() -> None:
    backfilled = pd.read_csv("data/dr17_missing_members.csv")["APOGEE_ID"]
    wanted = {str(i) for i in backfilled}

    with fits.open(ASTRA, memmap=True) as hdul:
        table = max(
            (h for h in hdul if isinstance(h, fits.BinTableHDU)), key=_n_rows,
        )
        data = table.data
        apogee_id = np.char.strip(np.asarray(data["sdss4_apogee_id"]).astype(str))
        sdss_id = np.asarray(data["sdss_id"])

    seen: dict[str, int] = {}
    for name, sid in zip(apogee_id.tolist(), sdss_id.tolist(), strict=True):
        if name in wanted and name not in seen:
            seen[name] = int(sid)

    missing = wanted - seen.keys()
    print(f"backfilled members      : {len(wanted)}")
    print(f"resolved to a DR19 id   : {len(seen)}")
    print(f"absent from DR19        : {len(missing)}")

    frame = pd.DataFrame(
        [{"APOGEE_ID": k, "url": mwmstar_url(v)} for k, v in sorted(seen.items())]
    )
    frame.to_csv(OUT, index=False)
    print(f"-> {OUT} ({len(frame)} URLs)")


if __name__ == "__main__":
    main()
