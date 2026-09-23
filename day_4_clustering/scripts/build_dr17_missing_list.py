"""DEPRECATED — built on a false premise. Use ``build_dr19_rerun_list.py``.

This script assumed DR19 did not contain these stars, so it backfilled their
spectra from DR17. That assumption is wrong: DR19 *reanalyses and includes*
DR17 (``astraAllStarASPCAP-0.6.0`` carries 717,689 rows with
``release='dr17'``), and DR19 serves a spectrum for 738/738 of the members
this script targeted.

Worse, the backfill mixed data *products*: DR17 ``aspcapStar`` is
continuum-normalised (median flux ~1.01) while the DR19 arm used raw
``apStar`` (~5.8e3). Feeding both to one autoencoder made the latent encode
the product, not the chemistry — see ``scripts/diagnose_product_mismatch.py``
and the § Provenance section of ``docs/spectral_benchmark_results.md``.

Kept only for provenance of the published numbers. Do not run it.

---

Build the DR17 aspcapStar download list for DR19 members missing DR19 spectra.

DR19 apStar (redux 1.3) covers only APO-North DR19 observations. The ~880
members not in it were observed in DR17 (APOGEE-2). Their spectra are the DR17
aspcapStar files:

    https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/<tel>/<field>/aspcapStar-dr17-<id>.fits

Writes a two-column CSV (APOGEE_ID, url) for a parallel downloader.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits

from cluster import config
from cluster.clusters import CLUSTERS
from cluster.data import prepare

BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec"


def main() -> None:
    settings = config.Settings()
    clusters = [c for c in CLUSTERS if c.name in settings.resolve_cluster_names(
        [c.name for c in CLUSTERS])]
    prep = prepare("data/astraAllStarASPCAP-0.6.0.fits.gz", settings, clusters,
                   seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
                   seed_parallax_frac=config.SEED_PARALLAX_FRAC,
                   seed_pm_tol=config.SEED_PM_TOL,
                   seed_rv_tol=config.SEED_RV_TOL,
                   n_refine_passes=config.N_REFINE_PASSES,
                   refine_sigma=config.REFINE_SIGMA)
    members = prep.df[prep.df["cluster"] != "field"]
    mem_ids = {i for i in members["APOGEE_ID"].astype(str) if i.strip()}

    # stars already embedded (DR19 apStar)
    dr19 = pd.read_parquet("data/embeddings/masked_latent.parquet")
    have = set(dr19["APOGEE_ID"].astype(str))
    missing = [i for i in mem_ids if i not in have]
    print(f"members: {len(mem_ids)} | have DR19: {len(mem_ids & have)} | missing: {len(missing)}")

    # telescope + field from the astra
    with fits.open("data/astraAllStarASPCAP-0.6.0.fits.gz", memmap=True) as hdul:
        tables = sorted([(i, h.data.shape[0]) for i, h in enumerate(hdul)
                         if isinstance(h, fits.BinTableHDU)], key=lambda x: -x[1])
        t = hdul[tables[0][0]].data
        aid = np.char.strip(t["sdss4_apogee_id"].astype(str))
        tel = np.char.strip(t["telescope"].astype(str))
        fld = np.char.strip(t["field"].astype(str))

    rows = []
    for i in missing:
        m = np.where(aid == i)[0]
        if len(m) == 0:
            continue
        k = m[0]
        url = f"{BASE}/{tel[k]}/{fld[k]}/aspcapStar-dr17-{i}.fits"
        rows.append({"APOGEE_ID": i, "url": url})
    df = pd.DataFrame(rows)
    out = "data/dr17_missing_members.csv"
    df.to_csv(out, index=False)
    print(f"→ {out} ({len(df)} URLs)")


if __name__ == "__main__":
    main()
