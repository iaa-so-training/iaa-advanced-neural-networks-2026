"""Bulk-download DR19 apStar spectra from the allStar-1.3 'uri' column.

Simple ThreadPoolExecutor + wget (no asyncio — the lightsurf asyncio
downloader crashes with 'Event loop is closed'). Skips already-downloaded
files; retries transient failures.

Usage:
    uv run python scripts/download_apstar_dr19.py \
        data/dr19_star_list.fits /path/to/apstar_out 20
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astropy.io import fits


def _table(hdul: fits.HDUList):
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.data is not None and len(hdu.data) > 0:
            return hdu.data
    raise ValueError("no non-empty table HDU")


def main() -> None:
    star_list = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    out_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(star_list, memmap=True) as hdul:
        uris = [str(u) for u in _table(hdul)["uri"] if str(u).strip() != ""]

    print(f"downloading {len(uris)} apStar files -> {out_dir} ({n_workers} workers)", flush=True)

    def download(uri: str) -> str:
        url = "https://dr19.sdss.org/sas/dr19/" + uri.replace(
            "apogee/spectro", "spectro/apogee", 1
        )
        fname = uri.split("/")[-1]
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            return "skip"
        r = subprocess.run(
            ["wget", "--no-check-certificate", "--tries=10", "--timeout=15",
             "--retry-connrefused", "--waitretry=1",
             "-O", str(dest), url],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return "ok"
        dest.unlink(missing_ok=True)
        return f"FAIL {url} :: {(r.stderr or '')[-160:]}"

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for status in ex.map(download, uris):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                if fail <= 10:
                    print(f"  {status}", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
