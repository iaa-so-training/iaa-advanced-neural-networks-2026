import asyncio
import os
import sys
from pathlib import Path
from typing import Any
import pandas as pd

BASE = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec"
OUT = Path(os.environ.get("BROAD_SPECTRA_DIR", "data/raw_data/broad_spectra"))
OUT.mkdir(parents=True, exist_ok=True)

STAR_LIST = os.environ.get("BROAD_STAR_LIST", "data/embeddings/broad_star_list.csv")
df = pd.read_csv(STAR_LIST)
sem = asyncio.Semaphore(16)


async def fetch(row: Any) -> str:
    fname = f"aspcapStar-dr17-{row['APOGEE_ID']}.fits"
    dest = OUT / fname
    if dest.exists() and dest.stat().st_size > 1000:
        return "cached"
    url = f"{BASE}/{row['TELESCOPE']}/{row['FIELD']}/{fname}"
    async with sem:
        proc = await asyncio.create_subprocess_shell(
            f"wget -q --no-check-certificate --http-user=sdss "
            f"--http-passwd=2.5-meters -O {dest} {url}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        code = await proc.wait()
    return "ok" if code == 0 else "fail"


async def main():
    rows = df.to_dict("records")
    results = await asyncio.gather(*(fetch(r) for r in rows))
    ok = results.count("ok")
    cached = results.count("cached")
    fail = results.count("fail")
    print(f"ok={ok} cached={cached} fail={fail}")
    print("done")


asyncio.run(main())
