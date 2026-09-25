"""Red-clump distance from APOGEE member giants (allStar directly).

Select kinematic members, then the member giants (2 <= logg <= 3.5), and
take the median K of the J-K clump slice (0.5 <= J-K <= 0.85) as the clump
magnitude. Distance modulus from the K-band clump absolute magnitude with a
metallicity correction (M_K ~ -1.6 + 0.25*[Fe/H]).

Usage:
    uv run python scripts/red_clump.py [--clusters "NGC 6819,NGC 7789"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.clusters import CLUSTERS
from cluster.data import apply_quality_cuts, load_allstar
from cluster.literature import fetch_literature
from cluster.membership import angular_separation, kinematic_members


def _clump_dm(giants: pd.DataFrame, feh: float) -> tuple[float, float, int]:
    giants = giants[np.isfinite(giants["J"]) & np.isfinite(giants["K"])]
    if len(giants) < 20:
        return float("nan"), float("nan"), len(giants)
    jk = (giants["J"] - giants["K"]).to_numpy(dtype=float)
    clump = giants[(jk >= 0.5) & (jk <= 0.85)]
    if len(clump) < 10:
        return float("nan"), float("nan"), len(giants)
    k_clump = float(np.median(clump["K"].to_numpy(dtype=float)))
    m_k = -1.6 + 0.25 * feh if np.isfinite(feh) else -1.6
    return k_clump - m_k, k_clump, len(giants)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all")
    args = ap.parse_args()
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    df = load_allstar(Path(args.allstar), settings.elements)
    df = apply_quality_cuts(df, settings)

    print(f'{"cluster":12} {"n_memb":>6} {"n_giants":>8} {"dm_clump":>8} {"dm_lit":>7} {"resid":>6}')
    for c in CLUSTERS:
        if names and c.name not in names:
            continue
        sep = angular_separation(df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
                                 c.ra_deg, c.dec_deg)
        reg = df[sep <= c.region_deg].copy()
        m = kinematic_members(reg, c, config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
                              config.SEED_PM_TOL, config.SEED_RV_TOL,
                              config.N_REFINE_PASSES, config.REFINE_SIGMA).to_numpy()
        giants = reg[m & (reg["LOGG"] >= 2.0) & (reg["LOGG"] <= 3.5)]
        lit = fetch_literature(c)
        dm_clump, k_clump, n = _clump_dm(giants, lit["feh"])
        resid = dm_clump - lit["dm"] if np.isfinite(dm_clump) else float("nan")
        print(f'{c.name:12} {int(m.sum()):6d} {n:8d} {dm_clump:8.2f} {lit["dm"]:7.2f} {resid:6.2f}')


if __name__ == "__main__":
    main()
