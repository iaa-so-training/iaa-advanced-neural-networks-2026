"""Fetch DSS2 colour composites for cluster slides via SkyView (NASA GSFC).

For each cluster: download DSS2-Red, DSS2-Blue, DSS2-IR JPEGs, combine as
R=IR, G=Red, B=Blue (the standard DSS2 photographic colour mapping), apply an
arcsinh stretch per channel, and save a square PNG into ``$DAY4_DECK_DIR``
(default ``results/deck/``). Also builds a 25-tile montage.

Usage:
    .venv/bin/python scripts/make_sky_images.py [--clusters "M 3,M 67,..."]

No API key needed; be polite (sequential, small sleep).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cluster.clusters import CLUSTERS, Cluster

# Where the PNGs land. Override with DAY4_DECK_DIR=... (e.g. a slides folder).
DECK_ASSETS = Path(os.environ.get("DAY4_DECK_DIR", "results/deck")).expanduser()
BANDS = {"R": "DSS2-Red", "G": "DSS2-Blue", "B": "DSS2-IR"}


# deg -> arcmin-per-degree at declination dec
def _pixels(cluster: Cluster, size_deg: float = 0.42, px: int = 420) -> list[str]:
    urls = []
    for _channel, survey in BANDS.items():
        urls.append(
            f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
            f"?position={cluster.ra_deg:.4f},{cluster.dec_deg:.4f}"
            f"&survey={survey.replace('-', '%20')}"
            f"&size={size_deg},{size_deg}&pixels={px}&return=jpg"
        )
    return urls


def _fetch(url: str, out: Path) -> None:
    subprocess.run(
        ["curl", "-s", "--max-time", "90", "-o", str(out), url], check=True,
    )


def _stretch(channel: np.ndarray) -> np.ndarray:
    """Arcsinh stretch + percentile clip to [0,1]."""
    lo, hi = np.percentile(channel, [1, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    x = (channel.astype(np.float32) - lo) / (hi - lo)
    x = np.arcsinh(x * 8.0) / np.arcsinh(8.0)
    return np.clip(x, 0, 1)


def make_colour(cluster: Cluster, size_deg: float, px: int, tmp: Path, out: Path) -> None:
    channels = []
    for band, url in zip(BANDS.values(), _pixels(cluster, size_deg, px), strict=True):
        f = tmp / f"{cluster.name.replace(' ', '')}_{band.split('-')[1]}.jpg"
        _fetch(url, f)
        channels.append(np.asarray(Image.open(f).convert("L"), dtype=np.float32))
        time.sleep(0.4)
    r, g, b = (_stretch(channels[2]), _stretch(channels[0]), _stretch(channels[1]))
    rgb = np.stack([r, g, b], axis=-1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out)
    print(f"  wrote {out} ({px}x{px}, {size_deg} deg)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="",
                        help="comma list; default: the 5 head-to-head clusters")
    parser.add_argument("--size", type=float, default=0.42)
    parser.add_argument("--px", type=int, default=420)
    parser.add_argument("--montage", action="store_true",
                        help="also build sky_montage_25.png (all 25 clusters)")
    parser.add_argument("--montage-px", type=int, default=300)
    parser.add_argument("--montage-size", type=float, default=0.30)
    args = parser.parse_args()

    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or [
        "Berkeley 66", "IC 166", "M 3", "M 67", "NGC 188",
    ]
    tmp = Path("/tmp/skyview_cache")
    tmp.mkdir(exist_ok=True)
    DECK_ASSETS.mkdir(parents=True, exist_ok=True)

    by_name = {c.name: c for c in CLUSTERS}
    for name in names:
        cluster = by_name.get(name)
        if cluster is None:
            print(f"  !! unknown cluster {name}", file=sys.stderr)
            continue
        out = DECK_ASSETS / f"sky_{name.lower().replace(' ', '')}.png"
        make_colour(cluster, args.size, args.px, tmp, out)

    if args.montage:
        tiles = []
        for cluster in CLUSTERS:
            out = tmp / f"mont_{cluster.name.replace(' ', '')}.png"
            make_colour(cluster, args.montage_size, args.montage_px, tmp, out)
            tiles.append(np.asarray(Image.open(out).convert("RGB")))
        grid = _montage(tiles)
        Image.fromarray(grid).save(DECK_ASSETS / "sky_montage_25.png")
        print(f"  wrote {DECK_ASSETS / 'sky_montage_25.png'}")


def _montage(tiles: list[np.ndarray], cols: int = 5, pad: int = 6) -> np.ndarray:
    """Stack tiles into a grid with padding and a dark frame."""
    rows = (len(tiles) + cols - 1) // cols
    h, w, _ = tiles[0].shape
    grid = np.full(
        (rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3),
        16, dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        grid[y:y + h, x:x + w] = tile
    return grid


if __name__ == "__main__":
    main()
