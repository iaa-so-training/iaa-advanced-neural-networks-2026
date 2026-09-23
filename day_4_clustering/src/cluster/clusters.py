"""Target cluster catalogue.

Open-cluster astrometry (RA, Dec, pmRA, pmDE, parallax→distance) follows
Cantat-Gaudin et al. (2018, A&A 618, A93). Globular-cluster parameters
follow the Harris catalogue / Vasiliev & Baumgardt (2021). Radial velocities
are literature values cross-checked against APOGEE's VHELIO_AVG. Pleiades
uses Gaia DR3 means (target paper's Appendix C box).

Angular diameters ``diam_arcmin``: open clusters from the Dias et al. (2014)
catalogue (VizieR B/ocl, field ``Diam``); globulars are the tidal diameters
from the Harris catalogue (2 × tidal radius); Pleiades and Berkeley 66 are
literature values (not resolved by B/ocl).

The per-cluster search region ``region_deg`` scales with the angular
diameter: ``max(3°, 10 × diam_deg)``.

Values are *seeds only*: ``membership.py`` refines them against the
APOGEE+Gaia data with iterative sigma clipping, so small seed errors are
corrected automatically.

``n_ref`` is the published member count from R. Garcia-Dias et al. (2019,
Table 1), kept only for cross-checking.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

REGION_MIN_DEG = 3.0
REGION_DIAMETER_FACTOR = 10.0


class Cluster(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    name: str
    kind: str          # "open" or "globular"
    ra_deg: float
    dec_deg: float
    dist_pc: float     # heliocentric distance; parallax = 1000 / dist_pc
    pmra: float        # mas/yr
    pmdec: float       # mas/yr
    rv: float          # km/s (heliocentric)
    diam_arcmin: float = 10.0  # angular diameter (arcmin)
    n_ref: int | None = None   # published member count (reference only)

    @property
    def parallax_mas(self) -> float:
        return 1000.0 / self.dist_pc

    @property
    def diam_deg(self) -> float:
        return self.diam_arcmin / 60.0

    @property
    def region_deg(self) -> float:
        """Per-cluster search radius, scaled to the angular diameter."""
        return max(REGION_MIN_DEG, REGION_DIAMETER_FACTOR * self.diam_deg)


# name, kind, RA, Dec, dist_pc, pmra, pmdec, rv, diam_arcmin, n_ref
_CLUSTER_ROWS: list[tuple[str, str, float, float, float, float, float, float, float, int | None]] = [
    # ------------------------------ open clusters ------------------------- #
    ("Pleiades",    "open",  56.601, +24.114,   136,  +20.077, -45.503,   +5.9, 110.0,  27),
    ("King 7",      "open",  59.783, +51.785,  3172,   +1.121,  -1.196,  -21.0,   7.0,   5),
    ("Berkeley 71", "open",  85.233, +32.272,  3930,   +0.650,  -1.614,   -1.0,   6.2,   5),
    ("IC 166",      "open",  28.094, +61.857,  4909,   -1.439,  +1.126,  -54.0,   7.0,   6),
    ("NGC 2158",    "open",  91.862, +24.099,  4535,   -0.177,  -2.002,  +27.0,   5.0,  18),
    ("NGC 1245",    "open",  48.691, +47.235,  3148,   +0.504,  -1.579,  -29.0,  32.0,  22),
    ("King 5",      "open",  48.682, +52.695,  2524,   -0.282,  -1.200,  -44.0,  14.2,   6),
    ("NGC 7789",    "open", 359.334, +56.726,  2075,   -0.922,  -1.933,  -54.0,  25.0,  32),
    ("NGC 1798",    "open",  77.914, +47.691,  4834,   +0.913,  -0.318,   -2.0,   5.0,   5),
    ("NGC 2420",    "open", 114.602, +21.575,  2553,   -1.190,  -2.125,  +73.0,   5.0,  15),
    ("NGC 6819",    "open", 295.327, +40.190,  2600,   -2.916,  -3.856,   +2.0,  13.0,  81),
    ("M 67",        "open", 132.846, +11.814,   859,  -10.986,  -2.964,  +33.6,  25.0,  46),
    ("Berkeley 66", "open",  46.030, +58.731,  5346,   -0.120,  -0.191,  -50.0,   7.0,   6),
    ("NGC 188",     "open",  11.798, +85.244,  1864,   -2.307,  -0.960,  -42.0,  17.0,  15),
    ("NGC 6791",    "open", 290.221, +37.778,  4531,   -0.421,  -2.269,  -47.0,  10.0,  58),
    ("Berkeley 17", "open",  80.130, +30.574,  3230,   +2.618,  -0.350,  -71.0,   7.0,   6),
    # southern intermediate-age clusters (GALAH + APOGEE sweet spot)
    ("NGC 2243",    "open",  97.395, -31.282,  4739,   -1.279,  +5.488,  +59.8,   5.0, None),
    ("Collinder 261", "open", 189.519, -68.377,  2190,   -6.351,  -2.705,  -30.0,   9.0, None),
    # ----------------------------- globular clusters ---------------------- #
    ("M 5",   "globular", 229.638,  +2.081,  7500,  +4.05, -6.18,  +52.1,  46.0, 27),
    ("M 3",   "globular", 205.548, +28.377, 10200,  -0.14, -2.66, -147.6,  76.0, 20),
    ("M 13",  "globular", 250.423, +36.461,  7100,  -3.19, -2.55, -246.6,  40.0,  9),
    ("M 15",  "globular", 322.493, +12.167, 10400,  -0.66, -3.68, -107.5,  44.0,  6),
    ("M 71",  "globular", 298.444, +18.779,  4000,  -3.71, -3.01,  -22.9,  14.0,  9),
    ("M 107", "globular", 248.133, -13.054,  6400,  -1.61, -4.10,  -33.8,  26.0, 12),
    ("M 92",  "globular", 259.281, +43.136,  8300,  -4.94, -0.57, -120.5,  30.0,  8),
]

def _make_cluster(row: tuple[str, str, float, float, float, float, float, float, float, int | None]) -> Cluster:
    name, kind, ra_deg, dec_deg, dist_pc, pmra, pmdec, rv, diam_arcmin, n_ref = row
    return Cluster(
        name=name,
        kind=kind,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        dist_pc=dist_pc,
        pmra=pmra,
        pmdec=pmdec,
        rv=rv,
        diam_arcmin=diam_arcmin,
        n_ref=n_ref,
    )


CLUSTERS: list[Cluster] = [_make_cluster(row) for row in _CLUSTER_ROWS]
CLUSTER_BY_NAME: dict[str, Cluster] = {c.name: c for c in CLUSTERS}
