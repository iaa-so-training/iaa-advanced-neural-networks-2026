"""Isochrone fitting via ASteCA — a quantitative membership-quality proxy.

For each membership source (catalogue / kinematic / combined) we fit a PARSEC
isochrone to the Gaia CMD of that source's members. A cleaner member list
traces a tighter sequence, which shows up as a *higher* maximum likelihood
and *smaller* posterior uncertainties on (metallicity, age, distance modulus,
extinction). So the fit quality is a proxy for membership quality.

Uses two PARSEC v1.2S grids (Gaia EDR3 filters), each with a fine 0.05-dex
age step and five metallicities, downloaded on first use and cached under
``data/isochrones/``:

* ``solar`` — Z = 0.010–0.030 (open clusters);
* ``metal_poor`` — [M/H] ≈ -2.3 … -0.3 (globulars).

The 400-isochrone CMD limit makes a single fine grid over both metallicity
ranges impossible, so the grid is chosen per cluster type.

Fit is an ``emcee`` run over 4 parameters (metallicity, log age, distance
modulus, extinction), seeded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clusters import Cluster
from .config import Settings
from .plots import thin_field

ISOCHRONE_URL = (
    "https://raw.githubusercontent.com/asteca/ASteCA/main/"
    "docs/_static/isochrones/parsec/gaia/parsec_12S_260510.dat"
)
ISOCHRONE_DIR = Path("data/isochrones/parsec")
ISOCHRONE_FILE = "parsec_solar_2color.dat"
ISOCHRONE_METALPOOR_FILE = "parsec_metalpoor_2color.dat"
_GAIA_2MASS_PHOTSYS = "YBC_tab_mag_odfnew/tab_mag_gaia_tycho2_2mass.dat"

_MAG = "Gmag"
_COLOR = ("G_BPmag", "G_RPmag")
_COLOR2 = ("Jmag", "Ksmag")   # 2MASS J-K breaks the age-metallicity degeneracy
_MAG_EFFL = 6390.7
_COLOR_EFFL = (5182.58, 7825.08)
_COLOR2_EFFL = (12350.0, 21590.0)

# fit parameters and their (name, lower, upper) uniform priors.
# met bounds are per-grid (see _GRID_MET); the rest are shared.
_PARAMS: dict[str, tuple[float, float]] = {
    "loga": (6.6, 10.1),     # log10(age / yr)
    "dm": (0.0, 20.0),       # distance modulus (re-centred per cluster)
    "Av": (0.0, 1.0),        # extinction (mag)
}
_GRID_MET: dict[str, tuple[float, float]] = {
    "solar": (0.008, 0.030),
    "metal_poor": (0.0002, 0.012),
}


@dataclass
class IsochroneFit:
    best: dict[str, float]
    std: dict[str, float]
    max_lkl: float
    n_stars: int
    curve_mag: np.ndarray | None = None
    curve_color: np.ndarray | None = None


def _download_grid(directory: str | Path, filename: str, meh: np.ndarray) -> Path:
    """Download a PARSEC grid (Gaia BP-RP + 2MASS J-K) from the CMD service."""
    directory = Path(directory)
    path = directory / filename
    if not path.exists():
        from . import _parsec

        directory.mkdir(parents=True, exist_ok=True)
        print(f"↓ downloading PARSEC grid -> {path}")
        cmd = _parsec.ParsecQuery()
        lgt = np.arange(6.6, 10.16, 0.05)
        raw = str(cmd.query_isochrones(
            lgt=lgt, MeH=meh, photsys_file=_GAIA_2MASS_PHOTSYS, ret_table=False,
        ))
        path.write_text(raw)
    return path


def ensure_isochrones(directory: str | Path = ISOCHRONE_DIR) -> Path:
    """Solar-metallicity PARSEC grid (5 metallicities, Gaia + 2MASS)."""
    return _download_grid(directory, ISOCHRONE_FILE, np.arange(-0.2, 0.33, 0.13))


def ensure_isochrones_metal_poor(directory: str | Path = ISOCHRONE_DIR) -> Path:
    """Metal-poor PARSEC grid ([M/H] -2.3…-0.3, Gaia + 2MASS)."""
    return _download_grid(directory, ISOCHRONE_METALPOOR_FILE, np.arange(-2.3, -0.25, 0.5))


def _distance_modulus(plx_mas: np.ndarray) -> float:
    finite = plx_mas[np.isfinite(plx_mas) & (plx_mas > 0)]
    if len(finite) == 0:
        return 9.5
    return float(5.0 * np.log10(1000.0 / np.median(finite)) - 5.0)


def fit_isochrone(
    members: pd.DataFrame,
    *,
    seed: int = 42,
    n_walkers: int = 32,
    n_steps: int = 500,
    isochrone_dir: str | Path = ISOCHRONE_DIR,
    grid: str = "solar",
    dm_center: float | None = None,
    dm_width: float = 2.0,
) -> IsochroneFit:
    """Fit a PARSEC isochrone to a membership source's CMD.

    ``members`` needs Gaia photometry — either the APOGEE column names
    (``GAIAEDR3_PHOT_G_MEAN_MAG``, ``GAIAEDR3_PHOT_BP_MEAN_MAG``,
    ``GAIAEDR3_PHOT_RP_MEAN_MAG``, ``GAIAEDR3_PARALLAX``) or the Gaia DR3 names
    (``phot_g_mean_mag``, ``phot_bp_mean_mag``, ``phot_rp_mean_mag``,
    ``parallax``). ``J``/``K`` (2MASS) are optional: when present a second
    colour J-Ks is fitted; otherwise the single BP-RP colour is used. ``grid`` is
    ``"solar"`` (open clusters, Z 0.010–0.030) or ``"metal_poor"`` (globulars,
    [M/H] -2.3…-0.3); it selects the isochrone set and the metallicity prior.
    """
    if grid not in _GRID_MET:
        raise ValueError(f"unknown grid {grid!r}; choose from {sorted(_GRID_MET)}")
    import asteca
    import emcee

    if len(members) < 25:
        raise ValueError(f"need >= 25 members for an isochrone fit, got {len(members)}")

    def _col(*candidates: str) -> str:
        for c in candidates:
            if c in members.columns:
                return c
        raise ValueError(f"missing column; need one of {candidates}")

    g_col = _col("GAIAEDR3_PHOT_G_MEAN_MAG", "phot_g_mean_mag")
    bp_col = _col("GAIAEDR3_PHOT_BP_MEAN_MAG", "phot_bp_mean_mag")
    rp_col = _col("GAIAEDR3_PHOT_RP_MEAN_MAG", "phot_rp_mean_mag")
    plx_col = _col("GAIAEDR3_PARALLAX", "parallax")
    has_jk = "J" in members.columns and "K" in members.columns

    keep = (
        np.isfinite(members[g_col].to_numpy(dtype=float))
        & np.isfinite(members[bp_col].to_numpy(dtype=float))
        & np.isfinite(members[rp_col].to_numpy(dtype=float))
        & (members[plx_col].to_numpy(dtype=float) > 0)
    )
    if has_jk:
        keep &= np.isfinite(members["J"].to_numpy(dtype=float))
        keep &= np.isfinite(members["K"].to_numpy(dtype=float))
    members = members[keep].copy()
    if len(members) < 25:
        raise ValueError(f"need >= 25 members with full photometry, got {len(members)}")

    g = members[g_col].to_numpy(dtype=float)
    bp = members[bp_col].to_numpy(dtype=float)
    rp = members[rp_col].to_numpy(dtype=float)
    plx = members[plx_col].to_numpy(dtype=float)
    color = bp - rp
    color2 = (
        members["J"].to_numpy(dtype=float) - members["K"].to_numpy(dtype=float)
        if has_jk else None
    )
    # Gaia photometric errors are not in allStar; use representative values
    e_g = np.full_like(g, 0.005)
    e_color = np.full_like(color, 0.01)
    e_color2 = np.full_like(color2, 0.02) if has_jk else None

    grid_path = (
        ensure_isochrones_metal_poor(isochrone_dir) if grid == "metal_poor"
        else ensure_isochrones(isochrone_dir)
    )
    isochs = asteca.Isochrones(
        model="parsec",
        isochs_path=str(grid_path),
        mag=_MAG,
        color=_COLOR,
        color2=_COLOR2 if has_jk else None,
        magnitude_effl=_MAG_EFFL,
        color_effl=_COLOR_EFFL,
        color2_effl=_COLOR2_EFFL if has_jk else None,
        verbose=0,
    )
    synth = asteca.Synthetic(isochs, seed=seed, verbose=0)
    cluster_kwargs: dict[str, Any] = dict(
        mag=g, e_mag=e_g, color=color, e_color=e_color,
    )
    if has_jk:
        cluster_kwargs["color2"] = color2
        cluster_kwargs["e_color2"] = e_color2
    cluster = asteca.Cluster(**cluster_kwargs, verbose=0)
    synth.calibrate(cluster)
    lkl = asteca.Likelihood(cluster)

    dm_guess = _distance_modulus(plx) if dm_center is None else dm_center
    bounds = {
        "met": _GRID_MET[grid],
        "loga": _PARAMS["loga"],
        "dm": (dm_guess - dm_width, dm_guess + dm_width),
        "Av": _PARAMS["Av"],
    }
    names = ["met", "loga", "dm", "Av"]
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])

    def _lnprob(theta: np.ndarray) -> float:
        for v, (blo, bhi) in zip(theta, (bounds[n] for n in names)):
            if not (blo <= v <= bhi):
                return -np.inf
        params = {"met": float(theta[0]), "loga": float(theta[1]),
                  "dm": float(theta[2]), "Av": float(theta[3])}
        try:
            gen = synth.generate(params)
            if isinstance(gen, tuple):
                gen = gen[0]
            dist = float(lkl.get(np.asarray(gen)))
        except (ValueError, IndexError):  # out-of-grid params
            return -np.inf
        return -dist  # lkl.get returns a distance (0 = perfect)

    # initialise walkers across the prior (log-uniform in metallicity, which
    # spans decades). Globulars are old, so bias their loga init high — the
    # age-metallicity degeneracy otherwise traps the chain in young modes.
    rng = np.random.default_rng(seed)
    pos = np.empty((n_walkers, 4))
    pos[:, 0] = 10.0 ** rng.uniform(np.log10(lo[0]), np.log10(hi[0]), n_walkers)
    loga_lo = 9.0 if grid == "metal_poor" else lo[1]
    pos[:, 1] = rng.uniform(loga_lo, hi[1], n_walkers)
    pos[:, 2] = rng.uniform(lo[2], hi[2], n_walkers)
    pos[:, 3] = rng.uniform(lo[3], hi[3], n_walkers)

    np.random.seed(seed)  # emcee 3.x draws from the global RNG
    sampler = emcee.EnsembleSampler(n_walkers, 4, _lnprob, vectorize=False)
    sampler.run_mcmc(pos, n_steps, progress=False)

    flat = sampler.get_chain(discard=n_steps // 2, flat=True)
    lp = sampler.get_log_prob(discard=n_steps // 2, flat=True)
    assert flat is not None and lp is not None
    best = flat[int(np.argmax(np.asarray(lp)))]
    std = flat.std(axis=0)
    best_dict = {n: float(best[i]) for i, n in enumerate(names)}

    curve_mag = curve_color = None
    try:
        iso_raw = synth.get_isochrone(best_dict)
        if iso_raw is not None:
            iso = np.asarray(iso_raw)
            curve_mag = iso[:, 0]
            curve_color = iso[:, 1]
    except Exception:
        pass

    return IsochroneFit(
        best=best_dict,
        std={n: float(std[i]) for i, n in enumerate(names)},
        max_lkl=float(np.amax(np.asarray(lp))),
        n_stars=len(members),
        curve_mag=curve_mag,
        curve_color=curve_color,
    )


def plot_isochrone_fit(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cluster_name: str,
    fit: IsochroneFit,
    highlight: str = "combined",
) -> Any:
    """Plotly CMD (apparent G vs BP−RP) with members + the fitted isochrone.

    The isochrone is in apparent magnitude (the distance modulus is a fitted
    parameter), so the observed points are shown in apparent G as well.
    """
    import plotly.graph_objects as go

    g = df["GAIAEDR3_PHOT_G_MEAN_MAG"].to_numpy(dtype=float)
    bp = df["GAIAEDR3_PHOT_BP_MEAN_MAG"].to_numpy(dtype=float)
    rp = df["GAIAEDR3_PHOT_RP_MEAN_MAG"].to_numpy(dtype=float)
    color = bp - rp
    hl = np.asarray(masks.get(highlight, masks["combined"]), dtype=bool)
    ok = np.isfinite(color) & np.isfinite(g)

    age = 10.0 ** (fit.best["loga"] - 9.0)
    keep = thin_field(hl)  # cap the grey field; members are never dropped
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=color[ok & ~hl & keep], y=g[ok & ~hl & keep], mode="markers", name="field",
        marker=dict(size=3, color="#c9c9c9", opacity=0.3), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=color[ok & hl], y=g[ok & hl], mode="markers", name=f"{highlight}",
        marker=dict(size=7, color="#ee1c2e", opacity=0.9),
    ))
    if fit.curve_color is not None and fit.curve_mag is not None:
        fig.add_trace(go.Scatter(
            x=fit.curve_color, y=fit.curve_mag, mode="lines", name="isochrone",
            line=dict(color="#00539f", width=2),
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=(
            f"{cluster_name} — {highlight}: age={age:.2f} Gyr (±{fit.std['loga']:.2f} dex), "
            f"met={fit.best['met']:.4f}±{fit.std['met']:.4f}, "
            f"dm={fit.best['dm']:.2f}±{fit.std['dm']:.2f}, Av={fit.best['Av']:.2f}"
        ),
        xaxis_title="BP − RP",
        yaxis_title="G (apparent)",
        height=460,
        legend=dict(font=dict(size=11)),
    )
    return fig


def isochrone_cell(
    df_hr: pd.DataFrame,
    cluster_name: str,
    method: str,
    settings: Any,
    mo: Any,
) -> Any:
    """Full isochrone fit + plot for the notebook (region cut + masks + emcee)."""
    from .catalog import membership_masks_for
    from .clusters import CLUSTER_BY_NAME
    from .data import complete_case, make_matrix
    from .membership import angular_separation

    cluster = CLUSTER_BY_NAME[cluster_name]
    sep = angular_separation(
        df_hr["RA"].to_numpy(dtype=float), df_hr["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    df = df_hr[sep <= 30.0].copy()
    df = complete_case(df, settings)
    X = make_matrix(df, settings)
    masks = membership_masks_for(df, cluster, X, settings)
    members = df[masks[method]]
    if len(members) < 25:
        return mo.md(f"⚠ only {len(members)} members in '{method}' — need ≥ 25 for a fit")
    fit = fit_isochrone(
        members,
        seed=settings.isofit_seed,
        n_walkers=settings.isofit_n_walkers,
        n_steps=settings.isofit_n_steps,
        grid="metal_poor" if cluster.kind == "globular" else "solar",
    )
    return plot_isochrone_fit(df, masks, cluster.name, fit, highlight=method)


def fit_isochrone_gaia(
    cluster: Cluster,
    *,
    radius_deg: float = 0.5,
    mag_min: float | None = None,
    cache_dir: str | Path = "data/gaia",
    seed: int = 42,
    n_walkers: int = 32,
    n_steps: int = 1000,
    isochrone_dir: str | Path = ISOCHRONE_DIR,
) -> IsochroneFit:
    """Fit an isochrone to a Gaia DR3 CMD down to the main sequence.

    Queries Gaia DR3 around ``cluster`` (full depth, G ≲ 21), selects members
    by a fixed proper-motion cut, and fits the single BP-RP colour. ``mag_min``
    drops bright stars: for globulars set it ≳ 15 to exclude the RGB and blue
    horizontal branch, which otherwise pull the age young — the main sequence
    / turnoff alone cleanly fixes the age.
    """
    from .gaia import gaia_members, query_gaia_region

    df = query_gaia_region(cluster, radius_deg, cache_dir)
    mask = gaia_members(df, cluster)
    members = df[mask]
    if mag_min is not None:
        members = members[members["phot_g_mean_mag"] > mag_min]
    if len(members) < 25:
        raise ValueError(
            f"only {len(members)} Gaia members in {radius_deg}° — need ≥ 25"
        )
    grid = "metal_poor" if cluster.kind == "globular" else "solar"
    return fit_isochrone(
        members,
        seed=seed,
        n_walkers=n_walkers,
        n_steps=n_steps,
        isochrone_dir=isochrone_dir,
        grid=grid,
    )


def plot_gaia_cmd(
    members: pd.DataFrame,
    fit: IsochroneFit,
    cluster_name: str,
) -> Any:
    """Plotly Gaia CMD (G vs BP-RP) with members + the fitted isochrone."""
    import plotly.graph_objects as go

    g = members["phot_g_mean_mag"].to_numpy(dtype=float)
    bp = members["phot_bp_mean_mag"].to_numpy(dtype=float)
    rp = members["phot_rp_mean_mag"].to_numpy(dtype=float)
    color = bp - rp
    ok = np.isfinite(color) & np.isfinite(g)
    age = 10.0 ** (fit.best["loga"] - 9.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=color[ok], y=g[ok], mode="markers", name="Gaia members",
        marker=dict(size=3, color="#ee1c2e", opacity=0.6),
    ))
    if fit.curve_color is not None and fit.curve_mag is not None:
        fig.add_trace(go.Scatter(
            x=fit.curve_color, y=fit.curve_mag, mode="lines", name="isochrone",
            line=dict(color="#00539f", width=2),
        ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=(
            f"{cluster_name} Gaia CMD: age={age:.2f} Gyr (±{fit.std['loga']:.2f} dex), "
            f"met={fit.best['met']:.4f}, dm={fit.best['dm']:.2f}"
        ),
        xaxis_title="BP − RP", yaxis_title="G", height=460,
        legend=dict(font=dict(size=11)),
    )
    return fig


def gaia_age_cell(cluster_name: str, settings: Settings) -> Any:
    """Full Gaia-only age fit + CMD plot for the notebook."""
    from .clusters import CLUSTER_BY_NAME
    from .gaia import gaia_members, query_gaia_region

    cluster = CLUSTER_BY_NAME[cluster_name]
    mag_min = 16.0 if cluster.kind == "globular" else None
    df = query_gaia_region(cluster, 0.5)
    members = df[gaia_members(df, cluster)]
    if mag_min is not None:
        members = members[members["phot_g_mean_mag"] > mag_min]
    grid = "metal_poor" if cluster.kind == "globular" else "solar"
    fit = fit_isochrone(
        members,
        seed=settings.isofit_seed,
        n_walkers=settings.isofit_n_walkers,
        n_steps=settings.isofit_n_steps,
        grid=grid,
    )
    return plot_gaia_cmd(members, fit, cluster.name)
