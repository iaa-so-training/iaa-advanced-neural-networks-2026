"""Figures for the workbook exercises.

The rule the rest of this package follows applies here too: **the notebook
presents, the module computes**. A deck never builds a plot inline — it calls
``figure()`` on the exercise's module, exactly as it calls ``solve()``. This
module holds the three plot families the exercises need, so an individual
exercise states *what* to draw and never *how*.

Three families, matching the three things the workbook actually measures:

``embedding_scatter``
    A 2-D embedding (t-SNE / UMAP / EVoC) coloured by cluster. This is the
    picture behind every "did the clusters separate?" question.

``cmd_diagram``
    A colour-magnitude diagram — Gaia ``BP-RP`` against ``G`` — optionally with
    a fitted isochrone over it. This is the picture behind chapter 15's ages.

``sky_cutout``
    A real survey image of the field with the member stars marked on it,
    through astropy's WCS machinery. This is the picture that answers "are
    these stars actually *there*, or is this an artefact of the abundances?"

Every function returns a matplotlib ``Figure``. Nothing here is interactive:
the decks are executed headless in CI and shipped as static artifacts, so a
plotly widget would render as an empty div for anyone reading the committed
notebook.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

__all__ = [
    "SKYVIEW_CACHE",
    "SkyImageUnavailable",
    "cmd_diagram",
    "embedding_scatter",
    "sky_cutout",
]

#: Where fetched sky cutouts are cached. Under ``results/`` because it is
#: derived data: gitignored, and safe to delete.
SKYVIEW_CACHE = Path("results/exercise_cache/skyview")

#: Set ``EXERCISES_NO_NETWORK=1`` to force the offline path even where a
#: connection exists — what CI does, and what a student on a train wants.
NO_NETWORK = os.environ.get("EXERCISES_NO_NETWORK", "") not in ("", "0")

#: Seconds to wait on SkyView before giving up and drawing the offline panel.
#: SkyView is a live service; a deck must not hang on it.
SKYVIEW_TIMEOUT = float(os.environ.get("EXERCISES_SKYVIEW_TIMEOUT", "30"))


class SkyImageUnavailable(RuntimeError):
    """No survey image could be obtained (offline, or the service failed)."""


# --------------------------------------------------------------------------- #
# Embeddings — the clustering runs
# --------------------------------------------------------------------------- #

def embedding_scatter(
    Z: np.ndarray,
    labels: Any,
    *,
    title: str = "",
    ax: Any = None,
    highlight: str | None = None,
    s: float = 8.0,
) -> Figure:
    """Scatter a 2-D embedding coloured by cluster label.

    ``Z`` is the (n, 2) embedding and ``labels`` the per-row label. Reuses the
    workshop's own palette via :func:`cluster.plots.scatter_embedding`, so an
    exercise figure and the corresponding figure in the workbook colour the
    same cluster the same way.

    ``highlight`` dims every cluster but the named one, which is how the
    single-cluster exercises (M 67, NGC 2243, …) point at their subject.
    """
    import matplotlib.pyplot as plt

    from cluster.plots import scatter_embedding

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 2:
        raise ValueError(f"embedding must be (n, 2); got {Z.shape}")
    names = np.asarray([str(v) for v in labels])
    if len(names) != len(Z):
        raise ValueError(
            f"{len(Z)} embedded rows but {len(names)} labels",
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 6.2), layout="constrained")
    else:
        fig = ax.figure

    if highlight is not None:
        # Everything that is not the subject becomes background, so the eye
        # goes to the one cluster the exercise is about.
        names = np.where(names == highlight, names, "field")

    frame = pd.DataFrame({"label": names})
    scatter_embedding(Z, frame, "label", ax=ax, title=title, s=s)
    return fig


# --------------------------------------------------------------------------- #
# Colour-magnitude diagrams — the isochrone fits
# --------------------------------------------------------------------------- #

def cmd_diagram(
    df: pd.DataFrame,
    *,
    member_mask: Any = None,
    curve_color: Any = None,
    curve_mag: Any = None,
    title: str = "",
    ax: Any = None,
    absolute: bool = False,
    label: str = "members",
) -> Figure:
    """Gaia colour-magnitude diagram, optionally with a fitted isochrone.

    ``df`` carries the Gaia columns the catalogue ships
    (``GAIAEDR3_PHOT_{G,BP,RP}_MEAN_MAG``, and ``GAIAEDR3_PARALLAX`` when
    ``absolute`` is set). ``member_mask`` picks the stars drawn as members;
    everything else is drawn as grey field behind them.

    Pass ``curve_color``/``curve_mag`` — the two arrays an
    :class:`cluster.isochrone.IsochroneFit` carries — to draw the fitted
    isochrone over the points. The fit works in *apparent* magnitude because
    the distance modulus is one of its free parameters, so ``absolute=False``
    is the axis that can be compared against a fit.
    """
    import matplotlib.pyplot as plt

    g = df["GAIAEDR3_PHOT_G_MEAN_MAG"].to_numpy(dtype=float)
    bp = df["GAIAEDR3_PHOT_BP_MEAN_MAG"].to_numpy(dtype=float)
    rp = df["GAIAEDR3_PHOT_RP_MEAN_MAG"].to_numpy(dtype=float)
    colour = bp - rp
    mag = g
    ylabel = "G (apparent)"

    if absolute:
        if curve_mag is not None:
            raise ValueError(
                "an isochrone fit is in apparent magnitude: pass absolute=False "
                "to plot it, or drop the curve",
            )
        parallax = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            # M = m + 5 log10(parallax["]) + 5, parallax in mas.
            mag = g + 5.0 * np.log10(np.where(parallax > 0, parallax, np.nan)) - 10.0
        ylabel = "$M_G$ (absolute)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 6.6), layout="constrained")
    else:
        fig = ax.figure

    finite = np.isfinite(colour) & np.isfinite(mag)
    members = np.ones(len(df), dtype=bool) if member_mask is None else np.asarray(member_mask, dtype=bool)

    field = finite & ~members
    if field.any():
        ax.scatter(
            colour[field], mag[field], s=6, c="#c9c9c9", alpha=0.45,
            linewidths=0, label="field", zorder=1,
        )
    shown = finite & members
    ax.scatter(
        colour[shown], mag[shown], s=18, c="#ee1c2e", alpha=0.85,
        linewidths=0, label=f"{label} ({int(shown.sum())})", zorder=3,
    )

    if curve_color is not None and curve_mag is not None:
        ax.plot(
            np.asarray(curve_color, dtype=float),
            np.asarray(curve_mag, dtype=float),
            color="#00539f", lw=2.0, label="fitted isochrone", zorder=4,
        )

    ax.invert_yaxis()  # brighter stars at the top, as astronomers read it
    ax.set_xlabel("BP − RP")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.15, linewidth=0.5)
    return fig


# --------------------------------------------------------------------------- #
# Sky cutouts — the members marked on a real survey image
# --------------------------------------------------------------------------- #

def _cache_path(cluster_name: str, survey: str, size_deg: float) -> Path:
    from exercises.utils import project_root

    stem = (
        f"{cluster_name.replace(' ', '_')}"
        f"__{survey.replace(' ', '_').replace('/', '-')}"
        f"__{size_deg:.3f}deg.fits"
    )
    return project_root() / SKYVIEW_CACHE / stem


def _fetch_cutout(
    ra_deg: float,
    dec_deg: float,
    cluster_name: str,
    survey: str,
    size_deg: float,
    pixels: int,
) -> Any:
    """An ``astropy`` HDU for the field, from the disk cache or from SkyView.

    Cached as FITS on first fetch: the decks are executed repeatedly (tests,
    CI, students re-running a chapter) and SkyView is a shared public service.
    Raises :class:`SkyImageUnavailable` rather than propagating a network
    error, so callers can draw the offline panel instead of dying.
    """
    from astropy.io import fits

    cached = _cache_path(cluster_name, survey, size_deg)
    if cached.is_file():
        with fits.open(cached) as hdul:
            return hdul[0].copy()

    if NO_NETWORK:
        raise SkyImageUnavailable(
            f"EXERCISES_NO_NETWORK is set and no cached cutout for "
            f"{cluster_name} ({survey}) exists at {cached}",
        )

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.utils.data import conf as data_conf
        from astroquery.skyview import SkyView

        # SkyView exposes no timeout of its own (astroquery 0.4.11): the fetch
        # goes through astropy's downloader, so this is the knob that actually
        # bounds it. Without it a deck can hang on the remote service.
        with data_conf.set_temp("remote_timeout", SKYVIEW_TIMEOUT):
            images = SkyView.get_images(
                position=SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg),
                survey=[survey],
                radius=size_deg * u.deg,
                pixels=str(pixels),
            )
    except Exception as exc:  # noqa: BLE001 - any failure means "draw offline"
        raise SkyImageUnavailable(
            f"could not fetch a {survey} cutout for {cluster_name}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc

    if not images:
        raise SkyImageUnavailable(f"SkyView returned no image for {cluster_name}")

    hdu = images[0][0]
    cached.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(cached, overwrite=True)
    return hdu


def _offline_panel(
    ax: Any,
    ra: np.ndarray,
    dec: np.ndarray,
    members: np.ndarray,
    reason: str,
) -> None:
    """Plain RA/Dec scatter, used when no survey image can be had.

    The exercise still works offline — the astrometry is in the catalogue. Only
    the background image is missing, and the panel says so rather than
    pretending it drew one.
    """
    ax.scatter(
        ra[~members], dec[~members], s=6, c="#c9c9c9", alpha=0.5,
        linewidths=0, label="field",
    )
    ax.scatter(
        ra[members], dec[members], s=26, facecolors="none",
        edgecolors="#ee1c2e", linewidths=1.1,
        label=f"members ({int(members.sum())})",
    )
    ax.invert_xaxis()  # RA increases to the east, i.e. leftwards on the sky
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.text(
        0.5, 0.01, f"no survey image: {reason}", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7, color="#8a8a8a", wrap=True,
    )


def sky_cutout(
    df: pd.DataFrame,
    cluster_name: str,
    *,
    member_mask: Any = None,
    survey: str = "DSS2 Red",
    size_deg: float = 0.6,
    pixels: int = 600,
    title: str = "",
    ax: Any = None,
) -> Figure:
    """A survey image of the field with the member stars circled on it.

    The image comes from SkyView and is drawn through astropy's WCS axes, so
    the markers land on the sky position the catalogue records rather than on
    an approximate pixel grid. The field is centred on the members' own median
    position.

    This is the reality check on a chemical-tagging result: a cluster the
    abundances group together should also be a visible concentration of stars
    on the sky. Where it is not — and for the dispersed open clusters it often
    is not — that is the finding, not a bug.

    Falls back to a plain RA/Dec scatter (clearly labelled) when the image
    cannot be fetched, so the figure still renders offline and in CI.
    """
    import matplotlib.pyplot as plt

    ra = df["RA"].to_numpy(dtype=float)
    dec = df["DEC"].to_numpy(dtype=float)
    members = np.ones(len(df), dtype=bool) if member_mask is None else np.asarray(member_mask, dtype=bool)
    if not members.any():
        raise ValueError(f"no members selected for {cluster_name}")

    centre_ra = float(np.nanmedian(ra[members]))
    centre_dec = float(np.nanmedian(dec[members]))

    try:
        hdu = _fetch_cutout(
            centre_ra, centre_dec, cluster_name, survey, size_deg, pixels,
        )
    except SkyImageUnavailable as exc:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 6.0), layout="constrained")
        else:
            fig = ax.figure
        _offline_panel(ax, ra, dec, members, str(exc).split(":")[0])
        ax.set_title(title or f"{cluster_name} — member positions")
        return fig

    from astropy.visualization import ImageNormalize, ZScaleInterval
    from astropy.wcs import WCS

    wcs = WCS(hdu.header)
    if ax is None:
        # No constrained layout here: it fights WCSAxes' own tick handling and
        # collapses the frame. WCSAxes wants an explicit, square-ish axes box.
        fig = plt.figure(figsize=(6.6, 6.4))
        ax = fig.add_subplot(111, projection=wcs)
    else:
        fig = ax.figure

    image = np.asarray(hdu.data, dtype=float)
    # WCSAxes formats tick labels over the whole frame; near the poles or at a
    # wrap the vectorised formatter evaluates invalid values and numpy warns.
    # The ticks it produces are correct, so silence the noise rather than let
    # it print over every deck that draws a cutout.
    with np.errstate(invalid="ignore"):
        ax.imshow(
            image, cmap="gray_r", origin="lower",
            norm=ImageNormalize(image, interval=ZScaleInterval()),
            aspect="equal", zorder=0,
        )
    # imshow on a WCSAxes leaves the limits at the data edges; make it explicit
    # so a stray marker outside the frame cannot rescale the whole image away.
    ax.set_xlim(-0.5, image.shape[1] - 0.5)
    ax.set_ylim(-0.5, image.shape[0] - 0.5)

    # world=True maps sky coordinates onto the image's own WCS, which is the
    # whole point of doing this through astropy rather than by hand.
    world = ax.get_transform("world")
    others = ~members
    if others.any():
        ax.scatter(
            ra[others], dec[others], transform=world,
            s=26, facecolors="none", edgecolors="#00539f",
            linewidths=0.7, alpha=0.65, label="other catalogue stars",
            zorder=3,
        )
    ax.scatter(
        ra[members], dec[members], transform=world,
        s=90, facecolors="none", edgecolors="#ee1c2e", linewidths=1.3,
        label=f"members ({int(members.sum())})", zorder=4,
    )

    ax.coords[0].set_axislabel("RA (J2000)")
    ax.coords[1].set_axislabel("Dec (J2000)")
    # WCSAxes puts RA ticks on top by default, where they collide with the
    # title; move them under the frame and give the title its own room.
    ax.coords[0].set_ticklabel_position("b")
    ax.coords[0].set_axislabel_position("b")
    ax.set_title(title or f"{cluster_name} — {survey} ({size_deg:g}°)", pad=12)
    ax.legend(fontsize=8, frameon=True, loc="upper right", framealpha=0.85)
    ax.coords.grid(color="#8a8a8a", alpha=0.25, linewidth=0.4)
    return fig
