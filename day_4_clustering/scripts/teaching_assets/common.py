"""Shared helpers for the IAA-SO teaching assets.

Synthetic toy datasets, a GIF writer, and the styling helpers that give every
figure the same didactic look: an iteration counter, a legend, and a footnote
reference so each animation is self-contained.

Usage (from the runner):
    from teaching_assets import common
    common.set_out("results/teaching")
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.datasets import make_blobs, make_moons

# Light-theme palette, keyed to the deck's light colours (RevealDeck.vue LIGHT
# block): purple (--accent), teal (--accent-cyan), green (--accent-green),
# orange (--accent-orange), magenta (--accent-pink), plus a deep blue for a
# sixth category when a plot needs more than five. Noise / unclustered points
# use slate (--comment) so "not a cluster" is never read as a cluster colour;
# field/background points use light slate so they stay in the background. Every
# colour is dark enough to read on a white slide.
PALETTE = ["#7c3aed", "#0e7490", "#c2410c", "#15803d", "#db2777", "#2563eb"]
NOISE = "#64748b"       # unclustered / outliers
FIELD = "#cbd5e1"       # background / field points
UNVISITED = "#e2e8f0"   # not-yet-processed (lightest)
INK = "#21222c"         # main ink (--r-main-color, light theme)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": INK,
        "xtick.color": NOISE,
        "ytick.color": NOISE,
    }
)

# Output directory, set once by the runner before any generator runs.
OUT = "results/teaching"


def set_out(path: str) -> None:
    global OUT
    OUT = path


def png_path(name: str) -> str:
    return f"{OUT}/{name}"


def gif_path(name: str) -> str:
    return f"{OUT}/{name}"


def ref(fig, text: str) -> None:
    """Small italic reference footnote, bottom-left of the figure."""
    fig.text(
        0.01, 0.012, text, fontsize=8.5, style="italic", color=NOISE,
        va="bottom", ha="left",
    )


def save_png(fig, name: str, tight: bool = True) -> str:
    """Write a PNG. `tight=False` keeps the declared figsize exactly, which is
    what a series of step figures needs: cropping to content makes each frame a
    slightly different size, so the image jumps as the presenter advances."""
    path = png_path(name)
    fig.savefig(path, bbox_inches="tight" if tight else None)
    plt.close(fig)
    return path


def save_gif(fig, update, frames: int, name: str, fps: float = 3.0) -> str:
    path = gif_path(name)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Synthetic toy data (all seeded, all 2-D, chosen to make one point each)
# --------------------------------------------------------------------------- #


def blobs(seed: int = 0, n: int = 300):
    """Three roughly-equal Gaussian blobs - the happy case for K-means."""
    X, y = make_blobs(
        n_samples=n, centers=3, cluster_std=[1.0, 0.8, 1.4], random_state=seed
    )
    return X, y


def moons(seed: int = 1, n: int = 300):
    """Two interleaved half-circles - non-spherical, breaks K-means."""
    return make_moons(n_samples=n, noise=0.09, random_state=seed)


def unequal_blobs(seed: int = 3):
    """One big and one small Gaussian - K-means ignores the imbalance."""
    X1, _ = make_blobs(n_samples=400, centers=[(0, 0)], cluster_std=1.0, random_state=seed)
    X2, _ = make_blobs(n_samples=60, centers=[(4.5, 0)], cluster_std=0.25, random_state=seed + 1)
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(400, int), np.ones(60, int)])
    return X, y


def noisy_blobs(seed: int = 2, n: int = 240, n_noise: int = 60):
    """Two blobs plus uniform noise - DBSCAN's 'some points are noise' case."""
    X, y = make_blobs(n_samples=n, centers=2, random_state=seed)
    rng = np.random.default_rng(seed + 1)
    lo, hi = X.min(0), X.max(0)
    noise = rng.uniform(lo - 0.5, hi + 0.5, size=(n_noise, 2))
    Xn = np.vstack([X, noise])
    yn = np.concatenate([y, -np.ones(n_noise, dtype=int)])
    return Xn, yn


def two_scale_blobs(seed: int = 4, n_dense: int = 420, n_sparse: int = 60):
    """A dense and a sparse cluster - one DBSCAN threshold cannot see both."""
    Xd, _ = make_blobs(n_samples=n_dense, centers=[(0, 0)], cluster_std=0.55, random_state=seed)
    Xs, _ = make_blobs(n_samples=n_sparse, centers=[(5, 0)], cluster_std=1.6, random_state=seed + 1)
    X = np.vstack([Xd, Xs])
    y = np.concatenate([np.zeros(n_dense, int), np.ones(n_sparse, int)])
    return X, y


def three_peaks_1d(seed: int = 7, n: int = 400):
    """1-D samples from three Gaussians of different weight (persistence demo)."""
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [
            rng.normal(-3.0, 0.5, 200),
            rng.normal(0.0, 0.7, 130),
            rng.normal(3.5, 0.45, 70),
        ]
    )
    return x
