"""Static per-step figures for the IAA-SO deck.

One PNG per logical step of each algorithm, so the presenter can walk through
the method slowly instead of watching a looping GIF. Reuses the frame/data
generators in iterative.py and embeddings.py, freezing the relevant state.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import common
from .common import PALETTE, NOISE, FIELD, UNVISITED, INK
from . import iterative, embeddings

# Full-slide step figure: wide so it sits under an eyebrow + heading with room
# for a one-line caption on a 1280x720 slide.
FIG = (10.4, 5.2)


def _ax(figsize=FIG):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(fig, ax, title, ref_text, fontsize=14, pad=None):
    ax.set_title(title, fontsize=fontsize, pad=pad)
    ax.set_xticks([])
    ax.set_yticks([])
    common.ref(fig, ref_text)
    return fig


# --------------------------------------------------------------------------- #
# K-means - assign/update loop, frozen one step at a time
#
# The teaching device (after Bishop-style textbook figures): draw the
# *assignment itself*, not just its outcome. A spoke from every point to the
# centre that owns it turns "nearest centre" from a claim into something the
# audience can see, and the bundle of spokes makes the straight-line boundary
# between clusters obvious without drawing it. Centres carry their cluster's
# colour AND its marker shape, so "which centre owns this colour" is never a
# guess; on the update panel the old centre stays as a hollow ghost with an
# arrow to the new one, so the movement is legible from a single still.
# --------------------------------------------------------------------------- #

# One marker per cluster, so the three groups survive a washed-out projector
# and colour-blind viewers - same rule as the DBSCAN core/border/noise figure.
KM_SHAPES = ["o", "s", "^"]
KM_COLORS = [PALETTE[0], PALETTE[1], PALETTE[3]]   # purple, teal, green
# Unassigned stars are the *subject* of the first panel, not background, so
# they get a mid slate rather than the pale FIELD grey - readable on a washed
# out projector, and still obviously "not yet a cluster".
KM_UNASSIGNED = NOISE
# Equal aspect is non-negotiable here (K-means measures plain Euclidean
# distance, so a stretched axis would make "nearest centre" look false), which
# means the axes box only fills a wide figure if the x-limits are padded to
# match. The slack goes mostly on the left, where the legend lives.
# Width is tuned to the padded x-range: with equal aspect matplotlib shrinks the
# axes box to the data ratio and centres it, so a figure wider than `ratio`
# just banks blank margin either side of the cloud.
KM_FIG = (7.6, 5.0)


def _kmeans_states():
    """Data, a deliberately mediocre init, and the first two iterations.

    The init is seeded from three data points bunched on the right of the
    field. That is on purpose: a lucky init converges in one step and the
    update panel then shows centres that barely twitch, which makes the slide
    titled "move each centre" a lie. A poor start gives visible motion, and it
    sets up the next slide's point that where you start decides where you land.
    """
    X, _ = common.blobs(seed=0, n=300)

    # Forgy initialisation: the centres are real data points, so snap the
    # chosen seed positions onto their nearest observations.
    wanted = np.array([[3.2, 5.2], [3.4, 1.2], [1.6, -1.0]])
    idx = [int(np.argmin(((X - w) ** 2).sum(1))) for w in wanted]
    centers0 = X[idx].copy()

    def assign(c):
        return np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), axis=1)

    labels1 = assign(centers0)
    centers1 = np.array([X[labels1 == j].mean(0) for j in range(3)])

    labels, c, n_iter = labels1.copy(), centers1.copy(), 1
    for _ in range(60):
        new_labels = assign(c)
        new = np.array([X[new_labels == j].mean(0) for j in range(3)])
        n_iter += 1
        if np.abs(new - c).max() < 1e-9 and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels, c = new_labels, new
    return X, centers0, labels1, centers1, labels, c, n_iter


def _km_limits(X, *centre_sets, ratio=1.72, left_share=0.5):
    """Data limits padded so an equal-aspect axes fills a `ratio`-wide box."""
    pts = np.vstack([X, *centre_sets])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.08 * (hi - lo)
    y0, y1 = lo[1] - pad[1], hi[1] + pad[1]
    x0, x1 = lo[0] - pad[0], hi[0] + pad[0]
    slack = ratio * (y1 - y0) - (x1 - x0)
    if slack > 0:
        x0 -= slack * left_share
        x1 += slack * (1 - left_share)
    return (x0, x1), (y0, y1)


def _km_points(ax, X, labels, size=26):
    """Points, shaped and coloured by cluster (or neutral when unassigned)."""
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], c=KM_UNASSIGNED, s=size, marker="o",
                   edgecolors="none", zorder=2)
        return
    for j in range(3):
        m = labels == j
        ax.scatter(X[m, 0], X[m, 1], c=KM_COLORS[j], s=size, marker=KM_SHAPES[j],
                   edgecolors="none", zorder=2)


def _km_spokes(ax, X, labels, centers, alpha=0.20, lw=0.6):
    """One faint line per point, to the centre that currently owns it."""
    from matplotlib.collections import LineCollection
    for j in range(3):
        m = labels == j
        if not m.any():
            continue
        segs = [[(p[0], p[1]), (centers[j, 0], centers[j, 1])] for p in X[m]]
        ax.add_collection(LineCollection(segs, colors=KM_COLORS[j], alpha=alpha,
                                         linewidths=lw, zorder=1))


def _km_centres(ax, centers, hollow=False, size=300):
    for j, c in enumerate(centers):
        # White halo so a centre stays readable on top of its own point cloud.
        ax.scatter(*c, marker=KM_SHAPES[j], s=size * 1.55, c="white",
                   edgecolors="none", zorder=4)
        ax.scatter(*c, marker=KM_SHAPES[j], s=size,
                   facecolors="white" if hollow else KM_COLORS[j],
                   edgecolors=KM_COLORS[j] if hollow else INK,
                   linewidths=2.0 if hollow else 1.6, zorder=5)


def _km_legend(ax, labels=None, extra=None):
    from matplotlib.lines import Line2D
    handles = []
    for j in range(3):
        n = "" if labels is None else f"  ({(labels == j).sum()})"
        handles.append(Line2D([], [], marker=KM_SHAPES[j], color="none",
                              markerfacecolor=KM_COLORS[j], markeredgecolor="none",
                              markersize=9, label=f"cluster {j + 1}{n}"))
    if extra:
        handles.extend(extra)
    # A key strip under the title, not a box inside the plot: with equal aspect
    # there is slack beside the cloud, and a floating box there reads as a hole.
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=len(handles), fontsize=11, frameon=False,
              handletextpad=0.4, columnspacing=1.6)


def _km_ax(xlim, ylim, figsize=KM_FIG):
    fig, ax = plt.subplots(figsize=figsize)
    # Equal aspect keeps the blobs round - a stretched axis invents structure
    # that the algorithm never sees, since K-means measures plain Euclidean
    # distance in the data's own units.
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # No frame: the padded x-range leaves slack beside the point cloud, and an
    # empty half-box would read as a missing panel rather than as margin.
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def _km_save(fig, ax, title, ref_text, name):
    """Fixed-canvas save, so the four step frames are pixel-identical in size."""
    _finish(fig, ax, title, ref_text, pad=30)
    fig.subplots_adjust(top=0.80, bottom=0.07, left=0.02, right=0.98)
    common.save_png(fig, name, tight=False)


def make_kmeans_steps():
    X, c0, lab1, c1, lab_f, c_f, n_iter = _kmeans_states()
    xlim, ylim = _km_limits(X, c0, c1, c_f)

    # Step 1 - the data plus K centres, nothing assigned yet. Centres already
    # carry their cluster's colour and shape so the next panel's colours are
    # read as "belongs to that centre", not as three arbitrary groups.
    fig, ax = _km_ax(xlim, ylim)
    _km_points(ax, X, None)
    _km_centres(ax, c0)
    _km_legend(ax, extra=[plt.Line2D([], [], marker="o", color="none",
                                     markerfacecolor=KM_UNASSIGNED,
                                     markeredgecolor="none",
                                     markersize=9, label="unassigned star")])
    _km_save(fig, ax, "Step 1 · choose K and scatter K centres at random",
             "K-means - Lloyd 1982. K is an input, never a result.",
             "kmeans_step1_init.png")

    # Step 2 - the assignment, drawn as spokes. Every star is tied to exactly
    # one centre and the bundles meet along straight lines.
    fig, ax = _km_ax(xlim, ylim)
    _km_spokes(ax, X, lab1, c0)
    _km_points(ax, X, lab1)
    _km_centres(ax, c0)
    _km_legend(ax, lab1)
    _km_save(fig, ax, "Step 2 · assign each star to its nearest centre",
             "Assignment: every star is tied to exactly one centre - the nearest in Euclidean distance.",
             "kmeans_step2_assign.png")

    # Step 3 - the update. Old centres stay as hollow ghosts, arrows show where
    # each one moved; the stars keep the colours they were given in step 2.
    fig, ax = _km_ax(xlim, ylim)
    _km_points(ax, X, lab1)
    _km_centres(ax, c0, hollow=True)
    for j in range(3):
        ax.annotate("", xy=c1[j], xytext=c0[j], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8,
                                    shrinkA=11, shrinkB=13))
    _km_centres(ax, c1)
    _km_legend(ax, lab1, extra=[
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=NOISE, markeredgewidth=2, markersize=9,
                   label="where the centre was"),
    ])
    _km_save(fig, ax, "Step 3 · move each centre to the mean of its stars",
             "Update: centre = mean of the stars assigned to it. The stars themselves do not move.",
             "kmeans_step3_update.png")

    # Step 4 - converged. Spokes are back, and this time no bundle crosses
    # another: every star is already with its nearest centre, so a further
    # pass would change nothing.
    fig, ax = _km_ax(xlim, ylim)
    _km_spokes(ax, X, lab_f, c_f)
    _km_points(ax, X, lab_f)
    _km_centres(ax, c_f)
    _km_legend(ax, lab_f)
    _km_save(fig, ax, f"Step 4 · repeat 2-3 until nothing changes ({n_iter} iterations)",
             "Converged: no star changes cluster, so no centre moves. SSE sits at a local minimum.",
             "kmeans_step4_converged.png")

    _kmeans_storyboard(X, c0, lab1, c1, lab_f, c_f)

    return ["kmeans_step1_init.png", "kmeans_step2_assign.png",
            "kmeans_step3_update.png", "kmeans_step4_converged.png",
            "kmeans_storyboard.png"]


def _kmeans_storyboard(X, c0, lab1, c1, lab_f, c_f):
    """The whole loop on one figure: six panels, read left to right.

    The per-step slides give each beat its own screen; this is the recap the
    audience can hold in their head afterwards, and it makes the repetition
    visible - (c)-(d) and (e)-(f) are the same two moves run twice.
    """
    labels2 = np.argmin(((X[:, None] - c1[None]) ** 2).sum(-1), axis=1)
    c2 = np.array([X[labels2 == j].mean(0) for j in range(3)])

    panels = [
        ("a", "the data", None, None, None, None),
        ("b", "drop K = 3 centres", None, c0, None, None),
        ("c", "assign", lab1, c0, lab1, None),
        ("d", "update", lab1, c1, None, c0),
        ("e", "assign again", labels2, c1, labels2, None),
        ("f", "update, and repeat to convergence", lab_f, c_f, None, None),
    ]
    sx, sy = _km_limits(X, c0, c1, c_f, ratio=1.16, left_share=0.5)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    for ax, (tag, cap, labels, centers, spoke_labels, ghost) in zip(axes.ravel(), panels):
        ax.set_aspect("equal")
        ax.set_xlim(*sx)
        ax.set_ylim(*sy)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if spoke_labels is not None:
            _km_spokes(ax, X, spoke_labels, centers, alpha=0.16, lw=0.5)
        _km_points(ax, X, labels, size=11)
        if ghost is not None:
            _km_centres(ax, ghost, hollow=True, size=95)
            for j in range(3):
                ax.annotate("", xy=centers[j], xytext=ghost[j], zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3,
                                            shrinkA=6, shrinkB=7))
        if centers is not None:
            _km_centres(ax, centers, size=95)
        ax.set_title(f"{tag})  {cap}", fontsize=11, loc="left", pad=5)
    fig.suptitle("One pass of K-means, twice over", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    common.ref(fig, "K-means - Lloyd 1982. Steps (c)-(d) repeat until no star changes colour.")
    common.save_png(fig, "kmeans_storyboard.png")

# --------------------------------------------------------------------------- #
# DBSCAN - knobs, classify, link, result
#
# The teaching device (after the textbook reachability diagrams): draw the
# eps-balls themselves and the arrows between them. Three things fall out that
# no scatter of coloured points can show:
#
#   * eps is a radius, so it has to be drawn as a circle - which means equal
#     aspect. The earlier version of these figures had a stretched axis, so the
#     "eps-ball" rendered as an ellipse on the very slide that defines it.
#   * A cluster is a connected chain of overlapping balls. Draw the balls and
#     the chain is visible; draw only the result and the audience has to take
#     "density-reachable" on trust.
#   * Reachability is asymmetric. A core reaches its border points, but a
#     border point reaches nobody - which is exactly why it joins a cluster and
#     never extends it. Double-headed arrows between cores and single-headed
#     arrows out to borders make that a picture rather than a caveat.
#
# Steps 1-3 share one canvas and one 19-point field, so only the annotation
# changes as the presenter advances; step 4 zooms out to 300 points.
# --------------------------------------------------------------------------- #

DB_EPS, DB_MIN_PTS = 1.0, 4
DB_FIG = (7.8, 4.9)
DB_CLUSTER_C = [PALETTE[0], PALETTE[1]]   # purple, teal
DB_BORDER_C = PALETTE[2]                  # orange
DB_NEUTRAL = NOISE

# Hand-placed so every role is visible and every ball can be drawn: a zigzag
# chain dense enough to be core along its middle, a tight clump far enough away
# to stay a separate cluster, borders hanging off both, and isolated points
# whose balls are visibly empty. Real data would need hundreds of points to
# make the same picture, and then no ball could be drawn at all.
DB_POINTS = np.array([
    (0.80, 2.35), (1.25, 2.62), (1.70, 2.30), (2.15, 2.60), (2.60, 2.28),
    (3.05, 2.58), (3.50, 2.28), (3.95, 2.55), (4.40, 2.32),   # the chain
    (2.85, 3.35),                                             # sits above it
    (6.05, 1.38), (5.59, 1.05), (5.77, 0.51), (6.33, 0.51), (6.51, 1.05),
    (0.55, 0.95), (1.85, 0.30), (3.95, 0.20), (6.60, 2.90),   # isolated
])


def _dbscan_schematic(eps=DB_EPS, min_pts=DB_MIN_PTS):
    """Labels and reachability for the schematic field, all computed.

    Nothing here is asserted by hand: the counts, the three-way split and the
    cluster membership all come out of the same rule the slide states, so the
    figure cannot drift from its own caption.
    """
    X = DB_POINTS
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    within = d <= eps
    counts = within.sum(1)                       # includes the point itself
    core = counts >= min_pts
    border = (~core) & within[:, core].any(1)
    noise = ~core & ~border

    # Clusters = connected components of the core graph, then borders attach to
    # a core that reaches them (whichever comes first - which is exactly the
    # arbitrariness that makes border points the shakiest part of DBSCAN).
    labels = np.full(len(X), -1)
    cid = 0
    for i in np.flatnonzero(core):
        if labels[i] >= 0:
            continue
        stack, cid = [i], cid + 1
        while stack:
            u = stack.pop()
            if labels[u] >= 0:
                continue
            labels[u] = cid - 1
            stack.extend(j for j in np.flatnonzero(core & within[u]) if labels[j] < 0)
    for i in np.flatnonzero(border):
        labels[i] = labels[np.flatnonzero(core & within[i])[0]]
    return X, counts, core, border, noise, labels, within


def _db_limits(X, eps, has_ball, ratio=1.62):
    """Room for the balls, then padded sideways so equal aspect fills the frame."""
    lo = np.minimum((X[has_ball] - eps * 1.02).min(0), (X - 0.3).min(0))
    hi = np.maximum((X[has_ball] + eps * 1.02).max(0), (X + 0.3).max(0))
    y0, y1 = lo[1], hi[1]
    x0, x1 = lo[0], hi[0]
    slack = ratio * (y1 - y0) - (x1 - x0)
    if slack > 0:
        x0, x1 = x0 - slack * 0.5, x1 + slack * 0.5
    return (x0, x1), (y0, y1)


def _db_ax(xlim, ylim, figsize=DB_FIG):
    fig, ax = plt.subplots(figsize=figsize)
    # Equal aspect is not cosmetic here: eps is a radius, so a stretched axis
    # would draw the neighbourhood as an ellipse and every "within eps" claim
    # on the slide would be false of the picture next to it.
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def _db_ball(ax, p, eps, color, fill=0.055, edge=0.55, ls="-", lw=1.0, zorder=0):
    from matplotlib.colors import to_rgba
    ax.add_patch(plt.Circle(tuple(p), eps, facecolor=to_rgba(color, fill),
                            edgecolor=to_rgba(color, edge), lw=lw, ls=ls,
                            zorder=zorder))


def _db_points(ax, X, core, border, noise, labels=None, size=90, legend_counts=False):
    """Core / border / noise, double-encoded by shape and hue."""
    from matplotlib.lines import Line2D
    ax.scatter(X[noise, 0], X[noise, 1], c=DB_NEUTRAL, s=size * 0.8, marker="x",
               linewidths=2.0, zorder=4)
    ax.scatter(X[border, 0], X[border, 1], marker="^", s=size * 1.25,
               facecolors="white", edgecolors=DB_BORDER_C, linewidths=2.0, zorder=4)
    if labels is None:
        ax.scatter(X[core, 0], X[core, 1], c=DB_CLUSTER_C[0], s=size, marker="o",
                   edgecolors="white", linewidths=0.8, zorder=5)
    else:
        for k in range(labels.max() + 1):
            m = core & (labels == k)
            ax.scatter(X[m, 0], X[m, 1], c=DB_CLUSTER_C[k % len(DB_CLUSTER_C)],
                       s=size, marker="o", edgecolors="white", linewidths=0.8, zorder=5)
    if legend_counts:
        handles = [
            Line2D([], [], marker="o", color="none", markerfacecolor=DB_CLUSTER_C[0],
                   markeredgecolor="none", markersize=10,
                   label=f"core — ≥ minPts inside  ({core.sum()})"),
            Line2D([], [], marker="^", color="none", markerfacecolor="white",
                   markeredgecolor=DB_BORDER_C, markeredgewidth=2, markersize=10,
                   label=f"border — inside a core's  ({border.sum()})"),
            Line2D([], [], marker="x", color=DB_NEUTRAL, linestyle="none",
                   markeredgewidth=2, markersize=9,
                   label=f"noise — inside none  ({noise.sum()})"),
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
                  ncol=3, fontsize=10.5, frameon=False, handletextpad=0.4,
                  columnspacing=1.5)


def _db_save(fig, ax, title, ref_text, name):
    _finish(fig, ax, title, ref_text, pad=30)
    fig.subplots_adjust(top=0.80, bottom=0.07, left=0.02, right=0.98)
    common.save_png(fig, name, tight=False)


def _db_note(ax, xy, text, xytext, color=INK, ha="left"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=11, color=color, ha=ha,
                va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0,
                                shrinkA=2, shrinkB=8, alpha=0.7))


def make_dbscan_steps():
    eps, min_pts = DB_EPS, DB_MIN_PTS
    X, counts, core, border, noise, labels, within = _dbscan_schematic()
    # Cores and noise both carry a ball in one step or another; borders never do.
    xlim, ylim = _db_limits(X, eps, core | noise)

    # ---- Step 1: the two knobs, applied to two points ---------------------- #
    # One point passes the test and one just misses it. A pass/fail pair is the
    # whole definition: the threshold only means something if you can see it
    # not being met.
    good, bad = 6, 0
    fig, ax = _db_ax(xlim, ylim)
    ax.scatter(X[:, 0], X[:, 1], c=DB_NEUTRAL, s=70, edgecolors="none", zorder=3)
    for q, colour in ((good, DB_CLUSTER_C[0]), (bad, DB_BORDER_C)):
        _db_ball(ax, X[q], eps, colour, fill=0.07, edge=0.75, lw=1.6)
        inside = np.flatnonzero(within[q])
        ax.scatter(X[inside, 0], X[inside, 1], c=colour, s=70, edgecolors="none", zorder=4)
        ax.scatter(*X[q], c=colour, s=180, edgecolors="white", linewidths=1.6, zorder=6)
    # eps drawn as what it is: a radius. Dashed arrow from the centre to the rim.
    ang = np.deg2rad(153)
    tip = X[good] + eps * np.array([np.cos(ang), np.sin(ang)])
    ax.annotate("", xy=tuple(tip), xytext=tuple(X[good]), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, ls="--", shrinkA=6))
    mid = X[good] + 0.58 * (tip - X[good])
    ax.text(mid[0], mid[1] + 0.16, "ε", fontsize=15, color=INK, ha="center", zorder=8)

    _db_note(ax, X[good] + (0, -eps), f"{counts[good]} points inside, counting itself\n"
             f"{counts[good]} ≥ minPts = {min_pts}   →   core",
             (X[good][0], ylim[0] + 0.42), color=DB_CLUSTER_C[0], ha="center")
    ax.text(X[bad][0], X[bad][1] + eps + 0.28,
            f"only {counts[bad]} inside\n{counts[bad]} < {min_pts}   →   not core",
            fontsize=11, color=DB_BORDER_C, ha="center", va="bottom", zorder=8)
    _db_save(fig, ax, f"Step 1 · pick ε, the radius, and minPts, the count",
             f"DBSCAN - Ester et al. 1996. Here ε = {eps}, minPts = {min_pts} "
             "(the point itself counts towards minPts).",
             "dbscan_step1_knobs.png")

    # ---- Step 2: the same test, applied to everything ---------------------- #
    fig, ax = _db_ax(xlim, ylim)
    for i in np.flatnonzero(core):
        _db_ball(ax, X[i], eps, DB_CLUSTER_C[0], fill=0.05, edge=0.32, lw=0.9)
    _db_points(ax, X, core, border, noise, legend_counts=True)
    _db_note(ax, X[9], "fails the count itself, but sits inside\na core's ball — so it joins that cluster",
             (xlim[1] - 0.25, ylim[1] - 0.22), color=DB_BORDER_C, ha="right")
    _db_note(ax, X[15], "inside nobody's ball → noise",
             (xlim[0] + 0.25, ylim[0] + 0.42), color=DB_NEUTRAL, ha="left")
    _db_save(fig, ax, "Step 2 · run that test on every point: core, border, or noise",
             "Every point gets a label from one rule. Only core points can start or extend a cluster.",
             "dbscan_step2_classify.png")

    # ---- Step 3: link the cores ------------------------------------------- #
    # The arrows are the definition. Two cores within eps of each other reach
    # both ways; a core reaches its borders one way only.
    fig, ax = _db_ax(xlim, ylim)
    for i in np.flatnonzero(noise):
        _db_ball(ax, X[i], eps, DB_NEUTRAL, fill=0.0, edge=0.3, lw=0.8, ls=(0, (3, 4)))
    for i in np.flatnonzero(core):
        _db_ball(ax, X[i], eps, DB_CLUSTER_C[labels[i] % len(DB_CLUSTER_C)],
                 fill=0.05, edge=0.30, lw=0.9)
    ci = np.flatnonzero(core)
    for a in ci:
        for b in ci:
            if b <= a or not within[a, b]:
                continue
            ax.annotate("", xy=tuple(X[b]), xytext=tuple(X[a]), zorder=6,
                        arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1,
                                        shrinkA=7, shrinkB=7, mutation_scale=11))
    for b in np.flatnonzero(border):
        a = np.flatnonzero(core & within[b])[0]
        ax.annotate("", xy=tuple(X[b]), xytext=tuple(X[a]), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=DB_BORDER_C, lw=1.4,
                                    shrinkA=7, shrinkB=9, mutation_scale=13))
    _db_points(ax, X, core, border, noise, labels=labels)
    _db_note(ax, X[18], "its ball is empty —\nno arrow ever arrives",
             (xlim[1] - 0.25, ylim[1] - 0.22), color=DB_NEUTRAL, ha="right")
    _db_note(ax, X[9], "one arrowhead only: the core reaches it,\nit reaches nothing back",
             (xlim[0] + 0.25, ylim[1] - 0.22), color=DB_BORDER_C, ha="left")
    _db_save(fig, ax, "Step 3 · link the cores: a cluster is one chain of overlapping balls",
             "Core to core the link runs both ways; core to border only one way, "
             "which is why a border joins a cluster but cannot grow it.",
             "dbscan_step3_grow.png")

    # ---- Step 4: the same algorithm, 300 points, a shape K-means cannot cut - #
    _dbscan_result()

    return ["dbscan_step1_knobs.png", "dbscan_step2_classify.png",
            "dbscan_step3_grow.png", "dbscan_step4_result.png"]


def _dbscan_result(eps=0.20, min_pts=6):
    """The payoff, on the half-moons K-means split down the middle (slide 11).

    Round blobs would not make the point: "clusters can be any shape" is only
    a claim until the shape is one a centroid cannot carve out.
    """
    from sklearn.cluster import DBSCAN

    Xm, _ = common.moons(seed=1, n=300)
    rng = np.random.default_rng(11)
    lo, hi = Xm.min(0) - 0.15, Xm.max(0) + 0.15
    Xn = np.vstack([Xm, rng.uniform(lo, hi, size=(45, 2))])
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(Xn)

    lo, hi = Xn.min(0), Xn.max(0)
    pad = 0.06 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    slack = 1.62 * (ylim[1] - ylim[0]) - (xlim[1] - xlim[0])
    if slack > 0:
        xlim = (xlim[0] - slack * 0.5, xlim[1] + slack * 0.5)
    fig, ax = _db_ax(xlim, ylim)
    m = lab == -1
    ax.scatter(Xn[m, 0], Xn[m, 1], c=DB_NEUTRAL, s=26, marker="x", linewidths=1.3,
               label=f"noise, left unlabelled ({m.sum()})")
    for k in range(lab.max() + 1):
        m = lab == k
        ax.scatter(Xn[m, 0], Xn[m, 1], c=DB_CLUSTER_C[k % len(DB_CLUSTER_C)], s=28,
                   marker=["o", "s", "^", "D"][k % 4], edgecolors="none",
                   label=f"cluster {k + 1} ({m.sum()})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, fontsize=10.5,
              frameon=False, handletextpad=0.4, columnspacing=1.5)
    _db_save(fig, ax, "Step 4 · read it off: any shape, and the outliers left out",
             f"Same two knobs (ε = {eps}, minPts = {min_pts}) on the half-moons K-means split "
             "down the middle. No centroid, so no straight boundary.",
             "dbscan_step4_result.png")


# --------------------------------------------------------------------------- #
# HDBSCAN* — tree + density sweep (core distance & mutual reachability reused)
# --------------------------------------------------------------------------- #


def make_hdbscan_steps():
    from scipy.cluster.hierarchy import linkage, dendrogram

    X, _ = common.two_scale_blobs(seed=4)
    # subsample for a readable dendrogram
    idx = np.random.default_rng(0).choice(len(X), 80, replace=False)
    Xs = X[idx]
    Z = linkage(Xs, method="single")

    # step 3: single-linkage tree
    fig, ax = _ax()
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0, above_threshold_color=INK)
    ax.axhline(2.6, color=PALETTE[2], lw=2, ls="--")
    ax.annotate("slice here = one density threshold", xy=(0.98, 0.62),
                xycoords="axes fraction", color=PALETTE[2], fontsize=12, ha="right")
    ax.set_xticks([])
    ax.set_yticks([])
    common.ref(fig, "HDBSCAN* - Campello et al. 2013, 2015. Single linkage on mutual-reachability distances.")
    common.save_png(fig, "hdbscan_step3_tree.png")

    # step 4: density sweep (one threshold frozen)
    frames, H, xe, ye = iterative._superlevel_components(X)
    cell_id, thresh, hmax = frames[len(frames) // 3]
    fig, ax = _ax()
    ax.scatter(X[cell_id < 0, 0], X[cell_id < 0, 1], c=UNVISITED, s=30, edgecolors="none")
    for c in range(cell_id.max() + 1):
        m = cell_id == c
        ax.scatter(X[m, 0], X[m, 1], c=PALETTE[c % len(PALETTE)], s=30, edgecolors="none")
    _finish(fig, ax, "Step 4 · slice the tree at every density, keep the branches",
            f"HDBSCAN* keeps every slice - the longest-lived branch is the cluster.")
    common.save_png(fig, "hdbscan_step4_sweep.png")
    return ["hdbscan_step3_tree.png", "hdbscan_step4_sweep.png"]


# --------------------------------------------------------------------------- #
# PLSCAN — density, persistence barcode, read the bars
# --------------------------------------------------------------------------- #


def make_plscan_steps():
    x = common.three_peaks_1d()
    centers, hist, peaks, masses = iterative._peak_masses(x)

    # step 1: density profile with the candidate peaks
    fig, ax = _ax()
    ax.fill_between(centers, hist, alpha=0.25, color=FIELD)
    ax.plot(centers, hist, color=NOISE, lw=1.6)
    for i, p in enumerate(peaks):
        ax.plot([centers[p]], [hist[p]], "o", color=PALETTE[i % len(PALETTE)], ms=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("density")
    common.ref(fig, "PLSCAN - Bot, McInnes & Aerts 2025. Each density peak is a candidate cluster.")
    common.save_png(fig, "plscan_step1_density.png")

    # step 2: persistence barcode (one bar per cluster)
    order = np.argsort(masses)[::-1]
    fig, ax = _ax()
    for i, oi in enumerate(order):
        m = masses[oi]
        ax.barh(i, m, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.text(m + 1, i, f"{m}", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(masses)))
    ax.set_yticklabels([f"cluster {i + 1}" for i in range(len(masses))], fontsize=11)
    ax.set_xlabel("persistence (lifetime over min-cluster-size)")
    common.ref(fig, "PLSCAN - Bot, McInnes & Aerts 2025. Bar length = a cluster's persistence.")
    common.save_png(fig, "plscan_step2_barcode.png")

    # step 3: keep the long bars
    fig, ax = _ax()
    threshold = float(np.median(masses))
    for i, oi in enumerate(order):
        m = masses[oi]
        keep = m >= threshold
        ax.barh(i, m, color=PALETTE[i % len(PALETTE)] if keep else UNVISITED, alpha=0.85)
        ax.text(m + 1, i, f"{m}", va="center", fontsize=11, color=INK)
    ax.axvline(threshold, color=PALETTE[2], lw=2, ls="--")
    ax.set_yticks(range(len(masses)))
    ax.set_yticklabels([f"cluster {i + 1}" for i in range(len(masses))], fontsize=11)
    ax.set_xlabel("persistence")
    common.ref(fig, "PLSCAN - Bot, McInnes & Aerts 2025. Long bars = stable clusters; short bars = noise.")
    common.save_png(fig, "plscan_step3_read.png")
    return ["plscan_step1_density.png", "plscan_step2_barcode.png", "plscan_step3_read.png"]


# --------------------------------------------------------------------------- #
# t-SNE — high-D similarities, low-D similarities, descent
# --------------------------------------------------------------------------- #


def make_tsne_steps():
    # step 1: high-D Gaussian similarities
    fig, ax = _ax()
    gx = np.linspace(-4, 4, 400)
    for sig, c, lab in [(0.5, PALETTE[0], "small σ: few neighbours"),
                        (1.6, PALETTE[1], "large σ: many neighbours")]:
        g = np.exp(-gx ** 2 / (2 * sig ** 2))
        ax.plot(gx, g, color=c, lw=2.4, label=lab)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    ax.set_ylabel("similarity p")
    common.ref(fig, "t-SNE - van der Maaten & Hinton 2008. Each point gets its own bandwidth sigma_i.")
    common.save_png(fig, "tsne_step1_highd.png")

    # step 2: heavy-tailed low-D similarities
    fig, ax = _ax()
    tx = np.linspace(-4, 4, 400)
    gauss = np.exp(-tx ** 2 / (2 * 1.0 ** 2))
    stud = 1.0 / (1.0 + tx ** 2)
    ax.plot(tx, gauss, color=NOISE, lw=2.2, label="Gaussian (would crowd)")
    ax.plot(tx, stud, color=PALETTE[0], lw=2.6, label="Student-t (heavy tail)")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    ax.set_ylabel("similarity q")
    common.ref(fig, "t-SNE - the 't' is the Student-t: a heavy tail lets points spread in 2-D.")
    common.save_png(fig, "tsne_step2_lowd.png")

    # step 3: gradient descent on KL
    X = embeddings._data3d(seed=0, n=220)
    frames = embeddings._tsne_frames(X, n_iter=450, seed=1)
    from sklearn.datasets import make_blobs
    true = make_blobs(n_samples=220, centers=4, n_features=3, cluster_std=0.9, random_state=0)[1]
    Y = frames[-1]
    fig, ax = _ax()
    for c in range(4):
        m = true == c
        ax.scatter(Y[m, 0], Y[m, 1], c=PALETTE[c], s=34, edgecolors="none", label=f"cluster {c + 1}")
    ax.legend(loc="best", fontsize=11, frameon=False)
    _finish(fig, ax, "Step 3 · minimise KL(P‖Q) by gradient descent",
            "t-SNE - gradient descent pulls similar points together, dissimilar apart.")
    common.save_png(fig, "tsne_step3_descent.png")
    return ["tsne_step1_highd.png", "tsne_step2_lowd.png", "tsne_step3_descent.png"]


# --------------------------------------------------------------------------- #
# UMAP — kNN graph, fuzzy edges, layout
# --------------------------------------------------------------------------- #


def make_umap_steps():
    X, _ = common.blobs(seed=0, n=200)
    k = 8
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, inds = nn.kneighbors(X)

    # step 1: kNN graph
    fig, ax = _ax()
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    for i in range(0, len(X), 3):
        for j in inds[i, 1:]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color=NOISE, lw=0.5, alpha=0.5, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    _finish(fig, ax, "Step 1 · join each point to its k nearest neighbours",
            f"UMAP - McInnes, Healy & Melville 2018. k = {k} here.")
    common.save_png(fig, "umap_step1_graph.png")

    # step 2: fuzzy edges (edge thickness = weight)
    fig, ax = _ax()
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    for i in range(0, len(X), 3):
        rho = dists[i, 1]
        for j in inds[i, 1:]:
            d = max(dists[i, list(inds[i]).index(j)] - rho, 0.0)
            w = float(np.exp(-d / 1.0))
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
                    color=PALETTE[0], lw=0.3 + 2.2 * w, alpha=0.5, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    _finish(fig, ax, "Step 2 · rescale each edge: a weight, not a distance",
            "UMAP - fuzzy edges normalise by local density, so sparse and dense regions agree.")
    common.save_png(fig, "umap_step2_fuzzy.png")

    # step 3: layout (attract/repel)
    X3 = embeddings._data3d(seed=0, n=220)
    frames = embeddings._umap_frames(X3, n_epochs=300, seed=2)
    from sklearn.datasets import make_blobs
    true = make_blobs(n_samples=220, centers=4, n_features=3, cluster_std=0.9, random_state=0)[1]
    Y = frames[-1]
    fig, ax = _ax()
    for c in range(4):
        m = true == c
        ax.scatter(Y[m, 0], Y[m, 1], c=PALETTE[c], s=34, edgecolors="none", label=f"cluster {c + 1}")
    ax.legend(loc="best", fontsize=11, frameon=False)
    _finish(fig, ax, "Step 3 · layout: attract along edges, repel non-neighbours",
            "UMAP - minimises cross-entropy against the fuzzy graph (t-SNE uses KL).")
    common.save_png(fig, "umap_step3_layout.png")
    return ["umap_step1_graph.png", "umap_step2_fuzzy.png", "umap_step3_layout.png"]


# --------------------------------------------------------------------------- #
# KNN - the nearest-neighbour primitive, one question at a time
#
# KNN is the only "algorithm" in the deck that is not a clusterer, so its steps
# answer a different question: not "where are the groups" but "what does a
# neighbourhood give you". Each panel hands one product to a later method -
# the k-th distance (DBSCAN's density test), the core distance kappa (HDBSCAN*'s
# mutual reachability), and the edge list itself (UMAP's and EVoC's graph).
# --------------------------------------------------------------------------- #


def make_knn_steps():
    from sklearn.neighbors import NearestNeighbors

    # A moderate density contrast on purpose: two_scale_blobs differ by ~30x in
    # kappa, which makes the dense circle a dot on the slide. ~3x reads.
    from sklearn.datasets import make_blobs
    Xd, _ = make_blobs(n_samples=200, centers=[(0, 0)], cluster_std=0.55, random_state=4)
    Xs, _ = make_blobs(n_samples=60, centers=[(4.6, 0)], cluster_std=1.15, random_state=5)
    X = np.vstack([Xd, Xs])
    k = 6
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, inds = nn.kneighbors(X)

    # Two probe points, one from each blob, taken near each blob's centre so the
    # contrast is the blob's density rather than an edge effect.
    dense_i = int(np.argmin(np.linalg.norm(X - np.array([0.0, 0.0]), axis=1)))
    sparse_i = int(np.argmin(np.linalg.norm(X - np.array([4.6, 0.0]), axis=1)))

    # ---- step 1: pick k, measure every distance ----
    fig, ax = _ax()
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    p = X[dense_i]
    for j in range(0, len(X), 2):
        ax.plot([p[0], X[j, 0]], [p[1], X[j, 1]], color=NOISE, lw=0.35, alpha=0.30, zorder=1)
    ax.scatter(*p, c=PALETTE[0], s=190, marker="*", edgecolors="white",
               linewidths=1.0, zorder=5, label="the point x")
    ax.legend(loc="best", fontsize=11, frameon=False)
    _finish(fig, ax, f"Step 1 \u00b7 pick k, then measure d(x, x\u1d62) to every other point",
            f"KNN - the only knob is k (here k = {k}). Naively O(N) per query; a space tree makes it ~O(log N).")
    common.save_png(fig, "knn_step1_distances.png")

    # ---- step 2: keep the k smallest ----
    fig, ax = _ax()
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    for j in inds[dense_i, 1:]:
        ax.plot([p[0], X[j, 0]], [p[1], X[j, 1]], color=PALETTE[0], lw=1.4, alpha=0.85, zorder=3)
    ax.scatter(X[inds[dense_i, 1:], 0], X[inds[dense_i, 1:], 1], c=PALETTE[0], s=64,
               edgecolors="white", linewidths=0.8, zorder=4, label=f"the {k} nearest")
    ax.scatter(*p, c=PALETTE[0], s=190, marker="*", edgecolors="white",
               linewidths=1.0, zorder=5, label="the point x")
    ax.legend(loc="best", fontsize=11, frameon=False)
    _finish(fig, ax, f"Step 2 \u00b7 keep the k smallest \u2014 that is the neighbourhood",
            "KNN - everything downstream is a different use of this one list.")
    common.save_png(fig, "knn_step2_neighbourhood.png")

    # ---- step 3: the k-th distance IS a density estimate ----
    fig, ax = _ax()
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=26, edgecolors="none", zorder=2)
    for i, col, tag in [(dense_i, PALETTE[0], "dense"), (sparse_i, PALETTE[2], "sparse")]:
        r = dists[i, k]
        ax.add_patch(plt.Circle(X[i], r, facecolor=col, alpha=0.10,
                                edgecolor=col, lw=1.4, zorder=1))
        ax.scatter(*X[i], c=col, s=180, marker="*", edgecolors="white",
                   linewidths=1.0, zorder=5,
                   label=f"{tag}: \u03ba(x) = {r:.2f}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=11, frameon=False)
    _finish(fig, ax, "Step 3 \u00b7 the k-th distance \u03ba(x) is a free density estimate",
            "KNN - small \u03ba = packed, large \u03ba = sparse. DBSCAN thresholds it; HDBSCAN* builds mutual reachability from it.")
    common.save_png(fig, "knn_step3_core_distance.png")

    # ---- step 4: keep the edges, and you have a graph ----
    fig, ax = _ax()
    for i in range(len(X)):
        for j in inds[i, 1:4]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
                    color=PALETTE[1], lw=0.45, alpha=0.45, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=PALETTE[1], s=26, edgecolors="none", zorder=2)
    _finish(fig, ax, "Step 4 \u00b7 keep the edges \u2014 now clustering is a graph question",
            "KNN - this graph is the object UMAP lays out and EVoC clusters. Same primitive, four different products.")
    common.save_png(fig, "knn_step4_graph.png")

    return ["knn_step1_distances.png", "knn_step2_neighbourhood.png",
            "knn_step3_core_distance.png", "knn_step4_graph.png"]


# --------------------------------------------------------------------------- #
# EVoC - the fusion, shown as the three borrowed parts plus the one new idea
#
# EVoC is the deck's destination, and the risk is that it reads as a black box
# after five slides of mechanism. So its steps deliberately re-show the SAME
# objects the audience has already met - the kNN graph from UMAP, the condensed
# tree from HDBSCAN*, the persistence barcode from PLSCAN - and the only new
# panel is the one that matters: scoring whole layers and keeping the winner.
# --------------------------------------------------------------------------- #


def _layer_stability(X, min_cluster_size):
    """HDBSCAN-style cluster stability for one layer of the hierarchy.

    sklearn's HDBSCAN exposes only labels_ and probabilities_, so the quantity
    EVoC and PLSCAN actually score has to be computed here. Works on the mutual
    reachability distance, whose single-linkage tree is HDBSCAN's hierarchy.

    Returns ``(labels, mean_stability)`` with stability measured in
    lambda = 1/distance units, so a cluster that survives a long span of
    density thresholds scores high.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.neighbors import NearestNeighbors

    k = max(2, min(min_cluster_size, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    core = nn.kneighbors(X)[0][:, -1]

    D = squareform(pdist(X))
    # mutual reachability: max(core_i, core_j, d_ij)
    MR = np.maximum(np.maximum(core[:, None], core[None, :]), D)
    Z = linkage(squareform(MR, checks=False), method="single")

    # Walk merge heights; a cluster is born when it first reaches
    # min_cluster_size and dies when it merges into something larger.
    heights = np.unique(Z[:, 2])
    if heights.size < 2:
        return np.full(len(X), -1), 0.0
    # Sample the height axis so the sweep is cheap but faithful.
    probe = np.quantile(heights, np.linspace(0.02, 0.98, 60))

    # Sweep from LARGE h to SMALL h, i.e. low lambda -> high lambda, which is
    # HDBSCAN's direction: a cluster is born when it separates out at
    # lambda_birth and dies at a higher lambda when it fragments, so
    # stability = (lambda_death - lambda_birth) * size is positive. Sweeping
    # the other way makes every stability negative and the figure all zeros.
    born, stab = {}, {}
    prev = {}
    for h in probe[::-1]:
        lab = fcluster(Z, t=h, criterion="distance")
        lam = 1.0 / h if h > 0 else 0.0
        sizes_here = {}
        for c in np.unique(lab):
            m = lab == c
            if m.sum() >= min_cluster_size:
                sizes_here[frozenset(np.flatnonzero(m))] = lam
        for key, lam_now in sizes_here.items():
            if key not in born:
                born[key] = lam_now
        # anything that vanished since the last probe has died
        for key, lam_birth in list(born.items()):
            if key not in sizes_here and key not in stab:
                lam_death = prev.get(key, lam_birth)
                stab[key] = max(0.0, lam_death - lam_birth) * len(key)
        prev = sizes_here

    # clusters still alive at the end
    for key, lam_birth in born.items():
        if key not in stab:
            stab[key] = max(0.0, prev.get(key, lam_birth) - lam_birth) * len(key)

    # The layer's own partition, for the cluster count on the slide.
    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    except Exception:
        lab = np.full(len(X), -1)

    vals = [v for v in stab.values() if v > 0]
    return lab, (float(np.mean(vals)) if vals else 0.0)


def make_evoc_steps():
    from sklearn.neighbors import NearestNeighbors

    X, y = common.two_scale_blobs(seed=4, n_dense=260, n_sparse=70)
    k = 10
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, inds = nn.kneighbors(X)

    # ---- step 1: the kNN graph, in the ORIGINAL space ----
    fig, ax = _ax()
    for i in range(len(X)):
        for j in inds[i, 1:4]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
                    color=NOISE, lw=0.4, alpha=0.40, zorder=1)
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=26, edgecolors="none", zorder=2)
    _finish(fig, ax, "Step 1 \u00b7 kNN graph on the raw features (cosine geometry)",
            "EVoC - borrowed from UMAP. No 2-D picture is ever made.")
    common.save_png(fig, "evoc_step1_graph.png")

    # ---- step 2: its own node embedding, not a 2-D picture ----
    Y = embeddings._umap_frames(
        np.column_stack([X, np.zeros(len(X))]), n_epochs=250, seed=3)[-1]
    fig, ax = _ax()
    ax.scatter(Y[:, 0], Y[:, 1], c=FIELD, s=30, edgecolors="none", zorder=2)
    _finish(fig, ax, "Step 2 \u00b7 embed the graph into 4\u201315-D of its own",
            "EVoC - sized for the clusterer, not for your eyes (shown here in 2-D only so it can be drawn).")
    common.save_png(fig, "evoc_step2_embed.png")

    # ---- step 3: density clustering on the embedding ----
    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=12).fit_predict(Y)
    except Exception:
        from sklearn.cluster import DBSCAN
        lab = DBSCAN(eps=0.8, min_samples=8).fit_predict(Y)
    fig, ax = _ax()
    noise = lab < 0
    ax.scatter(Y[noise, 0], Y[noise, 1], c=NOISE, s=22, edgecolors="none",
               zorder=2, label="noise")
    for c, lb in enumerate(sorted(set(lab[~noise]))):
        m = lab == lb
        ax.scatter(Y[m, 0], Y[m, 1], c=PALETTE[c % len(PALETTE)], s=32,
                   edgecolors="none", zorder=3, label=f"cluster {c + 1}")
    ax.legend(loc="best", fontsize=10, frameon=False)
    _finish(fig, ax, "Step 3 \u00b7 mutual-reachability MST + condensed tree on THAT embedding",
            "EVoC - borrowed from HDBSCAN*, unchanged. Field stars get a noise label.")
    common.save_png(fig, "evoc_step3_cluster.png")

    # ---- step 4: the one genuinely new idea - score whole layers ----
    # sklearn's HDBSCAN does not expose cluster_persistence_ (it is None), so
    # compute the real thing: HDBSCAN-style cluster stability on the mutual
    # reachability single-linkage tree, in lambda = 1/distance units.
    #   stability(C) = sum over members of (lambda_leave - lambda_birth)
    # A layer's score is the MEAN stability of its clusters - the mean, not the
    # sum, because summing rewards shattering the data into many specks.
    # Score the FEATURE space, not the 2-D picture. Scoring the 2-D UMAP view
    # rewards whatever the projection happened to merge - here it crowns a
    # 3-cluster layer over the true 5 - which is exactly the "never cluster the
    # 2-D view" mistake this deck warns about two slides earlier.
    Xs = (X - X.mean(0)) / X.std(0)
    # Range chosen so every bar is informative: above ~20 this dataset has no
    # cluster left to score and the layer is empty.
    sizes = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
    scores, counts = [], []
    for mcs in sizes:
        lb, score = _layer_stability(Xs, mcs)
        n_here = len(set(lb[lb >= 0]))
        # A layer where the clusterer itself returns nothing must not show a
        # stability bar - our own sweep and sklearn's labelling would otherwise
        # disagree on the slide ("0 clusters" over a tall bar).
        scores.append(score if n_here > 0 else 0.0)
        counts.append(n_here)

    best = int(np.argmax(scores))
    fig, ax = plt.subplots(figsize=FIG)
    bars = ax.bar(range(len(sizes)), scores, color=FIELD, zorder=2)
    bars[best].set_color(PALETTE[0])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("min_cluster_size (one layer of the hierarchy each)")
    ax.set_ylabel("mean cluster stability")
    top = max(scores) if max(scores) > 0 else 1.0
    for i, (s, n) in enumerate(zip(scores, counts)):
        ax.text(i, s + top * 0.03, f"{n}", ha="center", fontsize=9, color=INK)
    ax.text(best, scores[best] + top * 0.11, "winner", ha="center", fontsize=11,
            color=PALETTE[0], fontweight="bold")
    ax.set_ylim(0, top * 1.28)
    ax.set_title("Step 4 \u00b7 score every layer by persistence \u2014 keep the best one",
                 fontsize=14)
    ax.set_yticks([])
    common.ref(fig, "EVoC - borrowed from PLSCAN. Numbers above the bars are clusters found. No scale to guess.")
    common.save_png(fig, "evoc_step4_persistence.png")

    return ["evoc_step1_graph.png", "evoc_step2_embed.png",
            "evoc_step3_cluster.png", "evoc_step4_persistence.png"]
