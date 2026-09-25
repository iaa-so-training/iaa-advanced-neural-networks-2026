"""Iterative / partitional algorithms as didactic animations.

K-means (assign-update loop), KNN (the density probe), DBSCAN (region
growing), HDBSCAN* (density-threshold sweep) and PLSCAN (persistence over
min-cluster-size). Every GIF frames one decision the algorithm makes, with a
legend and a reference footnote.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from . import common

from .common import PALETTE, NOISE, FIELD, UNVISITED, INK


# --------------------------------------------------------------------------- #
# K-means
# --------------------------------------------------------------------------- #


def _kmeans_steps(X, k, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.full(len(X), -1)
    steps = [(labels.copy(), centers.copy())]  # initial: nothing assigned
    for _ in range(60):
        labels = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        steps.append((labels.copy(), centers.copy()))  # after assignment
        new_centers = np.array(
            [X[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(k)]
        )
        if np.abs(new_centers - centers).max() < 1e-6:
            break
        centers = new_centers
    steps.append((labels.copy(), centers.copy()))
    return steps


def make_kmeans_gif():
    X, _ = common.blobs(seed=0, n=300)
    steps = _kmeans_steps(X, 3, seed=42)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    def update(i):
        ax.clear()
        labels, centers = steps[i]
        if (labels == -1).all():
            ax.scatter(X[:, 0], X[:, 1], c=UNVISITED, s=18, edgecolors="none")
        else:
            for c in range(3):
                m = labels == c
                ax.scatter(X[m, 0], X[m, 1], c=PALETTE[c], s=18, edgecolors="none", label=f"cluster {c + 1}")
        ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=340, c=INK,
                   edgecolors="white", linewidths=0.8, zorder=3, label="centroids")
        ax.set_title(f"K-means - iteration {i} of {len(steps) - 1}", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        common.ref(fig, "K-means - Lloyd 1982; MacQueen 1967. Assignment: nearest centre; update: mean of assigned points.")

    return common.save_gif(fig, update, len(steps), "kmeans.gif", fps=2.2)


def make_kmeans_init():
    """Two different initialisations on the same data -> two different local minima.

    The blobs here are deliberately closer together than in make_kmeans_gif so a
    bad initialisation genuinely traps K-means in a worse minimum: the good run
    finds the 3-way split (SSE ~331), the bad run merges two clusters (SSE ~553).
    Initialisation indices are hardcoded so the two panels are reproducible.
    """
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=7, center_box=(-3.0, 3.0))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    def run(init):
        centers = X[list(init)].copy()
        labels = np.zeros(len(X), int)
        for _ in range(100):
            labels = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
            new = np.array([X[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(3)])
            if np.abs(new - centers).max() < 1e-6:
                break
            centers = new
        sse = sum(((X[labels == j] - centers[j]) ** 2).sum() for j in range(3))
        return labels, centers, sse

    # Good init: one representative point per true blob -> global minimum.
    # Bad init: two points in the same blob, one in another -> merged clusters.
    for ax, (init, tag) in zip(axes, [([153, 44, 290], "good init"), ([195, 10, 164], "bad init")]):
        labels, centers, sse = run(init)
        for c in range(3):
            m = labels == c
            ax.scatter(X[m, 0], X[m, 1], c=PALETTE[c], s=16, edgecolors="none")
        ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=300, c=INK,
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(f"{tag} - SSE = {sse:,.0f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("K-means: same data, different initialisation - different local minimum", fontsize=13)
    common.ref(fig, "K-means - Lloyd 1982. SSE = sum of squared error within clusters (chapter Fig. 13.3).")
    return common.save_png(fig, "kmeans_init.png")


def make_kmeans_fail():
    """K-means on non-spherical (moons) and unbalanced (blobs) data."""
    from sklearn.cluster import KMeans

    Xm, _ = common.moons(seed=1, n=300)
    Xu, _ = common.unequal_blobs(seed=3)
    km_m = KMeans(2, n_init=10, random_state=0).fit(Xm)
    km_u = KMeans(2, n_init=10, random_state=0).fit(Xu)

    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2))
    fig.subplots_adjust(hspace=0.5)
    for c in range(2):
        m = km_m.labels_ == c
        axes[0].scatter(Xm[m, 0], Xm[m, 1], c=PALETTE[c], s=16, edgecolors="none")
    axes[0].set_title("non-spherical: K-means splits each half-moon", fontsize=12)
    for c in range(2):
        m = km_u.labels_ == c
        axes[1].scatter(Xu[m, 0], Xu[m, 1], c=PALETTE[c], s=16, edgecolors="none")
    axes[1].set_title("unbalanced: K-means cuts the big cluster", fontsize=12)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("K-means assumes round, similar-size clusters", fontsize=13)
    common.ref(fig, "K-means - Lloyd 1982. Failure modes from the Clustering analysis chapter (Fig. 13.4).")
    return common.save_png(fig, "kmeans_fail.png")


# --------------------------------------------------------------------------- #
# KNN - the density probe
# --------------------------------------------------------------------------- #


def make_knn_probe():
    X, _ = common.two_scale_blobs(seed=4)
    k = 6
    # query points: one in the dense blob, one in the sparse blob
    dense_center = X[:420].mean(0)
    sparse_center = X[420:].mean(0)
    q_dense = X[np.argmin(((X[:420] - dense_center) ** 2).sum(1))]
    q_sparse = X[420 + np.argmin(((X[420:] - sparse_center) ** 2).sum(1))]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(X[:420, 0], X[:420, 1], c=FIELD, s=14, edgecolors="none", label="dense region")
    ax.scatter(X[420:, 0], X[420:, 1], c=UNVISITED, s=14, edgecolors="none", label="sparse region")

    for q, color, tag in [(q_dense, "#0e7490", "dense"), (q_sparse, "#c2410c", "sparse")]:
        d = np.linalg.norm(X - q, axis=1)
        order = np.argsort(d)
        nn = order[:k]
        kappa = d[order[k - 1]]
        ax.scatter(*q, c=color, s=90, zorder=5, edgecolors="white", linewidths=1)
        ax.scatter(X[nn, 0], X[nn, 1], c=color, s=26, alpha=0.85, edgecolors="white", linewidths=0.6)
        circ = plt.Circle(q, kappa, fill=False, color=color, lw=1.6, ls="--")
        ax.add_patch(circ)
        ax.annotate(
            f"core distance kappa = {kappa:.2f} ({tag})",
            (q[0], q[1] + 0.7), color=color, fontsize=12, ha="center",
        )

    ax.set_title(f"KNN: distance to the {k}-th neighbour is a local density probe", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    common.ref(fig, "Core distance kappa(x) = distance to the k-th nearest neighbour - McInnes & Healy 2017 (HDBSCAN).")
    return common.save_png(fig, "knn_core_distance.png")


# --------------------------------------------------------------------------- #
# DBSCAN - region growing
# --------------------------------------------------------------------------- #


def _dbscan_steps(X, eps, min_pts):
    n = len(X)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    neigh = [np.where(D[i] <= eps)[0] for i in range(n)]
    core = np.array([len(neigh[i]) >= min_pts for i in range(n)])
    labels = np.full(n, -1)  # -1 unvisited, -2 noise, >=0 cluster id
    steps = []
    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        if not core[i]:
            labels[i] = -2
            steps.append((labels.copy(), i, -1))
            continue
        cluster += 1
        labels[i] = cluster
        steps.append((labels.copy(), i, cluster))
        queue = list(neigh[i])
        for j in queue:
            if labels[j] == -2:
                labels[j] = cluster  # was noise, now border of this cluster
                steps.append((labels.copy(), j, cluster))
            if labels[j] != -1:
                continue
            labels[j] = cluster
            steps.append((labels.copy(), j, cluster))
            if core[j]:
                queue.extend(neigh[j])
    return steps, core


def make_dbscan_gif():
    X, _ = common.noisy_blobs(seed=2, n=240, n_noise=60)
    eps, min_pts = 1.1, 5
    steps, core = _dbscan_steps(X, eps, min_pts)

    # subsample to a fixed number of frames for a compact GIF
    target = 46
    if len(steps) > target:
        idx = np.unique(np.linspace(0, len(steps) - 1, target).astype(int))
    else:
        idx = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    def update(t):
        ax.clear()
        labels, active, cluster = steps[idx[t]]
        n_clusters = max([l for l in labels if l >= 0], default=0)
        for c in range(1, n_clusters + 1):
            m = labels == c
            ax.scatter(X[m, 0], X[m, 1], c=PALETTE[(c - 1) % len(PALETTE)], s=18, edgecolors="none")
        # noise and unvisited
        ax.scatter(X[labels == -2, 0], X[labels == -2, 1], c=NOISE, s=18, marker="x", label="noise")
        ax.scatter(X[labels == -1, 0], X[labels == -1, 1], c=UNVISITED, s=18, edgecolors="none", label="unvisited")
        if active >= 0 and core[active]:
            circ = plt.Circle(X[active], eps, fill=False, color="k", lw=1.2, ls=":")
            ax.add_patch(circ)
            ax.scatter(*X[active], c=INK, s=90, zorder=5, edgecolors="white", linewidths=1)
        ax.set_title(f"DBSCAN - growing clusters by density-reachability", fontsize=12)
        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        common.ref(fig, f"DBSCAN - Ester et al. 1996. eps = {eps}, minPts = {min_pts}. Core point: >= minPts neighbours within eps.")

    return common.save_gif(fig, update, len(idx), "dbscan.gif", fps=4.0)


def make_dbscan_static():
    from sklearn.cluster import DBSCAN

    X, _ = common.noisy_blobs(seed=2, n=240, n_noise=60)
    eps, min_pts = 1.1, 5
    lab = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(X)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for c in sorted(set(lab)):
        if c == -1:
            ax.scatter(X[lab == -1, 0], X[lab == -1, 1], c=NOISE, s=18, marker="x", label="noise")
        else:
            ax.scatter(X[lab == c, 0], X[lab == c, 1], c=PALETTE[c % len(PALETTE)], s=18,
                       edgecolors="none", label=f"cluster {c + 1}")
    ax.set_title("DBSCAN: arbitrary shape, and it labels outliers", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    common.ref(fig, f"DBSCAN - Ester et al. 1996. eps = {eps}, minPts = {min_pts}.")
    return common.save_png(fig, "dbscan.png")


# --------------------------------------------------------------------------- #
# HDBSCAN* - mutual reachability + density-threshold sweep
# --------------------------------------------------------------------------- #


def make_mutual_reachability():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(2, 2))
    X[1] = [3.2, 0.4]
    k = 4
    D = np.linalg.norm(X[:, None] - X[None], axis=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(*X[0], c=PALETTE[0], s=140, zorder=5, edgecolors="white", linewidths=1, label="point i")
    ax.scatter(*X[1], c=PALETTE[1], s=140, zorder=5, edgecolors="white", linewidths=1, label="point j")
    # a small cloud around each to make core distance meaningful
    for base, color in [(X[0], PALETTE[0]), (X[1], PALETTE[1])]:
        cloud = rng.normal(base, 0.35, size=(30, 2))
        ax.scatter(cloud[:, 0], cloud[:, 1], c=color, alpha=0.35, s=20, edgecolors="none")
    # core distance = k-th NN distance within each cloud
    kappa = np.array([0.85, 0.9])  # illustrative values
    for p, kk, color in [(X[0], kappa[0], PALETTE[0]), (X[1], kappa[1], PALETTE[1])]:
        ax.add_patch(plt.Circle(p, kk, fill=False, color=color, lw=1.6, ls="--"))
    ax.plot([X[0, 0], X[1, 0]], [X[0, 1], X[1, 1]], "k--", lw=1)
    d = np.linalg.norm(X[0] - X[1])
    mid = (X[0] + X[1]) / 2
    ax.annotate(f"d(i,j) = {d:.2f}", mid + (0.1, 0.25), fontsize=9, ha="center")
    ax.annotate("kappa(i) = distance to k-th neighbour", (X[0, 0], X[0, 1] - 1.5), color=PALETTE[0], fontsize=9, ha="center")
    ax.annotate("kappa(j)", (X[1, 0], X[1, 1] - 1.5), color=PALETTE[1], fontsize=9, ha="center")
    ax.set_title("Mutual reachability flattens dense regions", fontsize=12)
    ax.set_xlim(-2, 5.5); ax.set_ylim(-2.5, 3.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    common.ref(fig, "d_mut(i,j) = max(kappa_i, kappa_j, d_ij) - Campello et al. 2013, 2015 (HDBSCAN*).")
    return common.save_png(fig, "mutual_reachability.png")


def _superlevel_components(X, bins=55):
    """Cluster points by connected components of the density superlevel sets."""
    from scipy.ndimage import label as ndlabel

    H, xe, ye = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    xs = (X[:, 0] - xe[0]) / (xe[1] - xe[0])
    ys = (X[:, 1] - ye[0]) / (ye[1] - ye[0])
    bxi = np.clip((xs * bins).astype(int), 0, bins - 1)
    byi = np.clip((ys * bins).astype(int), 0, bins - 1)
    levels = np.linspace(H.max(), H[H > 0].min(), 36)[::-1]
    frames = []
    for t in levels:
        comps, ncomp = ndlabel(H >= t)
        cell_id = comps[bxi, byi] - 1  # -1 = below threshold
        frames.append((cell_id, t, H.max()))
    return frames, H, xe, ye


def make_hdbscan_sweep():
    X, _ = common.two_scale_blobs(seed=4)
    frames, H, xe, ye = _superlevel_components(X)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    def update(t):
        ax.clear()
        cell_id, thresh, hmax = frames[t]
        # below-threshold points grey
        ax.scatter(X[cell_id < 0, 0], X[cell_id < 0, 1], c=UNVISITED, s=18, edgecolors="none")
        for c in range(cell_id.max() + 1):
            m = cell_id == c
            ax.scatter(X[m, 0], X[m, 1], c=PALETTE[c % len(PALETTE)], s=18, edgecolors="none")
        ax.set_title(
            f"HDBSCAN*: clusters at every density level - lambda = {thresh / hmax:.2f} lambda_max",
            fontsize=11,
        )
        ax.set_xticks([]); ax.set_yticks([])
        common.ref(fig, "HDBSCAN* - Campello et al. 2013, 2015. One threshold -> a hierarchy over all density scales.")

    return common.save_gif(fig, update, len(frames), "hdbscan_density.gif", fps=4.0)


# --------------------------------------------------------------------------- #
# PLSCAN - persistence over min-cluster-size
# --------------------------------------------------------------------------- #


def _peak_masses(x, nbins=80):
    """Mode mass (lifetime in min-cluster-size) for each density peak.

    Smooths the histogram first so noise bins do not register as spurious
    modes; mass is the smoothed density integrated over the mode's basin.
    """
    from scipy.ndimage import gaussian_filter1d

    hist, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    peaks, _ = signal.find_peaks(smooth, prominence=smooth.max() * 0.05, distance=10)
    # order peaks by height desc
    order = peaks[np.argsort(smooth[peaks])[::-1]]
    masses = []
    for p in order:
        # basin = until the smoothed density stops rising on both sides
        left = p
        while left > 0 and smooth[left - 1] <= smooth[left]:
            left -= 1
        right = p
        while right < len(smooth) - 1 and smooth[right + 1] <= smooth[right]:
            right += 1
        masses.append(int(smooth[left : right + 1].sum()))
    return centers, smooth, order, np.array(masses)


def make_plscan_gif():
    x = common.three_peaks_1d()
    centers, hist, peaks, masses = _peak_masses(x)
    max_mass = masses.max()
    mcs = np.linspace(0, max_mass * 1.05, 30)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), height_ratios=[2, 1])

    def update(t):
        for ax in axes:
            ax.clear()
        mc = mcs[t]
        axes[0].fill_between(centers, hist, alpha=0.25, color=FIELD)
        axes[0].plot(centers, hist, color=NOISE, lw=1.4)
        alive = masses >= mc
        for i, p in enumerate(peaks):
            axes[0].axvspan(centers[max(0, p - 3)], centers[min(len(centers) - 1, p + 3)],
                            alpha=0.35, color=PALETTE[i % len(PALETTE)] if alive[i] else UNVISITED)
            axes[0].plot([centers[p]], [hist[p]], "o", color=PALETTE[i % len(PALETTE)] if alive[i] else NOISE)
        axes[0].set_ylabel("density")
        axes[0].set_title(f"PLSCAN: clusters alive at min-cluster-size m_c = {mc:.0f}", fontsize=11)
        axes[0].set_xticks([])
        # barcode
        for i, m in enumerate(masses):
            axes[1].barh(i, min(m, mc), color=PALETTE[i % len(PALETTE)], alpha=0.85)
            axes[1].barh(i, max(0, m - mc), left=min(m, mc), color=UNVISITED, alpha=0.5)
        axes[1].axvline(mc, color="k", lw=1, ls="--")
        axes[1].set_yticks(range(len(masses)))
        axes[1].set_yticklabels([f"cluster {i + 1}" for i in range(len(masses))], fontsize=8)
        axes[1].set_xlabel("min-cluster-size (persistence)")
        axes[1].set_xlim(0, max_mass * 1.05)
        common.ref(fig, "PLSCAN - Bot, McInnes & Aerts 2025. Persistence = a cluster's lifetime in min-cluster-size.")

    return common.save_gif(fig, update, len(mcs), "plscan_persistence.gif", fps=3.5)


def make_plscan_barcode():
    x = common.three_peaks_1d()
    centers, hist, peaks, masses = _peak_masses(x)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    axes[0].fill_between(centers, hist, alpha=0.25, color=FIELD)
    axes[0].plot(centers, hist, color=NOISE, lw=1.4)
    for i, p in enumerate(peaks):
        axes[0].plot([centers[p]], [hist[p]], "o", color=PALETTE[i % len(PALETTE)], ms=8)
    axes[0].set_title("density profile (3 peaks)", fontsize=11)
    axes[0].set_ylabel("density"); axes[0].set_xticks([])
    for i, m in enumerate(sorted(masses)[::-1]):
        axes[1].barh(i, m, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        axes[1].text(m + 1, i, f"{m}", va="center", fontsize=8)
    axes[1].set_yticks(range(len(masses)))
    axes[1].set_yticklabels([f"cluster {i + 1}" for i in range(len(masses))], fontsize=8)
    axes[1].set_xlabel("persistence (min-cluster-size)")
    axes[1].set_title("persistence barcode", fontsize=11)
    common.ref(fig, "PLSCAN - Bot, McInnes & Aerts 2025. Long bars = stable clusters; short bars = noise.")
    return common.save_png(fig, "plscan_barcode.png")
