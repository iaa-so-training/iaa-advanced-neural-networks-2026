"""Embedding algorithms (t-SNE, UMAP) as animations, plus the EVoC pipeline
panel and a synthetic C-space corner.

t-SNE and UMAP are implemented compactly here (numpy) so the per-iteration
embedding can be captured frame-by-frame; sklearn does not expose per-epoch
callbacks.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from . import common

from .common import PALETTE, NOISE, FIELD, UNVISITED, INK


def _data3d(seed=0, n=220):
    """Four Gaussian clusters in 3-D - structure the 2-D embedding must reveal."""
    X, _ = make_blobs(
        n_samples=n, centers=4, n_features=3, cluster_std=0.9, random_state=seed
    )
    return X


# --------------------------------------------------------------------------- #
# t-SNE (van der Maaten & Hinton 2008), manual for frame capture
# --------------------------------------------------------------------------- #


def _tsne_frames(X, perplexity=30.0, n_iter=450, seed=0, capture_every=6):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)

    P = np.zeros((n, n))
    for i in range(n):
        Di = np.concatenate([D2[i, :i], D2[i, i + 1 :]])
        beta, lo, hi = 1.0, None, None
        target = np.log(perplexity)
        for _ in range(100):
            Pi = np.exp(-Di * beta)
            s = Pi.sum()
            if s <= 0:
                beta *= 0.5
                continue
            H = np.log(s) + beta * (Di * Pi).sum() / s
            if abs(H - target) < 1e-5:
                break
            if H > target:
                lo = beta
                beta = beta * 2 if hi is None else (lo + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo is None else (lo + hi) / 2
        row = np.zeros(n)
        row[:i] = Pi[:i]
        row[i + 1 :] = Pi[i:]
        P[i] = row / row.sum()
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    Y = rng.standard_normal((n, 2)) * 1e-4
    prev_grad = np.zeros_like(Y)
    gains = np.ones((n, 2))
    lr = 50.0
    frames = [Y.copy()]
    for it in range(n_iter):
        d2 = ((Y[:, None] - Y[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, 1e-12)
        qraw = 1.0 / (1.0 + d2)
        np.fill_diagonal(qraw, 0.0)
        Q = qraw / qraw.sum()
        C = ((P * 12.0 if it < 250 else P) - Q) * qraw  # (p-q) * q_raw, early exaggeration then relax
        rowsum = C.sum(1)
        grad = 4.0 * (rowsum[:, None] * Y - C @ Y)
        same = (grad > 0) == (prev_grad > 0)
        gains = (gains + 0.2) * same + (gains * 0.8) * ~same
        gains = np.maximum(gains, 0.01)
        prev_grad = grad.copy()
        Y = Y - lr * gains * grad
        Y = Y - Y.mean(0)
        if (it + 1) % capture_every == 0:
            frames.append(Y.copy())
    return frames


def make_tsne_gif():
    X = _data3d(seed=0, n=220)
    frames = _tsne_frames(X, n_iter=450, seed=1)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    true = make_blobs(n_samples=220, centers=4, n_features=3, cluster_std=0.9, random_state=0)[1]

    def update(t):
        ax.clear()
        Y = frames[t]
        for c in range(4):
            m = true == c
            ax.scatter(Y[m, 0], Y[m, 1], c=PALETTE[c], s=16, edgecolors="none", label=f"cluster {c + 1}")
        ax.set_title(f"t-SNE - iteration {t * 6}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=False)
        common.ref(fig, "t-SNE - van der Maaten & Hinton 2008. Minimises KL divergence between high-D and low-D similarities.")

    return common.save_gif(fig, update, len(frames), "tsne.gif", fps=5.0)


# --------------------------------------------------------------------------- #
# UMAP (McInnes, Healy & Melville 2018), manual for frame capture
# --------------------------------------------------------------------------- #


def _umap_fuzzy_graph(X, k=15):
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, inds = nn.kneighbors(X)
    rho = dists[:, 1].copy()
    target = np.log2(k)
    graph = np.zeros((n, n))
    for i in range(n):
        d = np.maximum(dists[i, 1:] - rho[i], 0)
        sig, lo, hi = 1.0, 0.0, None
        for _ in range(60):
            w = np.exp(-d / sig)
            s = w.sum()
            if abs(s - target) < 1e-3:
                break
            if s > target:
                lo = sig
                sig = sig * 2 if hi is None else (lo + hi) / 2
            else:
                hi = sig
                sig = sig / 2 if lo is None else (lo + hi) / 2
        graph[i, inds[i, 1:]] = w
    g = graph + graph.T - graph * graph.T
    return g


def _umap_frames(X, n_epochs=300, seed=0, capture_every=5, neg_rate=5):
    g = _umap_fuzzy_graph(X, k=15)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(g)
    weights = g[rows, cols]
    upper = rows < cols
    rows, cols, weights = rows[upper], cols[upper], weights[upper]

    a, b = 1.576943, 0.895070  # min_dist = 0.1 curve
    # spectral init (as UMAP does): eigenvectors of the normalised graph Laplacian
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    d_ = np.asarray(g.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d_, 1e-12))
    Ln = csr_matrix(np.diag(dinv)) @ csr_matrix(np.diag(d_) - g) @ csr_matrix(np.diag(dinv))
    # v0 pinned: without a start vector ARPACK draws its own, so the spectral
    # init - and with it the whole layout - came out different on every run,
    # which made umap.gif and umap_step3_layout.png impossible to reproduce
    # from the documented rebuild command.
    _, vecs = eigsh(Ln, k=3, which='SM', v0=np.random.default_rng(0).normal(size=n))
    Y = vecs[:, 1:3].astype(float)
    lr = 1.0
    frames = [Y.copy()]

    for ep in range(n_epochs):
        grad = np.zeros((n, 2))
        for i, j, w in zip(rows, cols, weights):
            d = Y[i] - Y[j]
            d2 = float((d * d).sum())
            if d2 <= 1e-12:
                continue
            c = -2.0 * a * b * w * (d2 ** (b - 1.0)) / (1.0 + a * d2)
            gvec = c * d
            grad[i] += gvec
            grad[j] -= gvec
        for i in range(n):
            negs = rng.integers(0, n, size=neg_rate)
            for j in negs:
                if i == j:
                    continue
                d = Y[i] - Y[j]
                d2 = float((d * d).sum())
                if d2 < 0.001:
                    d2 = 0.001
                c = 2.0 * b / ((0.001 + d2) * (1.0 + a * d2))
                grad[i] += c * d
        Y += lr * grad
        if (ep + 1) % capture_every == 0:
            frames.append(Y.copy())
    return frames


def make_umap_gif():
    X = _data3d(seed=0, n=220)
    frames = _umap_frames(X, n_epochs=300, seed=2)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    true = make_blobs(n_samples=220, centers=4, n_features=3, cluster_std=0.9, random_state=0)[1]

    def update(t):
        ax.clear()
        Y = frames[t]
        for c in range(4):
            m = true == c
            ax.scatter(Y[m, 0], Y[m, 1], c=PALETTE[c], s=16, edgecolors="none", label=f"cluster {c + 1}")
        ax.set_title(f"UMAP - epoch {t * 5}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=False)
        common.ref(fig, "UMAP - McInnes, Healy & Melville 2018. Optimises a low-D layout against a fuzzy topological graph.")

    return common.save_gif(fig, update, len(frames), "umap.gif", fps=6.0)


# --------------------------------------------------------------------------- #
# EVoC pipeline panel (authentic: uses the evoc package internals)
# --------------------------------------------------------------------------- #


def make_evoc_panel():
    from evoc.clustering import knn_graph, evoc_clusters
    import umap

    X, _ = common.blobs(seed=0, n=240)
    Xf = X.astype(np.float32)
    rs = np.random.RandomState(0)

    nn_inds, nn_dists = knn_graph(Xf, n_neighbors=15, random_state=rs)
    layers, strengths, persist, _, _ = evoc_clusters(
        Xf, base_min_cluster_size=8, n_neighbors=15, n_epochs=50, node_embedding_dim=2, random_state=rs
    )
    labels = layers[0]
    emb = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    # 1 - kNN graph
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=14, edgecolors="none", zorder=1)
    for i in range(0, len(X), 4):
        for j in nn_inds[i, 1:6]:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], color=NOISE, lw=0.4, zorder=0)
    ax.scatter(X[:, 0], X[:, 1], c=FIELD, s=14, edgecolors="none", zorder=2)
    ax.set_title("1. kNN graph", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # 2 - node embedding
    ax = axes[1]
    ax.scatter(emb[:, 0], emb[:, 1], c=NOISE, s=14, edgecolors="none")
    ax.set_title("2. node embedding (UMAP-style)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # 3 - EVoC labels on the embedding canvas
    ax = axes[2]
    n_clusters = max([l for l in labels if l >= 0], default=0)
    for c in range(n_clusters + 1):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], c=PALETTE[c % len(PALETTE)], s=14,
                   edgecolors="none", label=f"c{c}" if c > 0 else None)
    noise = labels == -1
    ax.scatter(emb[noise, 0], emb[noise, 1], c=NOISE, s=14, marker="x")
    ax.set_title("3. cluster labels (EVoC)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # 4 - persistence scores
    ax = axes[3]
    scores = [p for p in persist[1:] if p > 0] or [0.0]
    ax.bar(range(len(scores)), scores, color=PALETTE[: len(scores)])
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([f"layer {i + 1}" for i in range(len(scores))], fontsize=7)
    ax.set_title("4. persistence per layer", fontsize=11)
    ax.set_ylabel("persistence", fontsize=8)

    fig.suptitle("EVoC: graph -> embedding -> density clustering -> persistence selection", fontsize=13, y=1.02)
    common.ref(fig, "EVoC - github.com/TutteInstitute/evoc. Fuses UMAP-style embedding + HDBSCAN* + PLSCAN persistence.")
    return common.save_png(fig, "evoc_pipeline.png")


# --------------------------------------------------------------------------- #
# Synthetic C-space corner (2 of 16 abundance dimensions)
# --------------------------------------------------------------------------- #


def make_cspace_corner():
    rng = np.random.default_rng(0)
    field = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.65], [0.65, 1.0]], 500)
    cluster = rng.multivariate_normal([1.6, 1.3], [[0.07, 0.02], [0.02, 0.07]], 40)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(field[:, 0], field[:, 1], c=FIELD, s=14, edgecolors="none", label="field stars")
    ax.scatter(cluster[:, 0], cluster[:, 1], c=PALETTE[4], s=22, edgecolors="white",
               linewidths=0.5, label="cluster members")
    ax.set_xlabel("[X/Fe]")
    ax.set_ylabel("[Y/Fe]")
    ax.set_title("C-space: 2 of 16 abundance dimensions", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    common.ref(fig, "APOGEE DR17 abundances - 16 elements define the chemical space (Kos et al. 2017).")
    return common.save_png(fig, "cspace_corner.png")
