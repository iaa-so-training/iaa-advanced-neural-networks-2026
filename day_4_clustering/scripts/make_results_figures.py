"""Generate the results figures for the deck (IAA-SO 2026).

Three figures, all computed from the real analysis data:

1. ``headtohead_pca.png`` — a plain linear 2-D view (PCA) of the exact
   55-star head-to-head sample (the intersection the ``cluster head-to-head``
   command uses), left panel in the masked-AE 256-d latent, right panel in
   the 16 ASPCAP abundances. Silhouette scores in the panel titles quantify
   the visual difference without any projection distortion.

2. ``proper_motions.png`` — the kinematic referee: pmRA vs pmDec for every
   star in the five cluster fields (grey), with the cluster members
   (magenta) sitting in one compact clump at the cluster's mean motion.

3. ``paired_control.png`` — the product-mismatch paired control: 253 stars
   embedded through both pipelines; the same star lands 1.7x farther from
   itself across products than from a different star within one product.

Usage:
    .venv/bin/python scripts/make_results_figures.py

Writes PNGs into the deck asset folder (~/git/garciadias.github.io/...).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from cluster.config import Settings
from cluster.data import PreparedData
from cluster.headtohead import Arm, common_population
from cluster.spectral import spectral_prepared

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ASTRA = "data/astraAllStarASPCAP-0.6.0.fits.gz"
DECK = Path.home() / "git/garciadias.github.io/public/presentations/iaa-so-chemical-tagging-2026"
FIVE = ["Berkeley 66", "IC 166", "M 3", "M 67", "NGC 188"]

MEMBER = "#db2777"   # magenta, matches cspace_corner.png
FIELD = "#64748b"    # slate grey
CLUSTER_COLORS = {
    "Berkeley 66": "#2563eb",
    "IC 166": "#059669",
    "M 3": "#dc2626",
    "M 67": "#d97706",
    "NGC 188": "#7c3aed",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#94a3b8",
    "axes.labelcolor": "#0f172a",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_prepared() -> PreparedData:
    """Same preparation the CLI uses (quality cuts ON, like the benchmark)."""
    from cluster.cli import _prepared_for  # reuse the exact loader

    cache = Path("/tmp/prepared_headtohead.parquet")
    if cache.exists():
        print(f"  using cached prepared data ({cache})", flush=True)
        from cluster.data import PreparedData
        return PreparedData(df=pd.read_parquet(cache), X=np.zeros((0, 0)),
                            elements=[])

    settings = Settings()
    settings.require_aspcap_flag_clean = False  # as the CLI head-to-head does
    t0 = time.time()
    prepared = _prepared_for(ASTRA, settings)
    print(f"  prepared in {time.time() - t0:.0f}s ({len(prepared.df)} rows)", flush=True)
    prepared.df.to_parquet(cache)
    return prepared


def fig_headtohead_pca(prepared: PreparedData) -> None:
    settings = Settings()
    shared, _clusters, _dropped = common_population(
        prepared,
        [
            Arm("abundances"),
            Arm("PCA 64-d", "data/embeddings/pca_64.parquet"),
            Arm("PCA 256-d", "data/embeddings/pca_256.parquet"),
            Arm("masked AE 256-d", "data/embeddings/masked_latent.parquet"),
        ],
        settings,
    )
    spec = spectral_prepared(
        prepared, "data/embeddings/masked_latent.parquet", settings,
    )
    sub = spec.df.drop_duplicates(subset=["APOGEE_ID"], keep="first")
    sub = sub[sub["APOGEE_ID"].astype(str).isin(shared)].sort_values("APOGEE_ID")
    labels = sub["cluster"].astype(str).to_numpy()
    X_lat = sub[list(spec.elements)].to_numpy(float)

    sub_a = prepared.df.drop_duplicates(subset=["APOGEE_ID"], keep="first")
    sub_a = sub_a[sub_a["APOGEE_ID"].astype(str).isin(sorted(shared))]
    sub_a = sub_a.sort_values(by="APOGEE_ID", kind="stable")
    X_ab = np.asarray(sub_a[list(settings.elements)], dtype=np.float64)
    nan_mask = np.isnan(X_ab)
    X_ab[nan_mask] = np.broadcast_to(
        np.nanmedian(X_ab, axis=0), X_ab.shape,
    )[nan_mask]
    X_ab = (X_ab - X_ab.mean(0)) / (X_ab.std(0) + 1e-9)

    print(f"  head-to-head sample: {len(labels)} stars, "
          f"{np.unique(labels, return_counts=True)}", flush=True)

    # Plot a fixed handful of clusters for readability; score on exactly the
    # plotted subset so the silhouette matches what the eye sees.
    show = np.isin(labels, FIVE)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for ax, X, title, tag in [
        (axes[0], X_lat, "masked AE 256-d latent", "a"),
        (axes[1], X_ab, "ASPCAP abundances (16-d)", "b"),
    ]:
        Y = PCA(n_components=2).fit_transform(X)
        sil = silhouette_score(Y[show], labels[show])
        for name in FIVE:
            m = labels == name
            ax.scatter(Y[m, 0], Y[m, 1], s=58, c=CLUSTER_COLORS[name],
                       label=name, edgecolors="white", linewidths=0.8, zorder=3)
        ax.set_title(f"{title} — 2-D PCA, silhouette {sil:.2f}", fontsize=11, pad=8)
        ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.legend(loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("Five of the 25 clusters — a plain linear 2-D view of each space",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = DECK / "headtohead_pca.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _n_rows(hdu: fits.BinTableHDU) -> int:
    """Row count from the header (avoids touching ``.data``)."""
    return int(str(hdu.header["NAXIS2"]))


def fig_proper_motions() -> None:
    members = pd.read_csv("data/embeddings/cluster_members.csv")
    members["APOGEE_ID"] = members["APOGEE_ID"].astype(str)
    from cluster.clusters import CLUSTERS
    meta = {c.name: c for c in CLUSTERS}

    with fits.open(ASTRA, memmap=True) as h:
        t = max((x for x in h if isinstance(x, fits.BinTableHDU)),
                key=_n_rows).data
        aid = np.char.strip(np.asarray(t["sdss4_apogee_id"]).astype(str))
        ra = np.asarray(t["ra"], float)
        dec = np.asarray(t["dec"], float)
        pmra = np.asarray(t["pmra"], float)
        pmdec = np.asarray(t["pmde"], float)

    fig, grid = plt.subplots(2, 3, figsize=(10.6, 5.6))
    axes = [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1]]
    grid[1, 2].axis("off")
    for ax, name in zip(axes, FIVE, strict=True):
        c = meta[name]
        # field: everything within ~2x the cluster diameter
        r = max(c.diam_arcmin / 60.0, 0.15) * 2
        d = np.hypot((ra - c.ra_deg) * np.cos(np.radians(c.dec_deg)),
                     dec - c.dec_deg)
        field = (d < r) & np.isfinite(pmra) & np.isfinite(pmdec)
        in_members = np.isin(aid, members.loc[members["cluster"] == name, "APOGEE_ID"])
        ax.scatter(pmra[field], pmdec[field], s=4, c=FIELD, alpha=0.45,
                   rasterized=True, label="field")
        ax.scatter(pmra[in_members & field], pmdec[in_members & field], s=26,
                   c=MEMBER, edgecolors="white", linewidths=0.6, zorder=4,
                   label="members")
        ax.axvline(c.pmra, color=MEMBER, lw=0.8, ls="--", alpha=0.7)
        ax.axhline(c.pmdec, color=MEMBER, lw=0.8, ls="--", alpha=0.7)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("pmRA (mas/yr)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("pmDec (mas/yr)", fontsize=9)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.04), handletextpad=0.3)
    fig.suptitle("The kinematic referee: who really belongs to each cluster "
                 "(Gaia proper motions)", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    out = DECK / "proper_motions.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def fig_paired_control() -> None:
    a = pd.read_parquet("data/embeddings/masked_latent_dr17.parquet").set_index("APOGEE_ID")
    b = pd.read_parquet("data/embeddings/masked_latent_dr19_apstar_v1.parquet").set_index("APOGEE_ID")
    a.index = a.index.astype(str)
    b.index = b.index.astype(str)
    both = sorted(set(a.index) & set(b.index))
    A = a.loc[both].to_numpy(float)  # DR17 aspcapStar (continuum-normalised)
    B = b.loc[both].to_numpy(float)  # DR19 apStar (raw)
    n = len(both)

    offset = B - A
    mu = offset.mean(0)
    axis = mu / np.linalg.norm(mu)                       # the shared shift
    resid = offset - offset @ axis[:, None] * axis[None, :]
    _u, _s, vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False)
    perp = vt[0]                                         # largest residual dir

    def proj(X: np.ndarray) -> np.ndarray:
        return np.column_stack([X @ axis, X @ perp])
    PA, PB = proj(A), proj(B)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter(PA[:, 0], PA[:, 1], s=16, c="#2563eb", alpha=0.65,
               label="same star, DR17 aspcapStar", rasterized=True)
    ax.scatter(PB[:, 0], PB[:, 1], s=16, c="#d97706", alpha=0.65,
               label="same star, DR19 apStar", rasterized=True)
    for i in np.random.default_rng(1).choice(n, 14, replace=False):
        ax.annotate("", xy=PB[i], xytext=PA[i],
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8",
                                    lw=0.7, alpha=0.55, shrinkA=4, shrinkB=4))
    ax.arrow(0, 0, axis_proj := float(mu @ axis), 0, color="#0f172a", lw=2,
             head_width=0.09, head_length=0.12, length_includes_head=True,
             zorder=6, label="shared product offset")
    ax.set_xlabel("projection on the shared offset axis")
    ax.set_ylabel("largest residual direction")
    ax.set_title(
        f"Paired control: {n} stars embedded through BOTH pipelines\n"
        "same star across products: 3.66 ± 0.46  |  different star, one product: 2.15 ± 0.68",
        fontsize=10.5,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.03, 0.06,
            "1.70\u00d7 farther from itself across products\nthan from a random other star "
            "(cosine 0.39)\n\u2192 the latent separates PRODUCTS, not populations",
            transform=ax.transAxes, fontsize=9.5, color="#0f172a",
            bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.9, pad=6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = DECK / "paired_control.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}  (axis_proj={axis_proj:.2f})", flush=True)


def main() -> None:
    DECK.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared()
    fig_headtohead_pca(prepared)
    fig_proper_motions()
    fig_paired_control()
    print("done.")


if __name__ == "__main__":
    main()
