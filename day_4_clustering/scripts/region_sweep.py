"""Region-mode sweep: benchmark t-SNE / UMAP / EVoC on every cluster.

Reproduces the target paper's setup — for each cluster, restrict to a sky
region (default 30 deg) around it, embed the abundances, cluster, and score
recovery against the kinematic ground truth. Prints and saves a per-cluster
comparison table.

Usage:
    uv run python scripts/region_sweep.py --radius 30 \
        --allstar /path/to/allStar-dr17-synspec_rev1.fits --out results/region_sweep.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cluster import config
from cluster.benchmark import fit_tsne
from cluster.clusters import CLUSTERS
from cluster.data import (
    apply_quality_cuts,
    complete_case,
    load_allstar,
    make_matrix,
)
from cluster.membership import angular_separation, kinematic_members

KIN_COLS = ["GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG"]


def _standardized_kinematics(reg: pd.DataFrame) -> np.ndarray:
    kin = reg[KIN_COLS].to_numpy(dtype=float)
    med = np.nanmedian(kin, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    kin = np.where(np.isfinite(kin), kin, med[np.newaxis, :])
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    return (kin - med) / scale


def _score(true_labels: np.ndarray, pred_labels: np.ndarray, cluster: str) -> dict[str, float | int]:
    """Precision/recall of the predicted group that best overlaps the cluster."""
    tmask = true_labels == cluster
    n_true = int(tmask.sum())
    best_overlap, best = 0, None
    for pc in np.unique(pred_labels):
        if pc == -1:
            continue
        overlap = int(((pred_labels == pc) & tmask).sum())
        if overlap > best_overlap:
            best_overlap, best = overlap, pc
    if best is None or n_true == 0:
        return {"recall": 0.0, "precision": 0.0, "overlap": 0}
    n_pred = int((pred_labels == best).sum())
    return {
        "recall": best_overlap / n_true,
        "precision": best_overlap / n_pred if n_pred else 0.0,
        "overlap": best_overlap,
    }


def _purity(Z: np.ndarray, true_labels: np.ndarray, cluster: str, k: int = 10) -> float:
    m = true_labels == cluster
    n_members = int(m.sum())
    if n_members < 3:
        return float("nan")
    k_c = min(k, n_members - 1)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z))).fit(Z)
    neighbors = nn.kneighbors(Z, return_distance=False)
    idx = np.asarray(neighbors)[:, 1:]
    neigh = true_labels[idx[m][:, :k_c]]
    return float((neigh == cluster).mean())


def sweep(allstar_path: Path, radius: float, settings: config.Settings, scaled: bool = False, kinematics: bool = False, embeddings: str | None = None, names: list[str] | None = None) -> pd.DataFrame:
    import hdbscan
    import umap
    from evoc import EVoC

    from cluster import spectral

    emb_frame = None
    emb_cols = None
    if embeddings is not None:
        emb_frame = spectral.load_embedding_frame(embeddings)
        emb_cols = spectral.embedding_columns(emb_frame)

    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)

    rows: list[dict[str, float | int | str]] = []

    for c in CLUSTERS:
        if names and c.name not in names:
            continue
        region = c.region_deg if scaled else radius
        sep = angular_separation(
            df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
            c.ra_deg, c.dec_deg,
        )
        reg = df[sep <= region].copy()
        if len(reg) < 10:
            continue

        m = kinematic_members(
            reg, c,
            config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
            config.SEED_PM_TOL, config.SEED_RV_TOL,
            config.N_REFINE_PASSES, config.REFINE_SIGMA,
        ).to_numpy()
        reg["cluster"] = np.where(m, c.name, "field")

        reg = complete_case(reg, settings)
        if len(reg) < 10:
            continue
        if emb_frame is not None:
            reg = spectral.align_embeddings(
                reg.drop_duplicates(subset=["APOGEE_ID"]), emb_frame,
            )
            if len(reg) < 10:
                continue
        X = make_matrix(reg, settings)
        if kinematics:
            X = np.hstack([X, _standardized_kinematics(reg)])
        true = reg["cluster"].to_numpy()
        n_members = int((true == c.name).sum())
        if len(X) < 50 or n_members < 3:
            rows.append({
                "cluster": c.name, "kind": c.kind, "n_field": len(reg) - n_members,
                "n_members": n_members,
            })
            continue

        z_tsne = fit_tsne(X, settings.tsne, settings.random_state)
        z_umap = umap.UMAP(n_components=2, random_state=settings.random_state, **settings.umap).fit_transform(X)
        evoc_lab = EVoC(random_state=settings.random_state, **settings.evoc).fit_predict(X)

        h_tsne = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z_tsne)
        h_umap = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z_umap)

        rows.append({
            "cluster": c.name, "kind": c.kind,
            "n_field": len(reg) - n_members, "n_members": n_members,
            "tsne_recall": _score(true, h_tsne, c.name)["recall"],
            "tsne_precision": _score(true, h_tsne, c.name)["precision"],
            "tsne_purity": _purity(z_tsne, true, c.name),
            "umap_recall": _score(true, h_umap, c.name)["recall"],
            "umap_precision": _score(true, h_umap, c.name)["precision"],
            "umap_purity": _purity(z_umap, true, c.name),
            "evoc_recall": _score(true, evoc_lab, c.name)["recall"],
            "evoc_precision": _score(true, evoc_lab, c.name)["precision"],
        })
        print(f"  {c.name:14} n={len(reg):6d} members={n_members:4d}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--scaled", action="store_true", help="Use per-cluster scaled regions (max(3 deg, 10 x diameter)).")
    ap.add_argument("--kinematics", action="store_true", help="Append standardised parallax/PM/RV.")
    ap.add_argument("--embeddings", default=None, help="Restrict to stars with embeddings (inner join on APOGEE_ID).")
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/region_sweep.csv")
    ap.add_argument("--clusters", default="", help="Comma-separated cluster names; empty = all.")
    args = ap.parse_args()
    names = [n.strip() for n in args.clusters.split(",") if n.strip()]

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # match the spectral sweep
    print(f"🧪 region sweep  radius={'scaled' if args.scaled else args.radius}  SNR_MIN={settings.snr_min}" + (" + kinematics" if args.kinematics else ""))
    table = sweep(Path(args.allstar), args.radius, settings, scaled=args.scaled, kinematics=args.kinematics, embeddings=args.embeddings, names=names)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\n=== region-mode sweep (recall | precision | kNN-purity) ===")
    cols = ["cluster", "n_members", "n_field",
            "tsne_recall", "tsne_precision", "tsne_purity",
            "umap_recall", "umap_precision", "umap_purity",
            "evoc_recall", "evoc_precision"]
    print(table[[c for c in cols if c in table.columns]].round(2).to_string(index=False))
    print(f"\nsaved to {args.out}")
    print("=== macro (mean over clusters with >=3 members) ===")
    for m in ("tsne", "umap", "evoc"):
        print(f"  {m}: recall={table[f'{m}_recall'].mean():.2f}  "
              f"precision={table[f'{m}_precision'].mean():.2f}" +
              (f"  purity={table[f'{m}_purity'].mean():.2f}" if f"{m}_purity" in table else ""))


if __name__ == "__main__":
    main()
