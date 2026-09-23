"""Spectral region-mode sweep: benchmark every cluster with scaled regions.

For each cluster, cut the sky region (scaled to the cluster angular
diameter), cross-match the spectral embeddings, embed+cluster, and score
recovery against the kinematic ground truth.

Usage:
    uv run python scripts/spectral_region_sweep.py \
        --embeddings data/embeddings/attention_broad_merged.parquet \
        --allstar data/astraAllStarASPCAP-0.6.0.fits.gz --out results/spectral_region_sweep.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cluster import config, spectral
from cluster.benchmark import fit_tsne
from cluster.clusters import CLUSTERS
from cluster.data import apply_quality_cuts, load_allstar
from cluster.membership import angular_separation, kinematic_members

KIN_COLS = ["GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG"]


def _score(true_labels: np.ndarray, pred_labels: np.ndarray, cluster: str) -> dict[str, float | int]:
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


def _standardized_kinematics(merged: pd.DataFrame) -> np.ndarray:
    kin = merged[KIN_COLS].to_numpy(dtype=float)
    med = np.nanmedian(kin, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    kin = np.where(np.isfinite(kin), kin, med[np.newaxis, :])
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    return (kin - med) / scale


def sweep(allstar_path: Path, embeddings_path: Path, settings: config.Settings, kinematics: bool = False, kin_only: bool = False) -> pd.DataFrame:
    import hdbscan
    import umap
    from evoc import EVoC

    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)
    emb = spectral.load_embedding_frame(embeddings_path)
    emb_cols = spectral.embedding_columns(emb)

    rows: list[dict[str, float | int | str]] = []

    for c in CLUSTERS:
        sep = angular_separation(
            df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
            c.ra_deg, c.dec_deg,
        )
        reg = df[sep <= c.region_deg].copy()
        m = kinematic_members(
            reg, c,
            config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
            config.SEED_PM_TOL, config.SEED_RV_TOL,
            config.N_REFINE_PASSES, config.REFINE_SIGMA,
        ).to_numpy()
        reg["cluster"] = np.where(m, c.name, "field")

        merged = spectral.align_embeddings(
            reg.drop_duplicates(subset=["APOGEE_ID"]), emb,
        )
        if len(merged) < 10:
            continue
        if len(merged) < 50:
            rows.append({"cluster": c.name, "kind": c.kind, "region": c.region_deg,
                         "n_field": int((merged["cluster"] == "field").sum()),
                         "n_members": int((merged["cluster"] == c.name).sum())})
            continue
        if kin_only:
            X = _standardized_kinematics(merged)
        else:
            X = spectral.embedding_matrix(merged, emb_cols, settings)
            if kinematics:
                X = np.hstack([X, _standardized_kinematics(merged)])
        true = merged["cluster"].to_numpy()
        n_members = int((true == c.name).sum())
        n_field = int((true == "field").sum())
        if n_members < 3:
            rows.append({"cluster": c.name, "kind": c.kind, "region": c.region_deg,
                         "n_field": n_field, "n_members": n_members})
            continue

        z_tsne = fit_tsne(X, settings.tsne, settings.random_state)
        z_umap = umap.UMAP(n_components=2, random_state=settings.random_state, **settings.umap).fit_transform(X)
        evoc_lab = EVoC(random_state=settings.random_state, **settings.evoc).fit_predict(X)
        h_tsne = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z_tsne)
        h_umap = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z_umap)

        rows.append({
            "cluster": c.name, "kind": c.kind, "region": c.region_deg,
            "n_field": n_field, "n_members": n_members,
            "tsne_recall": _score(true, h_tsne, c.name)["recall"],
            "tsne_precision": _score(true, h_tsne, c.name)["precision"],
            "tsne_purity": _purity(z_tsne, true, c.name),
            "umap_recall": _score(true, h_umap, c.name)["recall"],
            "umap_precision": _score(true, h_umap, c.name)["precision"],
            "umap_purity": _purity(z_umap, true, c.name),
            "evoc_recall": _score(true, evoc_lab, c.name)["recall"],
            "evoc_precision": _score(true, evoc_lab, c.name)["precision"],
        })
        print(f"  {c.name:14} region={c.region_deg:5.1f}  field={n_field:5d} members={n_members:3d}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings/attention_broad_merged.parquet")
    ap.add_argument("--kinematics", action="store_true", help="Append standardised parallax/PM/RV.")
    ap.add_argument("--kin-only", action="store_true", help="Use only the 4-d kinematics (no embeddings).")
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/spectral_region_sweep.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    print("🧪 spectral region sweep (scaled regions)" + (" + kinematics" if args.kinematics else "") + (" [kin-only]" if args.kin_only else ""))
    table = sweep(Path(args.allstar), Path(args.embeddings), settings, kinematics=args.kinematics, kin_only=args.kin_only)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    cols = ["cluster", "region", "n_members", "n_field",
            "tsne_recall", "tsne_precision", "tsne_purity",
            "umap_recall", "umap_precision", "umap_purity",
            "evoc_recall", "evoc_precision"]
    print("\n=== spectral region sweep ===")
    print(table[[c for c in cols if c in table.columns]].round(2).to_string(index=False))
    print(f"\nsaved to {args.out}")
    print("=== macro (clusters with >=3 members) ===")
    for mth in ("tsne", "umap", "evoc"):
        sub = table[table[f"{mth}_recall"].notna()]
        print(f"  {mth}: recall={sub[f'{mth}_recall'].mean():.2f}  "
              f"precision={sub[f'{mth}_precision'].mean():.2f}")


if __name__ == "__main__":
    main()
