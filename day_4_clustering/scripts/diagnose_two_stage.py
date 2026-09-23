"""Diagnose whether a two-stage pipeline is viable.

Stage 1 = kinematics-only (4-d) -> t-SNE -> HDBSCAN. It gives high recall
(0.93) but low precision (~0.63): it pulls in field stars that are
kinematic doppelgangers.

The question: are those false positives (FP) chemically / spectrally
distinct from the true members (TP)? If yes, Stage 2 (a chemical/spectral
distance cut) can reject them. If no, nothing can.

For each cluster we report, among the Stage-1 predicted members:
  - n_TP, n_FP
  - AUROC of "chemical distance to the true-member centroid" separating TP from FP
  - AUROC of "spectral (RNN-latent) distance" doing the same
AUROC > 0.5 means closer = more member-like; ~0.7+ means Stage 2 can work.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cluster import config, spectral
from cluster.benchmark import fit_tsne
from cluster.clusters import CLUSTERS
from cluster.data import apply_quality_cuts, load_allstar
from cluster.membership import angular_separation, kinematic_members

KIN_COLS = ["GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG"]


def _std_kin(df: pd.DataFrame) -> np.ndarray:
    kin = df[KIN_COLS].to_numpy(dtype=float)
    med = np.nanmedian(kin, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    kin = np.where(np.isfinite(kin), kin, med[np.newaxis, :])
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    return (kin - med) / scale


def _standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med[np.newaxis, :])
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    return (X - med) / scale


def _best_overlap(true: np.ndarray, pred: np.ndarray, cluster: str) -> tuple[np.ndarray, int, int]:
    """Return the mask of the predicted cluster that best overlaps the true one."""
    tmask = true == cluster
    best, best_overlap, best_pred = -1, 0, 0
    for pc in np.unique(pred):
        if pc < 0:  # HDBSCAN noise
            continue
        overlap = int(((pred == pc) & tmask).sum())
        if overlap > best_overlap:
            best, best_overlap, best_pred = pc, overlap, int((pred == pc).sum())
    if best < 0:
        return np.zeros(len(true), dtype=bool), 0, 0
    return (pred == best), best_overlap, best_pred


def _auroc(dist: np.ndarray, is_member: np.ndarray) -> float:
    if is_member.sum() == 0 or (~is_member).sum() == 0:
        return float("nan")
    # higher score = closer (member-like); AUROC > 0.5 means distance separates
    return float(roc_auc_score(is_member, -dist))


def diagnose(allstar_path: Path, embeddings_path: Path, settings: config.Settings, names: list[str]) -> pd.DataFrame:
    import hdbscan

    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)
    emb = spectral.load_embedding_frame(embeddings_path)
    emb_cols = spectral.embedding_columns(emb)

    rows: list[dict[str, float | int | str]] = []
    for c in CLUSTERS:
        if names and c.name not in names:
            continue
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

        merged = spectral.align_embeddings(reg.drop_duplicates(subset=["APOGEE_ID"]), emb)
        if len(merged) < 50:
            continue
        true = merged["cluster"].to_numpy()
        n_members = int((true == c.name).sum())
        if n_members < 3:
            continue

        # Stage 1: kinematics-only -> t-SNE -> HDBSCAN
        X_kin = _std_kin(merged)
        z = fit_tsne(X_kin, settings.tsne, settings.random_state)
        pred = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z)
        pred_mask, overlap, n_pred = _best_overlap(true, pred, c.name)
        if n_pred == 0:
            continue

        tp = pred_mask & (true == c.name)
        fp = pred_mask & (true == "field")
        n_tp, n_fp = int(tp.sum()), int(fp.sum())

        # Distances to the TRUE-member centroid (chemistry + spectral latent)
        chem = _standardize(merged, list(settings.elements))
        spec = _standardize(merged, emb_cols)
        out: dict[str, float | int | str] = {
            "cluster": c.name, "kind": c.kind, "region": c.region_deg,
            "n_members": n_members, "n_pred": n_pred, "n_tp": n_tp, "n_fp": n_fp,
        }
        for label, space in (("chem", chem), ("spectral", spec)):
            centroid = np.nanmedian(space[true == c.name], axis=0)
            dist = np.linalg.norm(space[pred_mask] - centroid, axis=1)
            is_member = (true[pred_mask] == c.name)
            out[f"auroc_{label}"] = _auroc(dist, is_member)
        rows.append(out)
        print(f"  {c.name:12} members={n_members:3d} pred={n_pred:3d} TP={n_tp:3d} FP={n_fp:3d} "
              f"AUROC_chem={out['auroc_chem']:.2f} AUROC_spec={out['auroc_spectral']:.2f}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--embeddings", default="data/embeddings/attention_broad_merged.parquet")
    ap.add_argument("--clusters", default="", help="comma-separated cluster names; empty = all")
    ap.add_argument("--out", default="results/two_stage_diagnosis.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False

    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    table = diagnose(Path(args.allstar), Path(args.embeddings), settings, names)
    table.to_csv(args.out, index=False)
    print(f"saved to {args.out}")
    if len(table):
        print("\n=== summary ===")
        print(f"  clusters: {len(table)}   mean AUROC_chem={table['auroc_chem'].mean():.2f}   "
              f"mean AUROC_spec={table['auroc_spectral'].mean():.2f}")


if __name__ == "__main__":
    main()
