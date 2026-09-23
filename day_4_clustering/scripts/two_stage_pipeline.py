"""Two-stage membership pipeline: kinematics recall -> spectral rejection.

Stage 1: kinematics-only (4-d) -> t-SNE -> HDBSCAN -> candidate members
(high recall, contaminated by kinematic doppelgangers).
Stage 2: reject candidates whose RNN-latent distance to the candidate
centroid is a robust outlier (median + k * 1.4826 * MAD).

Reports precision/recall before and after Stage 2, per cluster, so we can
check whether the measured precision gain follows the cluster age
(the through-line: young clusters blend into the field, old ones don't).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    tmask = true == cluster
    best, best_overlap, best_pred = -1, 0, 0
    for pc in np.unique(pred):
        if pc < 0:
            continue
        overlap = int(((pred == pc) & tmask).sum())
        if overlap > best_overlap:
            best, best_overlap, best_pred = pc, overlap, int((pred == pc).sum())
    if best < 0:
        return np.zeros(len(true), dtype=bool), 0, 0
    return (pred == best), best_overlap, best_pred


def two_stage(allstar_path: Path, embeddings_path: Path, settings: config.Settings,
              names: list[str], k: float) -> pd.DataFrame:
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

        # Stage 1: kinematics-only
        z = fit_tsne(_std_kin(merged), settings.tsne, settings.random_state)
        pred = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z)
        pred_mask, _, n_pred = _best_overlap(true, pred, c.name)
        if n_pred == 0:
            continue
        tp_before = int((pred_mask & (true == c.name)).sum())
        fp_before = int((pred_mask & (true == "field")).sum())

        # Stage 2: reject robust spectral outliers among candidates
        spec = _standardize(merged, emb_cols)
        centroid = np.nanmedian(spec[pred_mask], axis=0)
        dist = np.linalg.norm(spec[pred_mask] - centroid, axis=1)
        med = np.nanmedian(dist)
        mad = np.nanmedian(np.abs(dist - med))
        keep = dist <= med + k * 1.4826 * mad
        after = pred_mask.copy()
        after[pred_mask] = keep
        tp_after = int((after & (true == c.name)).sum())
        fp_after = int((after & (true == "field")).sum())

        rows.append({
            "cluster": c.name, "kind": c.kind,
            "n_members": n_members, "n_pred": n_pred,
            "rec_before": tp_before / n_members,
            "prec_before": tp_before / n_pred if n_pred else float("nan"),
            "n_after": int(keep.sum()),
            "rec_after": tp_after / n_members,
            "prec_after": tp_after / (tp_after + fp_after) if (tp_after + fp_after) else float("nan"),
            "fp_rejected": fp_before - fp_after,
            "tp_kept": tp_after,
        })
        print(f"  {c.name:12} prec {tp_before/n_pred:.2f}->{rows[-1]['prec_after']:.2f}  "
              f"rec {tp_before/n_members:.2f}->{tp_after/n_members:.2f}  "
              f"FP {fp_before:3d}->{fp_after:3d}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--embeddings", default="data/embeddings/attention_broad_merged.parquet")
    ap.add_argument("--clusters", default="", help="comma-separated cluster names; empty = all")
    ap.add_argument("--k", type=float, default=2.5, help="robust-outlier threshold (median + k*1.4826*MAD)")
    ap.add_argument("--out", default="results/two_stage_pipeline.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    table = two_stage(Path(args.allstar), Path(args.embeddings), settings, names, args.k)
    table.to_csv(args.out, index=False)
    print(f"\nsaved to {args.out}")
    if len(table):
        gain = table["prec_after"] - table["prec_before"]
        print(f"\n=== summary (n={len(table)}) ===")
        print(f"  precision {table['prec_before'].mean():.2f} -> {table['prec_after'].mean():.2f}  (+{gain.mean():.2f})")
        print(f"  recall    {table['rec_before'].mean():.2f} -> {table['rec_after'].mean():.2f}")


if __name__ == "__main__":
    main()
