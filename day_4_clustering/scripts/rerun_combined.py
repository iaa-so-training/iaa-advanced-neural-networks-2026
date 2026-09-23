"""Rerun the clustering benchmark on the combined GALAH + APOGEE sample.

For each covered cluster, score t-SNE / UMAP / EVoC across three signals:
  - abundances (14 common elements)
  - kinematics (parallax, pmra, pmdec, rv)
  - abundances + kinematics

against the kinematic ground truth (sigma-clip, same as the core pipeline).

Usage:
    uv run python scripts/rerun_combined.py [--clusters "M 67,Pleiades"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.benchmark import fit_tsne
from cluster.clusters import CLUSTERS
from cluster.membership import kinematic_members

ABUND = ["C_FE", "O_FE", "NA_FE", "MG_FE", "AL_FE", "SI_FE", "K_FE",
         "CA_FE", "TI_FE", "V_FE", "CR_FE", "MN_FE", "NI_FE", "FE_H"]
KIN = ["GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG"]


def _standardize(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].to_numpy(dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med[np.newaxis, :])
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    return (X - med) / scale


def _best_overlap(true: np.ndarray, pred: np.ndarray, cluster: str) -> np.ndarray:
    tmask = true == cluster
    best, best_overlap = -1, 0
    for pc in np.unique(pred):
        if pc < 0:
            continue
        overlap = int(((pred == pc) & tmask).sum())
        if overlap > best_overlap:
            best, best_overlap = pc, overlap
    if best < 0:
        return np.zeros(len(true), dtype=bool)
    return pred == best


def _score(mask: np.ndarray, true: np.ndarray, cluster: str) -> tuple[float, float]:
    n_pred = int(mask.sum())
    n_true = int((true == cluster).sum())
    tp = int((mask & (true == cluster)).sum())
    return (tp / n_true if n_true else 0.0, tp / n_pred if n_pred else 0.0)


def rerun(cluster_names: list[str], settings: config.Settings) -> pd.DataFrame:
    import hdbscan
    import umap
    from evoc import EVoC

    rows: list[dict[str, float | int | str]] = []
    for c in CLUSTERS:
        if cluster_names and c.name not in cluster_names:
            continue
        path = Path(f"data/galah_apogee_{c.name.replace(' ', '_')}.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # rename kinematics to the APOGEE names the membership function expects
        df = df.rename(columns={"PARALLAX": "GAIAEDR3_PARALLAX", "PMRA": "GAIAEDR3_PMRA",
                                "PMDEC": "GAIAEDR3_PMDEC", "RV": "VHELIO_AVG"})
        m = kinematic_members(
            df, c,
            config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
            config.SEED_PM_TOL, config.SEED_RV_TOL,
            config.N_REFINE_PASSES, config.REFINE_SIGMA,
        ).to_numpy()
        true = np.where(m, c.name, "field")
        n_members = int(m.sum())
        n_field = int((~m).sum())
        if n_members < 20:
            continue

        abund = _standardize(df, ABUND)
        kin = _standardize(df, KIN)
        signals: dict[str, np.ndarray] = {
            "abund": abund, "kin": kin, "abund_kin": np.hstack([abund, kin]),
        }
        print(f"\n=== {c.name}: members={n_members} field={n_field} ===", flush=True)
        for sig, X in signals.items():
            masks: dict[str, np.ndarray] = {
                "tsne": _best_overlap(true, hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(
                    fit_tsne(X, settings.tsne, settings.random_state)), c.name),
                "umap": _best_overlap(true, hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(
                    umap.UMAP(n_components=2, random_state=settings.random_state, **settings.umap).fit_transform(X)), c.name),
                "evoc": _best_overlap(true, EVoC(random_state=settings.random_state, **settings.evoc).fit_predict(X), c.name),
            }
            for mth, mask in masks.items():
                rec, prec = _score(mask, true, c.name)
                rows.append({"cluster": c.name, "kind": c.kind, "signal": sig,
                             "method": mth, "recall": rec, "precision": prec,
                             "n_members": n_members, "n_field": n_field})
                print(f"    {sig:9} {mth:5} recall={rec:.2f} precision={prec:.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all covered")
    ap.add_argument("--out", default="results/rerun_combined.csv")
    args = ap.parse_args()
    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    table = rerun(names, settings)
    table.to_csv(args.out, index=False)
    print(f"\nsaved to {args.out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
