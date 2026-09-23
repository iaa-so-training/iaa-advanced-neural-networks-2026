"""Isochrone-fit grid: does better membership narrow the isochrone parameters?

For every cluster, build member masks from every signal x method combination
(abundances / RNN latent / kinematics, and their kinematic concatenations, each
through t-SNE / UMAP / EVoC -> HDBSCAN), then fit a PARSEC isochrone (asteca +
emcee) to each mask and record the recovered parameters AND their posterior
width (std). Membership precision (vs the kinematic truth) is recorded
alongside, so we can test: does cleaner membership -> narrower, more accurate
isochrone parameters?

Usage:
    uv run python scripts/isochrone_grid.py \
        --clusters "M 67" --n-walkers 8 --n-steps 120   # pilot
    uv run python scripts/isochrone_grid.py --n-walkers 8 --n-steps 120  # all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config, spectral
from cluster.baseline import baseline_matrix
from cluster.benchmark import fit_tsne
from cluster.clusters import CLUSTERS
from cluster.data import apply_quality_cuts, load_allstar
from cluster.isochrone import IsochroneFit, fit_isochrone
from cluster.literature import fetch_literature
from cluster.membership import angular_separation, kinematic_members

KIN_COLS = ["GAIAEDR3_PARALLAX", "GAIAEDR3_PMRA", "GAIAEDR3_PMDEC", "VHELIO_AVG"]

SIGNALS = ("abund", "rnn", "kin", "abund_kin", "rnn_kin")
METHODS = ("tsne", "umap", "evoc")


def _std_kin(df: pd.DataFrame) -> np.ndarray:
    kin = df[KIN_COLS].to_numpy(dtype=float)
    med = np.nanmedian(kin, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    kin = np.where(np.isfinite(kin), kin, med[np.newaxis, :])
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    return (kin - med) / scale


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


def _predicted_masks(X: np.ndarray, true: np.ndarray, cluster: str,
                     settings: config.Settings, methods: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Run t-SNE / UMAP / EVoC -> HDBSCAN and return best-overlap masks."""
    import hdbscan
    import umap
    from evoc import EVoC

    masks: dict[str, np.ndarray] = {}
    if "tsne" in methods:
        z = fit_tsne(X, settings.tsne, settings.random_state)
        masks["tsne"] = _best_overlap(true, hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z), cluster)
    if "umap" in methods:
        z = umap.UMAP(n_components=2, random_state=settings.random_state, **settings.umap).fit_transform(X)
        masks["umap"] = _best_overlap(true, hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z), cluster)
    if "evoc" in methods:
        masks["evoc"] = _best_overlap(
            true, EVoC(random_state=settings.random_state, **settings.evoc).fit_predict(X), cluster,
        )
    return masks


def grid(allstar_path: Path, embeddings_path: Path, settings: config.Settings,
         names: list[str], methods: tuple[str, ...],
         n_walkers: int, n_steps: int) -> pd.DataFrame:
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
        true = merged["cluster"].to_numpy()
        n_members = int((true == c.name).sum())
        if n_members < 25:
            print(f"  {c.name}: {n_members} members < 25, skip", flush=True)
            continue

        # signal matrices (all standardise-only, consistent with baseline_matrix)
        abund = baseline_matrix(merged, settings, False, elements=list(settings.elements))
        rnn = baseline_matrix(merged, settings, False, elements=list(emb_cols))
        kin = _std_kin(merged)
        signals: dict[str, np.ndarray] = {
            "abund": abund,
            "rnn": rnn,
            "kin": kin,
            "abund_kin": np.hstack([abund, kin]),
            "rnn_kin": np.hstack([rnn, kin]),
        }

        masks: dict[str, np.ndarray] = {"truth": true == c.name}
        for sig in SIGNALS:
            for mth, mask in _predicted_masks(signals[sig], true, c.name, settings, methods).items():
                masks[f"{sig}_{mth}"] = mask

        grid = "metal_poor" if c.kind == "globular" else "solar"
        lit = fetch_literature(c)
        print(f"\n=== {c.name} ({c.kind})  n_members={n_members} ===", flush=True)
        for label, mask in masks.items():
            n = int(mask.sum())
            tp = int((mask & (true == c.name)).sum())
            precision = tp / n if n else float("nan")
            recall = tp / n_members
            row: dict[str, float | int | str] = {
                "cluster": c.name, "kind": c.kind, "combo": label, "n": n,
                "precision": precision, "recall": recall,
            }
            if n < 25:
                rows.append(row)
                print(f"    {label:14} n={n:3d} <25 skip", flush=True)
                continue
            try:
                fit: IsochroneFit = fit_isochrone(
                    merged[mask], n_walkers=n_walkers, n_steps=n_steps, grid=grid,
                )
                age = float(10.0 ** (fit.best["loga"] - 9.0))
                row.update({
                    "age": age, "std_loga": fit.std["loga"],
                    "dm": fit.best["dm"], "std_dm": fit.std["dm"],
                    "met": fit.best["met"], "std_met": fit.std["met"],
                    "max_lkl": fit.max_lkl,
                    "age_resid": abs(age - lit["age_Gyr"]) if np.isfinite(lit["age_Gyr"]) else float("nan"),
                    "dm_resid": abs(fit.best["dm"] - lit["dm"]),
                })
                rows.append(row)
                print(f"    {label:14} n={n:3d} prec={precision:.2f} age={age:5.2f} "
                      f"std_loga={fit.std['loga']:.3f} dm={fit.best['dm']:5.2f} std_dm={fit.std['dm']:.2f}", flush=True)
            except ValueError as exc:
                row["note"] = str(exc)[:60]
                rows.append(row)
                print(f"    {label:14} n={n:3d} fit failed: {exc}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--embeddings", default="data/embeddings/attention_broad_merged.parquet")
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all")
    ap.add_argument("--methods", default="tsne,umap,evoc")
    ap.add_argument("--n-walkers", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=120)
    ap.add_argument("--out", default="results/isochrone_grid.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    table = grid(Path(args.allstar), Path(args.embeddings), settings, names,
                 methods, args.n_walkers, args.n_steps)
    table.to_csv(args.out, index=False)
    print(f"\nsaved to {args.out}  ({len(table)} rows)")


if __name__ == "__main__":
    main()
