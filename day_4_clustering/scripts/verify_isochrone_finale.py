"""Verify the finale: does better membership -> better isochrone parameters?

Fits a PARSEC isochrone (asteca + emcee) to three membership sources for a
cluster:
  - truth   : kinematic_members (the label)
  - stage1  : kinematics-only t-SNE + HDBSCAN (contaminated)
  - twostage: stage1 + spectral-distance rejection (cleaner)

If the finale holds, the twostage fit recovers parameters closer to
literature than stage1 (and close to truth).
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
from cluster.isochrone import IsochroneFit, fit_isochrone
from cluster.literature import fetch_literature
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


def _age_gyr(fit: IsochroneFit) -> float:
    return float(10.0 ** (fit.best["loga"] - 9.0))


def verify(allstar_path: Path, embeddings_path: Path, settings: config.Settings,
           names: list[str], n_walkers: int, n_steps: int) -> None:
    import hdbscan

    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)
    emb = spectral.load_embedding_frame(embeddings_path)
    emb_cols = spectral.embedding_columns(emb)

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
        if (true == c.name).sum() < 25:
            print(f"  {c.name}: only {(true == c.name).sum()} members, skip")
            continue

        # masks
        z = fit_tsne(_std_kin(merged), settings.tsne, settings.random_state)
        pred = hdbscan.HDBSCAN(**settings.hdbscan).fit_predict(z)
        stage1 = _best_overlap(true, pred, c.name)

        spec = _standardize(merged, emb_cols)
        centroid = np.nanmedian(spec[stage1], axis=0)
        dist = np.linalg.norm(spec[stage1] - centroid, axis=1)
        med = np.nanmedian(dist)
        mad = np.nanmedian(np.abs(dist - med))
        keep = dist <= med + 2.5 * 1.4826 * mad
        twostage = stage1.copy()
        twostage[stage1] = keep

        grid = "metal_poor" if c.kind == "globular" else "solar"
        lit = fetch_literature(c)
        print(f"\n=== {c.name} ({c.kind}, grid={grid}) ===  lit age={lit['age_Gyr']:.2f} "
              f"dm={lit['dm']:.2f} feh={lit['feh']:.2f}", flush=True)
        for label, mask in [("truth", true == c.name), ("stage1", stage1), ("twostage", twostage)]:
            members = merged[mask]
            n = int(mask.sum())
            if n < 25:
                print(f"  {label:9} n={n:3d}  <25, skip", flush=True)
                continue
            fit = fit_isochrone(members, n_walkers=n_walkers, n_steps=n_steps, grid=grid)
            age = _age_gyr(fit)
            print(f"  {label:9} n={n:3d}  age={age:5.2f} (resid {abs(age - lit['age_Gyr']):.2f})  "
                  f"dm={fit.best['dm']:5.2f} (resid {abs(fit.best['dm'] - lit['dm']):.2f})  "
                  f"met={fit.best['met']:.4f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--embeddings", default="data/embeddings/attention_broad_merged.parquet")
    ap.add_argument("--clusters", default="", help="comma-separated; empty = all")
    ap.add_argument("--n-walkers", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=300)
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    names = [n.strip() for n in args.clusters.split(",") if n.strip()] or []
    verify(Path(args.allstar), Path(args.embeddings), settings, names, args.n_walkers, args.n_steps)


if __name__ == "__main__":
    main()
