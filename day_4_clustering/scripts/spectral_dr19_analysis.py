"""DR19 spectral full-analysis: baseline + field retrieval across feature sets.

Reproduces the DR17 spectral experiment matrix (docs/spectral_benchmark_results.md)
on the DR19 long re-train embeddings:

  features: abundances (16-d) | spectral (256-d) | combined (272-d) | kin-only (4-d)
  x kin    : no-kin | +kin (for the chemical/spectral blocks)
  benchmarks: cluster-only baseline (t-SNE/UMAP/EVoC) + field retrieval

Usage:
    uv run python scripts/spectral_dr19_analysis.py \
        --embeddings data/embeddings/attention_dr19_long.parquet \
        --allstar data/astraAllStarASPCAP-0.6.0.fits.gz \
        --out results/spectral_dr19
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.baseline import (
    KINEMATIC_COLUMNS,
    _fit_all,
    cluster_only,
    separation_scores,
)
from cluster.benchmark import run_benchmark
from cluster.catalog import attach_referee
from cluster.clusters import CLUSTERS
from cluster.data import PreparedData, make_matrix, prepare
from cluster.spectral import (
    embedding_matrix,
    load_embedding_frame,
    spectral_prepared,
)

SEED = dict(
    seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
    seed_parallax_frac=config.SEED_PARALLAX_FRAC,
    seed_pm_tol=config.SEED_PM_TOL,
    seed_rv_tol=config.SEED_RV_TOL,
    n_refine_passes=config.N_REFINE_PASSES,
    refine_sigma=config.REFINE_SIGMA,
)


def _standardize(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    return (X - med) / scale


def kin_matrix(df: pd.DataFrame) -> np.ndarray:
    return _standardize(df[KINEMATIC_COLUMNS].to_numpy(dtype=float))


def build_feature_matrices(
    prep_ab: PreparedData, embeddings_path: str, settings: config.Settings,
) -> dict[str, PreparedData]:
    """Build one PreparedData per feature set, all aligned on the spectral rows."""
    prep_spec = spectral_prepared(prep_ab, embeddings_path, settings)
    spec_df = prep_spec.df
    spec_cols = list(prep_spec.elements)

    # abundance matrix on the same rows (dedup + inner-join on APOGEE_ID)
    ab_df = prep_ab.df.drop_duplicates(subset=["APOGEE_ID"], keep="first")
    ab_df = ab_df.merge(spec_df[["APOGEE_ID"]], on="APOGEE_ID", how="inner")
    X_ab = make_matrix(ab_df, settings)          # impute→standardise→L2-normalise
    X_spec = prep_spec.X                          # standardise→L2-normalise

    # combined: both blocks already standardised+L2-normalised, concatenate
    X_comb = np.hstack([X_ab, X_spec])
    X_kin = kin_matrix(spec_df)

    return {
        "abundances": PreparedData(df=ab_df, X=X_ab, elements=list(settings.elements)),
        "spectral": prep_spec,
        "combined": PreparedData(df=spec_df, X=X_comb, elements=["combined"]),
        "kin-only": PreparedData(df=spec_df, X=X_kin, elements=["kin-only"]),
    }


def baseline_table(
    prep: PreparedData, settings: config.Settings, feature_name: str,
    min_members: int = 5,
) -> pd.DataFrame:
    """Cluster-only separation for a feature set, with +kin variant where meaningful."""
    rows: list[dict] = []

    def _score(X: np.ndarray, use_kin: bool) -> None:
        sub = cluster_only(prep.df)
        counts = sub["cluster"].value_counts()
        keep = [c for c in counts.index if counts[c] >= min_members]
        sub = sub[sub["cluster"].isin(keep)]
        if sub.empty:
            return
        # rebuild X on the surviving cluster-only rows (aligned by index)
        idx = sub.index
        Xsub = X[np.asarray([prep.df.index.get_loc(i) for i in idx])]
        if use_kin:
            Xsub = np.hstack([Xsub, kin_matrix(sub)])
        true = sub["cluster"].to_numpy()
        for name, pred in _fit_all(Xsub, settings).items():
            sc = separation_scores(true, pred)
            rows.append({"features": feature_name + (" + kin" if use_kin else ""),
                         "method": name, "n_stars": int(true.size),
                         "n_clusters": int(np.unique(true).size), **sc})

    X = prep.X
    _score(X, use_kin=False)
    if feature_name != "kin-only":
        _score(X, use_kin=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/spectral_dr19")
    ap.add_argument("--min-members", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # spectral reads the raw spectrum
    clusters = [c for c in CLUSTERS if c.name in settings.resolve_cluster_names(
        [c.name for c in CLUSTERS])]

    print("📦 preparing abundance data...")
    prep_ab = prepare(Path(args.allstar), settings, clusters, **SEED)

    print("🔀 building feature variants (aligned on spectral rows)...")
    variants = build_feature_matrices(prep_ab, args.embeddings, settings)
    for name, v in variants.items():
        print(f"   {name}: {v.X.shape[0]} stars × {v.X.shape[1]} dims")

    print("🧪 cluster-only baseline...")
    base_rows = []
    for name, v in variants.items():
        t = baseline_table(v, settings, name, min_members=args.min_members)
        base_rows.append(t)
    baseline_df = pd.concat(base_rows, ignore_index=True)
    baseline_df.to_csv(outdir / "baseline.csv", index=False)
    print(baseline_df.to_string(index=False))

    print("\n🏃 field retrieval (with Simbad referee)...")
    field_rows = []
    for name, v in variants.items():
        df = attach_referee(v.df, clusters, settings)
        v = PreparedData(df=df, X=v.X, elements=v.elements)
        res = run_benchmark(v, settings)
        macro = res.macro()
        for _, row in macro.iterrows():
            field_rows.append({"features": name, "method": row["method"],
                               "recall": row["recall"], "precision": row["precision"],
                               "n_clusters_scored": row["n_clusters_scored"],
                               "n_true_members": row["n_true_members"]})
    field_df = pd.DataFrame(field_rows)
    field_df.to_csv(outdir / "field_retrieval.csv", index=False)
    print(field_df.to_string(index=False))

    # combined markdown
    (outdir / "results.md").write_text(
        "# DR19 spectral re-analysis\n\n"
        "## Cluster-only baseline (homogeneity/completeness/v-measure/accuracy)\n\n"
        + baseline_df.to_string(index=False)
        + "\n\n## Field retrieval (recall/precision)\n\n"
        + field_df.to_string(index=False)
        + "\n",
    )
    print(f"\n💾 results → {outdir}/")


if __name__ == "__main__":
    main()
