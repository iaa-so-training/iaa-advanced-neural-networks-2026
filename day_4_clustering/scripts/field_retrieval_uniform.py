"""Field retrieval on one star list for every feature set.

`cluster run` (abundances) and `cluster run --spectral` do not score the same
stars: the abundance run keeps a 25,000-star subsample (about 3% members),
while the spectral run keeps only the stars that have an embedding (about 26%
members). Their precisions are therefore not comparable. This script scores
the ASPCAP abundances, PCA-64 and the masked AE on exactly the same stars:
every quality-cut star (one row per APOGEE_ID) that has an embedding in both
spectral artifacts. Each arm runs the benchmark pipeline (t-SNE -> HDBSCAN*,
UMAP -> HDBSCAN*, EVoC, default seed) and is scored against both ground
truths: the kinematic labels and the Simbad catalogue labels.

Usage (repo root):
    .venv/bin/python scripts/field_retrieval_uniform.py --out results/field_retrieval_uniform.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cluster import config  # noqa: E402
from cluster.benchmark import _score_one, knn_purity, run_benchmark  # noqa: E402
from cluster.catalog import attach_referee  # noqa: E402
from cluster.cli import _prepared_for  # noqa: E402
from cluster.clusters import CLUSTERS  # noqa: E402
from cluster.data import PreparedData, make_matrix  # noqa: E402
from cluster.spectral import (  # noqa: E402
    ID_COLUMN,
    embedding_columns,
    embedding_matrix,
    load_embedding_frame,
)

SPECTRAL_ARMS = {
    "PCA 64-d": "data/embeddings/pca_64.parquet",
    "masked AE 256-d": "data/embeddings/masked_latent.parquet",
}


def common_frame(allstar: str, settings: config.Settings) -> pd.DataFrame:
    """Quality-cut stars, one row per APOGEE_ID, that have every embedding."""
    prepared = _prepared_for(allstar, settings)
    df = prepared.df
    df = df[df[ID_COLUMN].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first")
    ids = set(df[ID_COLUMN].astype(str))
    for path in SPECTRAL_ARMS.values():
        ids &= set(load_embedding_frame(path)[ID_COLUMN].astype(str))
    df = df[df[ID_COLUMN].astype(str).isin(ids)]
    return df.sort_values(ID_COLUMN).reset_index(drop=True)


def _macro(truth: np.ndarray, labels: np.ndarray) -> tuple[float, float, int, int]:
    s = _score_one(truth, labels)
    s = s[s["n_true"] >= 1]
    return float(s["recall"].mean()), float(s["precision"].mean()), len(s), int(s["n_true"].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/field_retrieval_uniform.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # same as `cluster head-to-head`
    settings.max_stars = None  # the star list is set by the embeddings, not a subsample
    df = common_frame(args.allstar, settings)
    names = settings.resolve_cluster_names([c.name for c in CLUSTERS])
    df = attach_referee(df, [c for c in CLUSTERS if c.name in names], settings)
    truths = {"kinematic": df["cluster"].to_numpy(), "Simbad": df["referee"].to_numpy()}
    for tname, t in truths.items():
        n_mem = int((t != "field").sum())
        print(f"{tname}: {len(t)} stars, {n_mem} members in {len(set(t) - {'field'})} clusters, "
              f"chance precision {n_mem / len(t):.3f}", flush=True)

    matrices = {"abundances (16-d)": make_matrix(df, settings)}
    for arm, path in SPECTRAL_ARMS.items():
        frame = load_embedding_frame(path)
        frame = frame[frame[ID_COLUMN].astype(str).isin(set(df[ID_COLUMN].astype(str)))]
        # pca_*.parquet lists 60 stars twice (rows agree to ~1e-5); keep one, as head-to-head does
        frame = frame.drop_duplicates(subset=[ID_COLUMN], keep="first")
        merged = df[[ID_COLUMN]].merge(frame, on=ID_COLUMN, how="left")
        assert len(merged) == len(df) and merged[ID_COLUMN].equals(df[ID_COLUMN])
        matrices[arm] = embedding_matrix(merged, embedding_columns(frame), settings)

    rows = []
    base = df.drop(columns=["referee"])
    for arm, X in matrices.items():
        print(f"\n=== {arm}: {X.shape[0]} stars x {X.shape[1]} dims", flush=True)
        result = run_benchmark(PreparedData(df=base, X=X, elements=[]), settings)
        for method, r in result.results.items():
            for tname, truth in truths.items():
                rec, prec, n_cl, n_mem = _macro(truth, r.labels)
                pur = float("nan")
                if r.embedding is not None:
                    p = knn_purity(r.embedding, truth)
                    pur = sum(p.values()) / len(p) if p else float("nan")
                rows.append({"features": arm, "method": method, "truth": tname,
                             "n_stars": len(truth), "n_members": n_mem, "n_clusters": n_cl,
                             "recall": rec, "precision": prec, "knn_purity": pur})
                print(f"  {method:5} vs {tname:9}: recall {rec:.3f}  precision {prec:.3f}  "
                      f"kNN purity {pur:.3f}", flush=True)

    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("\n" + table.round(3).to_string(index=False))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
