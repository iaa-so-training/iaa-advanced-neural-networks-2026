"""Extra head-to-head rows on the uniform common population.

``cluster head-to-head`` scores the feature arms on the intersection of their
star sets (982 stars, 25 clusters for the uniform DR19 re-run). Two numbers
the deck quotes need that same population but are not produced by it:

1. a kinematics-only row (parallax, pmra, pmdec, rv) -- the physical
   reference the spectra are compared against;
2. the M 3 ablation for the abundance arm *and* the masked-AE arm on the
   same stars (``cluster ablate`` builds a separate population per arm, so
   its abundance and spectral rows are not directly comparable).

Usage (repo root):
    .venv/bin/python scripts/headtohead_extras.py \\
        --out results/head_to_head/extras_rerun.csv
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cluster import config
from cluster.baseline import KINEMATIC_COLUMNS, _fit_all, baseline_matrix, cluster_only
from cluster.cli import _prepared_for
from cluster.data import PreparedData
from cluster.headtohead import Arm, _arm_frame, common_population
from cluster.spectral import ID_COLUMN
from cluster.stability import DEFAULT_SEEDS, degeneracy, stability

# Same arms, same order as the `cluster head-to-head` run behind the deck.
ARMS: list[tuple[str, str | None]] = [
    ("abundances (16-d)", None),
    ("PCA 64-d", "data/embeddings/pca_64.parquet"),
    ("PCA 256-d", "data/embeddings/pca_256.parquet"),
    ("masked AE 256-d", "data/embeddings/masked_latent.parquet"),
]
KIN_LABEL = "kinematics only (4-d)"


def _write_kinematics(prepared: PreparedData, path: Path) -> set[str]:
    """Save the four kinematic columns as an embedding artifact.

    Returns the IDs whose four values are all finite.
    """
    df = prepared.df.drop_duplicates(subset=[ID_COLUMN], keep="first")
    out = df[[ID_COLUMN, *KINEMATIC_COLUMNS]].copy()
    out.columns = [ID_COLUMN, *(f"kin_{c.lower()}" for c in KINEMATIC_COLUMNS)]
    finite = np.isfinite(out.iloc[:, 1:].to_numpy(dtype=float)).all(axis=1)
    out = out[finite]
    out.to_parquet(path, index=False)
    return set(out[ID_COLUMN].astype(str))


def _score(
    prepared: PreparedData,
    arm: Arm,
    settings: config.Settings,
    shared: set[str],
    exclude: tuple[str, ...],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Seed-stability homogeneity for one arm, plus a collapse check.

    The collapse check fits once at the default seed and records, per
    method, how many clusters came out and how big the largest one is --
    a homogeneity number from a collapsed partition is a floor, not a
    comparison.
    """
    df, columns = _arm_frame(prepared, arm, settings)
    sub = cluster_only(df)
    sub = sub[sub[ID_COLUMN].astype(str).isin(shared)]
    sub = sub.sort_values(ID_COLUMN).drop_duplicates(subset=[ID_COLUMN], keep="first")
    for name in exclude:
        sub = sub[(sub["cluster"] != name).to_numpy()]
    y = sub["cluster"].to_numpy()
    X = baseline_matrix(sub, settings, False, elements=columns)
    stab = stability(X, y, settings, seeds=DEFAULT_SEEDS)
    stab.insert(0, "features", arm.label)
    stab.insert(1, "excluded", ",".join(exclude) or "-")
    stab.insert(2, "n_stars", len(y))
    stab.insert(3, "n_clusters", int(np.unique(y).size))
    degen = [
        {"features": arm.label, "excluded": ",".join(exclude) or "-", "method": m,
         **degeneracy(pred)}
        for m, pred in _fit_all(X, settings).items()
    ]
    return stab, degen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--out", default="results/head_to_head/extras_rerun.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # same as `cluster head-to-head`
    prepared = _prepared_for(args.allstar, settings)
    arms = [Arm(label, path) for label, path in ARMS]
    shared, keep, _ = common_population(prepared, arms, settings)
    print(f"common population: {len(shared)} stars, {len(keep)} clusters", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        kin_path = Path(tmp) / "kinematics.parquet"
        kin_ids = _write_kinematics(prepared, kin_path)
        missing = shared - kin_ids
        if missing:
            raise SystemExit(f"{len(missing)} shared members lack finite kinematics")
        jobs = [
            (Arm(KIN_LABEL, kin_path), ()),
            (arms[0], ()),
            (arms[0], ("M 3",)),
            (arms[3], ("M 3",)),
        ]
        rows, degen_rows = [], []
        for arm, exclude in jobs:
            print(f"scoring {arm.label} (excluded: {exclude or '-'})...", flush=True)
            stab, degen = _score(prepared, arm, settings, shared, exclude)
            rows.append(stab)
            degen_rows.extend(degen)

    table = pd.concat(rows, ignore_index=True)
    degen_table = pd.DataFrame(degen_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    degen_path = str(args.out).replace(".csv", "_degeneracy.csv")
    degen_table.to_csv(degen_path, index=False)
    print("\n=== homogeneity, mean ± std over seeds ===")
    for _, r in table.iterrows():
        print(
            f"  {r['features']:24} excl={r['excluded']:4} n={r['n_stars']:4d} "
            f"k={r['n_clusters']:2d}  {r['method']:6} {r['mean']:.3f} ± {r['std']:.3f}"
            f"  [{r['min']:.2f}, {r['max']:.2f}]",
        )
    print("\n=== collapse check (default seed) ===")
    print(degen_table.to_string(index=False))
    print(f"\nsaved to {args.out} and {degen_path}")


if __name__ == "__main__":
    main()
