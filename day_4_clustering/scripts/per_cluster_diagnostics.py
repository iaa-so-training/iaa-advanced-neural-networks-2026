"""Per-cluster diagnostics for the masked-AE latent (DR19).

For each APO cluster with spectral members: n_members, per-method homogeneity
(cluster purity in the full clustering), and the confusion-matrix row.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from cluster import config
from cluster.baseline import baseline_labels, cluster_only, separation_scores
from cluster.clusters import CLUSTERS
from cluster.data import prepare
from cluster.spectral import spectral_prepared

SEED = dict(
    seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
    seed_parallax_frac=config.SEED_PARALLAX_FRAC,
    seed_pm_tol=config.SEED_PM_TOL,
    seed_rv_tol=config.SEED_RV_TOL,
    n_refine_passes=config.N_REFINE_PASSES,
    refine_sigma=config.REFINE_SIGMA,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--min-members", type=int, default=1)
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    clusters = [c for c in CLUSTERS if c.name in settings.resolve_cluster_names(
        [c.name for c in CLUSTERS])]

    prep_ab = prepare("data/astraAllStarASPCAP-0.6.0.fits.gz", settings, clusters, **SEED)
    prep = spectral_prepared(prep_ab, args.embeddings, settings)
    sub = cluster_only(prep.df)

    counts = sub["cluster"].value_counts()
    keep = [c for c in counts.index if counts[c] >= args.min_members]
    sub = sub[sub["cluster"].isin(keep)]

    true = sub["cluster"].to_numpy()
    # reuse the baseline's fitting (t-SNE/UMAP/EVoC on the embedding matrix)
    from cluster.baseline import baseline_matrix, _fit_all
    X = baseline_matrix(sub, settings, use_kinematics=False, elements=list(prep.elements))
    preds = _fit_all(X, settings)

    rows = []
    for cl in keep:
        mask = true == cl
        row = {"cluster": cl, "n_members": int(mask.sum())}
        for method, pred in preds.items():
            # homogeneity of this cluster = fraction of its members put in its majority label
            member_pred = pred[mask]
            if len(member_pred) == 0:
                row[f"{method}_homog"] = np.nan
                continue
            majority = pd.Series(member_pred).value_counts().index[0]
            row[f"{method}_homog"] = float((member_pred == majority).mean())
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n_members", ascending=False)
    print(df.to_string(index=False))
    out = f"results/per_cluster_{Path(args.embeddings).stem}.csv" if False else None


if __name__ == "__main__":
    from pathlib import Path
    main()
