"""Paper baseline (cluster-only separation) on the uniform common population.

`cluster baseline` scores every member row of the Astra table. On DR19 that
table lists some stars more than once (separate rows whose abundances differ
by a median 0.04 dex), so its 1002 member rows are only 806 distinct stars
(+34 with no APOGEE ID). This script scores the same question on the
deduplicated 982-star / 25-cluster population used by the head-to-head, so
the baseline slide and the head-to-head slides describe the same stars.

For abundances (16-d) and abundances + kinematics (20-d) it reports:

* homogeneity / completeness / v-measure / accuracy, mean +- sd over 7 seeds;
* t-SNE and UMAP homogeneity over 8 random row orders (the seed spread alone
  misses that the t-SNE partition can depend on the order of the rows);
* a t-SNE confusion matrix (PNG + CSV) for the most common outcome across
  row orders.

Usage (repo root):
    .venv/bin/python scripts/baseline_uniform.py --outdir results/baseline_uniform
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster import config  # noqa: E402
from cluster.baseline import (  # noqa: E402
    _fit_all,
    baseline_matrix,
    cluster_only,
    confusion_matrix_frame,
    plot_confusion,
    separation_scores,
)
from cluster.benchmark import cluster_embedding, fit_tsne, fit_umap  # noqa: E402
from cluster.cli import _prepared_for  # noqa: E402
from cluster.headtohead import Arm, _arm_frame, common_population  # noqa: E402
from cluster.spectral import ID_COLUMN  # noqa: E402
from cluster.stability import DEFAULT_SEEDS  # noqa: E402

from headtohead_extras import ARMS  # noqa: E402

N_ORDERS = 8
METRICS = ("homogeneity", "completeness", "v_measure", "accuracy")


def _n_clusters(pred: np.ndarray) -> int:
    return int(len(set(pred.tolist()) - {-1}))


def uniform_population(allstar: str, settings: config.Settings) -> tuple[pd.DataFrame, list[str]]:
    """The deduplicated head-to-head population, one row per star, sorted by ID.

    ``settings.require_aspcap_flag_clean`` should be False, as in
    ``cluster head-to-head``. Returns the member frame and its abundance columns.
    """
    prepared = _prepared_for(allstar, settings)
    arms = [Arm(label, path) for label, path in ARMS]
    shared, _, _ = common_population(prepared, arms, settings)
    df, columns = _arm_frame(prepared, arms[0], settings)
    sub = cluster_only(df)
    sub = sub[sub[ID_COLUMN].astype(str).isin(sorted(shared))]
    sub = sub.sort_values(ID_COLUMN).drop_duplicates(subset=[ID_COLUMN], keep="first")
    return sub.reset_index(drop=True), list(columns)


def row_orders(n: int, n_orders: int = N_ORDERS, seed: int = 0) -> dict[str, np.ndarray]:
    """The sorted order plus ``n_orders`` fixed random permutations of ``n`` rows."""
    rng = np.random.default_rng(seed)
    orders = {"sorted": np.arange(n)}
    for k in range(n_orders):
        orders[f"perm{k}"] = rng.permutation(n)
    return orders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--outdir", default="results/baseline_uniform")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # same as `cluster head-to-head`
    sub, columns = uniform_population(args.allstar, settings)
    print(f"population: {len(sub)} stars, {sub['cluster'].nunique()} clusters", flush=True)
    orders = row_orders(len(sub))

    seed_rows, order_rows = [], []
    tsne_params = {k: v for k, v in settings.tsne.items() if k != "method"}
    for label, use_kin in (("abundances (16-d)", False), ("abundances + kinematics (20-d)", True)):
        y = sub["cluster"].to_numpy()
        X = baseline_matrix(sub, settings, use_kin, elements=columns)
        for seed in DEFAULT_SEEDS:
            local = copy.deepcopy(settings)
            local.random_state = seed
            for method, pred in _fit_all(X, local).items():
                seed_rows.append({"features": label, "seed": seed, "method": method,
                                  "n_clusters": _n_clusters(pred), **separation_scores(y, pred)})
        preds: dict[str, np.ndarray] = {}
        for tag, idx in orders.items():
            part = sub.iloc[idx]
            yp = part["cluster"].to_numpy()
            Xp = baseline_matrix(part, settings, use_kin, elements=columns)
            for method in ("t-SNE", "UMAP"):
                Z = (fit_tsne(Xp, tsne_params, settings.random_state) if method == "t-SNE"
                     else fit_umap(Xp, settings.umap, settings.random_state))
                pred = cluster_embedding(Z, settings.hdbscan)
                if method == "t-SNE":
                    preds[tag] = pred
                order_rows.append({"features": label, "order": tag, "method": method,
                                   "n_clusters": _n_clusters(pred), **separation_scores(yp, pred)})
        # confusion matrix for the modal t-SNE outcome across row orders
        counts = Counter(_n_clusters(p) for p in preds.values())
        modal = counts.most_common(1)[0][0]
        tag = next(t for t, p in preds.items() if _n_clusters(p) == modal)
        part = sub.iloc[orders[tag]]
        yp = part["cluster"].to_numpy()
        cm = confusion_matrix_frame(yp, preds[tag])
        suffix = "kin" if use_kin else "chem"
        title = f"t-SNE — {'abundances + kinematics' if use_kin else 'abundances only'}"
        plot_confusion(cm, outdir / f"confusion_tsne_{suffix}.png", title=title)
        cm.to_csv(outdir / f"confusion_tsne_{suffix}.csv")
        print(f"{label}: t-SNE cluster counts over row orders {dict(counts)}; "
              f"figure from '{tag}' ({modal} clusters)", flush=True)

    seeds = pd.DataFrame(seed_rows)
    ordr = pd.DataFrame(order_rows)
    seeds.to_csv(outdir / "seeds.csv", index=False)
    ordr.to_csv(outdir / "orders.csv", index=False)
    summ = seeds.groupby(["features", "method"])[list(METRICS)].agg(["mean", "std"])
    print("\n=== mean ± sd over 7 seeds (sorted order) ===")
    print(summ.round(3).to_string(), flush=True)
    osum = ordr.groupby(["features", "method"]).agg(
        homog_mean=("homogeneity", "mean"), homog_sd=("homogeneity", "std"),
        homog_min=("homogeneity", "min"), homog_max=("homogeneity", "max"),
        clusters_min=("n_clusters", "min"), clusters_max=("n_clusters", "max"),
    )
    print(f"\n=== homogeneity over {N_ORDERS + 1} row orders (default seed) ===")
    print(osum.round(3).to_string(), flush=True)
    print(f"\nsaved to {outdir}/", flush=True)


if __name__ == "__main__":
    main()
