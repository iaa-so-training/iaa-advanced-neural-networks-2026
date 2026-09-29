"""Casamiquela-style scores on OUR uniform cluster-only population (982 stars, 25 clusters).

For the abundance arm (16-d), per method and seed (sorted row order) and, for t-SNE,
per row order: homogeneity / completeness / V (must reproduce results/baseline_uniform/
seeds.csv), the chance-adjusted AMI and ARI, the recovery fraction RF40 / RF70 of
Casamiquela et al. 2021 (Sect. 4.1; ``cluster.baseline.recovery_fraction``, which
tests/test_baseline.py checks against their Table 2), the share of "statistical"
groups, and each run's own permutation null (same group sizes, labels shuffled).

Also prints the per-cluster star counts (with median logg / Teff, since Casamiquela et
al. used red-clump stars only to remove evolutionary-state systematics) and runs
K-means (K = number of true clusters) on the same matrix, since the workbook's
K-means row was on the 1 002-row matrix.

``--kinds open`` restricts the population to the 18 open clusters, the closest
analogue of their thin-disc open-cluster sample (they have no globulars).

Usage (repo root):
    .venv/bin/python scripts/casamiquela_comparison.py                      # 25 clusters
    .venv/bin/python scripts/casamiquela_comparison.py --kinds open \
        --out results/casamiquela_comparison_open.csv                        # 18 open clusters
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import homogeneity_completeness_v_measure as hcv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from baseline_uniform import row_orders, uniform_population  # noqa: E402
from cluster import config  # noqa: E402
from cluster.baseline import _fit_all, baseline_matrix, recovery_fraction  # noqa: E402
from cluster.benchmark import cluster_embedding, fit_tsne  # noqa: E402
from cluster.clusters import CLUSTERS  # noqa: E402
from cluster.stability import DEFAULT_SEEDS  # noqa: E402

rng = np.random.default_rng(7)
N_PERM = 200


def codes(x):
    return np.unique(np.asarray([str(v) for v in x]), return_inverse=True)[1]


def score(y, pred):
    h, c, v = hcv(codes(y), codes(pred))
    r40 = recovery_fraction(y, pred, 0.40)
    r70 = recovery_fraction(y, pred, 0.70)
    nulls = []
    for _ in range(N_PERM):
        q = rng.permutation(pred)
        hn, cn, vn = hcv(codes(y), codes(q))
        nulls.append((vn, recovery_fraction(y, q, 0.40)["rf"]))
    nulls = np.array(nulls)
    noise = float(np.mean(np.asarray([str(p) for p in pred]) == "-1"))
    return {"h": h, "c": c, "V": v, "AMI": ami(codes(y), codes(pred)), "ARI": ari(codes(y), codes(pred)),
            "RF40": r40["rf"], "RF70": r70["rf"], "stat40": r40["statistical_fraction"],
            "n_groups": r40["n_groups"], "noise_frac": noise,
            "V_null": nulls[:, 0].mean(), "RF40_null": nulls[:, 1].mean(),
            "RF40_null_p95": np.percentile(nulls[:, 1], 95),
            "recovered40": ";".join(sorted(r40["recovered"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--kinds", choices=("all", "open"), default="all")
    ap.add_argument("--out", default="results/casamiquela_comparison.csv")
    args = ap.parse_args()

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False  # as in `cluster head-to-head` / baseline_uniform.py
    sub, columns = uniform_population(args.allstar, settings)
    if args.kinds == "open":
        keep = {c.name for c in CLUSTERS if c.kind == "open"}
        sub = sub[sub["cluster"].isin(keep)].reset_index(drop=True)
    y = sub["cluster"].to_numpy()
    n_true = sub["cluster"].nunique()
    print(f"population ({args.kinds}): {len(sub)} stars, {n_true} clusters", flush=True)
    counts = sub["cluster"].value_counts()
    print(f"median {counts.median():.0f}, mean {counts.mean():.1f}, min {counts.min()}, max {counts.max()}", flush=True)
    per = sub.groupby("cluster").agg(n=("cluster", "size"), logg_med=("LOGG", "median"),
                                     teff_med=("TEFF", "median"),
                                     dwarf_frac=("LOGG", lambda v: float((v > 3.8).mean())))
    print("per-cluster stars, median logg / Teff, fraction with logg > 3.8:", flush=True)
    print(per.sort_values("n", ascending=False).round(2).to_string(), flush=True)

    X = baseline_matrix(sub, settings, False, elements=columns)
    rows = []
    for seed in DEFAULT_SEEDS:
        local = copy.deepcopy(settings)
        local.random_state = seed
        for method, pred in _fit_all(X, local).items():
            rows.append({"features": "abundances (16-d)", "order": "sorted", "seed": seed,
                         "method": method, **score(y, pred)})
        km = KMeans(n_clusters=n_true, n_init=10, random_state=seed).fit(X)
        rows.append({"features": "abundances (16-d)", "order": "sorted", "seed": seed,
                     "method": f"K-means (K={n_true})", **score(y, km.labels_)})
        print(f"  seed {seed} done", flush=True)
    tsne_params = {k: v for k, v in settings.tsne.items() if k != "method"}
    for tag, idx in row_orders(len(sub)).items():
        if tag == "sorted":
            continue
        part = sub.iloc[idx]
        yp = part["cluster"].to_numpy()
        Xp = baseline_matrix(part, settings, False, elements=columns)
        pred = cluster_embedding(fit_tsne(Xp, tsne_params, settings.random_state), settings.hdbscan)
        rows.append({"features": "abundances (16-d)", "order": tag, "seed": settings.random_state,
                     "method": "t-SNE", **score(yp, pred)})
    out = pd.DataFrame(rows)
    out.insert(0, "population", f"{args.kinds}: {len(sub)} stars / {n_true} clusters")
    dest = REPO / args.out
    out.to_csv(dest, index=False)
    cols = ["h", "c", "V", "AMI", "ARI", "RF40", "RF70", "stat40", "n_groups", "noise_frac",
            "V_null", "RF40_null", "RF40_null_p95"]
    seeds = out[out.order == "sorted"]
    print("\n=== 7 seeds, sorted row order: mean (sd) ===")
    g = seeds.groupby("method")[cols]
    summ = g.mean().round(3).astype(str) + " (" + g.std().round(3).astype(str) + ")"
    print(summ.T.to_string(), flush=True)
    t = out[out.method == "t-SNE"]
    t = t[(t.order != "sorted") | (t.seed == settings.random_state)]
    print("\n=== t-SNE over 9 row orders (seed 42) ===")
    print(t[["order"] + cols].round(3).to_string(index=False), flush=True)
    print("\nrecovered at 40% (seed 42, sorted):")
    for _, r in seeds[seeds.seed == settings.random_state].iterrows():
        print(f"  {r['method']:15s} {r['recovered40']}")
    print(f"\nhow often each cluster is recovered at 40% over the {len(DEFAULT_SEEDS)} seeds:")
    for method, g in seeds.groupby("method"):
        c = Counter(x for r in g["recovered40"].fillna("") for x in r.split(";") if x)
        print(f"  {method:15s} " + ", ".join(f"{k} {v}/{len(g)}" for k, v in c.most_common()))
    print(f"\nsaved to {dest}")


if __name__ == "__main__":
    main()
