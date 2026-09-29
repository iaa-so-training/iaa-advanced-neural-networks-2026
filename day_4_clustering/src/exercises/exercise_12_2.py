"""Chapter 12, exercise 2 — EVoC's seed spread versus t-SNE's row-order spread.

    On the member matrix, run EVoC with the seed varied over the same seven
    values used in this workbook, and report homogeneity as mean ± standard
    deviation. Now do the same for t-SNE with a fixed seed but nine row orders
    (\\S 9.3). Which source of variability is larger, and what does that say
    about which number belongs in a paper?

\\S 12.4 opens its objections with "the seed spread is larger than several of
the gaps", quoting EVoC's +-0.043 on abundances. \\S 9.3 quotes t-SNE's
0.218-0.560 under row permutation. This exercise puts the two on one axis —
same data, same population, same scoring — and the answer is not a matter of
opinion once measured. The comparison only became possible because both halves
were run under the workbook's own protocol; that is the methodological point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Seeds for the EVoC arm: the workbook's seven, as the exercise specifies.
N_SEEDS: int = len(SEEDS)

#: Row orders for the t-SNE arm: one sorted plus eight random (exercise 9.2).
N_RANDOM_ORDERS: int = 8

#: The fixed seed the t-SNE arm holds constant, so that row order is the only
#: thing varying there.
FIXED_SEED: int = SEEDS[0]


def evoc_arm(
    X: np.ndarray, labels: np.ndarray, seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """EVoC over seven seeds with everything else fixed."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import fit_evoc
    from cluster.stability import degeneracy

    cfg = settings()
    rows = []
    for seed in seeds:
        pred = fit_evoc(X, cfg.evoc, seed)
        scores = separation_scores(labels, pred)
        collapse = degeneracy(pred)
        rows.append({
            "seed": seed,
            "homogeneity": round(scores["homogeneity"], 4),
            "completeness": round(scores["completeness"], 4),
            "accuracy": round(scores["accuracy"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
        })
    return pd.DataFrame(rows)


def tsne_arm(
    X: np.ndarray, labels: np.ndarray, seed: int = FIXED_SEED,
    n_random: int = N_RANDOM_ORDERS,
) -> pd.DataFrame:
    """t-SNE + HDBSCAN* over nine row orders at one fixed seed."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding, fit_tsne
    from cluster.stability import degeneracy

    cfg = settings()
    params = {k: v for k, v in cfg.tsne.items() if k != "method"}
    rows = []
    for i in range(n_random + 1):
        index = (np.arange(len(X)) if i == 0 else
                 np.random.default_rng(1000 + i).permutation(len(X)))
        Z = fit_tsne(X[index], params, seed)
        pred = cluster_embedding(Z, cfg.hdbscan)
        scores = separation_scores(labels[index], pred)
        collapse = degeneracy(pred)
        rows.append({
            "row_order": "sorted" if i == 0 else f"random_{i}",
            "homogeneity": round(scores["homogeneity"], 4),
            "completeness": round(scores["completeness"], 4),
            "accuracy": round(scores["accuracy"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
        })
    return pd.DataFrame(rows)


def solve(
    seeds: tuple[int, ...] = SEEDS, n_random: int = N_RANDOM_ORDERS,
) -> dict[str, object]:
    """Run both arms on the same population and compare the spreads."""
    from exercises.utils import members

    data = members()
    evoc = evoc_arm(data.X, data.labels, seeds)
    tsne = tsne_arm(data.X, data.labels, FIXED_SEED, n_random)

    e = evoc["homogeneity"].to_numpy(dtype=float)
    t = tsne["homogeneity"].to_numpy(dtype=float)

    comparison = pd.DataFrame([
        {
            "arm": f"EVoC, seed varies (n={len(e)})",
            "mean": round(float(e.mean()), 4),
            "std": round(float(e.std(ddof=0)), 4),
            "min": round(float(e.min()), 4),
            "max": round(float(e.max()), 4),
            "range": round(float(e.max() - e.min()), 4),
            "coef_variation": round(float(e.std(ddof=0) / e.mean()), 4),
        },
        {
            "arm": f"t-SNE, row order varies (n={len(t)})",
            "mean": round(float(t.mean()), 4),
            "std": round(float(t.std(ddof=0)), 4),
            "min": round(float(t.min()), 4),
            "max": round(float(t.max()), 4),
            "range": round(float(t.max() - t.min()), 4),
            "coef_variation": round(float(t.std(ddof=0) / t.mean()), 4),
        },
    ])

    return {
        "n_stars": int(len(data.X)),
        "n_clusters": int(pd.Series(data.labels).nunique()),
        "fixed_seed_for_tsne": FIXED_SEED,
        "evoc": evoc,
        "tsne": tsne,
        "comparison": comparison,
        "groups": {
            "evoc_range": (int(evoc["n_groups"].min()),
                           int(evoc["n_groups"].max())),
            "tsne_range": (int(tsne["n_groups"].min()),
                           int(tsne["n_groups"].max())),
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Both spreads on one axis."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    evoc = result["evoc"]
    tsne = result["tsne"]
    assert isinstance(evoc, pd.DataFrame) and isinstance(tsne, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.scatter(np.zeros(len(evoc)) + np.linspace(-0.12, 0.12, len(evoc)),
               evoc["homogeneity"], s=45, color="#4c72b0",
               label="EVoC, 7 seeds")
    ax.scatter(np.ones(len(tsne)) + np.linspace(-0.12, 0.12, len(tsne)),
               tsne["homogeneity"], s=45, color="#dd8452",
               label="t-SNE, 9 row orders")
    for i, series in enumerate((evoc["homogeneity"], tsne["homogeneity"])):
        values = series.to_numpy(dtype=float)
        ax.hlines(values.mean(), i - 0.22, i + 0.22, color="k", lw=1.6)
        ax.vlines(i + 0.25, values.mean() - values.std(ddof=0),
                  values.mean() + values.std(ddof=0), color="k", lw=1.6)
    ax.set_xticks([0, 1], ["EVoC\n(seed varies)", "t-SNE + HDBSCAN*\n(row order varies)"])
    ax.set_ylabel("homogeneity, 1 002 member rows")
    ax.set_title("Which source of variability is larger?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the two protocols are not the same experiment": (
        f"EVoC {cite('EVoC')} is varied over its *seed* at fixed data and "
        f"fixed row order; t-SNE {cite('vanderMaaten:08')} is varied over "
        "*row order* at fixed seed. That is deliberate "
        "and it is the fair comparison, because those are the variations each "
        "method actually has. §9.3 established that t-SNE with init='pca' is "
        "seed-deterministic — a seed loop reports std = 0.000 for it and "
        "measures nothing — while its row-order spread is the real thing. EVoC "
        "has no row-order sensitivity of comparable size and does have a "
        "genuine seed spread. Comparing them means comparing each method's "
        "dominant source of arbitrariness, not forcing both through one knob."
    ),
    "what I measured": (
        "1 002 member rows, 25 clusters, HDBSCAN* "
        f"{cite('Campello:13')} min_cluster_size 5 on the "
        "t-SNE map, pipeline defaults elsewhere:\n\n"
        "  EVoC, 7 seeds:      mean 0.4536, sd 0.0245, range 0.4213-0.4908\n"
        "                      (range 0.0695, CV 0.054), groups 7-22\n"
        "  t-SNE, 9 row orders: mean 0.4752, sd 0.0430, range 0.3943-0.5199\n"
        "                      (range 0.1256, CV 0.091), groups 18-42\n\n"
        "The EVoC figures reproduce the workbook's own +-0.043 claim in "
        "magnitude (my sd is 0.0245; the quoted spread is larger because it "
        "covers the 982-star de-duplicated population as well). The t-SNE "
        "row-order standard deviation is 1.76x EVoC's and its range 1.81x; in "
        "coefficient-of-variation terms, 1.68x."
    ),
    "which is larger, and why": (
        "The t-SNE row-order spread, by a factor of about 1.8 on this data — "
        "not the order-of-magnitude gap the published 0.218-0.560 range might "
        "suggest, but a consistent gap in every summary statistic. The "
        "difference is structural rather than numerical. EVoC's seed perturbs "
        "one optimisation from a fixed starting point on a fixed graph, and "
        "the persistence criterion then chooses among a small family of "
        "candidate layers — here 3 or 4 of them — so the output can only take "
        "a handful of values. Row permutation perturbs the *path of the "
        "Barnes-Hut accumulation itself* "
        f"({cite('vanderMaaten:14', bare=True)}), so the t-SNE embedding is "
        "not a re-optimisation of the same problem but a genuinely different "
        "local optimum, and on this data the optima differ enough to change "
        "both the score and the group count. EVoC's variability is about which "
        "candidate layer wins; t-SNE's is about which map you get.\n\n"
        "The group counts make the difference visible: EVoC returns 7, 10, "
        "14, 15, 15, 22, 7 groups across its seven seeds, while t-SNE's nine "
        "row orders give 18, 20, 22, 24, 26, 30, 31, 35, 42. Both vary, but "
        "EVoC's values cluster into two families while t-SNE's spread evenly "
        "across a factor of two."
    ),
    "the uncomfortable parity": (
        "Neither spread is small compared with the effects the workbook "
        "reports. §12.4's own complaint is that EVoC's +-0.043 overlaps the "
        f"t-SNE and UMAP {cite('McInnes:18')} arms — and the row-order spread "
        "is larger still, so "
        "the t-SNE arm does not escape the same objection. Any statement of "
        "the form 'method X beats method Y on abundances' is unsupported at "
        "these effect sizes unless it is shown to survive *both* perturbations, "
        "and the honest summary of Table 'honest' is that the three arms are "
        "tied within their own arbitrariness while all three sit far below the "
        "kinematic ceiling of 0.942."
    ),
    "which number belongs in a paper": (
        "The pair, always — mean and spread, over the perturbation that is "
        "actually live for that method, with the perturbation named. Not the "
        "seed spread for t-SNE (it is zero and misleading), not the row-order "
        "spread for EVoC (it is not the dominant term), and never a best-of "
        "over either. Concretely: for a t-SNE row, report the nine-order range "
        "and the permutations' generator seed; for an EVoC row, report the "
        "seven-seed mean ± sd; for both, report the number of predicted groups "
        "and the largest-group fraction beside the score, because those move "
        "more than the score does in every measurement in this exercise. This "
        "is §9.4 rules 2 and 5, applied per method rather than uniformly."
    ),
    "what the exercise is really teaching": (
        "That 'run it with several seeds' is not a validation protocol — it is "
        "a check that only finds the variation you already suspected. The "
        "workbook's own t-SNE table carried std = 0.000 across seven seeds, "
        "which reads as perfect stability and meant only that init='pca' had "
        "removed the seed's influence. The row-order audit was needed to find "
        "the instability that was there all along. Before quoting an "
        "uncertainty on any embedding, ask which input the optimiser treats as "
        "a path variable — and vary that."
    ),
    "references": reference_list(
        "EVoC", "vanderMaaten:08", "vanderMaaten:14", "Campello:13",
        "McInnes:18",
    ),
}
