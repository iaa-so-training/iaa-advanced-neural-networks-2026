"""Chapter 12, exercise 1 — why the fused fit is more self-consistent, and why
the split one is easier to diagnose.

    EVoC clusters an embedding built from the graph, and HDBSCAN* clusters
    the embedding built by t-SNE or UMAP. Explain, in three sentences each,
    why the first is more self-consistent and why the second is easier to
    diagnose when it fails.

This is a conceptual question, so the module's job is to make the two claims
*checkable* rather than merely plausible. It runs both pipelines on the same
member matrix and instruments the thing the two sentences turn on: in EVoC
there is one objective and one seed, so the sources of arbitrary choice are
enumerable and countable; in t-SNE+UMAP -> HDBSCAN* there are two independent
stochastic stages, so each stage's contribution to the final answer can be
attributed separately — which is exactly what \\S 9.3's row-order audit did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Seeds used to demonstrate that EVoC's variation is one-dimensional.
SEED_NOTE: tuple[int, ...] = SEEDS[:4]

#: Row orders used to demonstrate that the two-stage pipeline has a second,
#: independent source of variation that EVoC's fit does not have.
N_ORDERS: int = 4


def fused_fit(
    X: np.ndarray, labels: np.ndarray, seeds: tuple[int, ...] = SEED_NOTE,
) -> pd.DataFrame:
    """EVoC: one fit, one seed, one set of decisions.

    Records the number of layers the fit considered and the persistence score
    of the layer it returned — the internals of ``cluster.layers`` that
    \\S 12.3 says the user normally cannot see.
    """
    from evoc import EVoC

    from cluster.baseline import separation_scores
    from cluster.stability import degeneracy

    cfg = settings()
    rows = []
    for seed in seeds:
        model = EVoC(**cfg.evoc, random_state=seed)
        pred = model.fit_predict(X)
        scores = separation_scores(labels, pred)
        collapse = degeneracy(pred)
        rows.append({
            "seed": seed,
            "homogeneity": round(scores["homogeneity"], 4),
            "completeness": round(scores["completeness"], 4),
            "accuracy": round(scores["accuracy"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
            "n_layers_considered": len(model.persistence_scores_),
            "chosen_layer": int(np.argmax(model.persistence_scores_)),
            "best_persistence": round(
                float(np.max(model.persistence_scores_)), 4),
        })
    return pd.DataFrame(rows)


def split_fit(
    X: np.ndarray, labels: np.ndarray, seed: int = SEEDS[0],
    n_orders: int = N_ORDERS,
) -> pd.DataFrame:
    """t-SNE -> HDBSCAN*: two stages, two independent sources of variation.

    Each row is one run; ``stage`` records which knob was turned, so the
    spread attributable to the embedding and the spread attributable to the
    clusterer can be read off separately. The embedding is re-fit for the
    seed rows and held fixed for the size-floor rows, which is the difference
    between the two experiments.
    """
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding, fit_tsne
    from cluster.stability import degeneracy

    cfg = settings()
    params = {k: v for k, v in cfg.tsne.items() if k != "method"}
    rows = []

    def record(stage: str, detail: str, Z: np.ndarray) -> None:
        pred = cluster_embedding(Z, cfg.hdbscan)
        scores = separation_scores(labels, pred)
        collapse = degeneracy(pred)
        rows.append({
            "stage": stage,
            "detail": detail,
            "homogeneity": round(scores["homogeneity"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
        })

    for i in range(n_orders):
        index = (np.arange(len(X)) if i == 0 else
                 np.random.default_rng(1000 + i).permutation(len(X)))
        record("stage 1: row order", f"order_{i}",
               fit_tsne(X[index], params, seed))

    Z = fit_tsne(X, params, seed)
    for size in (5, 15, 50):
        pred = cluster_embedding(Z, {
            "min_cluster_size": size, "min_samples": None,
            "cluster_selection_epsilon": 0.0,
        })
        scores = separation_scores(labels, pred)
        collapse = degeneracy(pred)
        rows.append({
            "stage": "stage 2: size floor",
            "detail": f"min_cluster_size={size}",
            "homogeneity": round(scores["homogeneity"], 4),
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
        })
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """Run both pipelines and tabulate where each one's freedom lives."""
    from exercises.utils import members

    data = members()
    fused = fused_fit(data.X, data.labels)
    split = split_fit(data.X, data.labels)

    fused_spread = fused["homogeneity"].std(ddof=0)
    stage1 = split[split["stage"].str.startswith("stage 1")]
    stage2 = split[split["stage"].str.startswith("stage 2")]

    return {
        "n_stars": int(len(data.X)),
        "fused": fused,
        "fused_spread": round(float(fused_spread), 4),
        "fused_groups_range": (
            int(fused["n_groups"].min()), int(fused["n_groups"].max()),
        ),
        "split": split,
        "variation_budget": pd.DataFrame([
            {
                "pipeline": "EVoC (fused)",
                "knob": "random_state",
                "n_settings": len(fused),
                "homogeneity_min": round(float(fused["homogeneity"].min()), 4),
                "homogeneity_max": round(float(fused["homogeneity"].max()), 4),
                "range": round(float(
                    fused["homogeneity"].max() - fused["homogeneity"].min()), 4),
            },
            {
                "pipeline": "t-SNE -> HDBSCAN*",
                "knob": "row order (stage 1)",
                "n_settings": len(stage1),
                "homogeneity_min": round(float(stage1["homogeneity"].min()), 4),
                "homogeneity_max": round(float(stage1["homogeneity"].max()), 4),
                "range": round(float(
                    stage1["homogeneity"].max() - stage1["homogeneity"].min()),
                    4),
            },
            {
                "pipeline": "t-SNE -> HDBSCAN*",
                "knob": "min_cluster_size (stage 2)",
                "n_settings": len(stage2),
                "homogeneity_min": round(float(stage2["homogeneity"].min()), 4),
                "homogeneity_max": round(float(stage2["homogeneity"].max()), 4),
                "range": round(float(
                    stage2["homogeneity"].max() - stage2["homogeneity"].min()),
                    4),
            },
        ]),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Where each pipeline's variability lives."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    budget = result["variation_budget"]
    assert isinstance(budget, pd.DataFrame)

    labels = [f"{row['pipeline']}\n{row['knob']}" for _, row in budget.iterrows()]
    y = np.arange(len(budget))
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.barh(y, budget["homogeneity_max"] - budget["homogeneity_min"],
            0.45, left=budget["homogeneity_min"], color="#4c72b0")
    for i, (_, row) in enumerate(budget.iterrows()):
        ax.plot([row["homogeneity_min"], row["homogeneity_max"]], [i, i],
                "|", color="crimson", ms=10)
        ax.text(row["homogeneity_max"] + 0.005, i,
                f"{row['range']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("homogeneity across settings")
    ax.set_title("One fit, or two independent places to lose an answer")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "why EVoC is more self-consistent — three sentences": (
        "(1) All three stages of the EVoC fit share one objective: the graph "
        "is built from the "
        "raw representation, the embedding is fitted to that same graph, and "
        "the clustering layers are read off that same embedding, so there is "
        "no point at which a decision made for stage n is invisible to stage "
        "n+1. (2) The number of clusters is an *output* of the fit rather than "
        "a knob — there is no min_cluster_size to set — so the user is not "
        "asked to supply a parameter whose value silently determines the "
        "answer, which §12.3 identifies as the single knob that most changed "
        "results in the DR17 sweeps. (3) The only free input left is "
        "``random_state``, so the method's uncertainty is genuinely "
        "one-dimensional: solve() varies it over four seeds and the whole "
        "spread is the 'fused_spread' and 'fused_groups_range' entries, with "
        "nothing hidden in a second stage."
    ),
    "why the split pipeline is easier to diagnose — three sentences": (
        "(1) Because the failure is *localisable*: when 't-SNE -> HDBSCAN*' "
        "returns 4 groups for 25 clusters you can hold the embedding fixed, "
        "sweep min_cluster_size, and see whether the collapse is the "
        "clusterer's doing or the map's — the 'variation_budget' table does "
        "exactly that and separates a stage-1 range from a stage-2 range. "
        "(2) Each stage has an interpretable output the other does not: the "
        "2-D map can be plotted and inspected for merged continents and "
        "orphaned points, and HDBSCAN*'s noise label "
        f"{cite('Campello:13')} is an explicit "
        "'I don't know' that a persistence-selected layer never returns. "
        "(3) Both stages are ordinary library calls with documented knobs "
        f"({cite('vanderMaaten:08', 'McInnes:18', 'McInnes:17', bare=True)}), "
        "so a reader can reproduce them one at a time; EVoC's persistence "
        "criterion is 'less inspectable: you cannot see which layers were "
        "considered' (§12.3), and even when you can reach "
        "``model.persistence_scores_`` — as this module does — the criterion "
        "still gives you no picture to look at, since §12.2 is explicit that "
        "the embedding is not for looking at."
    ),
    "the trade-off in one line": (
        f"EVoC {cite('EVoC')} has fewer places to lose the answer; the "
        "two-stage pipeline has "
        "more places to *find* it. That is not a distinction about quality but "
        "about where the arbitrary decisions live (§12.2), and it is why "
        "§12.4 reports all three arms: they answer different questions about "
        "the same data, and a method that is hard to diagnose is also hard to "
        "discredit."
    ),
    "what the instrumentation shows": (
        "solve() measures the free parameters rather than asserting them. "
        "EVoC's four seeds give homogeneity 0.4266, 0.4423, 0.4460 and 0.4774 "
        "— a range of 0.051 — with the fit considering 3 or 4 candidate layers "
        "each time and reporting which one persistence picked. The two-stage "
        "pipeline on the same data has *two* independent knobs: the row order "
        "(four orders, homogeneity 0.0732 to 0.5199, range 0.447) and the size "
        "floor (three values, 0.2945 to 0.5199, range 0.225). Put end to end, "
        "one fitted method's whole uncertainty is 0.05 and the two-stage "
        "pipeline's is 0.45 then another 0.23 — and they are *separable*, "
        "which is the diagnostic property. One degree of freedom is not the "
        "same as no variability: EVoC's spread is small here but it is real, "
        "and §12.4's case against the method rests on exactly this number "
        "(+-0.043 on the de-duplicated population) being larger than several "
        "of the gaps the workbook reports."
    ),
    "the caveat the chapter insists on": (
        "Self-consistency is not accuracy. §12.4 lists three results against "
        "EVoC, two of them the workbook's own: the seed spread of ±0.043 "
        "overlaps the t-SNE and UMAP arms, so any ranking among the three at "
        "that effect size is not a ranking; the self-supervision claim turns "
        "out to be a t-SNE/UMAP result rather than an EVoC one; and on the "
        "field task EVoC recovers 0.557 of true members at a precision of "
        "0.0033. Having one clean knob makes a method easy to trust and easy "
        "to report; it does not make it right, and the workbook's own "
        "favourite fails two of these three tests."
    ),
    "references": reference_list(
        "EVoC", "Campello:13", "vanderMaaten:08", "McInnes:18", "McInnes:17",
    ),
}
