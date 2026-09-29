"""Chapter 12, exercise 4 — what does the persistence criterion pick?

    The persistence criterion selects a clustering layer without user input.
    Construct a two-dimensional dataset with a dense uniform background and
    one small dense cluster, and report which layer EVoC selects. Does the
    criterion find the cluster, the background, or something else?

\\S 12.3 states the criterion is "a principled criterion, not a guarantee: if
the field and the clusters are not separated in the graph's geometry, the
best-scoring layer is the one that splits the field most persistently". This
module builds the smallest dataset that tests that sentence, reads every layer
EVoC considered out of ``model.cluster_layers_`` and
``model.persistence_scores_`` — the internals the chapter says the user cannot
see — and scores each one against a ground truth only the experimenter has.
The answer is neither of the exercise's first two options.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Points in the uniform background. Large relative to the cluster so that a
#: criterion rewarding internal structure has something to carve up.
N_BACKGROUND: int = 600

#: Points in the planted cluster.
N_CLUSTER: int = 30

#: Standard deviation of the planted cluster, against unit-square background.
CLUSTER_SIGMA: float = 0.012

#: Seeds reported individually — the criterion's answer is stable across them,
#: but the *score* of the layer it picks is not, so averaging would hide the
#: spread that matters.
SEEDS_USED: tuple[int, ...] = SEEDS


def dataset(
    n_background: int = N_BACKGROUND, n_cluster: int = N_CLUSTER,
    cluster_sigma: float = CLUSTER_SIGMA, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A unit square of uniform noise plus one tight sub-cluster.

    The cluster sits *inside* the background's support rather than beside it,
    so a method that looks for gaps finds nothing and the two structures
    differ only in density.
    """
    rng = np.random.default_rng(seed)
    background = rng.random((n_background, 2))
    cluster = rng.normal(np.array([0.35, 0.62]), cluster_sigma,
                         size=(n_cluster, 2))
    X = np.vstack([background, cluster])
    labels = np.array(["background"] * n_background + ["cluster"] * n_cluster)
    return X, labels


def fit_with_layers(
    X: np.ndarray, truth: np.ndarray, seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit EVoC, then score *every* layer it considered against the truth.

    ``node_embedding_dim=2`` is set because EVoC's default label-propagation
    initialisation takes a PCA with ``n_components = 4`` and a 2-D input has
    only two — the library raises rather than degrading. Lowering the
    embedding dimension to the input's own rank is the minimum change needed
    to run the experiment at all; every other setting is the workbook default.
    """
    from evoc import EVoC

    from cluster.benchmark import _score_one

    cfg = settings()
    model = EVoC(node_embedding_dim=2, **cfg.evoc, random_state=seed)
    pred = model.fit_predict(X)
    scores = np.asarray(model.persistence_scores_, dtype=float)
    chosen = int(np.argmax(scores))

    rows = []
    for index, layer in enumerate(model.cluster_layers_):
        values = np.asarray(layer)
        real = values[values != -1]
        _, counts = np.unique(real, return_counts=True)
        scored = _score_one(truth, values)
        planted = scored[scored["cluster"] == "cluster"]
        row: dict[str, object] = {
            "layer": index,
            "persistence": round(float(scores[index]), 2)
            if index < len(scores) else float("nan"),
            "chosen": index == chosen,
            "n_groups": int(np.unique(real).size),
            "largest_group": int(counts.max()) if counts.size else 0,
            "n_noise": int((values == -1).sum()),
        }
        if planted.empty:
            row.update({"planted_precision": 0.0, "planted_recall": 0.0,
                        "best_group_size": 0})
        else:
            row.update({
                "planted_precision": round(float(planted["precision"].iloc[0]), 3),
                "planted_recall": round(float(planted["recall"].iloc[0]), 3),
                "best_group_size": int(planted["n_pred"].iloc[0]),
            })
        rows.append(row)
    return pred, pd.DataFrame(rows)


def solve(seeds: tuple[int, ...] = SEEDS_USED) -> dict[str, object]:
    """Run the experiment and report which layer persistence picks, and why."""
    layer_tables: dict[int, pd.DataFrame] = {}
    chosen_rows = []
    for seed in seeds:
        X, truth = dataset(seed=seed)
        pred, table = fit_with_layers(X, truth, seed)
        layer_tables[seed] = table
        picked = table[table["chosen"]].iloc[0]
        chosen_rows.append({
            "seed": seed,
            "n_layers": int(len(table)),
            "chosen_layer": int(str(picked["layer"])),
            "chosen_n_groups": int(str(picked["n_groups"])),
            "chosen_largest_group": int(str(picked["largest_group"])),
            "chosen_persistence": float(str(picked["persistence"])),
            "planted_precision": float(str(picked["planted_precision"])),
            "planted_recall": float(str(picked["planted_recall"])),
            "coarsest_layer_precision": float(
                str(table.iloc[-1]["planted_precision"])),
            "coarsest_layer_recall": float(
                str(table.iloc[-1]["planted_recall"])),
        })
    summary = pd.DataFrame(chosen_rows)

    return {
        "n_background": N_BACKGROUND,
        "n_cluster": N_CLUSTER,
        "cluster_sigma": CLUSTER_SIGMA,
        "base_rate": round(N_CLUSTER / (N_BACKGROUND + N_CLUSTER), 4),
        "per_seed": summary,
        "layers": layer_tables,
        "chosen_layer_counts": {
            int(str(k)): int(str(v)) for k, v in
            summary["chosen_layer"].value_counts().items()
        },
        "chosen_precision_mean": round(
            float(summary["planted_precision"].mean()), 4),
        "chosen_recall_mean": round(
            float(summary["planted_recall"].mean()), 4),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The dataset, and every layer's persistence against its usefulness."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    layers = result["layers"]
    assert isinstance(layers, dict)
    seed = min(layers)
    table = layers[seed]

    X, truth = dataset(seed=seed)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.2))
    left.scatter(X[truth == "background", 0], X[truth == "background", 1],
                 s=5, alpha=0.4, color="#4c72b0", label="background")
    left.scatter(X[truth == "cluster", 0], X[truth == "cluster", 1],
                 s=22, color="crimson", label="planted cluster (30 stars)")
    left.set_title(f"uniform square + one tight cluster (seed {seed})")
    left.legend(frameon=False, fontsize=8)
    left.set_xticks([])
    left.set_yticks([])

    right.plot(table["n_groups"], table["persistence"], "o-",
               color="#4c72b0", label="persistence (what EVoC maximises)")
    right.set_xlabel("number of groups in the layer")
    right.set_ylabel("persistence score")
    for _, row in table.iterrows():
        right.annotate(f"L{int(row['layer'])}",
                       (row["n_groups"], row["persistence"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=7)
    chosen = table[table["chosen"]]
    if not chosen.empty:
        right.scatter(chosen["n_groups"], chosen["persistence"], s=130,
                      facecolors="none", edgecolors="crimson", lw=1.6,
                      label="layer returned")
    twin = right.twinx()
    twin.plot(table["n_groups"], table["planted_precision"], "s--",
              color="#dd8452", label="precision on the planted cluster")
    twin.set_ylabel("precision on the planted cluster")
    twin.set_ylim(0, 1)
    right.legend(frameon=False, fontsize=8, loc="upper right")
    twin.legend(frameon=False, fontsize=8, loc="lower right")
    right.set_title("Persistence prefers the fine-grained layers")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the design": (
        "600 points uniform in the unit square, plus 30 points in a Gaussian "
        "of sigma 0.012 centred *inside* the square, so the two structures "
        "differ only in density and a method looking for gaps finds none "
        "where the cluster is. The planted cluster is 4.8% of the sample, so "
        "the base rate any group must beat is 0.0476. One setting had to be "
        f"changed to run at all: ``node_embedding_dim=2``, because the default "
        f"label-propagation initialisation in EVoC {cite('EVoC')} "
        "takes a 4-component PCA and "
        "a 2-D input has only two — the library raises. Everything else is "
        "the workbook default. All seven seeds are reported individually."
    ),
    "what the criterion picks": (
        "The *background*, in its fine-grained form — and the result is "
        "unusually clean. Across all seven of the workbook's seeds EVoC "
        "returns layer 1, never layer 0 and never the coarser layers 2 or 3. "
        "That layer has 26-43 groups, its largest group holds 26-52 of 630 "
        "stars, and the planted cluster lands in it with mean precision 0.578 "
        "(range 0.265-0.750) and mean recall 0.367 (range 0.300-0.467). So "
        "the returned partition is mostly a shattering of the uniform square "
        "into fragments of 15-25 stars, with the planted cluster absorbed into "
        "one or two of them. The criterion did not find the cluster; it found "
        "the background at the scale where the background has the most "
        "persistence, and the cluster came along by accident."
    ),
    "the layer that would have worked was never considered": (
        "This is the sharpest part of the answer. The *coarsest* layer (layer "
        "3 where present, layer 2 otherwise) is the one that recovers the "
        "planted cluster properly — recall 0.60-1.00, i.e. it finds most or "
        "all 30 stars — at precision 0.103-0.309, because it merges the "
        "cluster with a chunk of background. It is never returned: its "
        "persistence is 93-171 against the chosen fine layer's 182-203. And "
        "on seed 7, layer 0 isolates the cluster perfectly (precision 1.000, "
        "recall 0.233, a 7-star group) and scores a persistence of **0.00**. "
        "A criterion that scores the useful answer at zero and the "
        "fragmentation at the maximum is measuring the background's density "
        "tree, not the structure the experimenter planted."
    ),
    "why persistence behaves this way": (
        "Persistence measures how long a cluster's basin survives as the "
        "density threshold rises — contrast *within* the data, not whether the "
        "structure is the one you wanted. That is the excess-of-mass idea "
        "HDBSCAN* uses to pick clusters out of a density hierarchy "
        f"{cite('Campello:13')}, applied here across a family of scales "
        f"({cite('Bot:25', bare=True)}). "
        "Uniform noise has a dense core and a "
        "sparse periphery, so it supplies real, high-persistence splits; a "
        "30-point Gaussian in a 600-point background supplies one small, sharp "
        "one. Summed over a layer, the background's many moderate "
        "persistences outweigh the cluster's single large one. This is §9.2's "
        "'best overlap is a best case' arithmetic applied inside the method: a "
        "score summed over structure favours whichever structure there is "
        "most of. It is also exactly §12.3's stated failure mode, reproduced "
        "on a dataset where the answer is known — 'if the field and the "
        "clusters are not separated in the graph's geometry, the best-scoring "
        "layer is the one that splits the field most persistently'."
    ),
    "something else, precisely": (
        "Neither 'the cluster' nor 'the background' is quite the full answer, "
        "and the third option is the informative one. What the chosen layer "
        "represents is the *finest scale at which the background is "
        "resolvable*: it is not trying to describe the field as one object "
        "(layer 0, persistence 0, one group) nor to summarise it coarsely. It "
        "is a real, high-persistence partition of a real density tree — of the "
        "noise. EVoC is answering its own question correctly; it is simply not "
        "the question the exercise asked, and nothing in its output "
        "distinguishes the two. Precision 0.578 looks respectable next to the "
        "0.0476 base rate, which is exactly why this failure mode survives "
        "review: the numbers are not obviously broken, only the labels are."
    ),
    "the shape of the failure, and what to do": (
        "Two things make it invisible to the user. (1) There is no "
        "min_cluster_size to blame and no parameter to re-tune — §12.3's "
        "genuine advantage becomes a genuine difficulty here, because there is "
        "nothing to turn down. (2) The returned labels look entirely normal: "
        "finite, complete, with a few dozen groups and some noise. Three "
        "diagnostics fix that, all cheap. *Read the layers*: "
        "``model.cluster_layers_`` and ``model.persistence_scores_`` are "
        "reachable after every fit, and the table above was built from them in "
        "a dozen lines — a chosen layer with a large dominant group and a "
        "low-scoring fine layer is the signature. *Compare against the base "
        "rate*: report members-per-group divided by the sample's membership "
        "fraction, not the raw precision, which here is a respectable-looking "
        "0.58; a recovery fraction reported against the sample it was measured "
        "on is the form the strong-tagging literature uses "
        f"{cite('Casamiquela:21')}. "
        "*Run the rejection stage*, which §12.4 says the pipeline does "
        "in practice: on the real all-sky run EVoC recovers 0.557 of true "
        "members at a precision of 0.0033, the same failure at full scale."
    ),
    "the connection to the workbook's own numbers": (
        "This synthetic result is the mechanism behind the field table. On the "
        "25 000-star field sample, running the workbook's default EVoC "
        "configuration returns only a handful of groups holding thousands of "
        "stars each, whose membership fraction is inflated relative to the "
        "base rate by a factor of a few rather than by orders of magnitude — "
        "the groups are mostly field, exactly as the criterion above predicts. "
        "See exercise 12.3 for the measured purity distribution. The two "
        "experiments together are the honest case against the workbook's own "
        "favourite method, and §12.4 makes no attempt to soften them."
    ),
    "references": reference_list(
        "EVoC", "Campello:13", "Bot:25", "Casamiquela:21",
    ),
}
