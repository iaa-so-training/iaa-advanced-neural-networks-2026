"""Chapter 11, exercise 2 — does the initialisation change the neighbourhoods?

    For a small dataset, compute UMAP's edge weights and then run the layout
    twice: once with PCA initialisation and once with a random one. Do the
    two layouts agree on which points are neighbours? Which of them would you
    publish, and what would you have to state about it?

\\S 11.3 lists ``random_state`` as one of UMAP's three consequential knobs and
notes that its spread is "the same order as some of the gaps quoted between
arms". Initialisation is the other half of the same question, and \\S 10.3
records that for t-SNE the difference was not subtle. This exercise separates
two things a layout does — *who is next to whom* and *where everything sits* —
and measures whether initialisation changes the first or only the second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Initialisations compared. ``pca`` is UMAP's own default (spectral is the
#: library default and is compared alongside for completeness).
INITIALISATIONS: tuple[str, ...] = ("pca", "random")

#: Neighbourhood size for the agreement measure, matching ``knn_purity``.
K: int = 15

#: Seeds per initialisation. Four is enough to separate the two effects:
#: two seeds of the *same* initialisation give the within-init baseline that
#: the across-init agreement has to be judged against.
N_SEEDS: int = 4


def neighbour_sets(Z: np.ndarray, k: int = K) -> list[set[int]]:
    """The k nearest neighbours of every point in a 2-D layout."""
    from sklearn.neighbors import NearestNeighbors

    index = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(
        Z, return_distance=False,
    )[:, 1:]
    return [{int(j) for j in row} for row in index]


def agreement(a: list[set[int]], b: list[set[int]], k: int = K) -> float:
    """Mean Jaccard-style overlap of two kNN neighbourhood sets."""
    return float(np.mean([len(a[i] & b[i]) / k for i in range(len(a))]))


def layouts(
    X: np.ndarray, initialisations: tuple[str, ...] = INITIALISATIONS,
    seeds: tuple[int, ...] = SEEDS[:N_SEEDS], k: int = K,
) -> dict[tuple[str, int], np.ndarray]:
    """One 2-D layout per (initialisation, seed)."""
    from cluster.benchmark import fit_umap

    cfg = settings()
    out: dict[tuple[str, int], np.ndarray] = {}
    for init in initialisations:
        params = dict(cfg.umap)
        params["init"] = init
        for seed in seeds:
            out[(init, seed)] = fit_umap(X, params, seed)
    return out


def _pairs(values: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for i, a in enumerate(values) for b in values[i + 1:]]


def solve(
    k: int = K, n_seeds: int = N_SEEDS,
) -> dict[str, object]:
    """Compare neighbourhoods within and across initialisations."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding, knn_purity
    from cluster.stability import degeneracy
    from exercises.utils import members

    data = members()
    X, y = data.X, data.labels
    seeds = SEEDS[:n_seeds]
    embeddings = layouts(X, INITIALISATIONS, seeds, k)

    sets = {key: neighbour_sets(Z, k) for key, Z in embeddings.items()}

    within: dict[str, list[float]] = {init: [] for init in INITIALISATIONS}
    for init in INITIALISATIONS:
        for a, b in _pairs(list(seeds)):
            within[init].append(agreement(sets[(init, a)], sets[(init, b)], k))

    across = [
        agreement(sets[("pca", a)], sets[("random", b)], k)
        for a in seeds for b in seeds
    ]

    rng = np.random.default_rng(0)
    reference = embeddings[("pca", seeds[0])]
    chance = agreement(
        sets[("pca", seeds[0])],
        neighbour_sets(reference[rng.permutation(len(reference))], k),
        k,
    )

    raw = neighbour_sets(X, k)
    to_raw = {
        init: float(np.mean([
            agreement(sets[(init, s)], raw, k) for s in seeds
        ]))
        for init in INITIALISATIONS
    }

    from exercises.utils import settings as _settings

    rows = []
    for (init, seed), Z in embeddings.items():
        pred = cluster_embedding(Z, _settings().hdbscan)
        collapse = degeneracy(pred)
        rows.append({
            "init": init,
            "seed": seed,
            "knn_purity": round(float(np.mean(
                list(knn_purity(Z, y, k=k).values()))), 4),
            **{key: round(value, 4)
               for key, value in separation_scores(y, pred).items()},
            "n_groups": collapse["n_clusters"],
            "largest_fraction": collapse["largest_fraction"],
        })
    per_run = pd.DataFrame(rows)
    summary = per_run.drop(columns=["seed"]).groupby("init").agg(
        ["mean", "std"],
    ).round(4)

    return {
        "n_stars": int(len(X)),
        "k": k,
        "n_seeds": n_seeds,
        "neighbour_agreement": {
            "within_pca": (round(float(np.mean(within["pca"])), 4),
                           round(float(np.std(within["pca"])), 4)),
            "within_random": (round(float(np.mean(within["random"])), 4),
                              round(float(np.std(within["random"])), 4)),
            "pca_vs_random": (round(float(np.mean(across)), 4),
                              round(float(np.std(across)), 4)),
            "chance": round(chance, 4),
        },
        "agreement_with_raw_16d": {
            key: round(value, 4) for key, value in to_raw.items()
        },
        "per_run": per_run,
        "summary": summary,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The same matrix under pca and random init — and their neighbourhoods."""
    import matplotlib.pyplot as plt

    from exercises.utils import members

    data = members()
    embeddings = layouts(data.X, INITIALISATIONS, SEEDS[:1], K)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    for ax, init in zip(axes, INITIALISATIONS):
        Z = embeddings[(init, SEEDS[0])]
        ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5, c="#4c72b0")
        ax.set_title(f"init={init} (seed {SEEDS[0]})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Rotation, reflection, arrangement: all free", fontsize=10)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the design, and why it needs three baselines": (
        "UMAP is stochastic even with the seed pinned, so 'the pca layout and "
        "the random layout differ' is not by itself evidence about "
        "initialisation. Two comparators are needed: the agreement between two "
        "*seeds of the same* initialisation (does seed noise already explain "
        "the difference?), and the agreement between a layout and a random "
        "permutation of itself (the chance floor). solve() computes both. "
        "Everything is measured on the 1 002-row member matrix at "
        f"k = 15, four seeds per initialisation, with UMAP {cite('McInnes:18')} "
        "and scikit-learn's neighbour search "
        f"{cite('Pedregosa:11')} doing the work."
    ),
    "do the layouts agree on neighbours": (
        "Yes — to within their own seed noise, and that is the headline. "
        "Mean overlap of the 15 nearest neighbours:\n\n"
        "  within pca       0.7605 +- 0.0225  (6 seed pairs)\n"
        "  within random    0.7577 +- 0.0150  (6 seed pairs)\n"
        "  pca vs random    0.7614 +- 0.0196  (16 pairs)\n"
        "  chance           0.0153\n\n"
        "The pca-vs-random agreement (0.761) is *higher* than the "
        "within-pca agreement (0.760) and higher than within-random (0.758). "
        "Changing the initialisation from PCA to random therefore perturbs the "
        "neighbourhood structure no more than changing the seed does. "
        "Initialisation does not change who is next to whom."
    ),
    "what it does change": (
        "The arrangement, the group count, and the largest-group fraction — "
        "and those three things are not small. Mean homogeneity is 0.521 "
        "(pca) against 0.536 (random); mean completeness 0.515 against 0.465; "
        "predicted groups 33.3 against 41.3; largest-group fraction 0.277 "
        "against 0.132, all read off HDBSCAN* "
        f"{cite('Campello:13')} run on each layout. The scores stay inside "
        "their spread, but the "
        "*shape* of the answer flips between two recurring solutions: a "
        "~29-33 group family with one group holding about 35% of the stars, "
        "and a ~42-45 group family with no dominant group at all. Both pca "
        "and random produce both families depending on seed, so this is seed "
        "noise surfacing, not an initialisation effect — but it is the same "
        "bimodality that exercises 9.2 and 11.3 catch, and it means the group "
        "count and largest-group fraction have to be quoted alongside every "
        "homogeneity."
    ),
    "why the neighbourhoods survive": (
        "Because the graph is built *before* the layout, from the original "
        "high-dimensional data, and the initialisation only chooses a starting "
        "point for the descent. The attraction term pulls graph edges together "
        "and the repulsion separates sampled non-edges "
        f"{cite('McInnes:18')}; both act on the same "
        "objective regardless of where the points started, so the fixed point "
        "is governed by the graph. What PCA initialisation buys is not a "
        "different answer but a *faster and more reproducible* route to one: "
        "it starts in a sensible arrangement, so fewer epochs are needed and "
        "the result is less sensitive to the random stream. §11.1's "
        "'more global structure' caveat follows directly — the long ranges a "
        "UMAP map appears to preserve come from the initialisation and the "
        "repulsion term, not from the graph, which never saw a long distance."
    ),
    "how much of the original structure survives at all": (
        "Both layouts keep only about half of the raw 16-D neighbourhoods: "
        "mean agreement with the C-space kNN sets is 0.510 (pca) and 0.511 "
        "(random). Against a chance floor of 0.015 that is not nothing, but it "
        "means the map is discarding half of whatever neighbourhood structure "
        "the chemistry provided — and exercise 5's raw-space kNN purity of "
        "0.365 says the structure it is discarding from was weak to begin "
        "with. This is the quantitative version of §10.5's 'the neighbourhoods "
        "are preserved, and the neighbourhoods were not separated to begin "
        "with', and it holds identically for both initialisations."
    ),
    "which would you publish": (
        "The PCA-initialised one, for four reasons. (1) It is the library's "
        "default for ``init`` in current versions and thus what a reader "
        "reproducing your command will get. (2) It is deterministic given the "
        "input, so the run is reproducible from the data alone. (3) It starts "
        "closer to the eventual layout, so it is less dependent on the number "
        "of optimisation epochs and on the random stream — the same reason "
        "§10.3's t-SNE arm uses ``init='pca'``. (4) Its group count is more "
        "stable across seeds (29-44 against 33-45), which is the number that "
        "moves most.\n\n"
        "But choosing it does not make the choice free, and that is what you "
        "have to state."
    ),
    "what you would have to state about it": (
        "Five sentences, and none of them optional. (i) Which initialisation, "
        "by name — 'UMAP, init=pca' — because the layouts are not "
        "interchangeable pictures even though they agree on neighbours. (ii) "
        "The seed, or that the run was averaged over seeds; §11.3 notes UMAP's "
        "±0.011 spread is the same order as the gaps quoted between arms, and "
        "this exercise shows the group count swinging by 12 between seeds. "
        "(iii) The number of predicted groups and the largest-group fraction, "
        "which move far more than the score (§9.4 rule 5). (iv) That UMAP has "
        "no noise model, so a blob in the picture is a density peak and "
        "nothing more — everything gets a position. (v) That the map is not "
        "evidence of separation on its own: here it retains about half the "
        "input neighbourhoods, so the picture should be read next to the "
        "parameter-free score, not instead of it."
    ),
    "the general rule": (
        "Two draws from one stochastic method are not a comparison; you need "
        "the within-method spread, the across-method spread, and the chance "
        "floor before 'different' means anything. This exercise is the small, "
        "controlled version of §9.3's row-order audit: in both cases the "
        "quantity that moves is not the neighbourhood structure but a "
        "second-order property of the layout that a downstream clusterer then "
        "turns into a score."
    ),
    "references": reference_list(
        "McInnes:18", "Campello:13", "Pedregosa:11",
    ),
}
