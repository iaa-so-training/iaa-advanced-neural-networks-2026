"""Chapter 5, exercise 3 — exact versus approximate nearest neighbours.

    ANN libraries advertise large speedups over exact kNN. Measure the
    disagreement rate between an approximate graph (UMAP's NN-descent) and an
    exact one on a random 5 000-star subsample of the DR19 matrix. Is the
    approximation good enough that the clustering result is unchanged?

\\S 5.3 warns that "UMAP is fast" is a claim about graph construction, not
about exact kNN, and that KD-trees degrade above roughly 20 dimensions. This
exercise measures both halves on the real matrix: how wrong the approximate
graph is (very slightly), and whether the advertised speedup exists at the
sample sizes this workbook actually uses (it does not).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, member_field

#: Graph size — the pipeline's UMAP/EVoC ``n_neighbors``.
K = 15

#: The exercise's subsample size.
N_SUBSAMPLE = 5_000

#: Sizes swept by :func:`scaling`. 25 000 is the whole FAST population.
SIZES: tuple[int, ...] = (2_000, 5_000, 10_000, 25_000)


def _subsample(n: int = N_SUBSAMPLE, seed: int = SEEDS[0]) -> tuple[np.ndarray, np.ndarray]:
    """``n`` random rows of the member+field matrix, with their labels."""
    data = member_field()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data.X), size=min(n, len(data.X)), replace=False)
    return np.ascontiguousarray(data.X[idx]), data.labels[idx]


def exact_graph(X: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force exact kNN — indices and distances, self excluded."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X)
    distances, indices = nn.kneighbors(X)
    return np.asarray(indices)[:, 1:], np.asarray(distances)[:, 1:]


def approximate_graph(
    X: np.ndarray, k: int = K, seed: int = SEEDS[0],
) -> tuple[np.ndarray, np.ndarray]:
    """NN-descent — the algorithm inside UMAP (pynndescent)."""
    from pynndescent import NNDescent

    index = NNDescent(X, n_neighbors=k + 1, metric="euclidean",
                      random_state=seed, n_jobs=1)
    graph = index.neighbor_graph
    if graph is None:  # pragma: no cover — pynndescent always populates it
        raise RuntimeError("NNDescent returned no neighbour graph")
    return np.asarray(graph[0])[:, 1:], np.asarray(graph[1])[:, 1:]


def disagreement(exact_idx: np.ndarray, approx_idx: np.ndarray) -> dict[str, float]:
    """Set recall, positional disagreement, and the perfect-row fraction."""
    k = exact_idx.shape[1]
    recall = np.array([
        len(set(exact_idx[r].tolist()) & set(approx_idx[r].tolist())) / k
        for r in range(len(exact_idx))
    ])
    return {
        "set_recall": float(recall.mean()),
        "set_disagreement": float(1.0 - recall.mean()),
        "positional_disagreement": float((exact_idx != approx_idx).mean()),
        "rows_with_perfect_graph": float((recall == 1.0).mean()),
    }


def _purity_from_indices(
    neighbour_idx: np.ndarray, labels: np.ndarray, k: int = 10,
    min_members: int = 5,
) -> float:
    """kNN purity read straight off a neighbour index array."""
    scores = []
    for cluster in np.unique(labels):
        if cluster == "field":
            continue
        mask = labels == cluster
        if int(mask.sum()) < min_members:
            continue
        scores.append(float((labels[neighbour_idx[mask][:, :k]] == cluster).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def scaling(sizes: tuple[int, ...] = SIZES, k: int = K) -> pd.DataFrame:
    """Wall-clock for brute force, sklearn's default, and NN-descent.

    The numba JIT is warmed once first, off the clock: the first NNDescent
    call in a process pays ~12 s of compilation, which is a real cost for a
    one-shot script but not a property of the algorithm.
    """
    from pynndescent import NNDescent
    from sklearn.neighbors import NearestNeighbors

    data = member_field()
    # Warm pynndescent's numba JIT so the timings below measure the graph
    # build, not compilation.
    _ = NNDescent(data.X[:500], n_neighbors=k + 1, metric="euclidean",
                  random_state=0, n_jobs=1).neighbor_graph

    rows = []
    for n in sizes:
        X = np.ascontiguousarray(data.X[:n])

        start = time.perf_counter()
        NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X).kneighbors(X)
        t_brute = time.perf_counter() - start

        model = NearestNeighbors(n_neighbors=k + 1).fit(X)
        start = time.perf_counter()
        model.kneighbors(X)
        t_auto = time.perf_counter() - start

        start = time.perf_counter()
        _ = NNDescent(X, n_neighbors=k + 1, metric="euclidean", random_state=42,
                  n_jobs=1).neighbor_graph
        t_approx = time.perf_counter() - start

        rows.append({
            "n": n,
            "exact_brute_s": round(t_brute, 3),
            "sklearn_default_s": round(t_auto, 3),
            "sklearn_algorithm": str(getattr(model, "_fit_method", "unknown")),
            "nn_descent_s": round(t_approx, 3),
            "speedup": round(t_brute / t_approx, 2),
        })
    return pd.DataFrame(rows)


def solve(
    n: int = N_SUBSAMPLE, k: int = K, seeds: tuple[int, ...] = SEEDS[:3],
) -> dict[str, object]:
    """Measure the disagreement, then ask whether it changes the answer."""
    X, labels = _subsample(n)
    exact_idx, exact_dist = exact_graph(X, k)

    per_seed = []
    for seed in seeds:
        approx_idx, approx_dist = approximate_graph(X, k, seed)
        stats = disagreement(exact_idx, approx_idx)
        stats["seed"] = float(seed)
        stats["mean_excess_edge_length"] = float(
            (approx_dist.sum(axis=1) / exact_dist.sum(axis=1) - 1.0).mean(),
        )
        per_seed.append(stats)
    table = pd.DataFrame(per_seed)

    approx_idx, _ = approximate_graph(X, k, seeds[0])
    return {
        "n_subsample": int(len(X)),
        "k": k,
        "per_seed": table.round(6),
        "mean_set_disagreement": round(float(table["set_disagreement"].mean()), 5),
        "mean_positional_disagreement": round(
            float(table["positional_disagreement"].mean()), 5,
        ),
        "mean_rows_perfect": round(float(table["rows_with_perfect_graph"].mean()), 4),
        "purity_exact": round(_purity_from_indices(exact_idx, labels), 4),
        "purity_approx": round(_purity_from_indices(approx_idx, labels), 4),
        "timing": scaling(),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Wall-clock against n for exact and approximate construction."""
    import matplotlib.pyplot as plt

    timing = scaling()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(timing["n"], timing["exact_brute_s"], "o-", color="#4c72b0",
            label="exact (brute force)")
    ax.plot(timing["n"], timing["nn_descent_s"], "s-", color="#dd8452",
            label="NN-descent (approximate)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n stars")
    ax.set_ylabel("wall clock (s)")
    ax.set_title("The ANN speedup does not exist at workbook sample sizes")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "how wrong the approximate graph is": (
        "Barely wrong. The exact neighbour rule is the classical one "
        f"{cite('Cover:67')}; the question is only what an approximation to "
        "it costs. On a random 5 000-star subsample of the DR19 "
        "member+field matrix with k=15, NN-descent — the approximate "
        f"neighbour search UMAP {cite('McInnes:18')} builds its graph with — "
        "recovers 99.3% of the "
        "exact neighbour sets — a set disagreement of 0.7%, stable across "
        "three seeds (0.0066, 0.0073, 0.0069). Per row, 92.6% of stars get an "
        "exactly correct neighbour set; the remaining 7% typically have one "
        "wrong entry out of fifteen. The positional disagreement is larger, "
        "3.1%, because a swap of two nearly-equidistant neighbours counts as "
        "two positional errors and zero set errors — quote which one you mean."
    ),
    "how wrong the geometry is": (
        "Much less wrong than the index disagreement suggests. The total "
        "length of a star's 15 edges is inflated by 2.9e-4 on average — "
        "three parts in ten thousand. The mistakes NN-descent makes are "
        "substitutions between neighbours at almost identical distances, "
        "which is exactly what you would expect in 16 dimensions where, per "
        "chapter 3, the nearest and furthest neighbours differ by only a "
        "factor of ~3. The approximation errs where the exact answer is "
        "least meaningful."
    ),
    "does the clustering result change": (
        "Not measurably. kNN purity at k=10 on the same subsample is 0.1359 "
        "from the exact graph and 0.1356 from the approximate one — a "
        "difference of 0.0003, three orders of magnitude below the "
        "seed-to-seed spread of any embedding in this workbook and far below "
        "the 0.20 swing S 2.3 gets from dropping one cluster. For the "
        "purposes of this project the approximation is free. That conclusion "
        "is specific to k=15 on standardised, row-normalised abundances; an "
        "algorithm that keys on the *identity* of the single nearest "
        "neighbour rather than on the neighbourhood as a whole would be more "
        "exposed."
    ),
    "the speedup does not exist here": (
        "This is the surprise, and it is the point S 5.3 is making. Measured "
        "with the numba JIT already warm: at n=5 000, brute force takes "
        "0.034 s and NN-descent 0.127 s — the approximation is about 3.7x "
        "*slower*. At n=25 000, the full FAST population, it is 0.49 s against "
        "0.78 s, still 1.6x slower. The absolute times move by tens of "
        "percent between runs on a shared machine, but the sign never does: "
        "exact wins at every size this workbook uses (scaling() returned "
        "speedup factors of 0.14, 0.27, 0.33 and 0.63 for n = 2 000 to "
        "25 000 — all below 1.0). Cold, the first NNDescent call in a process "
        "costs an extra ~12 s of JIT compilation, which at these sizes dwarfs "
        "the entire exact computation. Note also that sklearn's 'auto' "
        f"({cite('Pedregosa:11', bare=True)}) "
        "chooses brute force itself at d=16 — the KD-tree/ball-tree "
        "degradation S 5.3 mentions is not hypothetical, the library has "
        "already given up on spatial indices for this dimension."
    ),
    "when the speedup is real": (
        "Exact kNN is O(n^2 d); NN-descent is roughly O(n^1.14 d) with a large "
        "constant. The crossover is above the sizes in this workbook. S 5.3's "
        "own arithmetic gives the regime where it matters: 3.6e5 field stars "
        "at d=16 is ~1e12 distance evaluations, and there the approximation "
        "is the difference between minutes and hours. The lesson is to "
        "measure the crossover for your own n and d rather than adopting the "
        "approximation because a library advertises it — at n=5 000 you pay "
        "0.7% accuracy for a ~4x slowdown, which is a strictly worse deal "
        "than doing it exactly."
    ),
    "the caveat about what was measured": (
        "Single-threaded (n_jobs=1) on one machine, so the absolute times are "
        "not portable; the ratios and the accuracy numbers are the "
        "transferable part. UMAP does not expose the graph it builds "
        "directly, so this uses pynndescent — the same NNDescent "
        "implementation UMAP calls internally, with the same k, which is why "
        "the comparison is fair. The subsample is drawn from member+field, so "
        "it is 96% field stars: purity is correspondingly low (0.136) and "
        "should not be compared with the cluster-only number of exercise 2."
    ),
    "references": reference_list("McInnes:18", "Pedregosa:11", "Cover:67"),
}
