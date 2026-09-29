"""Chapter 7, exercise 1 — is mutual reachability a metric?

    For the four points a, b, c, d of a square grid, compute d_mreach with
    k=2 and verify that it is a valid metric. Which of the metric axioms does
    it break if you omit the kappa terms and use max{d(a,b), const}?

\\S 7.1 introduces mutual reachability (Equation 4) as "a floor" that makes
dense regions self-similar across scales. The exercise asks whether that
floor costs anything mathematically. The honest answer is more interesting
than the expected one: with the usual d(x,x) = 0 convention *both*
constructions satisfy all four axioms, including the triangle inequality, on
the square and on 300 random configurations. The axiom that breaks is the
identity of indiscernibles, and it breaks only if you take the formula
literally at x = y — which is why every implementation special-cases the
diagonal. What separates the two constructions is not an axiom; it is what
they do to the single-linkage hierarchy, which this module measures too.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: The unit square of the exercise.
SQUARE: np.ndarray = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
SQUARE_NAMES: tuple[str, ...] = ("a", "b", "c", "d")

#: A deliberately non-uniform configuration: a tight triple plus two isolated
#: points, so the core distances actually differ. On the square every kappa_2
#: is 1, which hides everything interesting about Equation 4.
UNEVEN: np.ndarray = np.array([
    (0.0, 0.0), (0.1, 0.0), (0.05, 0.05), (3.0, 0.0), (3.0, 2.0),
])
UNEVEN_NAMES: tuple[str, ...] = ("p1", "p2", "p3", "far", "further")


def core_distance(X: np.ndarray, k: int = 2) -> np.ndarray:
    """kappa_k: distance to the k-th nearest *other* point."""
    distance = np.linalg.norm(X[:, None] - X[None], axis=-1)
    return np.array([
        float(np.sort(np.delete(distance[i], i))[k - 1]) for i in range(len(X))
    ])


def mutual_reachability(
    X: np.ndarray, k: int = 2, zero_diagonal: bool = True,
) -> np.ndarray:
    """Equation 4: max{kappa_k(a), kappa_k(b), d(a,b)}.

    ``zero_diagonal`` applies the convention every implementation uses — the
    formula is only defined for distinct points, and d(x, x) is set to 0 by
    hand. Set it False to see what the literal formula does.
    """
    distance = np.linalg.norm(X[:, None] - X[None], axis=-1)
    kappa = core_distance(X, k)
    matrix = np.maximum(np.maximum(kappa[:, None], kappa[None, :]), distance)
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def floored_distance(
    X: np.ndarray, const: float | None = None, zero_diagonal: bool = True,
) -> np.ndarray:
    """The kappa-free variant: max{d(a,b), const}, one global floor."""
    distance = np.linalg.norm(X[:, None] - X[None], axis=-1)
    if const is None:
        off = ~np.eye(len(X), dtype=bool)
        const = float(np.median(distance[off]))
    matrix = np.maximum(distance, const)
    if zero_diagonal:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def check_axioms(matrix: np.ndarray, tol: float = 1e-12) -> dict[str, object]:
    """Test the four metric axioms exhaustively on a distance matrix."""
    n = len(matrix)
    violations = [
        (i, j, m) for i, j, m in itertools.product(range(n), repeat=3)
        if matrix[i, m] > matrix[i, j] + matrix[j, m] + tol
    ]
    off_diagonal_zeros = [
        (i, j) for i in range(n) for j in range(n)
        if i != j and abs(matrix[i, j]) <= tol
    ]
    return {
        "non_negativity": bool((matrix >= -tol).all()),
        "symmetry": bool(np.allclose(matrix, matrix.T, atol=tol)),
        "d_xx_is_zero": bool(np.allclose(np.diag(matrix), 0.0, atol=tol)),
        "distinct_points_positive": not off_diagonal_zeros,
        "triangle_inequality": not violations,
        "n_triangle_violations": len(violations),
    }


def square_table(k: int = 2) -> pd.DataFrame:
    """The full 4x4 mutual-reachability matrix of the square, labelled."""
    matrix = mutual_reachability(SQUARE, k)
    return pd.DataFrame(matrix.round(6), index=list(SQUARE_NAMES),
                        columns=list(SQUARE_NAMES))


def random_triangle_test(
    n_trials: int = 300, k: int = 2, seed: int = 42,
) -> dict[str, object]:
    """Brute-force the triangle inequality on random point sets.

    The square is too symmetric to be evidence: every kappa is equal, so the
    mutual-reachability matrix is just the Euclidean one. These trials use
    random dimensions and sizes so the kappa terms genuinely bind.
    """
    rng = np.random.default_rng(seed)
    worst_mreach = worst_floored = 0.0
    bad_mreach = bad_floored = 0
    for _ in range(n_trials):
        n = int(rng.integers(5, 12))
        points = rng.normal(size=(n, int(rng.integers(1, 5))))
        for matrix, which in (
            (mutual_reachability(points, k), "mreach"),
            (floored_distance(points), "floored"),
        ):
            for i, j, m in itertools.product(range(n), repeat=3):
                slack = float(matrix[i, j] + matrix[j, m] - matrix[i, m])
                if which == "mreach":
                    worst_mreach = min(worst_mreach, slack)
                    bad_mreach += slack < -1e-9
                else:
                    worst_floored = min(worst_floored, slack)
                    bad_floored += slack < -1e-9
    return {
        "n_trials": n_trials,
        "mreach_violations": int(bad_mreach),
        "mreach_worst_slack": float(worst_mreach),
        "floored_violations": int(bad_floored),
        "floored_worst_slack": float(worst_floored),
    }


def linkage_heights(matrix: np.ndarray) -> list[float]:
    """Single-linkage merge heights — the hierarchy the metric induces."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    tree = linkage(squareform(matrix, checks=False), method="single")
    return sorted({round(float(h), 6) for h in tree[:, 2]})


def solve(k: int = 2) -> dict[str, object]:
    """Compute on the square, then on a configuration where kappa varies."""
    square_matrix = mutual_reachability(SQUARE, k)
    square_distance = np.linalg.norm(SQUARE[:, None] - SQUARE[None], axis=-1)
    off = ~np.eye(4, dtype=bool)

    uneven_mreach = mutual_reachability(UNEVEN, k)
    uneven_floor = floored_distance(UNEVEN)
    uneven_distance = np.linalg.norm(UNEVEN[:, None] - UNEVEN[None], axis=-1)
    off5 = ~np.eye(5, dtype=bool)
    const = float(np.median(uneven_distance[off5]))

    return {
        "square": {
            "kappa_2": core_distance(SQUARE, k).round(6).tolist(),
            "matrix": square_table(k),
            "equals_euclidean_off_diagonal": bool(
                np.allclose(square_matrix[off], square_distance[off]),
            ),
            "axioms": check_axioms(square_matrix),
            "axioms_literal_formula": check_axioms(
                mutual_reachability(SQUARE, k, zero_diagonal=False),
            ),
        },
        "uneven": {
            "kappa_2": core_distance(UNEVEN, k).round(4).tolist(),
            "const_used": round(const, 4),
            "axioms_mreach": check_axioms(uneven_mreach),
            "axioms_floored": check_axioms(uneven_floor),
            "axioms_floored_literal": check_axioms(
                floored_distance(UNEVEN, zero_diagonal=False),
            ),
            "n_distinct_distances": int(
                len(np.unique(np.round(uneven_distance[off5], 9))),
            ),
            "n_distinct_mreach": int(
                len(np.unique(np.round(uneven_mreach[off5], 9))),
            ),
            "n_distinct_floored": int(
                len(np.unique(np.round(uneven_floor[off5], 9))),
            ),
            "pairs_collapsed_by_const": int((uneven_distance[off5] < const).sum()),
            "n_pairs": int(off5.sum()),
            "linkage_euclidean": linkage_heights(uneven_distance),
            "linkage_mreach": linkage_heights(uneven_mreach),
            "linkage_floored": linkage_heights(uneven_floor),
        },
        "random_trials": random_triangle_test(),
    }


def plot():  # pragma: no cover — figure
    """The square, its kappa_2 circles, and the three quantities of Eq. 4."""
    import matplotlib.pyplot as plt

    kappa = core_distance(SQUARE, 2)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_aspect("equal")
    for point, radius, name in zip(SQUARE, kappa, SQUARE_NAMES):
        ax.add_patch(plt.Circle(tuple(point), radius, color="#4c72b0",
                                alpha=0.10, fill=True, lw=1.0))
        ax.annotate(rf"{name}  $\kappa_2$={radius:.2f}", tuple(point),
                    textcoords="offset points", xytext=(6, 6))
    ax.scatter(SQUARE[:, 0], SQUARE[:, 1], c="#1f77b4", s=70, zorder=3)
    ax.plot([0, 1], [0, 1], color="crimson", ls="--", lw=1.2,
            label=r"$d(a,d)=\sqrt{2}$")
    ax.set_xlim(-1.4, 2.4)
    ax.set_ylim(-1.4, 2.4)
    ax.set_title(r"Unit square, $k=2$: every $\kappa_2 = 1$")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the square, worked": (
        "With a=(0,0), b=(1,0), c=(0,1), d=(1,1) the pairwise distances are 1 "
        "along each edge and sqrt(2) = 1.41421 on each diagonal. Every point "
        "has two neighbours at distance 1 and one at sqrt(2), so kappa_2 = 1 "
        "for all four. Equation 4 then gives d_mreach = max{1, 1, d}, which "
        "is 1 for the edges (unchanged, since the floor equals the distance) "
        "and sqrt(2) for the diagonals (unchanged, since the distance already "
        "exceeds the floor). The mutual-reachability matrix is identical to "
        "the Euclidean one off the diagonal — verified in solve(). The square "
        "is a degenerate example precisely because it is symmetric: every "
        "kappa is the same, so the floor never binds on anything."
    ),
    "verifying the axioms": (
        "All four hold. Non-negativity: a max of non-negative quantities. "
        "Symmetry: the formula is symmetric in a and b by inspection, and the "
        "matrix is verified symmetric. Identity: d_mreach(x,x) = 0 by the "
        "usual convention and d_mreach(x,y) > 0 for x != y whenever the "
        "points are distinct. Triangle inequality: checked exhaustively over "
        "all 64 ordered triples on the square, zero violations — and over 300 "
        "random configurations in 1-4 dimensions with 5-11 points each, still "
        "zero violations, worst slack -8.9e-16 (floating-point noise). "
        "Mutual reachability is a metric."
    ),
    "the axiom that actually breaks": (
        "The identity of indiscernibles, and only if you apply the formula "
        "literally at x = y. Taken at face value Equation 4 gives "
        "d_mreach(x,x) = max{kappa(x), kappa(x), 0} = kappa(x) > 0, so a "
        "point is at positive distance from itself — solve() reports "
        "d_xx_is_zero = False for the literal form and True once the diagonal "
        "is zeroed. This is a definitional wrinkle, not a mathematical "
        "problem: mutual reachability is introduced "
        f"{cite('Campello:13')} as a transform on pairs of *distinct* "
        "points, and every implementation — the reference one "
        f"{cite('McInnes:17')} included — sets the diagonal to zero. "
        "Worth knowing because it is the one thing that will make a "
        "hand-written mutual-reachability matrix fail a metric assertion."
    ),
    "max{d, const} does NOT break the triangle inequality": (
        "This is the answer the question seems to expect, and the "
        "computation refuses it. Floor everything at a global constant c and "
        "the triangle inequality still holds: if d(i,l) <= c then the "
        "left-hand side is c and the right-hand side is at least c + c = 2c; "
        "if d(i,l) > c then d(i,l) <= d(i,j) + d(j,l) <= max{d(i,j),c} + "
        "max{d(j,l),c}. Checked exhaustively on the square and on 300 random "
        "configurations: zero violations, worst slack -4.4e-16. Both "
        "constructions are metrics. Do not write down an axiom violation you "
        "have not verified — this module exists partly as that lesson."
    ),
    "what a global floor destroys instead": (
        "Information, not axioms. On a configuration where the core distances "
        "genuinely differ (a tight triple at scale 0.07-0.1 plus two isolated "
        "points at scale 2-3.6), the Euclidean matrix has 9 distinct pairwise "
        "values; mutual reachability has 7 (it lifts 6 pairs, each to its own "
        "*local* floor); a global floor at the median distance c = 2.925 has "
        "6, and it flattens 10 of the 20 ordered pairs onto exactly the same "
        "value. The consequence is visible in the hierarchy: single-linkage "
        "merge heights are [0.071, 2.0, 2.9] under Euclidean, [0.1, 2.9, "
        "3.52] under mutual reachability — the tight triple still merges "
        "first, at its own density scale — and a single height [2.925] under "
        "the global floor, i.e. the entire dataset becomes one cluster at one "
        "level. The hierarchy has been erased."
    ),
    "the point of Equation 4": (
        "A global floor is the same mistake as DBSCAN's global epsilon "
        f"({cite('Ester:96', bare=True)}, S 6.3), written into the "
        "metric instead of the algorithm: one scale for all data. The kappa "
        "terms make the floor *local*, so a dense region is pushed apart to "
        "its own density scale and a sparse one to its own. That is the "
        "property S 7.1 calls making dense regions 'self-similar across "
        "scales', and it is what lets the single MST of S 7.2 encode every "
        "density threshold at once "
        f"— the construction {cite('Campello:13', parenthetical=False)} "
        "introduced precisely to replace DBSCAN's single epsilon with a "
        "hierarchy. Both formulas are metrics; only one of them preserves "
        "the hierarchy that HDBSCAN then reads."
    ),
    "references": reference_list("Campello:13", "McInnes:17", "Ester:96"),
}
