"""Chapter 10, exercise 4 — a Student-t kernel with nu = 5.

    The low-dimensional kernel is q_ij ∝ (1 + ||y_i - y_j||^2)^{-1}. Replace
    it with a Student-t kernel of nu = 5 degrees of freedom, re-derive the
    gradient, and explain what you expect to change in the map's long-range
    behaviour.

\\S 10.2 argues that the heavy tail of the $t_1$ kernel is "the whole point":
it is what buys a point somewhere to go when the crowding problem leaves no
room at moderate distance. This exercise tests that claim by moving the one
knob that controls tail weight. The gradient is re-derived from scratch,
checked against finite differences, and then a full KL descent is run at
several nu on a synthetic three-cluster problem so the predicted long-range
behaviour can be measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: Degrees of freedom compared by :func:`solve`. nu = 1 is t-SNE's own kernel
#: (a Cauchy); nu -> infinity recovers a Gaussian map kernel, i.e. SNE.
NU_VALUES: tuple[float, ...] = (1.0, 2.0, 5.0, 30.0)

#: The exercise's kernel.
NU: float = 5.0


def kernel_weights(sq_distance: np.ndarray, nu: float) -> np.ndarray:
    """Unnormalised Student-t weights w_ij = (1 + d^2/nu)^{-(nu+1)/2}.

    At nu = 1 this is (1 + d^2)^{-1}, \\textbf{Equation~\\ref{eq:qlow}}
    exactly. The exponent is not optional decoration: it is what makes the
    weight a genuine t density, and it is what changes the tail.
    """
    return (1.0 + sq_distance / nu) ** (-(nu + 1.0) / 2.0)


def kl_and_gradient(
    Y: np.ndarray, P: np.ndarray, nu: float = NU,
) -> tuple[float, np.ndarray]:
    """KL(P||Q) and its gradient for a Student-t map kernel of nu d.o.f.

    Derivation. With ``w_ij = (1 + d_ij^2/nu)^{-(nu+1)/2}`` and
    ``Q_ij = w_ij / Z``, ``Z = sum_{k!=l} w_kl``, the standard SNE argument
    gives ``dC/dy_i = sum_j (P_ij - Q_ij) * (-d log w_ij / d d_ij^2) * 4 ...``
    — concretely, since

        d log w_ij / d d_ij^2 = -((nu+1)/(2 nu)) * (1 + d_ij^2/nu)^{-1},

    the gradient is

        dC/dy_i = 2 * (nu+1)/nu * sum_j (P_ij - Q_ij)
                  * (1 + d_ij^2/nu)^{-1} * (y_i - y_j).

    Setting nu = 1 recovers van der Maaten & Hinton's
    ``4 sum_j (P_ij - Q_ij)(1 + d_ij^2)^{-1}(y_i - y_j)``, which is the check
    that the algebra is right. Note the two distinct roles of nu: the prefactor
    (nu+1)/nu is a constant rescaling that the learning rate absorbs, while
    the ``(1 + d^2/nu)^{-1}`` factor is the one that changes behaviour.
    """
    diff = Y[:, None, :] - Y[None, :, :]
    sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq, 0.0)

    w = kernel_weights(sq, nu)
    np.fill_diagonal(w, 0.0)
    Z = w.sum()
    Q = np.maximum(w / Z, 1e-12)

    kl = float(np.sum(P * np.log(np.maximum(P, 1e-12) / Q)))

    inverse = 1.0 / (1.0 + sq / nu)
    np.fill_diagonal(inverse, 0.0)
    factor = (P - Q) * inverse
    grad = 2.0 * (nu + 1.0) / nu * (
        (np.diag(factor.sum(axis=1)) - factor) @ Y
    )
    return kl, grad


def check_gradient(
    nu: float = NU, n: int = 12, seed: int = 0, epsilon: float = 1e-6,
) -> float:
    """Max |analytic - finite-difference| on a small random problem."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, n))
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 0.0)
    P /= P.sum()
    Y = rng.normal(scale=0.5, size=(n, 2))

    _, analytic = kl_and_gradient(Y, P, nu)
    numeric = np.zeros_like(Y)
    for i in range(n):
        for k in range(2):
            up, down = Y.copy(), Y.copy()
            up[i, k] += epsilon
            down[i, k] -= epsilon
            numeric[i, k] = (
                kl_and_gradient(up, P, nu)[0] - kl_and_gradient(down, P, nu)[0]
            ) / (2 * epsilon)
    return float(np.max(np.abs(analytic - numeric)))


def high_dimensional_p(
    X: np.ndarray, perplexity: float = 15.0,
) -> np.ndarray:
    """Symmetrised P of \\textbf{Equation~\\ref{eq:phigh}} (reuses ex. 10.1)."""
    from scipy.spatial.distance import pdist, squareform

    from exercises.exercise_10_1 import binary_search_sigma

    sq = squareform(pdist(X)) ** 2
    n = len(X)
    conditional = np.zeros((n, n))
    for i in range(n):
        others = np.delete(sq[i], i)
        _, p, _ = binary_search_sigma(others, perplexity)
        conditional[i] = np.insert(p, i, 0.0)
    P = (conditional + conditional.T) / (2.0 * n)
    return np.maximum(P, 1e-12)


def descend(
    P: np.ndarray, nu: float = NU, n_iter: int = 1000,
    learning_rate: float = 50.0, seed: int = 42,
    early_exaggeration: float = 12.0, exaggeration_iter: int = 250,
) -> np.ndarray:
    """Momentum descent with adaptive gains, so the kernel is the only variable.

    The gains (Jacobs' delta-bar-delta, as in the reference implementation)
    are not cosmetic here: without them a learning rate that converges at
    nu = 1 diverges at nu >= 2, because the prefactor 2(nu+1)/nu and the
    shorter-range repulsion change the gradient's scale. A diverging run
    still produces a picture, which is precisely the trap — the KL is checked
    at the end so a failed descent cannot be quoted as a result.
    """
    rng = np.random.default_rng(seed)
    Y = rng.normal(scale=1e-4, size=(len(P), 2))
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for step in range(n_iter):
        scaled = P * early_exaggeration if step < exaggeration_iter else P
        _, grad = kl_and_gradient(Y, scaled, nu)
        gains = np.clip(
            np.where(np.sign(grad) != np.sign(velocity),
                     gains + 0.2, gains * 0.8),
            0.01, None,
        )
        momentum = 0.5 if step < 250 else 0.8
        velocity = momentum * velocity - learning_rate * gains * grad
        Y = Y + velocity
        Y -= Y.mean(axis=0)
    return Y


def synthetic_clusters(
    n_per: int = 40, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs in 8-D — a controlled test bed.

    Two blobs are placed close together and the third far away, so the map's
    *long-range* behaviour (the thing nu controls) has something to get right
    or wrong.
    """
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, 8))
    centres[1, 0] = 6.0
    centres[2, 0] = 30.0
    X = np.vstack([
        rng.normal(centres[i], 1.0, size=(n_per, 8)) for i in range(3)
    ])
    labels = np.repeat(np.arange(3), n_per)
    return X, labels


def _layout_statistics(Y: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Spread, separation, and how far the distant blob actually goes."""
    centroids = np.array([Y[labels == c].mean(axis=0) for c in np.unique(labels)])
    within = float(np.mean([
        np.linalg.norm(Y[labels == c] - centroids[i], axis=1).mean()
        for i, c in enumerate(np.unique(labels))
    ]))
    near = float(np.linalg.norm(centroids[0] - centroids[1]))
    far = float(np.linalg.norm(centroids[0] - centroids[2]))
    return {
        "map_radius": round(float(np.linalg.norm(Y, axis=1).max()), 3),
        "mean_within_cluster_spread": round(within, 3),
        "near_pair_separation": round(near, 3),
        "far_pair_separation": round(far, 3),
        "far_over_near": round(far / near if near > 0 else float("nan"), 3),
        "separation_over_spread": round(
            near / within if within > 0 else float("nan"), 3),
    }


def solve(
    nu_values: tuple[float, ...] = NU_VALUES, n_per: int = 40,
    seeds: tuple[int, ...] = (42, 0, 1),
) -> dict[str, object]:
    """Verify the gradient, then embed the same P at several nu."""
    gradient_errors = {
        f"nu={nu:g}": round(check_gradient(nu), 12) for nu in nu_values
    }

    X, labels = synthetic_clusters(n_per)
    P = high_dimensional_p(X)

    rows = []
    for nu in nu_values:
        for seed in seeds:
            Y = descend(P, nu, seed=seed)
            kl, _ = kl_and_gradient(Y, P, nu)
            rows.append({
                "nu": nu,
                "seed": seed,
                "final_kl": round(kl, 5),
                **_layout_statistics(Y, labels),
            })
    per_run = pd.DataFrame(rows)
    summary = per_run.drop(columns=["seed"]).groupby("nu").agg(
        ["mean", "std"],
    ).round(3)

    return {
        "gradient_max_abs_error": gradient_errors,
        "n_points": int(len(X)),
        "n_seeds": len(seeds),
        "true_geometry": {
            "near_pair_input_distance": 6.0,
            "far_pair_input_distance": 30.0,
            "input_far_over_near": 5.0,
        },
        "layouts": summary,
        "per_run": per_run,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The kernel's tail, and the layouts it produces."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    distance = np.linspace(0, 12, 400)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for nu in NU_VALUES:
        left.plot(distance, kernel_weights(distance ** 2, nu),
                  label=fr"$\nu={nu:g}$")
    left.plot(distance, np.exp(-distance ** 2 / 2), "k--", lw=1,
              label="Gaussian (SNE)")
    left.set_yscale("log")
    left.set_xlabel(r"map distance $\|y_i-y_j\|$")
    left.set_ylabel("unnormalised $q$")
    left.set_title("Higher nu = lighter tail")
    left.legend(frameon=False, fontsize=8)

    table = result["per_run"]
    assert isinstance(table, pd.DataFrame)
    grouped = table.groupby("nu")["map_radius"].mean()
    right.plot(grouped.index, grouped.to_numpy(), "o-")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel(r"$\nu$")
    right.set_ylabel("map radius")
    right.set_title("Lighter tail = more compact map")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the kernel": (
        "A Student-t density with nu degrees of freedom gives unnormalised map "
        "weights w_ij = (1 + d_ij^2/nu)^{-(nu+1)/2}, with "
        "q_ij = w_ij / sum_{k!=l} w_kl. At nu = 1 the exponent is -1 and the "
        "1/nu is 1, so this is exactly Equation (qlow) — t-SNE's kernel "
        f"{cite('vanderMaaten:08')} is the "
        "nu = 1 member of the family. As nu -> infinity the expression tends "
        "to exp(-d^2/2), the Gaussian map kernel of the original SNE "
        f"{cite('Hinton:02')}, so nu is "
        "a continuous dial between SNE and t-SNE. nu = 5 sits nearer the "
        "t-SNE end than people expect, but its tail is materially lighter."
    ),
    "the re-derived gradient": (
        "Writing C = KL(P||Q) and differentiating through the normaliser Z, "
        "the only nu-dependent piece is d log w_ij / d d_ij^2 = "
        "-((nu+1)/(2nu)) (1 + d_ij^2/nu)^{-1}. Substituting into the standard "
        "SNE chain rule gives\n\n"
        "    dC/dy_i = 2 (nu+1)/nu * sum_j (P_ij - Q_ij) "
        "(1 + d_ij^2/nu)^{-1} (y_i - y_j).\n\n"
        "Set nu = 1 and the prefactor becomes 4 and the bracket (1 + d^2)^{-1}: "
        f"the published t-SNE gradient {cite('vanderMaaten:08')}, which is "
        "the algebra check. Two things "
        "matter here. The prefactor 2(nu+1)/nu is a constant that the learning "
        "rate absorbs and changes nothing. The factor (1 + d_ij^2/nu)^{-1} is "
        "the *repulsion range*: it decides how strongly a pair that is already "
        "far apart still pushes, and that is what nu genuinely changes."
    ),
    "the gradient is verified, not asserted": (
        "check_gradient() compares the analytic gradient with central finite "
        "differences on a 12-point random problem at every nu in the sweep. "
        "The maximum absolute discrepancy is ~1.1e-10 at nu = 5 and between "
        "1.0e-10 and 1.5e-10 for nu = 1, 2 and 30 — consistent with the "
        "O(epsilon^2) truncation error of the difference quotient at "
        "epsilon = 1e-6, i.e. the derivation is right. Re-deriving a gradient "
        "without numerically checking it is how sign errors reach publication."
    ),
    "what changes at long range": (
        "The repulsion becomes shorter-ranged. With nu = 1 the weight decays "
        "as 1/d^2, so a pair at map distance 10 still exerts about 1% of the "
        "force of a coincident pair; with nu = 5 the weight decays as d^{-6} "
        "and at nu = 30 as d^{-31}, so distant pairs become nearly invisible "
        "to the objective. Two consequences follow. (1) *Cheap* long distances "
        "become *expensive*: placing a moderately-distant point far away now "
        "costs real divergence, so the crowding problem partly returns. "
        "(2) Well-separated groups stop being pushed apart once moderately "
        "separated, so the layout contracts. A third, practical consequence "
        "surfaced in the experiment: the gradient's scale changes with nu, and "
        "a learning rate tuned at nu = 1 diverges at nu >= 2 — the run still "
        "produces a plausible-looking picture with a KL twenty times worse."
    ),
    "measured on a controlled problem": (
        "solve() builds three 8-D Gaussian blobs, two centres 6 units apart "
        "and the third 30 away (input ratio 5.0), computes one shared P at "
        "perplexity 15, and runs the identical gain-adjusted descent under "
        "each nu at three seeds. The map contracts monotonically as the tail "
        "lightens: mean map radius 32.4 (nu=1) -> 20.6 (nu=2) -> 11.4 (nu=5) "
        "-> 7.7 (nu=30), and the mean within-cluster spread falls the same way "
        "(3.01 -> 2.13 -> 1.57 -> 1.29). The separation-to-spread ratio — how "
        "cleanly a clusterer could cut the map — collapses from 12.5 at nu = 1 "
        "to 7.1, 4.9 and 4.2: the clusters stay distinguishable but the empty "
        "space between them shrinks by a factor of three. Final KL is "
        "essentially flat (0.405, 0.402, 0.407) until nu = 30, where it "
        "degrades to 0.435, so nu = 5 fits P about as well as nu = 1 does; it "
        "simply draws the answer smaller. Note the scale of the experiment: "
        "this is an exact O(n^2) gradient on a few hundred points, which is "
        "why it can be written in twenty lines — production t-SNE reaches "
        "survey sizes only through the tree-based approximation of "
        f"{cite('vanderMaaten:14', parenthetical=False)}."
    ),
    "what does NOT change": (
        "The map's far/near centroid ratio is 1.24, 1.84, 1.89 and 1.74 at "
        "nu = 1, 2, 5, 30 against a true input ratio of 5.0, with seed-to-seed "
        "scatter of 0.07-0.32. No kernel recovers the input geometry, and the "
        "ordering across nu is not even monotone within that scatter. That is "
        "caveat (i) of 'How to read a t-SNE map' quantified: inter-cluster "
        "distances are not readable, and changing the tail does not make them "
        "readable — it changes how badly they are wrong, not whether."
    ),
    "would you use nu = 5": (
        "For visualisation, no, and the reason is the crowding argument of "
        "§10.2 rather than taste: a 2-D map of 16-D data needs the fattest "
        "tail it can get, because the volume mismatch between a shell at "
        "radius r in 2-D (grows as r) and in 16-D (grows as r^15) is exactly "
        "what the heavy tail compensates — this is the crowding problem the "
        f"heavy-tailed kernel was introduced to solve {cite('vanderMaaten:08')}. "
        "Lightening the tail re-imports the "
        "problem t-SNE was invented to solve. For clustering, though, the "
        "answer is less obvious: a more compact map with shorter-range "
        "repulsion may give HDBSCAN* denser, better-defined blobs. That is a "
        "measurable question and not a matter of opinion — and it would have "
        "to be answered with the row-order and seed protocol of §9.4, because "
        "the effect size is plausibly smaller than the instability of §9.3."
    ),
    "references": reference_list(
        "vanderMaaten:08", "Hinton:02", "vanderMaaten:14",
    ),
}
