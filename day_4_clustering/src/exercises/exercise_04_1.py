"""Chapter 4, exercise 1 — why Lloyd's algorithm stops, and where it stops.

    Show that the assignment and update steps each cannot increase J, and
    conclude that the algorithm terminates. Then construct a one-dimensional
    example (three points, two centres) where the global optimum and Lloyd's
    fixed point differ, and draw the two partitions.

The proof is \\S 4.2 written out, and the counterexample is the reason
\\S 4.3 treats initialisation as a modelling decision rather than a detail.
The construction is worth doing carefully: the obvious candidate, three
points at 0, 1, 3, does *not* work — every starting pair either reaches the
global optimum or collapses to one centre. The points 0, 1, 2.5 do work, and
:func:`solve` proves it by enumerating every partition and every start.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list

#: Three points whose Lloyd fixed points genuinely differ from the optimum.
POINTS: tuple[float, ...] = (0.0, 1.0, 2.5)

#: The textbook choice, which does *not* produce a suboptimal fixed point.
NAIVE_POINTS: tuple[float, ...] = (0.0, 1.0, 3.0)


def sse(groups: list[np.ndarray]) -> float:
    """The K-means objective J for an explicit partition (eq. 4.1).

    Each group contributes the sum of squared distances to its own mean,
    which is the optimal centre for a fixed assignment.
    """
    total = 0.0
    for group in groups:
        if group.size:
            total += float(((group - group.mean()) ** 2).sum())
    return total


def all_partitions(points: tuple[float, ...] = POINTS) -> pd.DataFrame:
    """Every 2-way partition of ``points`` with its J, best first.

    With three points there are only three distinct partitions, so the global
    optimum can be found by enumeration rather than asserted.
    """
    values = np.asarray(points, dtype=float)
    n = len(values)
    seen: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for mask in range(1, 2 ** n - 1):
        left = tuple(i for i in range(n) if (mask >> i) & 1)
        right = tuple(i for i in range(n) if not (mask >> i) & 1)
        key = (left, right) if left < right else (right, left)
        seen[key] = sse([values[list(left)], values[list(right)]])

    rows = [
        {
            "group_a": [float(values[i]) for i in key[0]],
            "group_b": [float(values[i]) for i in key[1]],
            "J": round(value, 6),
        }
        for key, value in seen.items()
    ]
    return pd.DataFrame(rows).sort_values("J").reset_index(drop=True)


def lloyd_1d(
    points: np.ndarray, centres: np.ndarray, max_iter: int = 200,
) -> dict[str, object]:
    """Lloyd's algorithm in one dimension, logging J after every half-step.

    The log is the point of the function: the proof says J never rises, and
    ``monotone`` checks that against the actual sequence.
    """
    current = np.asarray(centres, dtype=float).copy()
    trace: list[dict[str, object]] = []
    values: list[float] = []
    assign = np.zeros(len(points), dtype=int)

    for iteration in range(max_iter):
        assign = np.argmin(np.abs(points[:, None] - current[None, :]), axis=1)
        j_assign = _objective(points, current, assign)
        values.append(j_assign)
        trace.append({
            "iteration": iteration, "step": "assign",
            "J": j_assign, "centres": current.tolist(),
        })
        updated = current.copy()
        for k in range(len(current)):
            mask = assign == k
            if mask.any():
                updated[k] = float(points[mask].mean())
        j_update = _objective(points, updated, assign)
        values.append(j_update)
        trace.append({
            "iteration": iteration, "step": "update",
            "J": j_update, "centres": updated.tolist(),
        })
        if np.allclose(updated, current):
            current = updated
            break
        current = updated

    groups = [
        tuple(sorted(np.where(assign == k)[0].tolist()))
        for k in range(len(current))
        if (assign == k).any()
    ]
    return {
        "centres": current.tolist(),
        "assignment": assign.tolist(),
        "partition": tuple(sorted(groups)),
        "J": values[-1],
        "n_iterations": len(trace) // 2,
        "trace": pd.DataFrame(trace),
        "monotone": all(b <= a + 1e-12 for a, b in zip(values, values[1:])),
    }


def _objective(points: np.ndarray, centres: np.ndarray, assign: np.ndarray) -> float:
    return float(((points - centres[assign]) ** 2).sum())


def fixed_points(
    points: tuple[float, ...] = POINTS,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Every Lloyd fixed point reachable from a grid of starting centre pairs.

    Sweeping the starts is what turns "a local optimum exists" from a claim
    into a demonstration: the table below shows how many starts land in each.
    """
    values = np.asarray(points, dtype=float)
    grid = grid if grid is not None else np.arange(-2.0, 5.001, 0.1)

    found: dict[tuple[float, object], int] = {}
    example: dict[tuple[float, object], tuple[float, float]] = {}
    for a in grid:
        for b in grid:
            if a >= b:
                continue
            result = lloyd_1d(values, np.array([a, b]))
            j = cast(float, result["J"])
            key = (round(j, 6), result["partition"])
            found[key] = found.get(key, 0) + 1
            example.setdefault(key, (round(float(a), 2), round(float(b), 2)))

    best = min(j for j, _ in found)
    rows = [
        {
            "J": j,
            "partition": str(partition),
            "n_groups": len(partition) if isinstance(partition, tuple) else 0,
            "is_global_optimum": abs(j - best) < 1e-9,
            "n_starts": count,
            "example_start": example[(j, partition)],
        }
        for (j, partition), count in sorted(found.items())
    ]
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """Enumerate the optimum, then show Lloyd missing it."""
    values = np.asarray(POINTS, dtype=float)
    partitions = all_partitions(POINTS)
    reachable = fixed_points(POINTS)

    good = lloyd_1d(values, np.array([-2.0, 4.0]))
    bad = lloyd_1d(values, np.array([-2.0, 2.0]))

    naive = fixed_points(NAIVE_POINTS)

    return {
        "points": list(POINTS),
        "all_partitions": partitions,
        "global_optimum_J": float(partitions["J"].iloc[0]),
        "global_optimum": (
            partitions["group_a"].iloc[0], partitions["group_b"].iloc[0],
        ),
        "reachable_fixed_points": reachable,
        "good_start": {
            "start": [-2.0, 4.0], "J": good["J"],
            "partition": good["partition"], "monotone": good["monotone"],
        },
        "bad_start": {
            "start": [-2.0, 2.0], "J": bad["J"],
            "partition": bad["partition"], "monotone": bad["monotone"],
        },
        "penalty": round(
            cast(float, bad["J"]) - cast(float, good["J"]), 6,
        ),
        "monotone_on_every_start": all(
            bool(lloyd_1d(values, np.array([a, b]))["monotone"])
            for a in (-2.0, 0.3, 1.1) for b in (2.0, 3.0, 4.0)
        ),
        "naive_example_fails": {
            "points": list(NAIVE_POINTS),
            "reachable": naive,
            "note": (
                "0, 1, 3 has no suboptimal 2-group fixed point: every start "
                "reaches J=0.5 or collapses to one centre (J=4.667)"
            ),
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The two partitions, and J along both runs."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    values = np.asarray(POINTS, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0))
    ax = axes[0]
    for _j, (partition, colour, label, offset) in enumerate((
        (((0, 1), (2,)), "#4c72b0", "global optimum  J = 0.500", 0.1),
        (((0,), (1, 2)), "#c44e52", "Lloyd local optimum  J = 1.125", -0.1),
    )):
        for g, group in enumerate(partition):
            xs = values[list(group)]
            ax.scatter(xs, np.full(len(xs), offset), s=90,
                       color=colour, marker="o" if g == 0 else "s",
                       label=label if g == 0 else None)
            ax.scatter([xs.mean()], [offset], s=160, color=colour,
                       marker="*", edgecolor="k", linewidth=0.5)
    for x in values:
        ax.annotate(f"{x:g}", (x, 0), ha="center", fontsize=9)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_title("Three points, two centres: two different answers")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    for start, colour, label in (
        ([-2.0, 4.0], "#4c72b0", "start (-2, 4)"),
        ([-2.0, 2.0], "#c44e52", "start (-2, 2)"),
    ):
        trace = lloyd_1d(values, np.array(start))["trace"]
        assert isinstance(trace, pd.DataFrame)
        ax.plot(range(len(trace)), trace["J"], "o-", color=colour, label=label)
    ax.set_xlabel("half-step (assign, update, assign, ...)")
    ax.set_ylabel("J")
    ax.set_title("J never rises — but it stops in different places")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the assignment step cannot increase J": (
        "Hold the centres fixed. J = sum over points of ||x - mu(x)||^2 "
        "where mu(x) is the centre x is assigned to, and the terms are "
        "independent across points — no constraint couples them. Assigning "
        "each x to argmin_k ||x - mu_k||^2 therefore minimises every term "
        "separately, so it minimises the sum. The step replaces each term by "
        "a value that is by construction no larger than its previous value, "
        "hence J_new <= J_old. It is not merely non-increasing, it is "
        "*optimal* for fixed centres."
    ),
    "the update step cannot increase J": (
        "Hold the assignment fixed. J decomposes as a sum over clusters, "
        "sum_k sum_{x in C_k} ||x - mu_k||^2, and the clusters are now "
        "independent. For one cluster, f(mu) = sum ||x - mu||^2 is a convex "
        "quadratic in mu with grad f = -2 sum (x - mu) = 0 at "
        "mu = (1/|C_k|) sum x — the arithmetic mean, and the unique "
        "minimiser since the Hessian 2|C_k| I is positive definite. So "
        "moving each centre to its members' mean minimises each term, hence "
        "J_new <= J_old. This is §4.1's remark that the second half of "
        "eq. 4.1 is not a separate model choice but a consequence of the "
        "first."
    ),
    "why it terminates": (
        "J is non-increasing along the loop, bounded below by 0, and — the "
        "step that actually closes the argument — takes only finitely many "
        "values, because after each update the centres are determined by the "
        "assignment and there are at most K^n assignments of n points to K "
        "groups. A non-increasing sequence over a finite set must become "
        "constant. Once J stops changing the assignment stops changing "
        "(ties aside), so the centres stop moving and the loop has reached a "
        "fixed point. Note what this does *not* prove: nothing bounds how "
        "good that fixed point is, and nothing bounds the iteration count "
        "better than K^n (in practice a few dozen). Ties need a tie-break "
        "rule — assign to the lowest index, say — or a point can oscillate "
        "between two equidistant centres forever at constant J. The argument "
        "is the standard one for the two-move loop of "
        f"{cite('Lloyd:82', parenthetical=False)}, the modern formulation of "
        f"the method {cite('MacQueen:67', parenthetical=False)} proposed, and "
        "it has the same shape as the monotone-likelihood argument for the EM "
        f"algorithm {cite('Dempster:77')}, of which K-means is the "
        "hard-assignment limit: a bounded monotone sequence must settle, and "
        "in neither case does that say where."
    ),
    "the counterexample": (
        "Take three points on a line at x = 0, 1, 2.5 with K = 2. "
        "Enumerating all three partitions: {0,1}|{2.5} has J = 0.5, "
        "{0}|{1,2.5} has J = 1.125, {0,2.5}|{1} has J = 3.125. The global "
        "optimum is J = 0.5. Now run Lloyd from centres (-2, 4): the first "
        "assignment puts {0,1} left and {2.5} right, the centres move to 0.5 "
        "and 2.5, nothing changes, J = 0.5 — the optimum. Run it instead "
        "from (-2, 2): the first assignment puts {0} left and {1, 2.5} "
        "right, centres move to 0.0 and 1.75, and that is already a fixed "
        "point — 1 is closer to 1.75 than to 0.0, so no point changes hands. "
        "J = 1.125, which is 2.25x the optimum. Same data, same algorithm, "
        "different seed, and J decreased monotonically in both runs."
    ),
    "how common the bad basin is": (
        "Measured by sweeping every starting pair on a 0.1 grid over "
        "[-2, 5]: 1 117 starts reach the global optimum J = 0.500, 621 "
        "starts reach the local optimum J = 1.125, and 747 starts collapse "
        "to a single occupied centre (J = 3.167, the other centre keeps no "
        "points). So the suboptimal basin is not a measure-zero curiosity — "
        "it is a quarter of the starting space on a three-point problem, "
        "which is why n_init=10 is the default rather than a luxury, and why "
        "the D^2 seeding rule of "
        f"{cite('Arthur:07', parenthetical=False)} exists at all — exercise "
        "4.2 measures what each of the two is worth."
    ),
    "the example that does not work": (
        "Worth recording, because it is the obvious first try. Points at "
        "0, 1, 3 have partitions with J = 0.5, 2.0 and 4.5, and *every* "
        "two-centre start on the same grid reaches J = 0.5 or collapses to "
        "one centre — there is no suboptimal two-group fixed point at all. "
        "The reason is the geometry: for {0}|{1,3} to be stable, 1 must be "
        "closer to the right centre 2.0 than to the left centre 0.0, and it "
        "is not (distances 1.0 and 1.0 — an exact tie, which breaks the "
        "other way or oscillates). Moving the third point in to 2.5 makes "
        "the right centre 1.75, distance 0.75 against 1.0, and the fixed "
        "point becomes genuinely stable. A counterexample has to be checked, "
        "not assumed; this one was found by enumerating starts."
    ),
    "what it means for the workbook": (
        "§4.3's initialisation figure is this effect on real data, and §4.5's "
        f"citation of {cite('GarciaDias:18', parenthetical=False)} is it at "
        "scale: K-means on "
        "153 847 APOGEE spectra gives a solution that moves with the "
        "initialisation. Coordinate descent on a non-convex objective is "
        "guaranteed to stop and guaranteed to stop somewhere — those are two "
        "different guarantees, and only the first is a theorem. Hence the "
        "workbook's protocol of reporting a mean over seven seeds (§9.4 "
        "rule 2) rather than a single run."
    ),
    "references": reference_list(
        "MacQueen:67", "Lloyd:82", "Dempster:77", "Arthur:07",
        "GarciaDias:18",
    ),
}
