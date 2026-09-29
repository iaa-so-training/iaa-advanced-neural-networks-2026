"""Chapter 9, exercise 4 — circular chemical membership.

    Design a sigma-clipped chemical membership and use it to score a
    kinematic clustering. Based on \\S 2, explain in three sentences why the
    resulting number would be meaningless — and what the equivalent mistake
    would look like in your own field.

This is circularity made explicit, and it is the mirror image of \\S 9.4's
rule 3 ("kinematics are the ceiling, not a competitor"). The module builds the
circular labels honestly, scores the same kinematic clustering against them
and against the real \\S 2 labels, and adds two controls: a sweep over the
clipping threshold (which turns the reference itself into a tuning knob,
\\S 9.3's failure mode 4) and a sham label set carrying no membership
information at all (\\S 9.4's rule 6, the chance level).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Clipping thresholds swept by :func:`solve`. The point of sweeping is that
#: a reference definition with a free parameter is not a reference.
SIGMAS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)

#: The threshold quoted in the prose, chosen only because it maximises the
#: circular labels' agreement with the truth — which is itself the mistake.
SIGMA: float = 1.5

#: Passes of the clip, matching ``config.N_REFINE_PASSES`` for the kinematic
#: membership this parodies.
N_PASSES: int = 3


def sigma_clipped_membership(
    X: np.ndarray, seed_mask: np.ndarray,
    sigma: float = SIGMA, n_passes: int = N_PASSES,
) -> np.ndarray:
    """A chemical 'membership' by iterative sigma clipping in C-space.

    Starting from ``seed_mask``, compute the median C-space vector of the
    current members and the robust per-element scatter, then keep every star
    whose robust normalised distance to that centroid is below ``sigma``.
    Repeat ``n_passes`` times. This is deliberately the same recipe the
    pipeline uses for *kinematic* membership, applied to the wrong space.
    """
    group = np.asarray(seed_mask, dtype=bool).copy()
    for _ in range(n_passes):
        if group.sum() < 3:
            break
        centre = np.median(X[group], axis=0)
        scatter = np.median(np.abs(X[group] - centre), axis=0) * 1.4826
        scatter = np.where(scatter > 0, scatter, np.inf)
        distance = np.sqrt(np.mean(((X - centre) / scatter) ** 2, axis=1))
        group = distance < sigma
    return group


def circular_labels(
    X: np.ndarray, truth: np.ndarray, sigma: float = SIGMA,
) -> np.ndarray:
    """Chemical 'membership' labels, seeded from the true clusters.

    Each true cluster seeds one sigma-clipped chemical group; a star claimed
    by several goes to the nearest centroid. Everything unclaimed is 'field'.
    The labels are therefore a function of C-space *and* of the answer key —
    doubly dependent on things an honest reference may not touch.
    """
    names = [str(c) for c in np.unique(truth)]
    claims = np.zeros((len(names), len(X)), dtype=bool)
    centres = np.zeros((len(names), X.shape[1]))
    for i, name in enumerate(names):
        seed = truth == name
        claims[i] = sigma_clipped_membership(X, seed, sigma)
        centres[i] = (np.median(X[claims[i]], axis=0) if claims[i].any()
                      else np.median(X[seed], axis=0))

    out = np.full(len(X), "field", dtype=object)
    any_claim = claims.any(axis=0)
    distances = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
    distances = np.where(claims.T, distances, np.inf)
    winner = distances.argmin(axis=1)
    out[any_claim] = [names[w] for w in winner[any_claim]]
    return np.asarray(out)


def sham_labels(
    X: np.ndarray, n_groups: int = 25, seed: int = SEEDS[0],
) -> np.ndarray:
    """Equal-count slabs along a random C-space direction — the chance level.

    Constructed to carry no membership information whatsoever while having
    the same shape as a real label set (same n, comparable group sizes). Any
    score a method earns against these is free.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=X.shape[1])
    direction /= np.linalg.norm(direction)
    projection = X @ direction
    edges = np.quantile(projection, np.linspace(0, 1, n_groups + 1))
    index = np.clip(np.digitize(projection, edges[1:-1]), 0, n_groups - 1)
    return np.asarray([f"slab_{i}" for i in index])


def kinematic_clustering(
    df: pd.DataFrame, seed: int = SEEDS[0],
) -> np.ndarray:
    """Cluster the stars on kinematics alone — the thing being 'validated'."""
    from cluster.baseline import KINEMATIC_COLUMNS
    from cluster.benchmark import cluster_embedding, fit_umap

    kin = df[KINEMATIC_COLUMNS].to_numpy(dtype=float)
    median = np.nanmedian(kin, axis=0)
    kin = np.where(np.isfinite(kin), kin, median[None, :])
    scale = np.nanstd(kin, axis=0)
    scale[scale == 0] = 1.0
    kin = (kin - median) / scale

    cfg = settings()
    embedding = fit_umap(kin, cfg.umap, seed)
    return cluster_embedding(embedding, cfg.hdbscan)


def solve(
    seed: int = SEEDS[0], sigmas: tuple[float, ...] = SIGMAS,
) -> dict[str, object]:
    """Score one kinematic clustering against truth, circular labels and sham."""
    from cluster.baseline import separation_scores
    from exercises.utils import members

    data = members()
    truth = data.labels
    pred = kinematic_clustering(data.df, seed)

    rows: list[dict[str, object]] = [{
        "reference": "kinematic truth (§2)",
        "agreement_with_truth": 1.0,
        **separation_scores(truth, pred),
    }]
    for sigma in sigmas:
        labels = circular_labels(data.X, truth, sigma)
        rows.append({
            "reference": f"chemical {sigma:g}-sigma clip",
            "agreement_with_truth": float((labels == truth).mean()),
            **separation_scores(labels, pred),
        })
    sham = sham_labels(data.X, seed=seed)
    rows.append({
        "reference": "random C-space slabs (sham)",
        "agreement_with_truth": float((sham == truth).mean()),
        **separation_scores(sham, pred),
    })

    table = pd.DataFrame(rows).round(4)
    circular_only = table[table["reference"].str.startswith("chemical")]
    return {
        "n_stars": int(len(truth)),
        "seed": seed,
        "n_predicted_groups": int(np.unique(pred[pred != -1]).size),
        "n_noise": int((pred == -1).sum()),
        "scores": table,
        "homogeneity_range_over_sigma": (
            round(float(circular_only["homogeneity"].min()), 4),
            round(float(circular_only["homogeneity"].max()), 4),
        ),
        "best_agreement_sigma": float(
            sigmas[int(np.argmax(circular_only["agreement_with_truth"]))],
        ),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Bar chart: the same clustering scored against every label set."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["scores"]
    assert isinstance(table, pd.DataFrame)

    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(y - 0.2, table["homogeneity"], 0.4, label="homogeneity",
            color="#4c72b0")
    ax.barh(y + 0.2, table["v_measure"], 0.4, label="V-measure",
            color="#dd8452")
    ax.set_yticks(y, table["reference"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("score of the *same* kinematic clustering")
    ax.set_title("A score against circular labels is not a measurement")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the design": (
        "Seed one group per known cluster, take the median C-space vector and "
        "the robust per-element scatter of its members, keep every star within "
        "sigma robust standard deviations, iterate three times — the identical "
        "recipe the pipeline uses for kinematic membership "
        "(config.REFINE_SIGMA = 2.5, N_REFINE_PASSES = 3), applied to "
        "abundances instead of astrometry. Then cluster the 1 002 members on "
        "kinematics alone (UMAP "
        f"{cite('McInnes:18', bare=True)} + HDBSCAN on standardised Gaia "
        f"{cite('Gaia:23', bare=True)} parallax, pmra, "
        "pmdec, RV) and score that clustering against the chemical labels."
    ),
    "the numbers I measured": (
        "Seed 42, 52 predicted kinematic groups, 70 noise points. Scored "
        "against the real §2 labels the clustering gives h = 0.915, c = 0.668, "
        "V = 0.772 — the kinematic ceiling of §9.4 rule 3. Scored against the "
        "circular chemical labels at 1.5 sigma it gives h = 0.496, c = 0.364, "
        "V = 0.420. Scored against the sham labels — equal-count slabs along a "
        "random C-space direction, carrying no membership information by "
        "construction — it still gives h = 0.280, V = 0.258. So more than half "
        "of the circular 'result' is reproduced by labels that are definitionally "
        "meaningless, and none of that is visible in the number itself."
    ),
    "the reference is a tuning knob": (
        "Sweeping the clip over sigma = 1.0, 1.5, 2.0, 3.0 moves the circular "
        "homogeneity across 0.475, 0.496, 0.482, 0.739. The best-looking "
        "number, h = 0.739 at 3 sigma, comes from the labels that agree with "
        "the true membership *least* — 4.5% of stars, against 18.9% at 1.5 "
        "sigma. A generous clip produces few, huge chemical groups, and "
        "homogeneity rewards that shape regardless of whether the groups mean "
        "anything. This is §9.3's failure mode 4 with the lever moved into the "
        "reference: you can choose the threshold after seeing the score."
    ),
    "the three sentences": (
        "(1) The chemical labels are derived from the same 16 abundances that "
        "any chemical method under test would use — carrying the membership "
        "information is the founding premise of chemical tagging "
        f"{cite('Freeman:02')} — so reference and prediction "
        "are not independent and the 'validation' measures how well two "
        "functions of one dataset agree with each other. (2) Worse, the clip is "
        "seeded from the true membership, so the labels inherit the answer key "
        "and then re-derive a degraded copy of it — at the best threshold they "
        "still disagree with the truth about 81% of the time, so the reference "
        "is both circular and wrong. (3) §2's reason for using kinematics as "
        "ground truth is that proper motion, parallax and radial velocity are "
        "observables the chemical methods never see; substitute a chemical "
        "criterion and the labels stop being external, which was the one "
        "property that let them referee at all."
    ),
    "why it would pass review": (
        "Nothing in the reported number looks wrong. It is a homogeneity, from "
        "the same library call, on the same stars, quoted with the same "
        "seven-seed protocol, and h = 0.74 would read as a strong result next "
        "to the 0.52-0.58 the real chemical arms achieve. The circularity lives "
        "in the *provenance of the labels* — a sentence in the methods section, "
        "not a column in the table. That is why §9.4's rule 3 constrains inputs "
        "rather than scores: kinematics define the labels, so kinematics can "
        "never be a feature of a method scored against them, and symmetrically "
        "abundances can never define the labels for a chemical method."
    ),
    "the equivalent mistake elsewhere": (
        "The shape is always the same — the label is a function of the feature. "
        "In medical imaging: segmenting a lesion by thresholding the same "
        "intensity channel your classifier reads, then reporting Dice against "
        "that segmentation. In NLP: building a sentiment gold set with a "
        "lexicon, then evaluating a lexicon-feature model on it. In genomics: "
        "defining cell types by marker-gene expression, then reporting that "
        "clustering on expression recovers the cell types. In photometric "
        "redshifts: calibrating on a spectroscopic sample selected by the same "
        "colour cuts the photometric method uses. The test is one question — "
        "*could the label have been assigned without looking at any feature the "
        "method uses?* If not, the number is a consistency check, and should be "
        "called one."
    ),
    "what you could legitimately do instead": (
        "Chemical labels are not useless; they are just not independent of a "
        "chemical method. Two honest uses. (1) Ask explicitly 'do chemistry and "
        "kinematics agree?' and report it as an agreement statistic between two "
        "definitions, with neither called ground truth — and report the sham "
        "floor alongside it, because here that floor is h = 0.28. (2) Use an "
        "external catalogue: the pipeline's Simbad referee column, or a "
        "literature membership list built by people who never saw your matrix "
        f"(published compilations such as {cite('Dias:02', bare=True)} for "
        f"open clusters and {cite('Harris:96', bare=True)} for globulars). "
        "That is the only version of this experiment whose number can move your "
        "confidence in the method."
    ),
    "references": reference_list(
        "McInnes:18", "Gaia:23", "Freeman:02", "Dias:02", "Harris:96",
    ),
}
