"""Chapter 3, exercise 3 — what "K is known" would have to mean.

    Pick three clusters from Table 1 (one with more than 60 members, one with
    about 20, one with fewer than 10) and write down what "K is known" would
    even mean for a field of 25 clusters with these counts. Which of the four
    tasks of \\\\S 3.3 does a fixed K violate?

The exercise is really asking whether supplying the true number of clusters is
a gift to K-means or a red herring. The answer is measured: giving K-means
exactly K = 25 — the true count — and scoring the result against the truth
shows that the binding constraint is not the *number* of clusters but the
*size distribution*, which K-means cannot honour and which no choice of K
fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, members

__all__ = [
    "CHOSEN",
    "SEED",
    "SizePrior",
    "kmeans_with_true_k",
    "size_distribution",
    "solve",
]

#: One cluster per size class the exercise asks for, taken from the workbook's
#: own Table 1: > 60 members, about 20, fewer than 10.
CHOSEN: dict[str, str] = {
    "many (>60)": "M 67",          # 230
    "about 20": "NGC 2420",        # 19
    "few (<10)": "NGC 2158",       # 6
}

#: The workbook's repeat-run seed; K-means here has ``n_init=10`` restarts of
#: its own, so the result is stable across seeds — see :func:`seed_spread`.
SEED = SEEDS[0]


class SizePrior:
    """K-means' implicit claim about cluster sizes, stated as a number.

    A Voronoi partition of ``n`` points into ``K`` cells with the same mass
    gives each cell ``n/K``. The ratio between the true largest cluster and
    that ideal cell is what the equal-size assumption gets wrong.
    """

    def __init__(self, counts: pd.Series) -> None:
        self.counts = counts.sort_values(ascending=False)
        self.n = int(counts.sum())
        self.k = int(len(counts))

    @property
    def ideal_cell(self) -> float:
        """Members per cluster if every K-means cell held 1/K of the mass."""
        return self.n / self.k

    @property
    def largest_over_ideal(self) -> float:
        return float(self.counts.iloc[0] / self.ideal_cell)

    @property
    def smallest_over_ideal(self) -> float:
        return float(self.counts.iloc[-1] / self.ideal_cell)

    @property
    def dynamic_range(self) -> float:
        """Largest cluster size divided by smallest."""
        return float(self.counts.iloc[0] / self.counts.iloc[-1])

    def summary(self) -> dict[str, object]:
        return {
            "clusters": self.k,
            "members": self.n,
            "ideal_cell_size": round(self.ideal_cell, 1),
            "largest": int(self.counts.iloc[0]),
            "smallest": int(self.counts.iloc[-1]),
            "largest / ideal cell": round(self.largest_over_ideal, 2),
            "smallest / ideal cell": round(self.smallest_over_ideal, 3),
            "largest / smallest": round(self.dynamic_range, 1),
        }


def size_distribution() -> SizeCensus:
    """The size classes of Table 1, counted from the data.

    The table's counts are truth-side counts from the full-sky run; this reads
    them off the same member frame the rest of the workbook scores, so the two
    agree row for row (1 002 members, 25 clusters).
    """
    data = members()
    counts = pd.Series(
        [str(c) for c in data.labels],
    ).value_counts()
    return SizeCensus(counts=counts)


@dataclass(frozen=True)
class SizeCensus:
    """The sample's cluster-size distribution and its size classes."""

    counts: pd.Series

    @property
    def classes(self) -> dict[str, int]:
        """How many clusters fall in each size class the exercise names."""
        return {
            "> 60 members": int((self.counts > 60).sum()),
            "about 20 (15-25)": int(self.counts.between(15, 25).sum()),
            "< 10 members": int((self.counts < 10).sum()),
        }

    @property
    def chosen(self) -> dict[str, int]:
        """Member counts of the one-per-class clusters :data:`CHOSEN` names."""
        return {label: int(self.counts[name]) for label, name in CHOSEN.items()}


def kmeans_with_true_k(seed: int = SEED) -> dict[str, object]:
    """Score K-means given the true number of clusters.

    This is the generous version of the experiment: K is *not* estimated, it
    is handed over. What is scored is whether the resulting partition
    resembles the truth — per cluster, the largest overlap with any predicted
    group, and how many predicted groups that cluster is spread across.
    """
    from sklearn.cluster import KMeans

    data = members()
    truth = np.array([str(c) for c in data.labels])
    k = len(set(truth.tolist()))
    predicted = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(data.X)

    rows = []
    for name, group in pd.Series(truth).groupby(truth):
        mask = group.index.to_numpy()
        cells = pd.Series(predicted[mask]).value_counts()
        rows.append({
            "cluster": str(name),
            "n": int(len(mask)),
            "largest_overlap": int(cells.iloc[0]),
            "overlap_fraction": round(float(cells.iloc[0] / len(mask)), 3),
            "predicted_groups_touched": int(len(cells)),
        })
    table = pd.DataFrame(rows).set_index("cluster").sort_values("n", ascending=False)

    predicted_sizes = pd.Series(predicted).value_counts()
    return {
        "n_clusters_given": k,
        "table": table,
        "predicted_size_range": (int(predicted_sizes.max()), int(predicted_sizes.min())),
        "median_overlap_fraction": round(float(table["overlap_fraction"].median()), 3),
        "clusters_below_half_recovered": int((table["overlap_fraction"] < 0.5).sum()),
        "chosen": {
            label: table.loc[name].to_dict() for label, name in CHOSEN.items()
        },
    }


def seed_spread(seeds: tuple[int, ...] = SEEDS[:3]) -> dict[str, float]:
    """Is the K-means result above an artefact of one seed?

    K-means here runs with ``n_init=10`` restarts, so the answer should barely
    move. Reporting the spread keeps the headline number honest.
    """
    scores = [
        float(kmeans_with_true_k(seed)["median_overlap_fraction"])  # type: ignore[arg-type]
        for seed in seeds
    ]
    return {
        "median_of_median_overlap": round(float(np.mean(scores)), 3),
        "spread": round(float(np.max(scores) - np.min(scores)), 4),
    }


def solve() -> dict[str, object]:
    """Run the size census and the K-known experiment."""
    census = size_distribution()
    prior = SizePrior(census.counts)
    known = kmeans_with_true_k()
    return {
        "size distribution": prior.summary(),
        "the three classes": census.classes,
        "chosen clusters": census.chosen,
        "K-means given the true K": {
            "K handed over": known["n_clusters_given"],
            "predicted cell sizes (max, min)": known["predicted_size_range"],
            "median largest-overlap": known["median_overlap_fraction"],
            "clusters recovered below 50%": known["clusters_below_half_recovered"],
            "per-cluster": known["table"],
        },
        "chosen clusters under K-means": pd.DataFrame(known["chosen"]).T,
        "seed spread": seed_spread(),
    }


ANSWER: dict[str, object] = {
    "the three clusters": (
        "Taking the exercise's three size classes from Table 1: M 67 with 230 "
        "members (the >60 class; M 3 at 154, M 5 at 67 and NGC 6819 at 62 "
        "also qualify), NGC 2420 with 19 (the ~20 class), and NGC 2158 with 6 "
        "(the <10 class; Berkeley 17 has 7, King 7 has 8). The three are not "
        "an arbitrary pick — they span the actual dynamic range of the "
        "sample, which is 38.3x between the largest cluster and the smallest. "
        "Twelve of the 25 clusters hold fewer than 25 members while M 67 "
        "alone holds 230, so a quarter of the sample sits in one cluster and "
        "another half sits in clusters a tenth its size."
    ),
    "what 'K is known' would mean": (
        "It would mean knowing the answer to the wrong question. To set K you "
        "must already know that this field holds exactly 25 clusters — which "
        "is precisely what a blind search is trying to discover, and what "
        "you cannot know in a real field where the count of clusters is the "
        "unknown. But grant it anyway, and the assumption still fails, "
        "because K-means "
        f"{cite('MacQueen:67', 'Lloyd:82')} does not only assume a *number*: "
        "it assumes roughly "
        "equal, convex, similarly-sized cells. On this sample the ideal cell "
        "is 1 002/25 = 40 members, so the true largest cluster is 5.75x an "
        "ideal cell and the true smallest is 0.15x one. The largest-to-"
        "smallest ratio of 38.3 has to be absorbed somehow, and K-means "
        "cannot absorb it."
    ),
    "what happens when you hand it the true K": (
        "It still fails, and the failure is the answer. Given K = 25 — the "
        "exact truth — K-means returns cells between 7 and 98 members, a "
        "much flatter distribution than the 6-to-230 truth, and the median "
        "cluster keeps only half its members in any one predicted group "
        "(overlap fraction 0.500 at seed 42; 0.473 averaged over the first "
        "three seeds, spread 0.081, so this is not a seed artefact). Ten of "
        "the 25 clusters come back below 50% purity. The asymmetry is the "
        "informative part. M 67, the largest cluster, is shattered across 17 "
        "predicted groups with its largest overlap holding just 33 of its "
        "230 members (14%) — a cluster five times the ideal cell gets "
        "divided. NGC 2158, the smallest, is spread over 4 groups and keeps "
        "3 of its 6 members (50%) — a cluster an order of magnitude below "
        "the ideal cell gets merged into whatever is nearby. Neither is a "
        "tuning problem: there is no K that makes a 230-member cluster and a "
        "6-member cluster both look like one cell of a Voronoi partition."
    ),
    "who survives, and why that confirms the diagnosis": (
        "The clusters K-means recovers well are the ones that happen to sit "
        "near the ideal cell size or are unusually compact: M 71 (39 members) "
        "and M 107 (15) come back at 100%, the Pleiades (23) at 96%, NGC 1245 "
        "(21) at 91%. The ten failures are concentrated at both extremes — "
        "M 67 (230) and M 3 (154) at the top, King 5 (12), King 7 (8) and "
        "Berkeley 66 (12) at the bottom. That pattern is the equal-size prior "
        "made visible: the method is accurate where a cluster is already "
        "shaped like its assumption and wrong wherever it is not. A "
        "size-dependent failure mode, not a random one, which is what makes "
        "it a structural statement about the model rather than a complaint "
        "about a particular run."
    ),
    "which task a fixed K violates": (
        "Of the four tasks in §3.3 — membership determination, discovery, "
        "refinement, cluster-only separation — it violates *discovery*, and "
        "the violation is structural rather than practical. Discovery is "
        "defined as finding groups without knowing in advance where to look "
        "or what they should look like, so 'K is known' is not an "
        "approximation of discovery: it is the assumption discovery is "
        "defined against. Membership determination and refinement are not "
        "hurt by it, because both operate on one known cluster at a time, "
        "where the decision is binary (member or field) rather than a choice "
        "of K. Cluster-only separation — the experiment of "
        f"{cite('GarciaDias:19', parenthetical=False)}, recreated as the "
        "workbook's baseline — is where the workbook legitimately "
        "fixes K = 25, and §4 is explicit that this is a deliberate "
        "concession: 'in this project we cheat, deliberately. The catalogue "
        "tells us there are 25 clusters, so the benchmarking arms are handed "
        "the true number of groups.' That is why the workbook keeps field "
        "retrieval — a discovery-like task — and cluster-only separation — "
        "'a K-known task' — in separate tables. The same fixed K is honest in "
        "the baseline and dishonest in the field."
    ),
    "the practical consequence": (
        "This is why the pipeline's density methods take a *size* parameter "
        "and not a count: HDBSCAN's min_cluster_size says how small a group "
        "may be, not how many groups to look for, and PLSCAN replaces size "
        "with persistence altogether (§7-§8). Both are answering the "
        "question a fixed K gets wrong "
        f"({cite('Campello:13', bare=True)}) — how do you simultaneously "
        "allow a "
        "230-member cluster and a 6-member one to both be real, when you do "
        "not know either number in advance. The measured cost of the K "
        "assumption here is that even with the true K supplied, ten of the "
        "25 clusters come back at below half purity, and the ones that "
        "survive are the ones that already looked like the assumption."
    ),
    "references": reference_list(
        "MacQueen:67", "Lloyd:82", "GarciaDias:19", "Campello:13",
    ),
}
