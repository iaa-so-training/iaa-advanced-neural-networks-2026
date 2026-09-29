"""Chapter 16, exercise 2 — score a published result with this protocol.

    Take a published tagging result in the field --- a table of members, or a
    claimed purity --- and try to score it with this workbook's protocol. What
    information is missing from the paper, and what would you have asked the
    authors for?

The published result scored here is Casamiquela et al. (2021, A&A 654, A151):
their high-precision sample of 175 stars in 31 open clusters (their Table 1)
and the clustering metrics they quote for it (their Table 2). It is the right
target because the workbook already reproduces two of their numbers — the
recovery fraction RF40 = 29% and RF70 = 3% — from their own published tables,
which makes it possible to separate three different things: a *definition*
check (does our metric implementation agree with theirs?), a *chance-level*
check (is their metric score above what a random partition of the same shape
would give?), and the *data* check (does their claimed recovery survive when
the field is present in our pipeline?). The first two are computed here. The
third is not, and the module says why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, project_root

#: Casamiquela et al. 2021, Table 1: members per cluster, 31 clusters, 175 stars.
TABLE1_SIZES: tuple[int, ...] = (
    4, 12, 5, 5, 10, 4, 5, 6, 4, 4, 5, 6, 5, 6, 6, 7, 9, 6, 6, 4, 6, 4,
    5, 7, 5, 4, 5, 6, 4, 6, 4,
)
#: Their Table 2: (group, cluster, stars of that cluster assigned to the group).
TABLE2_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("H2", "NGC 6705", 7),
    ("H3", "Ruprecht 147", 4), ("H3", "NGC 188", 1), ("H3", "FSR 0278", 3),
    ("H4", "NGC 2420", 3), ("H4", "NGC 2354", 1),
    ("H5", "NGC 188", 2),
    ("H7", "NGC 6997", 3), ("H7", "NGC 6705", 1), ("H7", "NGC 2632", 2),
    ("H9", "NGC 6819", 2), ("H9", "NGC 2682", 1), ("H9", "Ruprecht 171", 1),
    ("H10", "NGC 2682", 5),
    ("H17", "UBC 3", 2),
    ("H30", "NGC 752", 3), ("H30", "NGC 6991", 1),
)
#: Their quoted metrics (Table 2 and the abstract), kept for the comparison.
PUBLISHED: dict[str, float] = {"h": 0.49, "c": 0.63, "V": 0.55, "RF40": 0.29, "RF70": 0.03}
#: Random partitions used for the chance level of their sample shape.
N_PERM = 200


def _table1_frame() -> pd.DataFrame:
    """Their per-cluster member counts, named the way their table is ordered.

    The paper's table does not name the clusters next to the counts in a
    machine-readable form; the workbook's ``tests/test_baseline.py`` carries
    the extraction, and this function reproduces its totals. Cluster *names*
    are needed only for the groups in Table 2, so the rest are anonymous.
    """
    names = [
        "UBC 3", "NGC 6705", "NGC 3532", "UBC 215", "NGC 2099", "NGC 7245",
        "NGC 6728", "NGC 6997", "NGC 2632", "NGC 6633", "NGC 2539", "UBC 6",
        "NGC 2266", "NGC 2355", "NGC 6811", "NGC 752", "IC 4756", "NGC 6940",
        "NGC 2354", "Skiff J1942+38.6", "NGC 7789", "NGC 6991", "NGC 6939",
        "NGC 2420", "NGC 7762", "NGC 6819", "FSR 0278", "Ruprecht 171",
        "Ruprecht 147", "NGC 2682", "NGC 188",
    ]
    if len(names) != len(TABLE1_SIZES):
        raise DataNotAvailable("Table 1 extraction and size list disagree in length")
    return pd.DataFrame({"cluster": names, "n_stars": TABLE1_SIZES})


def published_partition() -> tuple[np.ndarray, np.ndarray]:
    """Their labels and groups, reconstructed from Tables 1 and 2.

    Stars that appear in a Table 2 group get that group; stars in a cluster but
    in none of the groups are left as noise (``-1``), which is how the paper's
    own table presents them.
    """
    true: list[str] = []
    pred: list[str] = []
    placed: dict[str, int] = {}
    for group, cluster, count in TABLE2_GROUPS:
        true += [cluster] * count
        pred += [group] * count
        placed[cluster] = placed.get(cluster, 0) + count
    for _, row in _table1_frame().iterrows():
        cluster = str(row["cluster"])
        rest = int(row["n_stars"]) - placed.get(cluster, 0)
        if rest < 0:
            raise DataNotAvailable(f"{cluster}: Table 2 lists more stars than Table 1")
        true += [cluster] * rest
        pred += ["-1"] * rest
    return np.array(true), np.array(pred)


def _as_float(value: object) -> float:
    """``float`` of a value pyrefly cannot type (sklearn returns ``object``)."""
    return float(np.asarray(value, dtype=float))


def _recovery(true: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, object]:
    """The workbook's recovery fraction at a given recall/precision threshold."""
    from cluster.baseline import recovery_fraction

    return recovery_fraction(true, pred, threshold)


def recompute_published_metrics() -> pd.DataFrame:
    """Their RF40 and RF70, recomputed from their own tables."""
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv

    true, pred = published_partition()
    labels = np.unique(np.asarray([str(v) for v in true]), return_inverse=True)[1]
    groups = np.unique(np.asarray([str(v) for v in pred]), return_inverse=True)[1]
    h, c, v = hcv(labels, groups)
    rows = [
        {"metric": "stars", "published": float(sum(TABLE1_SIZES)),
         "recomputed": float(len(true)), "matches": len(true) == sum(TABLE1_SIZES)},
        {"metric": "clusters", "published": 31.0,
         "recomputed": float(len(set(true))), "matches": len(set(true)) == 31},
        {"metric": "h", "published": PUBLISHED["h"], "recomputed": round(float(h), 3),
         "matches": abs(h - PUBLISHED["h"]) < 0.05},
        {"metric": "c", "published": PUBLISHED["c"], "recomputed": round(float(c), 3),
         "matches": abs(c - PUBLISHED["c"]) < 0.05},
        {"metric": "V", "published": PUBLISHED["V"], "recomputed": round(float(v), 3),
         "matches": abs(v - PUBLISHED["V"]) < 0.05},
        {"metric": "RF40", "published": PUBLISHED["RF40"],
         "recomputed": round(_as_float(_recovery(true, pred, 0.40)["rf"]), 3),
         "matches": abs(_as_float(_recovery(true, pred, 0.40)["rf"])
                        - PUBLISHED["RF40"]) < 0.01},
        {"metric": "RF70", "published": PUBLISHED["RF70"],
         "recomputed": round(_as_float(_recovery(true, pred, 0.70)["rf"]), 3),
         "matches": abs(_as_float(_recovery(true, pred, 0.70)["rf"])
                        - PUBLISHED["RF70"]) < 0.01},
    ]
    return pd.DataFrame(rows)


def chance_level(n_perm: int = N_PERM, seed: int = 7) -> pd.DataFrame:
    """What a *random* partition of their sample shape scores.

    Their high-precision sample is 175 stars in 31 clusters. Mutual-information
    scores are not size-free: a random assignment of those labels already shares
    information with the truth, and on this sample shape that floor is high.
    The partitions permute the group labels while keeping the group sizes and
    the true group count, which is the null the metrics need to be read against.
    """
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv

    rng = np.random.default_rng(seed)
    true, _ = published_partition()
    labels = np.unique(np.asarray([str(v) for v in true]), return_inverse=True)[1]
    n_true = len(set(true))

    rows = []
    for tag, build in (
        ("true size profile, shuffled labels", "profile"),
        ("one group of 70 + 30 even groups", "blob"),
    ):
        h_list: list[float] = []
        c_list: list[float] = []
        v_list: list[float] = []
        rf40_list: list[float] = []
        for _ in range(n_perm):
            if build == "profile":
                random_labels = rng.permutation(labels)
            else:
                rest = len(labels) - 70
                counts = np.full(n_true - 1, rest // (n_true - 1))
                counts[: rest % (n_true - 1)] += 1
                random_labels = np.concatenate([
                    np.zeros(70, dtype=int),
                    np.repeat(np.arange(1, n_true), counts),
                ])
                random_labels = rng.permutation(random_labels)
            h, c, v = hcv(labels, random_labels)
            h_list.append(h)
            c_list.append(c)
            v_list.append(v)
            rf40_list.append(_as_float(_recovery(labels, random_labels, 0.40)["rf"]))
        rows.append({
            "null partition": tag,
            "n_perm": n_perm,
            "h": round(float(np.mean(h_list)), 3),
            "c": round(float(np.mean(c_list)), 3),
            "V": round(float(np.mean(v_list)), 3),
            "RF40": round(float(np.mean(rf40_list)), 3),
            "RF40_p95": round(float(np.percentile(rf40_list, 95)), 3),
        })
    return pd.DataFrame(rows)


def field_present_arm() -> pd.DataFrame:
    """The workbook's own recovery fraction with the field present, 25 clusters.

    Computed from the cached per-cluster sweep tables: for each cluster, the
    method is said to recover it at a threshold when its recall *and* its
    precision against that cluster's members both exceed it — the workbook's
    recovery-fraction definition, applied to the columns the sweep writes. The
    sweep itself (three embeddings, seven seeds, a 30-degree field around each
    cluster) is not re-run here; these are the workbook's cached numbers, read
    and scored rather than re-measured.
    """
    root = project_root()
    frames = []
    for name, note in (("region_sweep_dr19.csv", "row normalisation on (default)"),
                       ("region_sweep_dr19_nonorm.csv", "row normalisation off")):
        path = root / "results" / name
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        for method in ("tsne", "umap", "evoc"):
            recall = table[f"{method}_recall"].to_numpy(dtype=float)
            precision = table[f"{method}_precision"].to_numpy(dtype=float)
            at40 = (recall >= 0.40) & (precision >= 0.40)
            at70 = (recall >= 0.70) & (precision >= 0.70)
            frames.append({
                "sweep": note,
                "method": method,
                "n_clusters": int(len(table)),
                "RF40": round(float(at40.mean()), 3),
                "recovered40": int(at40.sum()),
                "RF70": round(float(at70.mean()), 3),
                "median_recall": round(float(np.median(recall)), 3),
                "median_precision": round(float(np.median(precision)), 3),
            })
    if not frames:
        raise DataNotAvailable(
            "results/region_sweep_dr19*.csv are not on disk; they are written by\n"
            "    uv run python scripts/region_sweep.py --allstar "
            "data/astraAllStarASPCAP-0.6.0.fits.gz\n",
        )
    return pd.DataFrame(frames)


def metric_treatment_variants() -> pd.DataFrame:
    """h/c/V for the three natural treatments of their unassigned stars.

    Their Table 2 assigns 42 of the 175 stars to nine groups; the other 133 are
    not grouped. How those 133 enter the mutual-information scores is not
    stated in the paper, and it decides the answer: as one noise block the
    scores collapse, dropped entirely they are very high, and as individuals
    they are higher still. Their published h = 0.49, c = 0.63, V = 0.55 match
    none of the three.
    """
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv

    true, pred = published_partition()
    labels = np.unique(np.asarray([str(v) for v in true]), return_inverse=True)[1]
    groups = np.unique(np.asarray([str(v) for v in pred]), return_inverse=True)[1]
    strings = np.asarray([str(p) for p in pred])
    noise = strings == "-1"

    dropped = hcv(labels[~noise], groups[~noise])
    alone = np.where(noise, np.arange(len(labels)) + 1_000, groups)
    rows = []
    for tag, values in (
        ("unassigned as one noise group", hcv(labels, groups)),
        ("unassigned dropped", dropped),
        ("each unassigned star its own group", hcv(labels, alone)),
    ):
        rows.append({
            "treatment of the 133 unassigned stars": tag,
            "h": round(float(values[0]), 3),
            "c": round(float(values[1]), 3),
            "V": round(float(values[2]), 3),
            "matches published (0.49/0.63/0.55)": bool(
                abs(values[0] - 0.49) < 0.03 and abs(values[1] - 0.63) < 0.03
                and abs(values[2] - 0.55) < 0.03
            ),
        })
    return pd.DataFrame(rows)


def solve(n_perm: int = N_PERM) -> dict[str, object]:
    """Definition check, chance level, the treatment variants, and the field arm."""
    metrics = recompute_published_metrics()
    chance = chance_level(n_perm=n_perm)
    try:
        field = field_present_arm()
    except DataNotAvailable as exc:
        field = pd.DataFrame([{"note": str(exc)}])
    return {
        "published_metrics": metrics,
        "chance_level": chance,
        "treatment_variants": metric_treatment_variants(),
        "field_present": field,
        "published": PUBLISHED,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Their published metrics against the chance floor of their sample shape."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    chance = result["chance_level"]
    assert isinstance(chance, pd.DataFrame)
    row = chance.iloc[0]

    names = ["h", "c", "V", "RF40"]
    published = [result["published"][k] for k in names]  # type: ignore[index]
    floor = [float(row[k]) for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(x - 0.2, published, 0.4, label="published (Casamiquela+21)", color="#4c72b0")
    ax.bar(x + 0.2, floor, 0.4, label="random partition, same sample shape",
           color="#c9c9c9")
    ax.set_xticks(x, names)
    ax.set_ylabel("metric value")
    ax.set_title("175 stars in 31 clusters: the floor is not zero")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what could be scored, and what could not": (
        "The published result scored here is "
        f"{cite('Casamiquela:21', parenthetical=False)}: 175 high-precision "
        "stars in 31 open clusters (their Table 1) and "
        "the clustering metrics they quote for it (h = 0.49, c = 0.63, "
        "V = 0.55, RF40 = 29%, RF70 = 3%). Three checks are possible. The "
        "*definition* check reconstructs their partition from their own two "
        "tables and runs this workbook's metric implementations on it: the star "
        "and cluster totals match exactly (175, 31), and the recovery fractions "
        "reproduce — RF40 = 0.290 against their 29%, RF70 = 0.032 against their "
        "3%. The *metric-treatment* check then fails, and that failure is the "
        "answer to 'what is missing'. The *data* check — does the claimed "
        "recovery survive with the field present? — needs their member table "
        "and their pipeline, neither of which is in this repository; what is "
        "computed instead is the same question asked of this workbook's own "
        "25-cluster field sweep."
    ),
    "the definition check reproduces RF40 and RF70, and nothing else": (
        "Their Table 2 assigns 42 of the 175 stars to nine groups and leaves the "
        "other 133 unassigned. RF40 and RF70 depend only on those groups' "
        "overlap with the true clusters, so they reproduce to the digit. The "
        "mutual-information scores depend on how the 133 unassigned stars are "
        "counted, and the paper does not say. All three natural treatments were "
        "run: as one noise group gives h = 0.227, c = 0.727, V = 0.346; dropped "
        "entirely gives 0.782, 0.914, 0.843; each unassigned star as its own "
        "group gives 0.962, 0.683, 0.799 (scikit-learn's implementation of the "
        f"three metrics, {cite('Pedregosa:11', bare=True)}). "
        "Their published 0.49, 0.63, 0.55 "
        "matches none of the three, and it sits between the first and second. "
        "So this workbook cannot verify their h/c/V from the published tables — "
        "not because the numbers are wrong, but because the assignment they were "
        "computed over is not stated. That is a reporting gap of exactly the kind "
        "the rubric is written against, and it is invisible until someone tries "
        "to recompute."
    ),
    "the chance level, which changes how their headline reads": (
        "Mutual-information metrics are not size-free: for 175 stars in 31 "
        "clusters, a random assignment of those labels already scores h = 0.508, "
        "c = 0.508, V = 0.508 (200 random partitions, labels shuffled, group "
        "sizes preserved). So a reported h = 0.49 is at or below the floor of "
        "its own sample shape, and V = 0.55 is barely above it. The recovery "
        "fraction does not have this problem: random partitions score RF40 = "
        "0.041 with a 95th percentile of 0.097, because 'is there a group whose "
        "recall *and* precision both exceed 40%' is a question a random "
        "partition essentially never answers yes to. Their RF40 = 0.29 therefore "
        "means something at this sample size, while their h and V cannot be "
        "distinguished from chance — which is the reason this workbook's "
        "headline metric is the recovery fraction, and the reason any reported "
        "mutual-information score should be printed with the sample shape beside "
        "it."
    ),
    "the field-present arm, scored on this workbook's own sweep": (
        "The workbook's DR19 30-degree region sweep puts 25 clusters against a "
        "field of ~28 000 stars each and scores them per cluster. Recovery at "
        "40% with the default row normalisation: 0 of 25 for t-SNE, 0 of 25 for "
        "UMAP, 0 of 25 for EVoC — RF40 = 0.000. With row normalisation off, "
        "UMAP recovers 1 of 25 (RF40 = 0.040) and the others none. Their "
        "field-present arm reports RF40 = 0.29 on the same kind of task. The "
        "median columns show why the two cannot be the same operating point: "
        "with normalisation on, UMAP's median recall is 0.740 against a median "
        "precision of 0.008 — it is finding the members and burying them in "
        "field stars, which the recovery fraction refuses to call a recovery "
        "while a recall-only table would report 74%. That is the whole argument "
        "for printing both halves, made concrete on this repository's own data. "
        "Whether a chemistry-only selection can retrieve clusters from a field "
        "at all is the contested question of this literature, argued in both "
        f"directions {cite('Hogg:16', 'Casamiquela:21', 'Spina:25')}, which is "
        "why the operating point has to be stated before the scores are "
        "compared."
    ),
    "what is missing from the paper": (
        "In order of how much they block a reproduction: the per-star member "
        "table (identifiers, cluster label, group label, membership "
        "probability) — their Table 1 gives counts, not stars; the treatment of "
        "unassigned stars in the metrics, which the exercise just showed decides "
        "h/c/V by 0.5 or more; the field population's selection function (which "
        "stars, over what area and magnitude range, and how drawn) — without it "
        "the field-present arm is uninterpretable; the abundance uncertainties "
        "and whether they entered the clustering as weights; the survey data "
        "release, since the workbook's own DR16/DR17/DR19 comparison "
        f"({cite('Majewski:17', 'Abdurrouf:22', 'Almeida:23', bare=True)}) "
        "shows both "
        "a photometric rescaling and a product mismatch between releases; "
        "whether the epsilon/min_samples grid was tuned on the labels before or "
        "after the score was read (their methods text says the grid maximises "
        "recovered clusters, which is a selection on the reported metric); and "
        "the number of runs and the spread over them."
    ),
    "what to ask the authors for": (
        "Six things: (1) the per-star member table in a machine-readable "
        "format, with the group assignments; (2) the code that computes h, c, V "
        "and RF, or just a statement of how unassigned stars are counted — the "
        "one-line answer that makes the exercise's first check pass; (3) the "
        "field sample's selection function; (4) the scores for the untuned "
        "clustering operating point, if the grid was tuned on the labels; (5) "
        "the abundance uncertainties and their use; (6) the runs and seeds "
        "behind each quoted number. Note the shape of the request: (2) is one "
        "sentence of prose, and it is the difference between a reproducible "
        "metric and an unreproducible one. Goodwill is not the bottleneck; "
        "reporting convention is."
    ),
    "the honest caveat about this exercise": (
        "Everything computed here is computed on their *published tables*, so it "
        "inherits any error in the extraction: the cluster names of Table 1 are "
        "transcribed in tests/test_baseline.py (including the documented "
        "mislabelled NGC 752 row) and reused here, and the groups of Table 2 are "
        "taken as printed. The workbook reproduces their RF definitions, which "
        "is a check on this repository's code — not a check on their stars, "
        "which are not available. And the field-present comparison is between "
        "two different pipelines on two different samples: it establishes that "
        "the workbook cannot reproduce their field-present recovery, not that "
        "their number is wrong. Saying which of those two things you have shown "
        "is the exercise."
    ),
    "references": reference_list(
        "Casamiquela:21", "Pedregosa:11", "Hogg:16", "Spina:25",
        "Majewski:17", "Abdurrouf:22", "Almeida:23",
    ),
}
