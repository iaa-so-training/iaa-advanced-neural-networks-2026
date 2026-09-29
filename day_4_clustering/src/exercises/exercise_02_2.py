"""Chapter 2, exercise 2 — strict complete-case, and a claim that no longer holds.

    MIN_FINITE_ELEMENTS defaults to 8 of 16. Set it to 16 (strict
    complete-case) and count how many of the 25 clusters survive with at
    least five members. Which clusters are lost, and what does that do to any
    conclusion about the surviving ones?

The exercise expects a massacre: \\S 2.3 step 3 says "dropping every star
with a NaN deletes the metal-poor globulars *entirely* --- M 15 and M 92
vanish from the sample". Run it on DR19 and that does not happen. All 25
clusters survive, and only 4 member rows are lost. This module reports the
measurement rather than the expectation, finds the cut that *does* delete
M 15 and M 92, and treats the discrepancy itself as the lesson: a data-
handling claim inherited from one release has to be re-measured on the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import settings

#: The floor the exercise asks for: every one of the 16 elements finite.
STRICT_FLOOR = 16

#: Minimum members for a cluster to survive the workbook's sample selection.
MIN_MEMBERS = 5

#: Floors swept by :func:`solve`, spanning the default to the strict case.
FLOORS: tuple[int, ...] = (8, 9, 10, 12, 14, 15, 16)


def _prepared(**overrides: object):
    """Run the pipeline end to end under one configuration.

    Not cached: every call re-reads the 1.17 GB catalogue, which is the price
    of changing a flag that acts *before* the cached member frame is built.
    Budget ~25 s per configuration.
    """
    from cluster import config
    from cluster.clusters import CLUSTERS
    from cluster.data import prepare
    from exercises.utils import catalogue_path

    cfg = settings(**overrides)
    return prepare(
        catalogue_path(), cfg, list(CLUSTERS),
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )


def _survivors(df: pd.DataFrame, min_members: int = MIN_MEMBERS) -> dict[str, int]:
    """Clusters with at least ``min_members`` member rows, name -> count."""
    member = pd.DataFrame(df[df["cluster"] != "field"])
    counts = member["cluster"].value_counts()
    return {
        str(name): int(value)
        for name, value in counts.items()
        if int(value) >= min_members
    }


def _by_count(item: tuple[str, int]) -> int:
    """Sort key: most populous cluster first."""
    return -item[1]


def nan_rates() -> dict[str, object]:
    """Per-element NaN rate, and the finite-per-row histogram, on DR19.

    The answer to "why is strict complete-case so cheap on DR19" is here: the
    NaN pattern is almost entirely all-or-nothing, so requiring 16 finite
    elements costs barely more than requiring 8.
    """
    from cluster.data import apply_quality_cuts, load_allstar
    from exercises.utils import catalogue_path

    cfg = settings()
    elements = list(cfg.elements)
    frame = apply_quality_cuts(load_allstar(catalogue_path(), elements), cfg)
    values = frame[elements].to_numpy(dtype=float)
    finite = np.isfinite(values)

    table = pd.DataFrame({
        "element": elements,
        "nan_rate": np.round(1.0 - finite.mean(axis=0), 4),
    }).sort_values("nan_rate", ascending=False).reset_index(drop=True)

    per_row = finite.sum(axis=1)
    return {
        "n_clean_rows": int(len(frame)),
        "per_element": table,
        "finite_per_row_histogram": np.bincount(per_row, minlength=17),
        "fraction_complete": round(float((per_row == 16).mean()), 4),
        "fraction_at_least_8": round(float((per_row >= 8).mean()), 4),
    }


def solve(floors: tuple[int, ...] = FLOORS) -> dict[str, object]:
    """Sweep the floor, then find the cut that really does delete a cluster."""
    rows: list[dict[str, object]] = []
    survivors: dict[str, dict[str, int]] = {}

    for floor in floors:
        prepared = _prepared(min_finite_elements=floor, impute_missing=True)
        keep = _survivors(prepared.df)
        survivors[f"min_finite={floor}"] = keep
        rows.append({
            "configuration": f"impute, min_finite={floor}",
            "n_rows": int(len(prepared.df)),
            "n_member_rows": int((prepared.df["cluster"] != "field").sum()),
            "n_clusters_ge5": len(keep),
            "member_rows_kept": sum(keep.values()),
        })

    strict = _prepared(impute_missing=False)
    strict_keep = _survivors(strict.df)
    survivors["strict (impute_missing=False)"] = strict_keep
    rows.append({
        "configuration": "strict complete-case (impute_missing=False)",
        "n_rows": int(len(strict.df)),
        "n_member_rows": int((strict.df["cluster"] != "field").sum()),
        "n_clusters_ge5": len(strict_keep),
        "member_rows_kept": sum(strict_keep.values()),
    })

    flagged = _prepared(impute_missing=False, require_element_flag_clean=True)
    flagged_keep = _survivors(flagged.df)
    survivors["strict + per-element flags"] = flagged_keep
    rows.append({
        "configuration": "strict + REQUIRE_ELEMENT_FLAG_CLEAN",
        "n_rows": int(len(flagged.df)),
        "n_member_rows": int((flagged.df["cluster"] != "field").sum()),
        "n_clusters_ge5": len(flagged_keep),
        "member_rows_kept": sum(flagged_keep.values()),
    })

    baseline = survivors["min_finite=8"]
    ordered = sorted(baseline.items(), key=_by_count)
    per_cluster = pd.DataFrame({"cluster": [name for name, _ in ordered]})
    for label, keep in survivors.items():
        per_cluster[label] = [keep.get(c, 0) for c in per_cluster["cluster"]]

    return {
        "sweep": pd.DataFrame(rows),
        "per_cluster": per_cluster,
        "lost_at_strict": sorted(set(baseline) - set(strict_keep)),
        "lost_with_element_flags": sorted(set(baseline) - set(flagged_keep)),
        "member_rows_default": sum(baseline.values()),
        "member_rows_strict": sum(strict_keep.values()),
        "member_rows_with_element_flags": sum(flagged_keep.values()),
        "chapter_claim": (
            "§2.3 step 3: dropping every star with a NaN deletes M 15 and "
            "M 92 entirely"
        ),
        "chapter_claim_reproduces_on_dr19": bool(
            {"M 15", "M 92"} - set(strict_keep),
        ),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Member rows kept against the finite-element floor."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    sweep = result["sweep"]
    assert isinstance(sweep, pd.DataFrame)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(sweep))
    ax.bar(x, sweep["member_rows_kept"], color="#4c72b0")
    for i, (kept, n_cl) in enumerate(
        zip(sweep["member_rows_kept"], sweep["n_clusters_ge5"]),
    ):
        ax.text(i, kept + 8, f"{n_cl} cl", ha="center", fontsize=8)
    ax.set_xticks(x, [str(c) for c in sweep["configuration"]],
                  rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("member rows kept (clusters with >=5)")
    ax.set_title("What the missing-value rule actually costs on DR19")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the measured answer": (
        "All 25 clusters survive, and the sample loses 4 member rows out of "
        "1 002. Setting the floor to 16 finite elements (or equivalently "
        "impute_missing=False) leaves 998 member rows in 25 clusters, every "
        "one still above the five-member threshold. The floor does nothing "
        "at all between 8 and 15: 1 002 member rows at every one of "
        "min_finite = 8, 9, 10, 12, 14, 15, and only at 16 does anything "
        "change. The 4 rows lost are all NGC 6791, which drops 45 -> 41; two "
        "of them fail because [V/Fe] is NaN, and the other two are complete "
        "rows that fall out because complete_case runs before the membership "
        "sigma-clip, so removing their neighbours moves the cluster "
        "centroid. That last detail is worth noticing: a quality cut can "
        "change *membership*, not just sample size."
    ),
    "the chapter claim does not reproduce": (
        "§2.3 step 3 says dropping every star with a NaN 'deletes the "
        "metal-poor globulars entirely --- M 15 and M 92 vanish from the "
        "sample'. On DR19 they do not. Under strict complete-case M 15 keeps "
        "all 31 of its member rows and M 92 keeps all 25; in the default "
        "sample every single one of their stars already has all 16 elements "
        "finite. Report the measurement, not the expectation. The claim is "
        "very likely true of the DR17 ASPCAP catalogue the project started "
        f"on {cite('Abdurrouf:22')} — §2.1 notes DR19 "
        f"{cite('Almeida:23')} reanalyses rather than replaces DR17, and the "
        "Astra pipeline's abundance coverage for metal-poor giants is "
        "visibly different — but it is not true of the data this workbook "
        "actually ships, and the chapter text should be corrected or scoped "
        "to the release it came from."
    ),
    "why strict complete-case is so cheap here": (
        "Because the NaN pattern is all-or-nothing rather than per-element. "
        "In the 399 883 clean DR19 rows of the APOGEE spectroscopic sample "
        f"{cite('Majewski:17')} the per-element NaN rate is 0.104 "
        "for thirteen elements, 0.108 for [V/Fe] and 0.017 for [Mg/Fe] and "
        "[Fe/H]; but the histogram of finite-elements-per-row is bimodal: "
        "356 453 rows have all 16, 34 610 have exactly 2, 6 868 have 0, and "
        "fewer than 2 000 rows sit anywhere in between. Astra either fits "
        "the abundances or it does not. A floor of 8 and a floor of 16 "
        "therefore select almost the same rows — 89.1% complete against "
        "89.5% with at least 8 — which is why the lever the chapter treats "
        "as decisive is nearly inert on this release."
    ),
    "the cut that does delete M 15 and M 92": (
        "REQUIRE_ELEMENT_FLAG_CLEAN, the per-element quality flag. Turn it "
        "on together with strict complete-case and the sample collapses from "
        "1 002 member rows in 25 clusters to 705 rows in 23: M 15 and M 92 "
        "are gone, exactly the objects §2.3 warns about, and the survivors "
        "are gutted too — M 3 falls 154 -> 64, M 5 67 -> 40, M 13 34 -> 17, "
        "and the Pleiades 23 -> 7. So the chapter's *physics* is right (weak "
        "lines in metal-poor giants are the mechanism, and it is the "
        "metal-poor globulars that pay) while its attribution to the NaN "
        "floor is wrong for DR19: on this release the flags carry the "
        "information the NaNs used to."
    ),
    "what it does to conclusions about the survivors": (
        "Two distinct kinds of damage, and only the second is subtle. "
        "(1) Loss of range. Deleting M 15 and M 92 removes the two most "
        "metal-poor clusters in the sample ([Fe/H] = -2.26 and -2.19), both "
        "globulars drawn from the catalogue the target list is built on "
        f"{cite('Harris:96')}; every "
        "remaining cluster is within about 1.5 dex of solar, so any claim "
        "about tagging 'across metallicity' loses the half of the axis that "
        "made it interesting, and — per chapter 1's exercise 1 — the "
        "globular/open contrast that the sample mean rests on is thinned "
        "from seven globulars to five. (2) Selection on the outcome. The "
        "stars that survive a strict cut are the ones with the best spectra, "
        "so the surviving sample is biased toward high SNR and well-behaved "
        "atmospheres. A homogeneity measured on that sample is measured on "
        "an easier problem, and comparing it with a number from the "
        "permissive sample compares two different populations. Measured: "
        "K-means at K=25 gives homogeneity 0.4490 on the 1 002-row default "
        "and 0.4271 on the 998-row strict sample — a small move here, but "
        "the point is that it moved at all from deleting 4 rows."
    ),
    "the methodological rule": (
        "Re-measure inherited claims after a data release changes. This "
        "exercise exists because §2.3 calls the missing-value rule 'the "
        "difference between a sample that contains the interesting objects "
        "and one that does not' — a strong, checkable, load-bearing claim, "
        "and on DR19 it is false as stated. The claim was not careless; it "
        "was true once. Data-handling defaults carry the fingerprint of the "
        "catalogue they were tuned on, and §2.1's warning that the DR17 to "
        "DR19 schema changes 'bite silently rather than loudly' applies to "
        "the pipeline's own configuration as much as to its column names."
    ),
    "reproduce it": (
        "solve() runs the whole sweep, but each configuration re-reads the "
        "1.17 GB catalogue (the floor acts before the cached member frame "
        "exists), so budget ~25 s per row and ~4 minutes overall. The single "
        "check is: settings(min_finite_elements=16, impute_missing=False), "
        "pass it to cluster.data.prepare, and count value_counts() on the "
        "member rows. The equivalent CLI form is "
        "CLUSTER_IMPUTE_MISSING=0 uv run cluster run --fast, and the "
        "per-floor population counts already on disk are "
        "results/population_counts_floor*.csv."
    ),
    "references": reference_list(
        "Almeida:23", "Abdurrouf:22", "Majewski:17", "Harris:96",
    ),
}
