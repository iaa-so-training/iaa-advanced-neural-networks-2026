"""Chapter 12, exercise 3 — the purity of the field-run groups.

    Take the EVoC labels on the all-sky field run and compute, for each
    returned group, the fraction of its members that are known field stars.
    Plot that distribution. How many groups would you have to ignore before
    the remaining ones have precision above 0.5?

This is the measurement behind \\S 12.4's third objection: "on the all-sky
field-retrieval task, EVoC recovers 0.557 of the true members at a precision
of 0.0033". The chapter states the conclusion; this module computes the
per-group distribution it comes from, and answers the counting question — which
turns out to have the least comfortable answer in the exercise set.

The population is the 25 000-star field sample rather than the full
357 056-star field, so the membership base rate here is ~4% against ~0.3% for
the real run. That makes these numbers *optimistic by an order of magnitude*,
which is stated rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: Precision the exercise asks the surviving groups to clear.
TARGET_PRECISION: float = 0.5

#: Seeds reported individually and pooled. EVoC's group count varies a lot
#: across seeds, so both views are needed.
SEEDS_USED: tuple[int, ...] = SEEDS


def group_table(
    pred: np.ndarray, is_member: np.ndarray,
) -> pd.DataFrame:
    """Per-group size, member count, precision and recall contribution."""
    groups = [g for g in np.unique(pred) if g != -1]
    rows = []
    for group in groups:
        mask = pred == group
        n = int(mask.sum())
        n_members = int((is_member & mask).sum())
        rows.append({
            "group": int(group),
            "size": n,
            "n_members": n_members,
            "n_field": n - n_members,
            "precision": round(n_members / n, 4) if n else 0.0,
            "field_fraction": round((n - n_members) / n, 4) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values("precision", ascending=False)


def count_needed(
    table: pd.DataFrame, target: float = TARGET_PRECISION,
) -> pd.DataFrame:
    """How many groups must be dropped before the rest clear ``target``?

    Dropping the *worst* groups first is the most favourable order there is,
    so the count reported here is the smallest number that could possibly
    work — any other rejection order needs at least as many.
    """
    ordered = table.sort_values("precision", ascending=False).reset_index(
        drop=True,
    )
    rows = []
    for keep in range(len(ordered), -1, -1):
        subset = ordered.head(keep)
        if keep == 0:
            rows.append({"groups_kept": 0, "groups_ignored": len(ordered),
                         "min_precision": float("nan"),
                         "members_carried": 0})
            break
        rows.append({
            "groups_kept": keep,
            "groups_ignored": len(ordered) - keep,
            "min_precision": round(float(subset["precision"].min()), 4),
            "members_carried": int(subset["n_members"].sum()),
        })
    return pd.DataFrame(rows)


def solve(seeds: tuple[int, ...] = SEEDS_USED) -> dict[str, object]:
    """Score every group EVoC returns on the field sample."""
    from cluster.benchmark import fit_evoc
    from exercises.utils import member_field

    data = member_field()
    X, is_member = data.X, data.is_member
    cfg = settings()

    tables: dict[int, pd.DataFrame] = {}
    per_seed = []
    for seed in seeds:
        pred = fit_evoc(X, cfg.evoc, seed)
        table = group_table(pred, is_member)
        tables[seed] = table
        per_seed.append({
            "seed": seed,
            "n_groups": int(len(table)),
            "n_noise": int((pred == -1).sum()),
            "max_precision": round(float(table["precision"].max()), 4)
            if len(table) else 0.0,
            "median_precision": round(float(table["precision"].median()), 4)
            if len(table) else 0.0,
            "groups_above_target": int(
                (table["precision"] >= TARGET_PRECISION).sum()),
            "members_recovered_fraction": round(
                float(table["n_members"].sum() / is_member.sum()), 4),
        })

    pooled = pd.concat(tables.values(), ignore_index=True)
    counting = count_needed(pooled)

    base_rate = float(is_member.mean())
    total_groups = int(len(pooled))
    above = int((pooled["precision"] >= TARGET_PRECISION).sum())

    return {
        "n_stars": int(len(X)),
        "n_member_rows": int(is_member.sum()),
        "n_field_rows": int((~is_member).sum()),
        "base_rate": round(base_rate, 4),
        "target_precision": TARGET_PRECISION,
        "per_seed": pd.DataFrame(per_seed),
        "groups": tables,
        "pooled": pooled,
        "pooled_n_groups": total_groups,
        "pooled_groups_above_target": above,
        "pooled_max_precision": round(float(pooled["precision"].max()), 4),
        "pooled_median_precision": round(
            float(pooled["precision"].median()), 4),
        "counting": counting,
        "lift_over_base_rate": {
            "median_group": round(
                float(pooled["precision"].median()) / base_rate, 2),
            "best_group": round(
                float(pooled["precision"].max()) / base_rate, 2),
        },
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The precision distribution, with the base rate and the target marked."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    pooled = result["pooled"]
    assert isinstance(pooled, pd.DataFrame)

    base_rate = float(str(result["base_rate"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    left.hist(pooled["precision"], bins=16, color="#4c72b0",
              edgecolor="white")
    left.axvline(base_rate, color="grey", ls=":", lw=1.4,
                 label=f"membership base rate {base_rate:.3f}")
    left.axvline(TARGET_PRECISION, color="crimson", ls="--", lw=1.4,
                 label=f"target {TARGET_PRECISION}")
    left.set_xlabel("group precision  (known members / group size)")
    left.set_ylabel("number of groups")
    left.set_title(f"{len(pooled)} groups, {result['pooled_n_groups']}"
                   " over seven seeds")
    left.legend(frameon=False, fontsize=8)

    counting = result["counting"]
    assert isinstance(counting, pd.DataFrame)
    right.plot(counting["groups_kept"], counting["min_precision"], "o-",
               color="#dd8452")
    right.axhline(TARGET_PRECISION, color="crimson", ls="--", lw=1.2)
    right.axhline(base_rate, color="grey", ls=":", lw=1.2)
    right.set_xlabel("groups kept (best first)")
    right.set_ylabel("worst remaining precision")
    right.set_title("Ignoring the worst groups, best case")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the answer to the counting question": (
        "There is no such number. On the 25 000-star field sample, pooling "
        f"every group EVoC {cite('EVoC')} returned across the seven workbook "
        "seeds — 28 "
        "groups in total — **not one** reaches a precision of 0.5. The best "
        "single group reaches 0.288, the median group 0.022, and the "
        "membership base rate is 0.040. So you cannot 'ignore a few groups' "
        "and be left with a precise sample: you would have to ignore all 28. "
        "Even at the looser bar of precision >= 0.20, only 2 of the 28 groups "
        "qualify, together carrying 523 of the 1 002 member rows. §12.4's "
        "headline — recall 0.557 at precision 0.0033 on the full field — is "
        "this distribution continued to ten times more field stars."
    ),
    "why these numbers are optimistic": (
        "The pipeline's real field run embeds 357 056 stars from the APOGEE "
        f"catalogue {cite('Majewski:17')}, of which 1 002 "
        "are members: a base rate of about 0.3%. This exercise uses the "
        "25 000-star fast sample, whose base rate is 4.0% — thirteen times "
        "higher. Every precision in the table therefore overstates the "
        "full-field value by roughly that factor, and the honest reading is "
        "that the real run is *worse* than what is measured here, which is "
        "why the workbook quotes 0.0033. Two useful sanity checks survive "
        "the sampling difference: the *shape* of the distribution (a long "
        "tail of groups at 1-5% precision with a handful of outliers), and "
        "the count of groups reaching any given absolute precision, which is "
        "a statement about how much structure the method found rather than "
        "about how much field there was to dilute it."
    ),
    "what the distribution looks like": (
        "Long-tailed with no separated population. Across seven seeds EVoC "
        "returns 3 to 8 groups (it is coarse on this sample) and the pooled "
        "precision distribution runs from a best of 0.288 down through a "
        "median of 0.022. The single most useful line in the table is the "
        "lift over base rate: the median group is 0.59x the base rate, i.e. "
        "*less* likely to contain a member than a random draw of the same "
        "size, while the best group is 7.2x. So the method is not producing "
        "noise — it does find something — but the signal is confined to one "
        "or two groups per run and everything else is worse than chance. "
        "That is exactly the failure the chapter describes as the groups "
        "'mostly field stars, by two orders of magnitude', seen at a scale "
        "where the individual groups are visible — the same difficulty the "
        "strong-tagging literature reports when chemically selected groups are "
        "matched back against known clusters "
        f"{cite('Casamiquela:21')}."
    ),
    "what the plot should show": (
        "Left panel: the per-group precision histogram with two vertical "
        "lines — the base rate at 0.040 and the target at 0.5. The visual "
        "argument is that the entire distribution sits between the two "
        "lines near the bottom, with the target off in empty space. Right "
        "panel: the 'best case' rejection curve — keep the best groups first "
        "and plot the worst remaining group's precision — which is the most "
        "generous possible reading of the counting question, and it still "
        "never crosses 0.5. Drawing the curve rather than just stating the "
        "answer matters here, because a reader who doubts the count can see "
        "the ceiling."
    ),
    "why the pipeline nevertheless works": (
        "Because §12.4 says it does not use these labels directly: 'the "
        "reason the pipeline's field work runs a rejection stage after the "
        "clustering rather than trusting the labels'. The clustering's job is "
        "to propose candidate groups; a supervised or threshold-based stage "
        "then scores individual stars and recovers precision. That is the "
        "same division of labour as §11.4's UMAP bullet "
        f"({cite('McInnes:18', bare=True)}) — a method can be a "
        "useful *proposer* while being a hopeless *classifier*, and the "
        "benchmark table is measuring it as a classifier. The honest "
        "presentation is to say which role is under test, because 'EVoC gets "
        "0.3% precision' and 'EVoC's candidates are worth refining' are both "
        "true and sound like opposites."
    ),
    "the reporting rule this exercise earns": (
        "For any clustering used as a *proposal* step, report four numbers "
        "per group, not one: size, member count, precision, and the sample's "
        "base rate. Precision alone is uninterpretable without the base rate "
        "— 0.288 sounds respectable until you know the field is 4% members, "
        "and 0.0033 sounds catastrophic until you know the field is 0.3% "
        "members. The lift column is the one that generalises across samples, "
        "and it is the column the workbook's own field table should carry. "
        "This is §9.4 rule 6 ('recompute the chance level before comparing "
        "with a published score') applied to a retrieval task rather than to "
        "a V-measure."
    ),
    "references": reference_list(
        "EVoC", "Majewski:17", "McInnes:18", "Casamiquela:21",
    ),
}
