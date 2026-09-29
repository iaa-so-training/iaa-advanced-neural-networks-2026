"""Chapter 14, exercise 2 — the linear provenance probe.

    Reproduce the linear probe of §14.5: fit a logistic regression from a
    256-dimensional latent to a binary flag identifying the data product, and
    report the AUC on held-out member stars. Now repeat it for a latent
    trained on a single product. What is the AUC you would consider acceptable
    before publishing a field-retrieval number?

The probe is :func:`cluster.provenance.provenance_auc` — the module was written
for this exercise — run three ways: on the mixed latent the disaster was
measured on, on the single-product re-run, and against a permuted-label null
that shows the AUC is not an artefact of 256 dimensions on ~800 stars. Two of
those three runs are cheats on the *label*, and the module says which: the
member list selects on the same sky and the same survey, so "which product
vouched for this star" is partly a statement about stellar parameters, not only
about the pipeline. That is the honest reason a single-product probe cannot be
reported as a clean zero, and it is why the answer to the exercise's last
question is a ceiling and not a point value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import embedding_path, member_field, members

#: The mixed latent the §14.5 disaster was measured on: DR19 ``apStar`` spectra
#: for the field plus a DR17 ``aspcapStar`` backfill for members.
MIXED_LATENT = "masked_latent_all_mixed_v1.parquet"
#: The two products the source map is built from (disjoint identifier sets).
#: ``masked_latent.parquet`` cannot be used here: the re-run overlaps the
#: backfill by 736 stars, so provenance would be ambiguous and
#: ``cluster.provenance.load_source_map`` raises rather than guessing.
DR19_LATENT = "masked_latent_dr19_apstar_v1.parquet"
DR17_LATENT = "masked_latent_dr17_missing.parquet"
#: The single-product re-run ("736 mwmStar spectra already downloaded").
RERUN_LATENT = "masked_latent.parquet"

#: What the chapter considers an acceptable provenance signature. Anything at
#: or below this and a member-vs-field score is measuring chemistry, not
#: pipeline.
ACCEPTABLE_AUC = 0.65
#: Permuted-label nulls run per latent (each is one 5-fold cross-validated probe).
N_NULL = 20


def _as_float(value: object) -> float:
    """``float`` of a value pyrefly cannot type (pandas/numpy returns ``object``)."""
    return float(np.asarray(value, dtype=float))


def _embedding(path_name: str) -> pd.DataFrame:
    """Load a latent artifact, keyed by string ``APOGEE_ID``.

    Reads every column: the probe needs all 256 dimensions. The artifacts are
    50-110 MB, so this is the slow step and it is done once per arm.
    """
    frame = pd.read_parquet(embedding_path(path_name))
    frame["APOGEE_ID"] = frame["APOGEE_ID"].astype(str)
    return frame.drop_duplicates(subset=["APOGEE_ID"])


def _members() -> pd.DataFrame:
    """The member frame, one row per star."""
    _ = members().df  # ensure the shared cache is warm before the fields are used
    frame = member_field().df
    frame = pd.DataFrame(frame[frame["cluster"] != "field"]).copy()
    frame["APOGEE_ID"] = frame["APOGEE_ID"].astype(str)
    return frame.drop_duplicates(subset=["APOGEE_ID"])


def source_labels(frame: pd.DataFrame) -> np.ndarray:
    """``DR17``/``DR19`` per row from the two disjoint product artifacts."""
    backfill = set(
        pd.read_parquet(
            embedding_path(DR17_LATENT), columns=["APOGEE_ID"],
        )["APOGEE_ID"].astype(str),
    )
    return np.where(frame["APOGEE_ID"].isin(sorted(backfill)), "DR17", "DR19")


def probe(
    latent: str, n_null: int = N_NULL, seed: int = 0,
) -> dict[str, object]:
    """Members-only provenance AUC for one latent, plus its permuted null.

    The members-only form is the one that means something: it strips the
    member/field imbalance, so a high AUC there cannot be explained by "the
    field is a different population".
    """
    from cluster.provenance import (
        PROVENANCE_AUC_CEILING,
        provenance_auc,
        separating_dimensions,
    )

    frame = _members()
    embedded = _embedding(latent)
    columns = [c for c in embedded.columns if c != "APOGEE_ID"]
    merged = frame.merge(embedded, on="APOGEE_ID", how="inner")
    source = source_labels(merged)

    X = merged[columns].to_numpy(dtype=float)
    auc, std = provenance_auc(X, source, random_state=seed)

    rng = np.random.default_rng(seed)
    nulls = [
        provenance_auc(X, rng.permutation(source), random_state=seed)[0]
        for _ in range(max(0, n_null))
    ]
    finite_nulls = [v for v in nulls if np.isfinite(v)]

    return {
        "latent": latent,
        "n_stars": int(len(merged)),
        "n_dr17": int((source == "DR17").sum()),
        "n_dr19": int((source == "DR19").sum()),
        "auc": round(float(auc), 4),
        "auc_std": round(float(std), 4),
        "null_mean": round(float(np.mean(finite_nulls)), 4) if finite_nulls else float("nan"),
        "null_std": round(float(np.std(finite_nulls)), 4) if finite_nulls else float("nan"),
        "null_p95": round(float(np.percentile(finite_nulls, 95)), 4) if finite_nulls else float("nan"),
        "n_separating_dims": int(separating_dimensions(X, source)),
        "n_dims": int(X.shape[1]),
        "ceiling": PROVENANCE_AUC_CEILING,
        "confounded": bool(np.isfinite(auc) and auc > PROVENANCE_AUC_CEILING),
    }


def physics_control(seed: int = 0) -> dict[str, object]:
    """Can the *stellar parameters* recover the product flag, without a latent?

    The backfilled members are the stars DR19's APO-North star list missed —
    fainter, hotter, lower-SNR ones — so the label partly encodes physics.
    Two answers are reported. ``univariate`` is the direct ROC AUC of one raw
    parameter against the flag, which is the honest version of "how much of
    the label is this parameter". ``four_parameters`` is the same logistic
    probe as the main experiment, fed Teff/logg/[Fe/H]/SNR instead of a
    latent: that is the floor a latent's product AUC must clear before it says
    anything about the pipeline. ``within_cluster`` repeats the four-parameter
    probe inside one cluster, which removes the "different clusters are
    different stars" part of the signature.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    frame = _members()
    source = source_labels(frame)
    y = (source == "DR19").astype(int)

    univariate: dict[str, float] = {}
    for column in ("TEFF", "LOGG", "FE_H", "SNR"):
        values = frame[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        univariate[column] = round(float(roc_auc_score(y[finite], values[finite])), 3)

    columns = ["TEFF", "LOGG", "FE_H", "SNR"]
    P = frame[columns].to_numpy(dtype=float)
    P = np.where(np.isfinite(P), P, np.nanmedian(P, axis=0))
    P = (P - P.mean(axis=0)) / np.where(P.std(axis=0) == 0, 1.0, P.std(axis=0))

    def _auc(features: np.ndarray, target: np.ndarray) -> float:
        minority = int(min(np.bincount(target)))
        if minority < 5:
            return float("nan")
        scores = cross_val_score(
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            features, target,
            cv=StratifiedKFold(max(2, min(5, minority)), shuffle=True,
                               random_state=seed),
            scoring="roc_auc",
        )
        return float(scores.mean())

    counts = frame["cluster"].value_counts()
    within: dict[str, float] = {}
    for name in (str(counts.index[0]), "M 3"):
        sub = (frame["cluster"] == name).to_numpy()
        if sub.sum() >= 20:
            within[name] = round(_auc(P[sub], y[sub]), 4)

    return {
        "univariate": univariate,
        "four_parameters": round(_auc(P, y), 4),
        "within_cluster": within,
    }


def one_bit_flag_auc() -> float:
    """AUC of the single-bit product flag against member-versus-field.

    The chapter's second measured consequence: on the mixed population the
    one-bit flag ("which product vouched for this star?") separates members
    from field stars almost as well as the whole 256-dimensional latent does.
    """
    from sklearn.metrics import roc_auc_score

    population = member_field()
    frame = population.df.copy()
    frame["APOGEE_ID"] = frame["APOGEE_ID"].astype(str)
    frame = frame.drop_duplicates(subset=["APOGEE_ID"])
    source = source_labels(frame)
    member = (frame["cluster"] != "field").to_numpy(dtype=bool)
    score = roc_auc_score(member.astype(int), (source == "DR17").astype(int))
    return round(_as_float(score), 4)


def solve(n_null: int = N_NULL) -> dict[str, object]:
    """The mixed latent, the single-product re-run, and the physics control."""
    rows = [
        probe(MIXED_LATENT, n_null=n_null),
        probe(RERUN_LATENT, n_null=n_null),
    ]
    return {
        "probes": pd.DataFrame([
            {k: v for k, v in row.items() if k != "confounded"} for row in rows
        ]),
        "mixed": rows[0],
        "rerun": rows[1],
        "physics_control": physics_control(),
        "one_bit_flag_auc": one_bit_flag_auc(),
        "acceptable_auc": ACCEPTABLE_AUC,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The probe AUCs against their permuted nulls and the acceptable ceiling."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["probes"]
    assert isinstance(table, pd.DataFrame)

    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - 0.15, table["auc"], 0.3, yerr=table["auc_std"],
           label="latent $\\to$ product AUC", color="#c44e52", capsize=3)
    ax.bar(x + 0.15, table["null_mean"], 0.3, yerr=table["null_std"],
           label="permuted-label null", color="#8c8c8c", capsize=3)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.axhline(_as_float(result["acceptable_auc"]), color="crimson", ls="--", lw=1.2,
               label=f"acceptable ceiling ({result['acceptable_auc']})")
    ax.set_xticks(x, ["mixed latent\n(2 products)", "re-run\n(1 product)"])
    ax.set_ylabel("cross-validated AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("A linear probe reads the data product off the latent")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the published probe, reproduced": (
        "On the mixed latent (masked_latent_all_mixed_v1.parquet: 806 distinct "
        f"member stars, 738 DR17 aspcapStar {cite('Abdurrouf:22')} and 68 "
        f"DR19 apStar {cite('Almeida:23')}) the members-only probe recovers "
        "the data product at cross-validated AUC 0.9997 ± 0.0006, "
        "with 135 of the 256 dimensions individually separating the two products "
        f"at p < 1e-6 (logistic regression and cross-validation as implemented "
        f"by scikit-learn, {cite('Pedregosa:11', bare=True)}). The chapter "
        "reports 0.999 and 223 of 256 from the same "
        "artifact family; the AUC reproduces exactly, the dimension count does "
        "not, because it is a threshold count on a Welch t-test whose power "
        "depends on the artifact and the alpha. A permuted-label null on the "
        "same matrix gives 0.507 ± 0.042 (95th percentile 0.564), so 0.9997 is "
        "not 256 dimensions memorising 806 stars: it is a real, near-perfect "
        "linear signature of the pipeline that produced the spectrum. The "
        "chapter's other measured number reproduces too — the one-bit product "
        "flag alone separates members from field stars at AUC 0.957."
    ),
    "the single-product repeat, and why it cannot be a clean zero": (
        "On the re-run latent — one product, DR19 mwmStar for every star — the "
        "same label gives AUC 0.9729 ± 0.0141 over 804 member stars (37 of 256 "
        "dimensions separating), against a permuted null of 0.485 ± 0.053. That "
        "looks like the fix failed, and it is not, because the label is no "
        "longer a product label: it is 'was this star in the DR17 backfill "
        "list'. The backfill exists for the members DR19's APO-North star list "
        "missed, and those are systematically different stars — the three "
        "strongest single-parameter signatures of the label are SNR at AUC "
        "0.647, [Fe/H] at 0.408 and Teff at 0.218 (logg 0.253), and a probe on "
        "those four parameters with no latent at all already reaches AUC 0.7966 "
        "on all members, 0.879 within M 67 and 0.882 within M 3. So the "
        "honest reading is a comparison of two levels: the mixed latent's "
        "0.9997 sits far above the 0.80-0.88 physics floor, while the re-run's "
        "0.973 sits at it. The mixed latent encodes the product; the re-run "
        "latent encodes the selection. And a single-product probe can never "
        "prove the absence of a product effect, because there is no second "
        "product inside it to separate — which is exactly why the fix is a "
        "re-run (uniform mwmStar spectra for everything) rather than a caveat."
    ),
    "the AUC to demand before publishing": (
        "The repo's own ceiling is PROVENANCE_AUC_CEILING = 0.65, and it is the "
        "right kind of answer: below it a member-vs-field score cannot be "
        "dominated by the product, because the product is barely recoverable at "
        "all. Three qualifications the exercise should carry. First, the "
        "ceiling applies to the members-only probe — on a mixed population the "
        "member/field imbalance alone gives the flag AUC 0.957, which the "
        "chapter quotes as the size of the confound rather than as a probe "
        "result. Second, 0.65 is a convention, not a derivation, so the "
        "defensible rule is to publish the provenance AUC next to the "
        "field-retrieval number, on the same population, and let the reader see "
        "both. Third, the probe has to be run with the label it claims: a probe "
        "on a single-product latent answers a different question and returns a "
        "number that is not comparable to a cross-product one."
    ),
    "the transferable lesson": (
        "A probe can only falsify, and what it falsifies here is a claim about "
        f"a masked-autoencoder latent {cite('He:22')} rather than about the "
        "probe. The protocol the repo implements is the one "
        "to copy: build the source map from two *disjoint* artifacts ("
        "load_source_map raises on overlap rather than guessing, which matters "
        "because the current re-run overlaps the old backfill by 736 stars), "
        "report the members-only AUC and the separating-dimension count, and "
        "re-score any member-vs-field metric on a single-product subsample as "
        "well. Note what the workbook did with the result: it withdrew the "
        "'twice as pure' precision claim, which depended on the contaminated "
        "field-retrieval number, and kept the cluster-only homogeneity, which "
        "is computed between members and therefore within one product. The "
        "honest response to a batch effect is to quarantine the affected "
        "number, not the whole result."
    ),
    "references": reference_list(
        "He:22", "Abdurrouf:22", "Almeida:23", "Pedregosa:11",
    ),
}
