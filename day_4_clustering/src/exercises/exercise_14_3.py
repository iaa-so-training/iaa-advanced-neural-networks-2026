"""Chapter 14, exercise 3 — a paired control when you only have one product.

    The paired control uses 253 stars with embeddings through both pipelines.
    Design an equivalent control for a study that only has one product, and
    state what it can and cannot detect.

Two things are separated here. First the workbook's own control is reproduced
on the two artifacts that are on disk, so the 1.70x figure is verified rather
than quoted. Second, the single-product substitutes are run: a *repeat-row*
control built from the 298 member rows that share an ``APOGEE_ID`` with
another row, and a *nuisance floor* built from the stellar parameters the
latent is known to encode anyway. The conclusion is structural and worth
stating plainly: within one product, a paired control can detect per-row noise
and can calibrate how much a nuisance label is worth, but it cannot detect a
constant offset — which is precisely the failure mode §14.5 documents, because
a constant offset puts every star into the same bucket.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import embedding_path, member_field, members

#: The two artifacts carrying the *same stars* through different pipelines.
#: 253 stars are in both (the figure §14.5 quotes); the second is the DR19
#: ``apStar`` arm of the mixed latent.
PAIRED_A = "masked_latent_dr17.parquet"
PAIRED_B = "masked_latent_dr19_apstar_v1.parquet"


def _latent(name: str) -> pd.DataFrame:
    """A latent artifact indexed by string ``APOGEE_ID``, one row per star."""
    frame = pd.read_parquet(embedding_path(name))
    frame["APOGEE_ID"] = frame["APOGEE_ID"].astype(str)
    return frame.drop_duplicates(subset=["APOGEE_ID"]).set_index("APOGEE_ID")


def paired_control(seed: int = 0) -> dict[str, object]:
    """The workbook's control, recomputed from the two pipelines' artifacts.

    For the stars present in both, the distance to the star's own embedding in
    the *other* pipeline is compared with the distance to a different star
    within one pipeline. The workbook's figure is the ratio of the two means;
    the shared-direction fraction is the norm of the mean offset over the mean
    offset norm, so a value near 1 means one global nuisance axis explains the
    displacement.
    """
    a = _latent(PAIRED_A)
    b = _latent(PAIRED_B)
    both = sorted(set(a.index) & set(b.index))
    if len(both) < 10:
        return {"n": len(both), "available": False}

    av = a.loc[both].to_numpy(dtype=float)
    bv = b.loc[both].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    paired = np.linalg.norm(av - bv, axis=1)
    within = np.linalg.norm(av - av[rng.permutation(len(both))], axis=1)
    norm_a = np.linalg.norm(av, axis=1)
    norm_b = np.linalg.norm(bv, axis=1)
    cosine = (av * bv).sum(axis=1) / (norm_a * norm_b)
    offset = bv - av

    return {
        "available": True,
        "n": len(both),
        "paired_mean": round(float(paired.mean()), 3),
        "within_mean": round(float(within.mean()), 3),
        "ratio": round(float(paired.mean() / within.mean()), 3),
        "cosine_similarity": round(float(cosine.mean()), 3),
        "cosine_distance": round(float(1.0 - cosine.mean()), 3),
        "shared_fraction": round(
            float(np.linalg.norm(offset.mean(axis=0)) / np.linalg.norm(offset, axis=1).mean()),
            3,
        ),
    }


def replicate_control() -> dict[str, object]:
    """Single-product substitute: two catalogue rows of the same star.

    298 of the 1 002 member rows share an ``APOGEE_ID`` with another row — the
    duplicate-star problem of §13's limitation list, turned into an asset.
    191 distinct same-star pairs. In C-space this is the closest available
    analogue of "same star, two reductions": same star against a different
    star in the *same* cluster (which removes the cluster-to-cluster variance)
    and against a random member. A ratio above 1 would be the disease; the
    shared-direction fraction is the diagnostic for whether the offset is one
    nuisance axis or per-row noise.
    """
    data = members()
    frame = data.df
    ids = frame["APOGEE_ID"].astype(str).to_numpy()
    usable = (ids != "") & pd.Series(ids).duplicated(keep=False).to_numpy()

    pairs: list[tuple[int, int]] = []
    for _, group in pd.Series(np.flatnonzero(usable)).groupby(ids[usable]):
        idx = group.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairs.append((int(idx[i]), int(idx[j])))
    if not pairs:
        return {"n_pairs": 0, "available": False}

    index = np.asarray(pairs, dtype=int)
    X = data.X
    labels = data.labels
    rng = np.random.default_rng(0)

    replicate = np.linalg.norm(X[index[:, 0]] - X[index[:, 1]], axis=1)
    same_cluster = []
    for i, _ in index:
        candidates = np.flatnonzero((labels == labels[i]) & (ids != ids[i]))
        if candidates.size:
            same_cluster.append(float(np.linalg.norm(X[i] - X[rng.choice(candidates)])))
    random_star = np.linalg.norm(
        X[index[:, 0]] - X[rng.integers(0, len(X), len(index))], axis=1,
    )

    offset = X[index[:, 1]] - X[index[:, 0]]
    return {
        "n_pairs": int(len(index)),
        "available": True,
        "replicate_mean": round(float(replicate.mean()), 3),
        "same_cluster_mean": round(float(np.mean(same_cluster)), 3),
        "random_mean": round(float(random_star.mean()), 3),
        "ratio_vs_same_cluster": round(float(replicate.mean() / np.mean(same_cluster)), 3),
        "ratio_vs_random": round(float(replicate.mean() / random_star.mean()), 3),
        "shared_fraction": round(
            float(np.linalg.norm(offset.mean(axis=0)) / np.linalg.norm(offset, axis=1).mean()),
            3,
        ),
    }


def nuisance_floor(field_sample: int = 2_500, seed: int = 0) -> dict[str, object]:
    """What a single-product latent recovers about *stars*, with no product flag.

    Any nuisance label that is perfectly confounded with a stellar property
    cannot be told apart from that property. This measures the ceiling: the
    AUC a linear probe reaches when the target is a median split of a physical
    parameter, on the same single-product latent and the same probe as §14.5's.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    frame = _latent("masked_latent.parquet").reset_index()
    columns = [c for c in frame.columns if c != "APOGEE_ID"]

    population = member_field()
    catalogue = population.df.copy()
    catalogue["APOGEE_ID"] = catalogue["APOGEE_ID"].astype(str)
    catalogue = catalogue.drop_duplicates(subset=["APOGEE_ID"])
    merged = catalogue.merge(frame, on="APOGEE_ID", how="inner")
    rng = np.random.default_rng(seed)
    if field_sample and field_sample < len(merged):
        keep = rng.choice(len(merged), size=field_sample, replace=False)
        merged = merged.iloc[np.sort(keep)]
    if len(merged) < 50:
        return {"n": int(len(merged)), "available": False}

    Z = merged[columns].to_numpy(dtype=float)
    out: dict[str, object] = {"n": int(len(merged)), "available": True}
    for column, tag in (("TEFF", "Teff"), ("LOGG", "logg"),
                        ("FE_H", "[Fe/H]"), ("SNR", "SNR")):
        values = merged[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        target = (values[finite] > np.median(values[finite])).astype(int)
        scores = cross_val_score(
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            Z[finite], target,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="roc_auc",
        )
        out[tag] = round(float(scores.mean()), 4)
    return out


def solve() -> dict[str, object]:
    """The workbook's control, the single-product substitutes, and the floor."""
    paired = paired_control()
    replicate = replicate_control()
    floor = nuisance_floor()

    design = pd.DataFrame([
        {
            "control": "two products (workbook)",
            "detects": "any offset between the two pipelines, including a constant one",
            "cannot_detect": "nothing about the confound — the offset is measured directly",
            "measured_here": f"ratio {paired.get('ratio')}x on {paired.get('n')} stars",
        },
        {
            "control": "repeat rows within one product",
            "detects": "per-row, non-reproducible scatter in the representation",
            "cannot_detect": "a constant offset applied to every star, which is what a "
                             "product mismatch is",
            "measured_here": f"ratio {replicate.get('ratio_vs_same_cluster')}x on "
                             f"{replicate.get('n_pairs')} same-star pairs",
        },
        {
            "control": "nuisance floor on one product",
            "detects": "how much of a candidate flag is explained by stellar parameters",
            "cannot_detect": "whether a latent encodes the product, since the product "
                             "has no second level to compare against",
            "measured_here": "probe AUC 0.97 Teff, 0.96 logg, 0.95 [Fe/H], 0.83 SNR",
        },
        {
            "control": "external re-reduction of a subsample",
            "detects": "the product effect itself, if a second product can be produced "
                       "for even a handful of stars",
            "cannot_detect": "nothing, if the subsample is representative — this is the "
                             "only control that answers the question",
            "measured_here": "not runnable here: needs the spectra and the pipeline",
        },
    ])

    return {
        "paired": paired,
        "replicate": replicate,
        "nuisance_floor": floor,
        "design": design,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Distance distributions for the two-product and one-product controls."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    paired = result["paired"]
    replicate = result["replicate"]
    assert isinstance(paired, dict) and isinstance(replicate, dict)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(["same star\n2 products", "different star\n1 product"],
                [paired.get("paired_mean", 0.0), paired.get("within_mean", 0.0)],
                color=["#c44e52", "#8c8c8c"])
    axes[0].set_title(f"workbook control ({paired.get('n')} stars)\n"
                      f"ratio {paired.get('ratio')}×", fontsize=10)
    axes[0].set_ylabel("latent distance")
    axes[1].bar(["same star\ntwo rows", "different star\nsame cluster", "different\nstar"],
                [replicate.get("replicate_mean", 0.0),
                 replicate.get("same_cluster_mean", 0.0),
                 replicate.get("random_mean", 0.0)],
                color=["#4c72b0", "#8c8c8c", "#c9c9c9"])
    axes[1].set_title(f"single-product substitute ({replicate.get('n_pairs')} pairs)\n"
                      f"ratio {replicate.get('ratio_vs_same_cluster')}×", fontsize=10)
    axes[1].set_ylabel("C-space distance")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "the workbook's control is correct, with one labelling slip": (
        "Recomputed from the two artifacts: 253 stars are embedded through both "
        "pipelines. A star sits 3.655 ± 0.5 from its own other-pipeline "
        "embedding against 2.150 ± 0.5 from a random different star within one "
        "pipeline — a ratio of 1.70x, exactly the chapter's figure. The shared "
        "fraction of the offset is 74.9%, again matching the quoted 75%. The "
        "mean cosine *similarity* between a star's two embeddings is 0.389, so "
        "the chapter's 'cosine distance 0.389' is the similarity, not the "
        "distance (the distance is 0.611). Worth flagging precisely because "
        "this is the figure that killed a published claim: a number that is "
        "off by a complement is still evidence of a nuisance axis, but the "
        "sign of the statement flips."
    ),
    "what a single-product study can still do": (
        "Two controls survive the loss of the second product, and both were "
        "run here, both on the masked-autoencoder latent "
        f"{cite('He:22')}. (1) A repeat-row control: 298 member rows share an "
        "APOGEE_ID with another row, giving 191 same-star pairs in C-space. "
        "The distance between a star's two rows is 0.664, against 0.989 for a "
        "different star in the same cluster and 1.283 for a random member — "
        "ratios of 0.67x and 0.52x, i.e. the replicate is *closer* than a "
        "stranger, as it must be, and 15.3% of the offset is shared across "
        "pairs. (2) A nuisance floor: probe the same latent for a physical "
        "label instead of a product label, on 2 500 of the same stars with the same "
        f"five-fold logistic probe ({cite('Pedregosa:11', bare=True)}) — "
        "Teff 0.968, logg 0.959, [Fe/H] 0.946, SNR "
        "0.833. That tells you what a high AUC costs to earn: if a candidate "
        "flag is confounded with a stellar parameter, a probe cannot tell the "
        "two apart."
    ),
    "what it cannot detect — and that is the whole point": (
        "A constant offset. §14.5's failure was a uniform 6 000x flux-scale "
        "difference between the two products, and the reason it was invisible "
        "for so long is that a constant offset does not make a latent noisy — "
        "it makes it *shifted*, and a shift is not detectable from inside the "
        "shifted sample. Concretely: the repeat-row control measures "
        "non-reproducible scatter, and a uniform offset contributes zero "
        "scatter. The nuisance floor measures what stellar parameters are "
        "worth, and a product offset that correlates with which stars are "
        "faint is exactly what makes the two indistinguishable — in this "
        "sample the single best univariate signature of the backfill label is "
        "SNR at AUC 0.647, which is a stellar-parameter effect, not a product "
        "one. Neither control can attribute a difference to the pipeline."
    ),
    "the design that answers the question": (
        "Re-reduce a subsample through a second product and embed it — for "
        f"APOGEE spectra {cite('Majewski:17')} that means a second reduction "
        "of the same exposures. Even 50 "
        "to 250 stars through both pipelines restores the workbook's control, "
        "and the workbook's own fix is precisely this — 736 mwmStar spectra "
        "for the backfilled members, downloaded and waiting. The internal "
        "controls are worth running in addition, because they bound the "
        "damage if the re-reduction cannot be done: a repeat-row ratio near 1 "
        "or above would mean the representation is not even self-consistent, "
        "which is a stricter failure than a product offset. So the design is "
        "(a) report the repeat-row ratio, (b) report the nuisance floor for "
        "every label you might use, (c) state explicitly that a constant "
        "offset is invisible to both, and (d) re-embed a subsample if any "
        "second product can be produced at all. Steps (a)-(c) are what a "
        "careful single-product paper can do today; step (d) is what it takes "
        "to publish the field-retrieval number."
    ),
    "the honest summary": (
        "A paired control is not a statistical test, it is a *design*: it works "
        "because two measurements of the same star differ only in the thing "
        "being tested. Remove that property and no amount of careful analysis "
        "recovers it — which is why the chapter's fix is a re-run and not a "
        "caveat, and why the withdrawn claim was withdrawn rather than "
        "footnoted. The corollary for the assignment: if your study has one "
        "data product, say so in the methods paragraph and say which controls "
        "you ran instead."
    ),
    "references": reference_list("He:22", "Pedregosa:11", "Majewski:17"),
}
