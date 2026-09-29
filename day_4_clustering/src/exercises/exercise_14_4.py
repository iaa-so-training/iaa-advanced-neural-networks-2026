"""Chapter 14, exercise 4 — what each latent encodes.

    The masked latent and PCA-64 tie on field retrieval. Working from the two
    latents, characterise what each one encodes: fit a linear model from each
    latent to Teff, logg and [Fe/H] in turn, and compare the cross-validated
    errors. Which representation spends more of its capacity on chemistry?

The tie on field retrieval is a statement about one operating point; this
exercise asks what the two representations are *for*. Three probes are run on
the same 3 208 stars: the cross-validated ridge error for each of Teff, logg
and [Fe/H]; the [Fe/H] error at fixed atmospheric parameters, which is the
part of metallicity the cluster signal actually needs; and the share of each
latent's variance that Teff+logg explains, which is the capacity the chapter
says PCA spends on the continuum. PCA-256 is carried as a control because it
separates "PCA is the wrong tool" from "64 dimensions is too few".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import embedding_path, member_field

#: The three arms compared, with the dimension each one carries.
ARMS: dict[str, str] = {
    "masked AE (256-d)": "masked_latent.parquet",
    "PCA-64": "pca_64.parquet",
    "PCA-256": "pca_256.parquet",
}

#: Regression targets and their units.
TARGETS: dict[str, str] = {"TEFF": "K", "LOGG": "dex", "FE_H": "dex"}


def _latent(name: str) -> pd.DataFrame:
    """A latent artifact keyed by string ``APOGEE_ID``, one row per star."""
    frame = pd.read_parquet(embedding_path(name))
    frame["APOGEE_ID"] = frame["APOGEE_ID"].astype(str)
    return frame.drop_duplicates(subset=["APOGEE_ID"])


def common_population() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The stars present in the catalogue and in *every* arm, sorted by id."""
    catalogue = member_field().df.copy()
    catalogue["APOGEE_ID"] = catalogue["APOGEE_ID"].astype(str)
    catalogue = catalogue.drop_duplicates(subset=["APOGEE_ID"])

    frames = {arm: _latent(path) for arm, path in ARMS.items()}
    shared = set(catalogue["APOGEE_ID"])
    for frame in frames.values():
        shared &= set(frame["APOGEE_ID"])
    ids = sorted(shared)
    base = catalogue[catalogue["APOGEE_ID"].isin(ids)].set_index("APOGEE_ID").loc[ids]
    return base, frames


def _ridge() -> Any:
    """The estimator used for every probe: standardise, then ridge with CV.

    Linear by design — the exercise asks what a *linear* probe recovers, which
    is the fair comparison between a linear representation (PCA) and a
    nonlinear one (the autoencoder). A boosted or kernel probe would confound
    "more capacity in the probe" with "more chemistry in the latent".
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)),
    )


def regression_errors(
    base: pd.DataFrame, frames: dict[str, pd.DataFrame], seed: int = 0,
) -> pd.DataFrame:
    """Cross-validated ridge error per arm and target, in the target's units."""
    from sklearn.model_selection import KFold, cross_val_predict

    rows: list[dict[str, object]] = []
    for arm, frame in frames.items():
        Z = frame.set_index("APOGEE_ID").loc[base.index].to_numpy(dtype=float)
        for target, unit in TARGETS.items():
            y = base[target].to_numpy(dtype=float)
            finite = np.isfinite(y)
            prediction = cross_val_predict(
                _ridge(), Z[finite], y[finite],
                cv=KFold(5, shuffle=True, random_state=seed),
            )
            residual = prediction - y[finite]
            rmse = float(np.sqrt(np.mean(residual ** 2)))
            rows.append({
                "arm": arm,
                "dim": int(Z.shape[1]),
                "target": target,
                "unit": unit,
                "n": int(finite.sum()),
                "rmse": round(rmse, 4),
                "mae": round(float(np.mean(np.abs(residual))), 4),
                "r2": round(float(1 - np.sum(residual ** 2) / np.sum((y[finite] - y[finite].mean()) ** 2)), 4),
                "target_sd": round(float(np.std(y[finite])), 4),
                "rmse_over_sd": round(rmse / float(np.std(y[finite])), 4),
            })
    return pd.DataFrame(rows)


def chemistry_at_fixed_parameters(
    base: pd.DataFrame, frames: dict[str, pd.DataFrame], seed: int = 0,
) -> pd.DataFrame:
    """[Fe/H] with the atmospheric-parameter trend regressed out first.

    A latent that predicts [Fe/H] well may be predicting Teff and getting
    metallicity for free, since the two correlate across a 25-cluster sample.
    Regressing [Fe/H] on (Teff, logg) linearly and probing the *residual*
    removes that, and the residual is what a cluster-separation score needs.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    teff = base["TEFF"].to_numpy(dtype=float)
    logg = base["LOGG"].to_numpy(dtype=float)
    feh = base["FE_H"].to_numpy(dtype=float)
    usable = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh)
    parameters = np.column_stack([teff[usable], logg[usable]])
    residual = feh[usable] - LinearRegression().fit(parameters, feh[usable]).predict(parameters)

    rows: list[dict[str, object]] = []
    for arm, frame in frames.items():
        Z = frame.set_index("APOGEE_ID").loc[base.index].to_numpy(dtype=float)[usable]
        prediction = cross_val_predict(
            _ridge(), Z, residual, cv=KFold(5, shuffle=True, random_state=seed),
        )
        rows.append({
            "arm": arm,
            "n": int(usable.sum()),
            "sd_feh": round(float(np.std(feh[usable])), 4),
            "sd_residual": round(float(np.std(residual)), 4),
            "rmse": round(float(np.sqrt(np.mean((prediction - residual) ** 2))), 4),
            "r2": round(float(1 - np.sum((prediction - residual) ** 2) / np.sum((residual - residual.mean()) ** 2)), 4),
        })
    return pd.DataFrame(rows)


def capacity_share(
    base: pd.DataFrame, frames: dict[str, pd.DataFrame], seed: int = 0,
) -> pd.DataFrame:
    """Share of each latent's variance explained by (Teff, logg), then +[Fe/H].

    The chapter's claim is that PCA maximises variance and the leading
    variance in a stellar spectrum is the temperature-sensitive continuum.
    This measures the share directly: a quadratic ridge on (Teff, logg) is
    fitted to every latent dimension, and the variance-weighted R² is reported
    before and after [Fe/H] is added to the design.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    def design(*columns: np.ndarray) -> Any:
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(2, include_bias=False),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 15)),
        )

    teff = base["TEFF"].to_numpy(dtype=float)
    logg = base["LOGG"].to_numpy(dtype=float)
    feh = base["FE_H"].to_numpy(dtype=float)
    usable = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh)
    parameters = np.column_stack([teff[usable], logg[usable]])
    both = np.column_stack([teff[usable], logg[usable], feh[usable]])

    rows: list[dict[str, object]] = []
    for arm, frame in frames.items():
        Z = frame.set_index("APOGEE_ID").loc[base.index].to_numpy(dtype=float)[usable]
        Z = (Z - Z.mean(axis=0)) / np.where(Z.std(axis=0) == 0, 1.0, Z.std(axis=0))
        total = float(np.sum(Z ** 2))

        def share(features: np.ndarray, Z: np.ndarray = Z, total: float = total) -> float:
            prediction = cross_val_predict(
                design(), features, Z, cv=KFold(5, shuffle=True, random_state=seed),
            )
            return 1.0 - float(np.sum((prediction - Z) ** 2)) / total

        without = share(parameters)
        with_chemistry = share(both)
        rows.append({
            "arm": arm,
            "dim": int(Z.shape[1]),
            "explained_by_Teff_logg": round(without, 4),
            "explained_by_Teff_logg_FeH": round(with_chemistry, 4),
            "chemistry_increment": round(with_chemistry - without, 4),
        })
    return pd.DataFrame(rows)


def solve() -> dict[str, object]:
    """The three probes on the common population."""
    base, frames = common_population()
    errors = regression_errors(base, frames)
    pivot = errors.pivot_table(
        index=["arm", "dim"], columns="target", values="rmse_over_sd",
    ).round(4)

    return {
        "n_stars": int(len(base)),
        "n_members": int((base["cluster"] != "field").sum()),
        "errors": errors,
        "errors_normalised": pivot,
        "chemistry_at_fixed_parameters": chemistry_at_fixed_parameters(base, frames),
        "capacity": capacity_share(base, frames),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Cross-validated error per arm, normalised by the target's own spread."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["errors"]
    assert isinstance(table, pd.DataFrame)

    arms = list(ARMS)
    targets = list(TARGETS)
    x = np.arange(len(targets))
    width = 0.8 / len(arms)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, arm in enumerate(arms):
        values = [
            float(table[(table["arm"] == arm) & (table["target"] == t)]["rmse_over_sd"].iloc[0])
            for t in targets
        ]
        ax.bar(x + (i - (len(arms) - 1) / 2) * width, values, width, label=arm)
    ax.set_xticks(x, targets)
    ax.set_ylabel("cross-validated RMSE / target s.d.")
    ax.set_title("What each latent recovers with a linear probe")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what was run": (
        "3 208 stars (804 of them cluster members) present in the catalogue "
        "and in all three arms. For each arm and each of Teff, logg and [Fe/H], "
        "a standardise-then-ridge pipeline with an internal 13-point alpha grid "
        f"({cite('Pedregosa:11', bare=True)}) is fitted with 5-fold "
        "cross-validation and the error is normalised by "
        "the target's own spread, so the three targets are comparable. The "
        f"three targets are the APOGEE pipeline's own labels "
        f"{cite('Majewski:17')}, which is what a data-driven spectral model "
        f"is normally trained to reproduce {cite('Ness:15', 'Ting:19')}."
    ),
    "the headline comparison": (
        f"Normalised RMSE (lower is better), masked AE {cite('He:22', bare=True)} "
        "/ PCA-64 / PCA-256: Teff "
        "0.358 / 0.444 / 0.291; logg 0.367 / 0.362 / 0.276; [Fe/H] 0.351 / "
        "0.437 / 0.341. In units: Teff 222 K / 275 K / 180 K, logg 0.409 / "
        "0.404 / 0.308 dex, [Fe/H] 0.166 / 0.207 / 0.162 dex. The masked "
        "latent beats PCA-64 on Teff by a clear margin and on [Fe/H] by a "
        "narrower one, and the two are level on logg. PCA-256 beats both "
        "everywhere — which is the control that matters, because it says the "
        "PCA deficit is a *dimension* effect as much as a representation "
        "effect."
    ),
    "chemistry at fixed atmospheric parameters": (
        "Regressing [Fe/H] on (Teff, logg) and probing the residual removes the "
        "shared trend: the residual's s.d. is 0.383 dex against 0.474 dex for "
        "raw [Fe/H], so about a third of the apparent metallicity spread is "
        "atmospheric. On that residual the masked latent reaches RMSE 0.146 "
        "dex (R² 0.856) and PCA-64 reaches 0.182 dex (R² 0.776) — a 20% "
        "advantage to the autoencoder on the quantity cluster separation "
        "actually uses. PCA-256 reaches 0.134 dex (R² 0.877), so again the "
        "masked latent is ahead of PCA-64 but behind the wider PCA."
    ),
    "which one spends more capacity on chemistry": (
        "The masked AE, but not by much, and the dimension control changes the "
        "conclusion. The share of each latent's variance explained by (Teff, "
        "logg) alone is 0.196 for the masked latent, 0.079 for PCA-64 and "
        "0.030 for PCA-256; adding [Fe/H] raises those to 0.236, 0.098 and "
        "0.039, so the chemistry increment is 0.040 / 0.020 / 0.010. In other "
        "words the *fraction* of capacity spent on chemistry is larger for the "
        "masked latent, but the *absolute* amount is larger in the masked "
        "latent too. Note the direction this cuts against the naive "
        "expectation: PCA's leading components are dominated by parameters "
        "(0.079 of variance is Teff+logg for the first 64 components), but its "
        "256-dimensional version explains *less* of the parameters, not more — "
        "the extra components are not continuum, they are noise, which is why "
        "PCA-256 is the best regressor here and the worst retrieval arm in "
        "§13."
    ),
    "the answer to the tie": (
        "On field retrieval the masked AE (256-d) and PCA-64 tie at "
        "0.418/0.619 and 0.417/0.631. On 'what does the latent know about the "
        "star', they do not tie: the autoencoder recovers more chemistry per "
        "dimension. The reconciliation is the one §14.4 states — if a linear "
        "64-dimensional projection reaches the same field-retrieval precision "
        "as an 8.6-million-parameter convolutional latent, the limit is not "
        "the model's capacity, it is the information the retrieval metric "
        "extracts. Both representations carry the chemistry; the metric throws "
        "most of it away, because a precision of 0.62 is set by how many field "
        "stars sit on top of the cluster in whatever space the clusterer "
        "builds, and the two latents put them in the same place."
    ),
    "the methodological caveat": (
        "Every number here is a *linear* probe, and the difference between the "
        "arms is a difference measured by a linear map. A representation can "
        "hold information a linear probe cannot reach — that is the case for "
        "adding layers to a decoder — so 'the masked latent encodes more "
        "chemistry' should be read as 'more chemistry is linearly decodable "
        "from it'. The comparison is still the right one for this workbook, "
        "because everything downstream (HDBSCAN, EVoC, the recovery fraction) "
        "operates on the latent without a learned decoder. And the probe is "
        "fitted on a population that is 75% field stars chosen by "
        "SNR-selected sampling, not a magnitude-complete sample, so the errors "
        "are conditional on being bright enough for APOGEE."
    ),
    "references": reference_list(
        "He:22", "Majewski:17", "Ness:15", "Ting:19", "Pedregosa:11",
    ),
}
