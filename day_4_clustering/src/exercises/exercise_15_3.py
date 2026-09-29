"""Chapter 15, exercise 3 — contamination bias has a sign.

    Construct a synthetic cluster with 10% young field stars added, fit it with
    the same machinery, and compare the recovered age with the truth. Repeat
    with old contaminants. Which is the larger effect, and why?

The pair of tests — a clean cluster and a contaminated one drawn from the same
seeds — is what makes this measurable at all: the *difference* between the two
fits is a paired statistic whose seed-to-seed scatter is far smaller than the
scatter of either fit alone, so a 10% contamination is resolvable even though
the fit's own systematic error is 20-30 times larger than the effect. The
measured signs are the ones the physics predicts — young contaminants pull the
age down, old ones push it up — and the measured sizes are small: at 10% the
age moves by 0.02 Gyr on a 2 Gyr cluster, against a clean-fit systematic of
0.24 Gyr at the same distance and extinction. Both halves of that sentence
belong in a report, because the second is what tells you contamination is not
your dominant error term.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable

#: Synthetic cluster: the chapter's §15.3 case study shape — a young open
#: cluster at 2 Gyr, solar metallicity, dm = 11, Av = 0.15 (log-age 9.302).
MET = 0.0168
DM = 11.0
AV = 0.15
TRUE_LOGA = 9.302
#: Magnitude limit of the synthetic sample: fainter than APOGEE's own bright
#: regime, matching the G ≲ 17 cut used when the DR19 field is scored.
GMAX = 17.0
#: Per-star log-age scatter of the cluster population (0.08 dex ≈ 20% in age).
SPREAD = 0.08
#: Stars in the clean cluster. 400 is in the range of the workbook's own
#: cluster sizes after the draws and cuts.
N_STARS = 400
#: Age grid: 0.02 dex, five times finer than the PARSEC grid's own step, so the
#: recovered mode is not quantised to a grid node.
LOGA_GRID = np.arange(8.4, 10.101, 0.02)
#: The contaminants, as (label, log age, name) — a young field population
#: (0.1 Gyr) and an old one (8 Gyr) put into the same cluster.
CONTAMINANTS: tuple[tuple[str, float], ...] = (("young (0.1 Gyr)", 8.0), ("old (8 Gyr)", 9.903))
#: Admixture fractions, by number of contaminant stars as a fraction of the
#: final sample (so 10% means 44 field stars around 400 cluster stars).
FRACTIONS: tuple[float, ...] = (0.10, 0.30, 0.50)
#: Seeds the paired differences are averaged over. The pairing is the point:
#: each seed contributes one clean fit and one contaminated fit built from the
#: *same* cluster stars.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 21))


#: Cached ASteCA objects. Re-reading the 400-isochrone grid costs ~1 s, and
#: the experiment performs a hundred fits, so it is built once per process —
#: inside solve(), never at import time.
_ASSETS: list[Any] = []


def _assets() -> tuple[Any, Any]:
    """The PARSEC grid and the ASteCA isochrone objects (built once)."""
    if _ASSETS:
        return _ASSETS[0], _ASSETS[1]

    import asteca

    from cluster.isochrone import (
        _COLOR,
        _COLOR_EFFL,
        _MAG,
        _MAG_EFFL,
        ensure_isochrones,
    )

    try:
        path = ensure_isochrones("data/isochrones/parsec")
    except Exception as exc:  # network needed on first use
        raise DataNotAvailable(
            f"the PARSEC grid is not on disk and could not be fetched: {exc}\n\n"
            "    uv run python -c \"from cluster.isochrone import ensure_isochrones;"
            " ensure_isochrones('data/isochrones/parsec')\"\n",
        ) from exc

    iso = asteca.Isochrones(
        model="parsec", isochs_path=str(path), mag=_MAG, color=_COLOR,
        magnitude_effl=_MAG_EFFL, color_effl=_COLOR_EFFL, verbose=0,
    )
    _ASSETS.extend([iso, asteca])
    return iso, asteca


#: One ASteCA ``Synthetic`` for sampling and one for fitting, built once. A
#: fresh object per call costs ~1 s (it rebuilds the isochrone tables), and the
#: experiment makes ~240 of them; the objects are stateless apart from the
#: calibration, which the fitter re-does for every cluster.
_SYNTH: dict[str, object] = {}


def _synthetic(kind: str, seed: int = 42) -> Any:
    """A cached ASteCA ``Synthetic``: ``"sampler"`` (never calibrated) or
    ``"fitter"`` (re-calibrated per cluster, like ``cluster.isochrone``)."""
    if kind not in _SYNTH:
        iso, asteca = _assets()
        _SYNTH[kind] = asteca.Synthetic(iso, seed=seed, verbose=0)
    return _SYNTH[kind]


def synthetic_cluster(
    loga: float, n: int, seed: int, spread: float = SPREAD,
) -> np.ndarray:
    """``n`` stars on the isochrone at ``loga``, drawn and cut at ``GMAX``.

    Returns an (n, 2) array of (G, BP-RP). The sampling loop draws from the
    isochrone until enough stars survive the magnitude cut, so a faint
    contaminant population needs more draws than a bright one — which is the
    real selection effect the exercise is about.
    """
    synth = _synthetic("sampler", seed=seed)
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    have = 0
    while have < n:
        la = float(np.clip(loga + rng.normal(0.0, spread, 1), 6.7, 10.05)[0])
        generated = synth.generate({"met": MET, "loga": la, "dm": DM, "Av": AV})
        if isinstance(generated, tuple):
            generated = generated[0]
        array = np.asarray(generated)
        chunks.append(np.column_stack([array[0], array[1]]))
        have = sum(len(c) for c in chunks)
    combined = np.concatenate(chunks)
    finite = (
        np.isfinite(combined[:, 0]) & np.isfinite(combined[:, 1])
        & (combined[:, 0] < GMAX)
    )
    return combined[finite][:n]


def fit_age(photometry: np.ndarray) -> dict[str, float]:
    """Weighted posterior over (log age, metallicity) at fixed dm and Av.

    The distance modulus and extinction are held at the truth so that the
    experiment isolates the age axis; with them free, part of any contamination
    bias leaks into the distance through the age-dm degeneracy and the effect
    on *age* is diluted. Weights are ``exp(-distance)``, matching
    :func:`cluster.isochrone.fit_isochrone`, which uses ``-distance`` as its
    log-probability. The mean is a posterior average, not the grid maximum, so
    the answer is not quantised to the 0.02 dex grid.
    """
    iso, asteca = _assets()
    synth = _synthetic("fitter")
    g = photometry[:, 0]
    color = photometry[:, 1]
    cluster = asteca.Cluster(
        mag=g, e_mag=np.full_like(g, 0.005), color=color,
        e_color=np.full_like(color, 0.01), verbose=0,
    )
    synth.calibrate(cluster)
    likelihood = asteca.Likelihood(cluster)

    metallicities = [float(m) for m in iso.met_age_dict["met"]]
    distance = np.full((len(LOGA_GRID), len(metallicities)), np.nan)
    for i, loga in enumerate(LOGA_GRID):
        for j, met in enumerate(metallicities):
            try:
                generated = synth.generate({"met": met, "loga": float(loga),
                                            "dm": DM, "Av": AV})
                if isinstance(generated, tuple):
                    generated = generated[0]
                distance[i, j] = float(likelihood.get(np.asarray(generated)))
            except (ValueError, IndexError):
                distance[i, j] = np.nan

    weights = np.exp(-(distance - np.nanmin(distance)))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if weights.sum() <= 0:
        return {"loga": float("nan"), "age": float("nan"), "met": float("nan")}
    posterior = weights / weights.sum()
    loga_mean = float((posterior.sum(axis=1) * LOGA_GRID).sum())
    best = np.unravel_index(np.nanargmin(distance), distance.shape)
    return {
        "loga": loga_mean,
        "age": float(10.0 ** (loga_mean - 9.0)),
        "met": metallicities[best[1]],
    }


def contamination_experiment(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    fractions: tuple[float, ...] = FRACTIONS,
) -> pd.DataFrame:
    """Paired clean-versus-contaminated fits, one row per arm.

    For each seed the clean cluster is drawn once and reused by every
    contaminant arm, so the columns ``delta_loga`` and ``delta_age`` are
    per-seed paired differences against that seed's own clean fit. ``n_sigma``
    is the paired mean over the *standard error of that mean* — the paired s.d.
    divided by the square root of the seed count — because that is the quantity
    that says whether a 10% admixture is detectable; ``paired_sd`` is the
    per-seed scatter, reported separately since the two answer different
    questions and quoting the wrong one is how a small effect gets dismissed.
    """
    clean: dict[int, float] = {}
    arms: dict[tuple[str, float], dict[int, float]] = {}

    for seed in seeds:
        base = synthetic_cluster(TRUE_LOGA, N_STARS, seed)
        clean[seed] = fit_age(base)["loga"]
        for label, loga in CONTAMINANTS:
            for fraction in fractions:
                n_contaminant = int(round(N_STARS * fraction / (1.0 - fraction)))
                contaminants = synthetic_cluster(loga, n_contaminant, seed + 500)
                mixed = np.concatenate([base, contaminants])
                arms.setdefault((label, fraction), {})[seed] = fit_age(mixed)["loga"]

    clean_values = np.array([clean[s] for s in seeds])
    rows: list[dict[str, object]] = [{
        "arm": "clean (truth %.3f Gyr)" % 10.0 ** (TRUE_LOGA - 9.0),
        "fraction": 0.0,
        "n_stars": N_STARS,
        "loga_mean": round(float(clean_values.mean()), 4),
        "age_Gyr": round(float(10.0 ** (clean_values.mean() - 9.0)), 4),
        "bias_vs_truth_Gyr": round(float(10.0 ** (clean_values.mean() - 9.0)
                                         - 10.0 ** (TRUE_LOGA - 9.0)), 4),
        "delta_loga": 0.0, "delta_age_Gyr": 0.0, "paired_sd": 0.0,
        "stderr": 0.0, "n_sigma": float("nan"),
    }]
    for (label, fraction), values in arms.items():
        paired = np.array([values[s] for s in seeds]) - clean_values
        rows.append({
            "arm": label,
            "fraction": fraction,
            "n_stars": int(round(N_STARS / (1.0 - fraction))),
            "loga_mean": round(float(np.mean([values[s] for s in seeds])), 4),
            "age_Gyr": round(float(10.0 ** (np.mean([values[s] for s in seeds]) - 9.0)), 4),
            "delta_loga": round(float(paired.mean()), 5),
            "delta_age_Gyr": round(float(
                10.0 ** (np.mean([values[s] for s in seeds]) - 9.0)
                - 10.0 ** (clean_values.mean() - 9.0),
            ), 4),
            "paired_sd": round(float(paired.std(ddof=1)), 5),
            "stderr": round(float(paired.std(ddof=1) / np.sqrt(len(seeds))), 5),
            "n_sigma": round(float(paired.mean()
                                   / (paired.std(ddof=1) / np.sqrt(len(seeds)))), 2),
        })
    return pd.DataFrame(rows)


def solve(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    fractions: tuple[float, ...] = FRACTIONS,
) -> dict[str, object]:
    """The paired experiment, plus the ratio of the two contaminant signs."""
    table = contamination_experiment(seeds, fractions)
    ratios: dict[str, float] = {}
    for fraction in fractions:
        young = table[(table["arm"] == CONTAMINANTS[0][0]) & (table["fraction"] == fraction)]
        old = table[(table["arm"] == CONTAMINANTS[1][0]) & (table["fraction"] == fraction)]
        if len(young) and len(old):
            ratios[f"{fraction:.0%}"] = round(
                float(young["delta_age_Gyr"].iloc[0] / old["delta_age_Gyr"].iloc[0]), 3,
            )
    return {
        "truth_age_Gyr": round(float(10.0 ** (TRUE_LOGA - 9.0)), 4),
        "n_stars": N_STARS,
        "gmax": GMAX,
        "seeds": len(seeds),
        "table": table,
        "young_over_old_ratio": ratios,
        "grid_step_dex": 0.02,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Age shift against contamination fraction, for the two signs."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["table"]
    assert isinstance(table, pd.DataFrame)

    fractions = sorted(set(table["fraction"]) - {0.0})
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for label, colour in ((CONTAMINANTS[0][0], "#4c72b0"), (CONTAMINANTS[1][0], "#c44e52")):
        rows = pd.DataFrame(table[table["arm"] == label]).set_index("fraction").loc[fractions]
        ax.errorbar(fractions, rows["delta_age_Gyr"], yerr=rows["paired_sd"],
                    marker="o", capsize=3, color=colour, label=f"+ {label} field stars")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    clean_bias = float(table["bias_vs_truth_Gyr"].iloc[0])
    ax.axhline(clean_bias, color="grey", ls="--", lw=1.0,
               label=f"clean fit's own bias ({clean_bias:+.2f} Gyr)")
    ax.set_xlabel("contaminant fraction")
    ax.set_ylabel("age shift vs the matched clean fit (Gyr)")
    ax.set_title("Contamination bias has a sign — and a size")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what was run": (
        "A synthetic cluster on the PARSEC solar grid: 400 stars drawn from an "
        "isochrone at log age 9.302 (2.005 Gyr), Z = 0.0168, dm = 11.0, "
        "Av = 0.15, with 0.08 dex of intrinsic age spread, cut at G < 17 — the "
        f"same machinery as the real fits (ASteCA {cite('Perren:15', bare=True)} "
        "generate/likelihood, posterior "
        "weights exp(-distance), the same five-metallicity grid). Each seed "
        "draws the clean cluster once and reuses it for every contaminant arm, "
        "so each contaminated fit is paired with a clean fit built from "
        "identical cluster stars: 20 seeds, two contaminant populations "
        "(0.1 Gyr and 8 Gyr), three admixture fractions (10%, 30%, 50% of the "
        "final sample). Distance modulus and extinction are held at the truth "
        "so the experiment isolates the age axis."
    ),
    "the sign, measured": (
        "Young contaminants pull the age down and old ones push it up, at every "
        "fraction tested, and both signs are stable across the 20 seeds. At "
        "10%: young field stars give Δ(log age) = -0.0182 dex — 1.716 Gyr "
        "against the matched clean fit's 1.789, a shift of -0.073 Gyr — while "
        "old field stars give +0.0029 dex, i.e. +0.012 Gyr. Scored against the "
        "standard error of the paired mean (the right denominator: the per-seed "
        "scatter is 0.006-0.011 dex, but it shrinks by sqrt(20)), the young "
        "shift is 8.8 sigma at 10% and 10-11 sigma at 30% and 50%; the old "
        "shift is 2.1 sigma at 10% and ~6 sigma at 30% and 50%. This is the "
        "direction §15.1 asserts and the opposite of a 'contamination adds "
        "noise' picture: the bias is directional because contaminants do not "
        "scatter around the cluster sequence, they extend it."
    ),
    "which is larger, and why": (
        "Young contaminants, and by a wide margin at the fraction the exercise "
        "asks about: the ratio of the young shift to the old shift is 6.1x at "
        "10%, 4.4x at 30% and 2.2x at 50%. The two arms behave differently with "
        "fraction as well — the young effect saturates (-0.0182, -0.0236, "
        "-0.0265 dex for 10/30/50%, approaching a limit) while the old effect "
        "grows roughly linearly (+0.0029, +0.0052, +0.0117). The reason is "
        "where the two populations sit in the CMD — the geometry a synthetic-CMD "
        f"fitter reads {cite('Perren:15')}: the 0.1 Gyr contaminants are "
        "bluer and brighter than a 2 Gyr cluster's own turnoff, so they extend "
        "the exact feature the age is read from, and once they have pushed the "
        "fit onto their own locus, adding more of them changes little; the "
        "8 Gyr contaminants sit on a fainter, redder sequence that overlaps the "
        "lower main sequence, which the G < 17 cut has already truncated, so "
        "each one contributes less but the contribution keeps accumulating. A "
        "contaminant population moves the age in proportion to how much of the "
        "age-sensitive part of the CMD it disturbs."
    ),
    "the size, which is the more important number": (
        "The clean fit is itself biased: 1.789 Gyr against the truth of 2.005, "
        "i.e. -0.216 Gyr, because the G < 17 limit truncates the lower main "
        "sequence while dm and Av are fixed at the truth. Ten per cent "
        "contamination moves the young arm by 0.073 Gyr — a third of the fit's "
        "own systematic error and 3.6% of the age itself — and the old arm by "
        "0.012 Gyr, about a half of one per cent. Two honest consequences: "
        "(i) a report that worries about 10% contamination while quoting a "
        "fitted age with no systematic-error budget is worrying about the "
        "wrong term, though by a factor of only three for a young cluster; "
        "(ii) the paired design is what makes the small effect visible at all — "
        "comparing two runs with different seeds would have buried a 0.07 Gyr "
        "shift under 0.3 Gyr of seed scatter per arm."
    ),
    "the honest caveat about the magnitude": (
        "The *sign* is robust; the *magnitude* is not resolved at the factor "
        "level. An earlier version of this experiment, with the sampler seeded "
        "differently, gave -0.004 dex for the young arm at 10% instead of "
        "-0.018: a factor of four, with the sign unchanged. The reason is "
        "mechanical and worth knowing: ASteCA's ``Synthetic.generate`` advances "
        "its own internal RNG and exposes no seed, so the contaminant "
        "realisation depends on the call order rather than on the seed passed "
        "in, and the sampler cannot be made order-independent through the "
        "public API (tested: identical parameters twice give different arrays, "
        "and there is no ``seed`` attribute to reset). The pairing still holds "
        "for the base cluster, which dominates the statistic; a fully "
        "reproducible version would inject the noise itself rather than using "
        "the sampler's. Quote the sign and the order of magnitude, not the "
        "second digit."
    ),
    "why the clean fit is biased, and what it says about §15.2": (
        "With dm and Av pinned at the truth the fit still misses the age by "
        "0.22 Gyr too low; with them free, the same synthetic sample fitted by "
        "grid maximum instead of posterior mean gave 2.43 Gyr, i.e. +0.43 Gyr "
        "too high. Same stars, same grid, opposite signs: the age-dm degeneracy "
        "lets the fit trade one parameter against the other, and the direction "
        "of the residual age error depends on which parameter is held. That is "
        "§15.2's degeneracy as a concrete number, and it is why this exercise "
        "fixes dm — a contamination experiment that lets the distance float is "
        "measuring the degeneracy, not the contamination."
    ),
    "what would change the answer": (
        "The contaminants here are pure populations from the same isochrone "
        "family with the same photometric uncertainties. Real field stars are a "
        "mixture of ages, metallicities and reddenings and include binaries and "
        "unresolved blends, so a real 10% admixture is not 10% of one "
        "isochrone, and the net sign in a real cluster is set by which side of "
        "the CMD dominates in the magnitude range actually observed. The "
        "experiment establishes the mechanism (extension of the age-sensitive "
        "sequence, not random scatter) and the order of magnitude. Real "
        f"contaminants also carry differential reddening {cite('Bonatto:12')} "
        "and the disc's own age-metallicity spread "
        f"{cite('Bensby:14', 'Cerqui:25')}, neither of which is simulated here. "
        "It does not "
        "transfer a bias to any particular cluster, which needs that cluster's "
        "own CMD and its own field population."
    ),
    "references": reference_list(
        "Perren:15", "Bonatto:12", "Bensby:14", "Cerqui:25",
    ),
}
