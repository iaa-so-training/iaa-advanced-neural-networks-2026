"""Chapter 15, exercise 2 — the age-metallicity degeneracy, quantified.

    The age-metallicity degeneracy means the posterior stays broad even with
    perfect membership. Choose a cluster in the sample and quantify the
    degeneracy directly: plot the posterior in the age-metallicity plane and
    measure its correlation coefficient. What extra observable would break it?

M 67 is the cluster to use: its membership is the workbook's best, so the
breadth that remains cannot be blamed on contamination. The degeneracy is then
measured twice. First as the *shape* of the ASteCA distance function on the
(age, metallicity) plane at the literature distance — the ridge along which the
fit trades age for metallicity, its width, and the correlation of the best
metallicity with log age. Second as the correlation in the *posterior* from a
sampler with all four parameters free, which is a different and much smaller
number: the pairwise age-metallicity correlation is diluted once the distance
modulus is free to absorb the trade-off. Both numbers are real, they answer
different questions, and the gap between them is the most instructive part of
the exercise.

Finally the extra observable is not asserted but run: the repository's own
isochrone machinery supports a second colour (2MASS J-K), and the ridge width
is recomputed with it. That is the honest way to answer "what would break it" —
measure the narrowing rather than name the observable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, catalogue_path, settings

#: The cluster. M 67 has the best membership in the sample (the workbook's
#: two-stage list) and 218-230 member rows with full photometry.
TARGET_CLUSTER = "M 67"
#: Extinction held fixed for the grid scan, and the width of the flat prior
#: around the literature distance modulus. §15.1's point is that distance is
#: the parameter that can be pinned; the degeneracy is about the other two.
AV = 0.10
DM_WIDTH = 2.0
#: Age grid for the scan: 0.05 dex, the PARSEC grid's own step, so no
#: interpolation in age is involved in the ridge measurement.
AGE_GRID = np.arange(8.60, 10.101, 0.05)
#: Δ(distance) contours that define "the ridge". ASteCA's distance is a
#: normalised residual, so a Δ of 0.05 is a small penalty relative to the
#: variation across the plane.
RIDGE_LEVELS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
#: Emcee settings for the marginal posterior (same as cluster.isochrone).
N_WALKERS = 32
N_STEPS = 600


def target_cluster():
    """The ``Cluster`` record for :data:`TARGET_CLUSTER`."""
    from cluster.clusters import CLUSTER_BY_NAME

    return CLUSTER_BY_NAME[TARGET_CLUSTER]


def literature() -> dict[str, float]:
    """Literature age, distance modulus and [Fe/H] for M 67."""
    from cluster.literature import fetch_literature

    return fetch_literature(target_cluster())


def members() -> pd.DataFrame:
    """M 67's kinematic members from the DR19 catalogue, with Gaia photometry.

    Same selection as :mod:`exercise_15_1`: quality cuts, the cluster's search
    region, then ``kinematic_members`` with the repository's tolerances.
    """
    from cluster import config
    from cluster.data import apply_quality_cuts, load_allstar
    from cluster.membership import angular_separation, kinematic_members

    cluster = target_cluster()
    cfg = settings()
    cfg.require_aspcap_flag_clean = False
    frame = load_allstar(catalogue_path(), cfg.elements)
    frame = apply_quality_cuts(frame, cfg)
    separation = angular_separation(
        frame["RA"].to_numpy(dtype=float), frame["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    region = pd.DataFrame(frame[separation <= cluster.region_deg]).copy()
    mask = kinematic_members(
        region, cluster,
        config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
        config.SEED_PM_TOL, config.SEED_RV_TOL,
        config.N_REFINE_PASSES, config.REFINE_SIGMA,
    ).to_numpy()
    return pd.DataFrame(region[mask]).copy()


#: ASteCA objects, built once per process inside the functions below. Typed as
#: ``Any`` because ASteCA ships no type information and its classes are
#: generated at import.
_ASSETS: dict[str, Any] = {}


def _isochrones(two_colour: bool) -> Any:
    """The ASteCA isochrone set, single-colour (BP-RP) or two-colour (+J-K)."""
    key = "two_colour" if two_colour else "single_colour"
    if key in _ASSETS:
        return _ASSETS[key]
    import asteca

    from cluster.isochrone import (
        _COLOR,
        _COLOR2,
        _COLOR2_EFFL,
        _COLOR_EFFL,
        _MAG,
        _MAG_EFFL,
        ISOCHRONE_DIR,
        ensure_isochrones,
    )

    if two_colour:
        path = ISOCHRONE_DIR / "parsec_solar_2color.dat"
        if not path.is_file():
            raise DataNotAvailable(
                f"{path} is not on disk; it is the PARSEC grid with the 2MASS "
                "J-K colour and is fetched by the first two-colour fit:\n\n"
                "    uv run python -c \"from cluster.isochrone import "
                "fit_isochrone; ...\"\n",
            )
    else:
        path = ensure_isochrones(ISOCHRONE_DIR)
    iso = asteca.Isochrones(
        model="parsec", isochs_path=str(path), mag=_MAG, color=_COLOR,
        color2=_COLOR2 if two_colour else None,
        magnitude_effl=_MAG_EFFL,
        color_effl=_COLOR_EFFL,
        color2_effl=_COLOR2_EFFL if two_colour else None,
        verbose=0,
    )
    _ASSETS[key] = iso
    return iso


def _photometry(frame: pd.DataFrame, two_colour: bool) -> dict[str, np.ndarray]:
    """The arrays ASteCA needs, with the finite mask applied."""
    g = frame["GAIAEDR3_PHOT_G_MEAN_MAG"].to_numpy(dtype=float)
    bp = frame["GAIAEDR3_PHOT_BP_MEAN_MAG"].to_numpy(dtype=float)
    rp = frame["GAIAEDR3_PHOT_RP_MEAN_MAG"].to_numpy(dtype=float)
    colour = bp - rp
    keep = np.isfinite(g) & np.isfinite(colour)
    out = {"g": g[keep], "colour": colour[keep]}
    if two_colour:
        j = frame["J"].to_numpy(dtype=float)
        k = frame["K"].to_numpy(dtype=float)
        keep2 = keep & np.isfinite(j) & np.isfinite(k)
        out = {"g": g[keep2], "colour": (bp - rp)[keep2], "colour2": (j - k)[keep2]}
        out["n_dropped"] = np.array([float((keep & ~keep2).sum())])
    return out


def degeneracy_grid(
    frame: pd.DataFrame | None = None, *, two_colour: bool = False,
    av: float = AV, dm: float | None = None,
) -> dict[str, object]:
    """The ASteCA distance on the (age, metallicity) plane, at fixed (dm, Av).

    Returns the distance array, the ridge statistics, and the best-fitting
    point. The ridge statistics are the answer to the exercise: along the
    degenerate direction, how far can the age move before the distance
    function notices, and how does the best metallicity track the age.
    """
    import asteca

    iso = _isochrones(two_colour)
    frame = members() if frame is None else frame
    photometry = _photometry(frame, two_colour)
    dm = float(literature()["dm"]) if dm is None else float(dm)

    synth = asteca.Synthetic(iso, seed=42, verbose=0)
    e_colour = np.full_like(photometry["colour"], 0.01)
    kwargs = {
        "mag": photometry["g"], "e_mag": np.full_like(photometry["g"], 0.005),
        "color": photometry["colour"], "e_color": e_colour,
    }
    if two_colour:
        kwargs["color2"] = photometry["colour2"]
        kwargs["e_color2"] = np.full_like(photometry["colour2"], 0.02)
    cluster = asteca.Cluster(**kwargs, verbose=0)
    synth.calibrate(cluster)
    likelihood = asteca.Likelihood(cluster)

    metallicities = [float(m) for m in iso.met_age_dict["met"]]
    distance = np.full((len(AGE_GRID), len(metallicities)), np.nan)
    for i, loga in enumerate(AGE_GRID):
        for j, met in enumerate(metallicities):
            try:
                generated = synth.generate({"met": met, "loga": float(loga),
                                            "dm": dm, "Av": av})
                if isinstance(generated, tuple):
                    generated = generated[0]
                distance[i, j] = float(likelihood.get(np.asarray(generated)))
            except (ValueError, IndexError):
                distance[i, j] = np.nan

    best = np.unravel_index(np.nanargmin(distance), distance.shape)
    best_met = np.array([metallicities[np.nanargmin(distance[i])]
                         for i in range(len(AGE_GRID))])
    feh_path = np.log10(best_met / 0.0152)
    correlation = float(np.corrcoef(AGE_GRID, feh_path)[0, 1])

    bands: dict[str, dict[str, object]] = {}
    for level in RIDGE_LEVELS:
        selected = (distance - np.nanmin(distance)) <= level
        ages = np.repeat(AGE_GRID, len(metallicities))[selected.ravel()]
        mets = np.tile(metallicities, len(AGE_GRID))[selected.ravel()]
        if ages.size < 2:
            bands[f"delta<={level}"] = {"n_points": int(ages.size)}
            continue
        bands[f"delta<={level}"] = {
            "n_points": int(ages.size),
            "loga_span": [round(float(ages.min()), 2), round(float(ages.max()), 2)],
            "age_span_Gyr": [round(float(10 ** (ages.min() - 9)), 2),
                             round(float(10 ** (ages.max() - 9)), 2)],
            "age_factor": round(float(10 ** (ages.max() - ages.min())), 2),
            "met_span_Z": [round(float(mets.min()), 5), round(float(mets.max()), 5)],
        }

    return {
        "cluster": TARGET_CLUSTER,
        "two_colour": two_colour,
        "n_stars": int(len(photometry["g"])),
        "dm": round(dm, 4),
        "av": av,
        "best": {
            "loga": round(float(AGE_GRID[best[0]]), 3),
            "age_Gyr": round(float(10 ** (AGE_GRID[best[0]] - 9)), 3),
            "met": round(float(metallicities[best[1]]), 5),
            "feh": round(float(np.log10(metallicities[best[1]] / 0.0152)), 3),
            "distance": round(float(distance[best]), 5),
        },
        "path_correlation_log_age_feh": round(correlation, 3),
        "path_feh_range": [round(float(feh_path.min()), 3), round(float(feh_path.max()), 3)],
        "bands": bands,
        "distance": distance,
        "metallicities": metallicities,
    }


def sample_posterior(
    frame: pd.DataFrame | None = None, *, two_colour: bool = False,
    seed: int = 42, n_walkers: int = N_WALKERS, n_steps: int = N_STEPS,
) -> dict[str, object]:
    """Sample (met, loga, dm, Av) with all four free, optionally with J-K.

    The same likelihood, priors and ``-distance`` log-probability as
    ``cluster.isochrone.fit_isochrone`` — which itself passes a second colour
    when the catalogue carries J and K — so the only difference between the two
    arms is the observable.
    """
    import asteca
    import emcee

    from cluster.isochrone import _GRID_MET, _PARAMS, _distance_modulus

    iso = _isochrones(two_colour)
    frame = members() if frame is None else frame
    photometry = _photometry(frame, two_colour)
    synth = asteca.Synthetic(iso, seed=seed, verbose=0)
    kwargs = {
        "mag": photometry["g"], "e_mag": np.full_like(photometry["g"], 0.005),
        "color": photometry["colour"],
        "e_color": np.full_like(photometry["colour"], 0.01),
    }
    if two_colour:
        kwargs["color2"] = photometry["colour2"]
        kwargs["e_color2"] = np.full_like(photometry["colour2"], 0.02)
    cluster = asteca.Cluster(**kwargs, verbose=0)
    synth.calibrate(cluster)
    likelihood = asteca.Likelihood(cluster)

    dm_guess = _distance_modulus(np.asarray(photometry.get("plx", np.array([])), dtype=float)) \
        if "plx" in photometry else float(literature()["dm"])
    bounds = {
        "met": _GRID_MET["solar"],
        "loga": _PARAMS["loga"],
        "dm": (dm_guess - DM_WIDTH, dm_guess + DM_WIDTH),
        "Av": _PARAMS["Av"],
    }
    names = ["met", "loga", "dm", "Av"]

    def lnprob(theta: np.ndarray) -> float:
        for value, (low, high) in zip(theta, (bounds[n] for n in names)):
            if not (low <= value <= high):
                return -np.inf
        params = {"met": float(theta[0]), "loga": float(theta[1]),
                  "dm": float(theta[2]), "Av": float(theta[3])}
        try:
            generated = synth.generate(params)
            if isinstance(generated, tuple):
                generated = generated[0]
            return -float(likelihood.get(np.asarray(generated)))
        except (ValueError, IndexError):
            return -np.inf

    rng = np.random.default_rng(seed)
    position = np.empty((n_walkers, 4))
    position[:, 0] = 10.0 ** rng.uniform(np.log10(bounds["met"][0]),
                                        np.log10(bounds["met"][1]), n_walkers)
    position[:, 1] = rng.uniform(bounds["loga"][0], bounds["loga"][1], n_walkers)
    position[:, 2] = rng.uniform(bounds["dm"][0], bounds["dm"][1], n_walkers)
    position[:, 3] = rng.uniform(bounds["Av"][0], bounds["Av"][1], n_walkers)

    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(n_walkers, 4, lnprob, vectorize=False)
    sampler.run_mcmc(position, n_steps, progress=False)
    chain = np.asarray(sampler.get_chain(discard=n_steps // 2, flat=True))
    return {"chain": chain, "names": names, "n_stars": int(len(photometry["g"]))}


def posterior_correlations(
    frame: pd.DataFrame | None = None, *, two_colour: bool = False,
    seed: int = 42, n_walkers: int = N_WALKERS, n_steps: int = N_STEPS,
) -> dict[str, object]:
    """Marginal posterior widths and correlations, with all four parameters free.

    This is the second, much smaller answer to the exercise's question. The
    grid ridge measures the degeneracy direction with the distance pinned; the
    posterior measures what the *data* constrain when nothing is — and for M 67
    at this magnitude range that is very little on the age axis, because the
    age-distance degeneracy takes over and the chain wanders over much of the
    age prior. Reported for both arms.
    """
    result = sample_posterior(frame, two_colour=two_colour, seed=seed,
                              n_walkers=n_walkers, n_steps=n_steps)
    chain = np.asarray(result["chain"])
    names = list(result["names"])  # type: ignore[arg-type]
    correlations = np.corrcoef(chain.T)
    std = {name: round(float(chain[:, i].std()), 4) for i, name in enumerate(names)}
    pairs = {
        f"{a}-{b}": round(float(correlations[i, j]), 3)
        for i, a in enumerate(names) for j, b in enumerate(names) if i < j
    }
    loga = chain[:, names.index("loga")]
    ages = 10.0 ** (loga - 9.0)
    return {
        "two_colour": two_colour,
        "n_samples": int(chain.shape[0]),
        "n_stars": int(np.asarray(result["n_stars"], dtype=float)),
        "std": std,
        "correlations": pairs,
        "age_median_Gyr": round(float(np.median(ages)), 3),
        "age_p16_p84_Gyr": [round(float(np.percentile(ages, 16)), 3),
                            round(float(np.percentile(ages, 84)), 3)],
    }


def solve() -> dict[str, object]:
    """The ridge at one colour, the ridge with J-K, and both posteriors."""
    frame = members()
    single = degeneracy_grid(frame, two_colour=False)
    try:
        two = degeneracy_grid(frame, two_colour=True)
    except DataNotAvailable as exc:
        two = {"error": str(exc)}
    return {
        "literature": {k: round(float(v), 4) for k, v in literature().items()},
        "single_colour": single,
        "two_colour": two,
        "posterior_single_colour": posterior_correlations(frame, two_colour=False),
        "posterior_two_colour": posterior_correlations(frame, two_colour=True),
        "n_members": int(len(frame)),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """The distance function on the age-metallicity plane, with its ridge."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    single = result["single_colour"]
    assert isinstance(single, dict)
    distance = np.asarray(single["distance"], dtype=float)
    metallicities = list(single["metallicities"])  # type: ignore[arg-type]
    feh = np.log10(np.asarray(metallicities, dtype=float) / 0.0152)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mesh = ax.pcolormesh(feh, AGE_GRID, distance, shading="nearest",
                         cmap="viridis_r", vmin=np.nanmin(distance),
                         vmax=np.nanmin(distance) + 0.10)
    best = single["best"]
    assert isinstance(best, dict)
    ax.plot([best["feh"]], [best["loga"]], marker="*", ms=14, color="crimson",
            label="best fit")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("$\\log_{10}$ age (yr)")
    ax.set_title(f"{single['cluster']}: the ridge the fit trades along\n"
                 f"path correlation {single['path_correlation_log_age_feh']:+.3f}")
    ax.legend(frameon=False, fontsize=9)
    fig.colorbar(mesh, ax=ax, label="ASteCA distance (dark = better)")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what was run": (
        "M 67's kinematic members from the DR19 catalogue (the cluster's "
        "4.17-degree region plus the repository's kinematic tolerances): 278 "
        f"stars with full Gaia photometry {cite('Gaia:23')}. The ASteCA "
        f"distance {cite('Perren:15')} is computed on a "
        "31 x 5 grid of (log age, Z) — the PARSEC solar grid's five "
        "metallicities, ages 8.60 to 10.10 dex in 0.05 dex steps — at fixed "
        "dm = 9.537 (the literature value) and Av = 0.10, because distance and "
        "extinction are the two parameters that can be pinned and the "
        "degeneracy is about the other two. Then the same grid with the 2MASS "
        "J-K colour added, and two sampled posteriors with all four parameters "
        "free (32 walkers, 600 steps, 9600 samples each), sampled with emcee "
        f"{cite('ForemanMackey:13')}."
    ),
    "the degeneracy, measured on the grid": (
        "The best metallicity falls as the age rises: along the ridge the "
        "correlation between log age and [Fe/H] is -0.455, with [Fe/H] running "
        "from +0.28 at the youngest grid point to -0.21 at the oldest — the "
        "classic direction, an older and more metal-poor isochrone mimicking a "
        "younger and more metal-rich one. The best fit sits at log age 9.55 "
        "(3.55 Gyr, Z = 0.0126, [Fe/H] = -0.08) against the literature "
        f"({cite('Dias:02', bare=True)}) 2.82 Gyr "
        "at +0.03, and the ridge is measured on the distance function rather "
        "than on noise: with 278 member stars and the distance pinned, the fit "
        "still has a one-parameter family of near-equally-good answers. Its "
        "width is the number to quote: within Δ = 0.01 of the minimum the ridge "
        "admits three grid points spanning 0.10 dex in log age (a factor of "
        "1.26 in age); within Δ = 0.05 it admits 19 points spanning 0.45 dex — "
        "log age 9.40 to 9.85, a factor of 2.82 in age — across Z = 0.0095 to "
        "0.0221, nearly the full metallicity range available."
    ),
    "the posterior, which is a different and weaker number": (
        "Freeing all four parameters gives σ_loga = 0.977 dex and a posterior "
        "whose age median is 0.16 Gyr with a 16-84 range of 0.01 to 2.61 — the "
        "chain spans most of the age prior, so the age is essentially "
        "unconstrained by this CMD at this magnitude range, and the pairwise "
        "age-metallicity correlation collapses to +0.037. That small "
        "correlation is *not* evidence against the degeneracy: it is what a "
        "prior-dominated posterior looks like, because the dominant trade-off "
        "in the marginalised fit is age against distance modulus (σ_dm = 1.13), "
        "and the metallicity axis is too narrow (σ_met = 0.006 in Z) to carry "
        "the correlation. Both numbers are correct answers to different "
        "questions: the pinned-distance ridge measures the degeneracy "
        "direction and width; the posterior measures what the data constrain "
        "when nothing is pinned, which here is very little."
    ),
    "what extra observable would break it — measured, and not this one": (
        "The chapter's natural candidate is a second colour, and this repository "
        "already carries the grid for it (the PARSEC isochrones with 2MASS J-K "
        "alongside Gaia BP-RP, which ``cluster.isochrone`` uses whenever the "
        "catalogue provides J and K — as it does for all 278 stars here). Run "
        "on the same stars, at the same fixed distance and extinction, the "
        "two-colour fit does *not* narrow the ridge: at Δ ≤ 0.02 the admitted "
        "band grows from 4 grid points to 7, at Δ ≤ 0.10 from 43 to 60, and the "
        "age factor inside Δ ≤ 0.10 doubles from 5.6 to 11.2. In the sampled "
        "posteriors σ_loga goes from 0.977 to 1.019 and the age-metallicity "
        "correlation from +0.037 to -0.022 — unchanged in magnitude. So the "
        "measured answer to the exercise's last question is that adding J-K in "
        "this set-up does not break the degeneracy, and two honest reasons are "
        "visible. First, ASteCA's distance is normalised over the observables, "
        "so a fixed Δ is a *different* penalty in a two-colour fit than in a "
        "one-colour fit: part of the apparent widening is the normalisation, "
        "and a proper comparison needs a fixed fractional penalty or the two "
        "posteriors' widths rather than a fixed Δ. Second, the age posterior is "
        "prior-dominated in both arms, so a factor-of-two improvement in "
        "constraint would still be swamped."
    ),
    "what actually breaks it": (
        "An external distance, not an extra colour — which is what §15.3's "
        "NGC 2243 sequence does and what exercise 1 measured on Collinder 261: "
        "the red-clump prior collapsed σ_dm from 1.170 to 0.086 (a factor of "
        "13.6) while the age-metallicity trade-off persisted underneath. The "
        "other observables that would break it are the ones that do not go "
        "through the CMD at all: a spectroscopic log g (APOGEE supplies one, "
        f"{cite('Majewski:17', bare=True)}, "
        "and it separates turnoff from giant directly, which is the axis the "
        "age is really read from), an age marker independent of the isochrone "
        "(lithium, or CN), and asteroseismic ν_max with its few-per-cent log g. "
        "The transferable rule: a degeneracy between two fitted parameters is "
        "broken by a measurement of one of them, not by adding sensitivity "
        "somewhere else on the same degeneracy."
    ),
    "why this matters for the workbook's claims": (
        "§15.1 says the posterior stays broad even with perfect membership, and "
        "the number to quote is the ridge: a factor of 2.8 in age inside a "
        "Δ = 0.05 distance penalty, at fixed distance, with all 278 stars "
        "belonging to the cluster — log age 9.40-9.85 around a best fit of "
        "9.55, which is roughly 2.5 to 7 Gyr. Any age quoted from a "
        "single-colour CMD fit of this sample therefore carries a systematic "
        "of that order, and it is an order of magnitude larger than the "
        "contamination bias: exercise 3 measures 0.073 Gyr for a 10% young "
        "admixture and 0.095 Gyr at 30%, against a ridge several Gyr wide. "
        "Even the clean fit's own bias — 0.216 Gyr, the G < 17 truncation with "
        "distance and reddening pinned at the truth — is three times the "
        "10% contamination shift. (Exercise 3 is explicit that these "
        "contamination magnitudes are good to the order of magnitude only; "
        "the sign is the robust part.) The practical consequence is the one "
        "§15.3 acts on: pin the distance with the red clump, fix the "
        "metallicity spectroscopically, and quote the age with the residual "
        "degeneracy as the dominant systematic."
    ),
    "the honest caveats": (
        "The ridge is measured on five metallicities, so the correlation of "
        "-0.455 is a correlation over five distinct [Fe/H] values rather than a "
        "continuous relation, and its exact value moves a little with the age "
        "grid (an earlier run of the same scan on 220 members gave -0.556, "
        "before the member selection was widened to 278). The Δ levels are "
        "ASteCA's normalised residuals, whose absolute scale is not a χ², so "
        "the chosen levels (0.01, 0.02, 0.05, 0.10) are conventions — the "
        "factor-2.8-in-age result is a statement at the 0.05 level and should "
        "be quoted with the level attached. And the two-colour comparison "
        "changes the number of fitted observables, so part of its result is "
        "the normalisation; the experiment isolates the observable, not the "
        "information content per measurement."
    ),
    "references": reference_list(
        "Perren:15", "ForemanMackey:13", "Gaia:23", "Majewski:17", "Dias:02",
    ),
}
