"""Chapter 15, exercise 1 — the two-survey fit, on a second cluster.

    Reproduce the two-survey fit for one of the other clusters in the combined
    sample. Report age, distance and their posterior widths, with and without
    the red-clump prior. How much of the age improvement comes from the main
    sequence and how much from the prior?

Collinder 261 is the cluster to try: it is the workbook's track-C sweet spot
and it has 213 kinematic GALAH members with Gaia photometry, against 27 APOGEE
members in the region. Four fits are run — APOGEE giants alone, GALAH main
sequence alone, both together, and both plus the red-clump distance prior.

The answer is a partial reproduction and says so. The distance prior does what
it did for NGC 2243 (the posterior width on the distance modulus collapses by
more than a factor of ten) and the age does *not* improve the way §15.3
describes. Reporting that as a discrepancy is the point of the exercise: the
mechanism that fixed NGC 2243's age is a wrong-by-a-half age moving *towards*
the literature value, and on Collinder 261 the same prior moves an already
young age in the wrong direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import DataNotAvailable, catalogue_path, settings

#: The cluster fitted. Chosen because it is the workbook's own track-C example
#: and carries both surveys: GALAH main-sequence members and APOGEE giants.
TARGET_CLUSTER = "Collinder 261"

#: The four arms of the answer.
ARMS: tuple[str, ...] = (
    "APOGEE giants only",
    "GALAH main sequence only",
    "both surveys, no prior",
    "both surveys + red-clump prior",
)

#: Walker/step budget. ``cluster.config.Settings`` defaults are 32 x 1000
#: (§15.1 quotes ~1-2 min per fit); 600 steps keeps the four-arm, five-seed
#: sweep at a few minutes while leaving the chain converged for four
#: parameters on 240 points.
DEFAULT_WALKERS = 32
DEFAULT_STEPS = 600
#: Seeds the sweep is averaged over (the workbook's rule: never quote one run).
DEFAULT_SEEDS: tuple[int, ...] = (42, 0, 1, 2, 7)
#: Half-width of the flat distance-modulus prior when no clump prior is used.
DM_WIDTH_NO_PRIOR = 2.0
#: Half-width once the clump measurement is treated as a prior (mag).
DM_WIDTH_WITH_PRIOR = 0.15


def target_cluster():
    """The ``Cluster`` record for :data:`TARGET_CLUSTER`."""
    from cluster.clusters import CLUSTER_BY_NAME

    return CLUSTER_BY_NAME[TARGET_CLUSTER]


def literature() -> dict[str, float]:
    """Literature age, distance modulus and [Fe/H] for the target."""
    from cluster.literature import fetch_literature

    return fetch_literature(target_cluster())


def apogee_members():
    """APOGEE kinematic members of the target inside its search region.

    Runs the same selection the pipeline uses: quality cuts on the DR19
    catalogue, the cluster's ``region_deg`` cone, then ``kinematic_members``.
    """
    from cluster import config
    from cluster.data import apply_quality_cuts, load_allstar
    from cluster.membership import angular_separation, kinematic_members

    cluster = target_cluster()
    cfg = settings()
    # the DR17-era re-run in cluster.isochrone used the less strict ASPCAP gate
    cfg.require_aspcap_flag_clean = False
    frame = load_allstar(catalogue_path(), cfg.elements)
    frame = apply_quality_cuts(frame, cfg)
    sep = angular_separation(
        frame["RA"].to_numpy(dtype=float), frame["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    region = pd.DataFrame(frame[sep <= cluster.region_deg]).copy()
    mask = kinematic_members(
        region, cluster,
        config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
        config.SEED_PM_TOL, config.SEED_RV_TOL,
        config.N_REFINE_PASSES, config.REFINE_SIGMA,
    ).to_numpy()
    return pd.DataFrame(region[mask]).copy()


def galah_members():
    """GALAH kinematic members of the target from the cross-match parquet.

    The combined file carries both surveys' photometry in different columns;
    the Gaia columns are filled from the GALAH copies where APOGEE's names are
    empty, which is the same reconciliation ``scripts/rerun_combined.py`` does
    before scoring.
    """
    from cluster import config
    from cluster.membership import kinematic_members
    from exercises.utils import project_root

    cluster = target_cluster()
    path = project_root() / "data" / f"galah_apogee_{TARGET_CLUSTER.replace(' ', '_')}.parquet"
    if not path.is_file():
        raise DataNotAvailable(
            f"{path} is not on disk; build it with\n\n"
            "    uv run python scripts/build_galah_apogee.py\n",
        )
    frame = pd.read_parquet(path)
    frame = frame.rename(columns={
        "PARALLAX": "GAIAEDR3_PARALLAX", "PMRA": "GAIAEDR3_PMRA",
        "PMDEC": "GAIAEDR3_PMDEC", "RV": "VHELIO_AVG",
    })
    for target, source in (
        ("GAIAEDR3_PARALLAX", "parallax"), ("GAIAEDR3_PMRA", "pmra"),
        ("GAIAEDR3_PMDEC", "pmdec"),
        ("GAIAEDR3_PHOT_G_MEAN_MAG", "phot_g_mean_mag"),
        ("GAIAEDR3_PHOT_BP_MEAN_MAG", "phot_bp_mean_mag"),
        ("GAIAEDR3_PHOT_RP_MEAN_MAG", "phot_rp_mean_mag"),
    ):
        if source in frame.columns:
            frame[target] = frame[target].fillna(frame[source])
    mask = kinematic_members(
        frame, cluster,
        config.SEED_POSITION_RADIUS_DEG, config.SEED_PARALLAX_FRAC,
        config.SEED_PM_TOL, config.SEED_RV_TOL,
        config.N_REFINE_PASSES, config.REFINE_SIGMA,
    ).to_numpy()
    selected = frame[mask & (frame["survey"] == "GALAH").to_numpy()]
    return pd.DataFrame(selected).copy()


#: The Gaia photometry columns every arm is reduced to before fitting: the
#: fit is single-colour (BP-RP), so 2MASS J-K is dropped for all arms.
PHOTOMETRY = [
    "GAIAEDR3_PHOT_G_MEAN_MAG", "GAIAEDR3_PHOT_BP_MEAN_MAG",
    "GAIAEDR3_PHOT_RP_MEAN_MAG", "GAIAEDR3_PARALLAX",
]


def red_clump_distance() -> dict[str, float]:
    """Distance modulus from the member giants' K-band red clump.

    The recipe of ``scripts/red_clump.py``: member giants (2 ≤ logg ≤ 3.5),
    the J-K clump slice 0.5-0.85, the median K of that slice as the clump
    magnitude, and M_K = -1.6 + 0.25 [Fe/H].
    """
    members = apogee_members()
    giants = members[(members["LOGG"] >= 2.0) & (members["LOGG"] <= 3.5)]
    giants = giants[np.isfinite(giants["J"]) & np.isfinite(giants["K"])]
    jk = (np.asarray(giants["J"], dtype=float) - np.asarray(giants["K"], dtype=float))
    clump = pd.DataFrame(giants[(jk >= 0.5) & (jk <= 0.85)])
    feh = float(np.nanmedian(members["FE_H"])) if len(members) else float("nan")
    if len(clump) < 5:
        return {"dm": float("nan"), "n_clump": len(clump), "n_giants": len(giants),
                "feh": feh, "k_clump": float("nan"), "m_k": float("nan")}
    k_clump = float(np.median(np.asarray(clump["K"], dtype=float)))
    m_k = -1.6 + 0.25 * feh if np.isfinite(feh) else -1.6
    return {
        "dm": k_clump - m_k, "n_clump": int(len(clump)), "n_giants": int(len(giants)),
        "feh": feh, "k_clump": k_clump, "m_k": m_k,
    }


def posterior_chain(
    members: pd.DataFrame,
    *,
    seed: int = 42,
    n_walkers: int = DEFAULT_WALKERS,
    n_steps: int = DEFAULT_STEPS,
    dm_center: float | None = None,
    dm_width: float = DM_WIDTH_NO_PRIOR,
    grid: str = "solar",
) -> dict[str, object]:
    """Sample the (met, loga, dm, Av) posterior and return the flat chain.

    The same likelihood as :func:`cluster.isochrone.fit_isochrone` — ASteCA's
    distance on the (G, BP-RP) CMD against the PARSEC grid, with the same
    priors — but returning the samples rather than only the mode and the
    marginal widths, because this exercise needs the posterior pairs.
    ``cluster.isochrone.fit_isochrone`` is called instead wherever the chain
    is not needed.
    """
    import asteca
    import emcee

    from cluster.isochrone import (
        _COLOR,
        _COLOR_EFFL,
        _GRID_MET,
        _MAG,
        _MAG_EFFL,
        _PARAMS,
        _distance_modulus,
        ensure_isochrones,
        ensure_isochrones_metal_poor,
    )

    g = np.asarray(members["GAIAEDR3_PHOT_G_MEAN_MAG"], dtype=float)
    bp = np.asarray(members["GAIAEDR3_PHOT_BP_MEAN_MAG"], dtype=float)
    rp = np.asarray(members["GAIAEDR3_PHOT_RP_MEAN_MAG"], dtype=float)
    plx = np.asarray(members["GAIAEDR3_PARALLAX"], dtype=float)
    finite = np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp)
    g, bp, rp, plx = g[finite], bp[finite], rp[finite], plx[finite]
    color = bp - rp

    path = (
        ensure_isochrones_metal_poor("data/isochrones/parsec") if grid == "metal_poor"
        else ensure_isochrones("data/isochrones/parsec")
    )
    isochrones = asteca.Isochrones(
        model="parsec", isochs_path=str(path), mag=_MAG, color=_COLOR,
        magnitude_effl=_MAG_EFFL, color_effl=_COLOR_EFFL, verbose=0,
    )
    synth = asteca.Synthetic(isochrones, seed=seed, verbose=0)
    cluster = asteca.Cluster(
        mag=g, e_mag=np.full_like(g, 0.005),
        color=color, e_color=np.full_like(color, 0.01), verbose=0,
    )
    synth.calibrate(cluster)
    likelihood = asteca.Likelihood(cluster)

    guess = _distance_modulus(plx) if dm_center is None else float(dm_center)
    bounds = {
        "met": _GRID_MET[grid], "loga": _PARAMS["loga"],
        "dm": (guess - dm_width, guess + dm_width), "Av": _PARAMS["Av"],
    }
    names = ["met", "loga", "dm", "Av"]
    lo = np.array([bounds[n][0] for n in names])
    hi = np.array([bounds[n][1] for n in names])

    def lnprob(theta: np.ndarray) -> float:
        for value, (low, high) in zip(theta, (bounds[n] for n in names)):
            if not (low <= value <= high):
                return -np.inf
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        try:
            generated = synth.generate(params)
            if isinstance(generated, tuple):
                generated = generated[0]
            return -float(likelihood.get(np.asarray(generated)))
        except (ValueError, IndexError):
            return -np.inf

    rng = np.random.default_rng(seed)
    position = np.empty((n_walkers, 4))
    position[:, 0] = 10.0 ** rng.uniform(np.log10(lo[0]), np.log10(hi[0]), n_walkers)
    position[:, 1] = rng.uniform(9.0 if grid == "metal_poor" else lo[1], hi[1], n_walkers)
    position[:, 2] = rng.uniform(lo[2], hi[2], n_walkers)
    position[:, 3] = rng.uniform(lo[3], hi[3], n_walkers)

    np.random.seed(seed)  # emcee 3.x draws from the global RNG
    sampler = emcee.EnsembleSampler(n_walkers, 4, lnprob, vectorize=False)
    sampler.run_mcmc(position, n_steps, progress=False)
    flat = np.asarray(sampler.get_chain(discard=n_steps // 2, flat=True))
    log_prob = np.asarray(sampler.get_log_prob(discard=n_steps // 2, flat=True))
    return {
        "chain": flat, "log_prob": log_prob, "names": names, "n_stars": int(len(g)),
        "mode": {n: float(flat[int(np.argmax(log_prob)), i]) for i, n in enumerate(names)},
        "std": {n: float(flat[:, i].std()) for i, n in enumerate(names)},
    }


def _as_float(value: object) -> float:
    """``float`` of a value pyrefly cannot type (ASteCA/emcee return ``object``)."""
    return float(np.asarray(value, dtype=float))


def _fit_arm(
    frame: pd.DataFrame, name: str, dm_center: float | None,
    dm_width: float, seeds: tuple[int, ...],
) -> dict[str, object]:
    """One arm: fit at every seed and summarise age, distance and widths."""
    ages: list[float] = []
    distances: list[float] = []
    sigma_loga: list[float] = []
    sigma_dm: list[float] = []
    metallicities: list[float] = []
    n_stars = 0
    for seed in seeds:
        result = posterior_chain(
            frame, seed=seed, dm_center=dm_center, dm_width=dm_width,
        )
        n_stars = int(_as_float(result["n_stars"]))
        mode = result["mode"]
        std = result["std"]
        assert isinstance(mode, dict) and isinstance(std, dict)
        ages.append(10.0 ** (mode["loga"] - 9.0))
        distances.append(mode["dm"])
        sigma_loga.append(std["loga"])
        sigma_dm.append(std["dm"])
        metallicities.append(mode["met"])
    return {
        "arm": name,
        "n_stars": n_stars,
        "age_Gyr": round(float(np.median(ages)), 3),
        "age_seed_sd": round(float(np.std(ages)), 3),
        "dm": round(float(np.median(distances)), 3),
        "dm_seed_sd": round(float(np.std(distances)), 3),
        "sigma_loga": round(float(np.mean(sigma_loga)), 3),
        "sigma_dm": round(float(np.mean(sigma_dm)), 3),
        "met": round(float(np.median(metallicities)), 4),
        "age_residual": round(float(abs(np.median(ages) - literature()["age_Gyr"])), 3),
        "dm_residual": round(float(np.median(distances) - literature()["dm"]), 3),
    }


def solve(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, object]:
    """The four-arm sweep with and without the red-clump prior."""
    lit = literature()
    clump = red_clump_distance()

    apogee = pd.DataFrame(apogee_members()[PHOTOMETRY])
    galah = pd.DataFrame(galah_members()[PHOTOMETRY])
    combined = pd.concat([apogee, galah], ignore_index=True)

    rows = [
        _fit_arm(apogee, ARMS[0], None, DM_WIDTH_NO_PRIOR, seeds),
        _fit_arm(galah, ARMS[1], None, DM_WIDTH_NO_PRIOR, seeds),
        _fit_arm(combined, ARMS[2], None, DM_WIDTH_NO_PRIOR, seeds),
        _fit_arm(combined, ARMS[3], clump["dm"], DM_WIDTH_WITH_PRIOR, seeds),
    ]
    table = pd.DataFrame(rows)

    return {
        "cluster": TARGET_CLUSTER,
        "literature": {k: round(float(v), 4) for k, v in lit.items()},
        "red_clump": {k: (round(float(v), 4) if isinstance(v, float) else v)
                      for k, v in clump.items()},  # type: ignore[arg-type]
        "n_apogee": int(len(apogee)),
        "n_galah": int(len(galah)),
        "n_combined": int(len(combined)),
        "fits": table,
        "seeds": list(seeds),
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Age and distance-modulus posterior widths, arm by arm."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    table = result["fits"]
    assert isinstance(table, pd.DataFrame)

    x = np.arange(len(table))
    reference = result["literature"]
    assert isinstance(reference, dict)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    axes[0].bar(x - 0.2, table["age_Gyr"], 0.4, yerr=table["age_seed_sd"],
                color="#4c72b0", capsize=3, label="fitted age")
    axes[0].axhline(float(reference["age_Gyr"]), color="crimson",
                    ls="--", lw=1.2, label="literature")
    axes[0].set_xticks(x, table["arm"], rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("age (Gyr)")
    axes[0].set_title("Age: the prior does not fix a young fit")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, table["sigma_dm"], 0.6, color="#dd8452")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, table["arm"], rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("$\\sigma_{dm}$ (mag, posterior)")
    axes[1].set_title("Distance: the prior is what buys the width")
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what was run": (
        "Collinder 261 — the workbook's track-C cluster — with four arms, five "
        "seeds each (42, 0, 1, 2, 7), 32 walkers and 600 steps per fit, on the "
        "PARSEC solar grid with the same likelihood and priors as "
        f"cluster.isochrone — ASteCA's synthetic-CMD likelihood "
        f"{cite('Perren:15')} sampled with emcee {cite('ForemanMackey:13')}. "
        "The samples are 27 APOGEE kinematic members "
        f"{cite('Majewski:17')} from "
        "the DR19 catalogue's region cone and 213 GALAH kinematic members "
        f"{cite('DeSilva:15', 'Buder:21')} from "
        "the cross-match parquet (240 together). The APOGEE arm is 27 giants, "
        "logg median 2.23 and no dwarf at all, G = 12.0-15.3 (median 13.4); the "
        "GALAH arm is 213 mostly faint main-sequence stars, G = 11.3-16.8 with a "
        "median of 16.2, so its turnoff is barely sampled by the bright limit. "
        "Literature: age 8.91 Gyr, dm 11.70, [Fe/H] -0.03 — the open-cluster "
        f"catalogue values {cite('Dias:02')} the repository caches."
    ),
    "the age, and where it comes from": (
        "APOGEE giants alone: 6.00 ± 1.28 Gyr (residual 2.92 against the "
        "literature's 8.91). GALAH main sequence alone: 4.87 ± 1.22 Gyr "
        "(residual 4.04). Both together: 5.18 ± 1.05 Gyr (residual 3.73). "
        "Combined plus the red-clump prior: 5.62 ± 0.90 Gyr (residual 3.30). "
        "Read the directions carefully, because neither is the direction §15.3 "
        "leads you to expect. Adding the main sequence moves the fit *down* by "
        "0.82 Gyr, towards the main-sequence-only value and away from the "
        "literature; the prior then moves it back *up* by 0.43 Gyr. Every arm "
        "is 2.9-4.0 Gyr too young, and the seed-to-seed scatter (0.9-1.3 Gyr) "
        "is a third of the residual, so the four arms are not separated by more "
        "than the seeds."
    ),
    "the distance, where the prior is decisive": (
        "Posterior widths on the distance modulus: 0.938 (APOGEE giants), 1.082 "
        "(GALAH main sequence), 1.170 (both, no prior), 0.086 (both + clump "
        "prior). That is a factor of 13.6 collapse and it is the one part of "
        "§15.3 that reproduces cleanly. With a ±2 mag flat prior the CMD alone "
        "cannot fix the distance of this cluster; the clump measurement can, "
        "because it is a standard candle rather than a fit. The catch is that "
        "the candle is misread: dm_clump = 12.31 against the literature 11.70, "
        "so the tight posterior is tight around the wrong value, which is why "
        "the fitted dm only moves from 12.39 to 12.38 and its residual stays at "
        "+0.68 mag."
    ),
    "the part that does not reproduce, stated plainly": (
        "§15.3's NGC 2243 sequence works because the giant-only fit is wrong by "
        "half — 1.59 Gyr against a literature 1.07 — and adding the main "
        "sequence pulls it up to 0.98. On Collinder 261 the giants-only fit is "
        "the *closest* arm to the literature (6.00 against 8.91) and the main "
        "sequence pulls the fit away from it, because the GALAH members are "
        "overwhelmingly faint main-sequence stars (median G = 16.2) whose turnoff "
        "is at the faint end of the sampled range. Two candidate explanations, "
        "both testable and neither settled here: (a) the clump distance is "
        "biased — dm = 12.31 against 11.70, from 11 giants with K_clump = "
        "10.707, [Fe/H] = +0.008 and the script's M_K = -1.6 + 0.25[Fe/H] = "
        "-1.598 — so the prior pulls the fit to the wrong distance; (b) the "
        "APOGEE arm has only 27 members (against 220-260 for M 67), so the "
        "giant-only fit is Poisson-limited and its apparent agreement with the "
        "literature may be luck. The honest report is: the distance-width "
        "result reproduces, the age result does not reproduce in sign or in "
        "size, and the prior's own 0.60 mag error is the first thing to fix."
    ),
    "how much from the main sequence, how much from the prior": (
        "From the arm-versus-arm differences, on the same seeds and the same "
        "fitting machinery: the main sequence moves the age mode by -0.82 Gyr "
        "(6.00 → 5.18), the clump prior by +0.43 Gyr (5.18 → 5.62), and the two "
        "partly cancel, so the residual against the literature moves 2.92 → "
        "3.73 → 3.30 Gyr without ever leaving the 'wrong by a factor of 1.5-1.6' "
        "regime. In the uncertainty budget the split is different again: adding "
        "the main sequence *widens* σ_loga from 0.884 to 1.003 dex (it does not "
        "help — at this sample size the age is prior-dominated, and 213 fainter "
        "stars whose turnoff is barely sampled add scatter rather than "
        "constraint), while the clump prior collapses σ_dm by 13.6x. So the "
        "answer to the exercise's question: the main sequence moves the age "
        "mode and the prior fixes the distance — and neither fixes the age "
        "*uncertainty*, which is §15.2's degeneracy doing its work."
    ),
    "the transferable lesson": (
        "A two-survey fit improves what the second survey is sensitive to. "
        "GALAH reaches the turnoff, so it moves the age mode; APOGEE's giants "
        "reach the clump, so a prior built from them pins the distance. Neither "
        "survey is redundant, and neither is a fix for the other's bias — which "
        "is why §15.5 lists 'one cluster, one pipeline' as the thing that is "
        "not established. Before quoting any of these numbers as a measurement "
        "of Collinder 261, re-measure the clump distance on a deeper giant "
        "sample (the 736 staged mwmStar spectra will not help; this needs more "
        "members, not more pixels) and re-run with the literature distance as "
        "the prior to separate 'the fit is wrong' from 'the clump is wrong'."
    ),
    "references": reference_list(
        "Perren:15", "ForemanMackey:13", "Majewski:17", "DeSilva:15",
        "Buder:21", "Dias:02",
    ),
}
