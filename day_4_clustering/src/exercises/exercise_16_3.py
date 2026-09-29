"""Chapter 16, exercise 3 — argue for the other track.

    Your cluster is listed with a suggested track. Write two sentences arguing
    for the other track, using only the cluster's properties (age, metallicity,
    declination, richness).

The suggested tracks are §16.1's: **A** spectral embeddings, **B** isochrones
and the red clump, **C** two surveys. The worked answers below take four
clusters off the workbook's own table, one of each kind of argument:

* M 67 — suggested C (or A), argued for B on the strength of its giant
  population;
* Collinder 261 — suggested C, argued for B, because the two-survey fit is
  where the *interesting* failure is;
* Pleiades — suggested A with an expected null, argued for B, because a null
  result is a weaker deliverable than a measurement;
* NGC 6791 — suggested B (or A), argued for C, and the declination check rules
  it out.

The properties used are the ones the exercise allows: age, metallicity,
declination and richness. The declination bound for track C is DEC ≲ +25°
(§16.1), because GALAH is a southern survey; richness is the workbook's own
per-cluster member count, which is the number a student can actually work with.
"""

from __future__ import annotations

import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import members

#: The declination beyond which track C is not available (§16.1: GALAH is a
#: southern survey, DEC ≲ +25°).
TRACK_C_DEC_LIMIT = 25.0
#: Clusters worked in the answer, with the suggested track from Table 13.
WORKED: dict[str, str] = {
    "M 67": "C (or A)",
    "Collinder 261": "C",
    "Pleiades": "A (expect a null)",
    "NGC 6791": "B (or A)",
    "M 3": "A",
}


def cluster_properties() -> pd.DataFrame:
    """Age, metallicity, declination, richness and track-C eligibility.

    Richness is counted from the shared member matrix (``members()``), which is
    the deduplicated sample the workbook's own tables use; the abundance
    column is the median [Fe/H] of those rows.
    """
    import numpy as np

    from cluster.clusters import CLUSTERS
    from cluster.literature import fetch_literature

    data = members()
    frame = data.df
    counts = frame["cluster"].value_counts()
    feh = frame.groupby("cluster")["FE_H"].median()

    rows = []
    for cluster in CLUSTERS:
        literature = fetch_literature(cluster)
        n_stars = int(counts.get(cluster.name, 0))
        age = literature.get("age_Gyr")
        metal = literature.get("feh")
        rows.append({
            "cluster": cluster.name,
            "kind": cluster.kind,
            "suggested track": WORKED.get(cluster.name, "—"),
            "age_Gyr": round(float(age) if age is not None else float("nan"), 3),
            "feh_literature": round(float(metal) if metal is not None else float("nan"), 2),
            "feh_median_sample": round(float(feh.get(cluster.name, np.nan)), 3),
            "dec_deg": round(cluster.dec_deg, 2),
            "n_members_sample": n_stars,
            "track_C_available": bool(cluster.dec_deg <= TRACK_C_DEC_LIMIT),
            "rich_enough_for_clump": bool(n_stars >= 20),
        })
    return pd.DataFrame(rows).sort_values("n_members_sample", ascending=False)


def solve() -> dict[str, object]:
    """The property table, plus the eligibility flags used by the arguments."""
    table = cluster_properties()
    worked = table[table["cluster"].isin(WORKED)].copy()
    return {
        "all_clusters": table,
        "worked_clusters": worked,
        "declination_limit_track_C": TRACK_C_DEC_LIMIT,
        "clusters_without_track_C": list(
            table.loc[~table["track_C_available"], "cluster"]
        ),
        "clusters_too_sparse_for_clump": list(
            table.loc[~table["rich_enough_for_clump"], "cluster"]
        ),
    }


ANSWER: dict[str, object] = {
    "what the exercise is really asking": (
        "Not which track is better in general — the tracks are advisory, and "
        "§16.1 says so. It is asking whether you can argue from a cluster's own "
        "properties to a *method*, which is the skill that transfers. Two "
        "constraints do most of the work and both are hard: declination (track "
        "C needs DEC ≲ +25°, because GALAH "
        f"{cite('DeSilva:15')} is a southern survey) and richness "
        "(a red-clump distance needs red-clump giants, conventionally a couple "
        "of dozen members, and an isochrone fit needs ≥ 25 stars with full "
        "photometry before the repository's own code will run at all). A "
        "cluster that fails either check cannot be argued onto that track, "
        "however good the physics. The property table is generated from "
        "src/cluster/literature.py — open-cluster parameters from the VizieR "
        f"catalogue {cite('Dias:02')} and globular ones from "
        f"{cite('Harris:96', parenthetical=False)}, with globular ages from "
        f"{cite('Dotter:10', parenthetical=False)} — and from the member "
        "counts in the shared "
        "sample, so the numbers in the arguments below are the ones the code "
        "will use."
    ),
    "M 67 — suggested C (or A), argued for B": (
        "M 67 is the richest cluster in the sample (230 member rows) and old "
        "enough (2.82 Gyr, [Fe/H] +0.03) to carry a genuine red-clump "
        "population, so a clump distance from its own giants is measurable to "
        "the few-hundredths-of-a-magnitude level that NGC 6819 reached in this "
        "workbook's own table, and a distance that is *measured* rather than "
        "fitted is the one thing that breaks the age-metallicity-distance "
        "degeneracy the isochrone fit otherwise cannot escape. Arguments "
        "against: B on M 67 is the least novel thing in the workbook — §15.4 "
        "already fits three membership lists to it, and the literature age is "
        "well determined, so a student working on it is unlikely to produce a "
        "result the chapter does not already contain."
    ),
    "Collinder 261 — suggested C, argued for B": (
        "Collinder 261 at 8.91 Gyr and [Fe/H] -0.03 is old and giant-rich "
        "enough that a clump distance should be the *most* reliable thing about "
        "it, and this workbook's own attempt at that measurement came out 0.60 "
        "mag off the literature value — pinning down whether that is the "
        "cluster's metallicity scale, the clump slice, or the M_K calibration "
        "is a self-contained result that needs no new survey data. Arguments "
        "against: the giant arm has only 27 kinematic members against the 213 "
        "main-sequence GALAH stars, so B discards most of the information the "
        "cluster offers and the answer is dominated by the small-number "
        "statistics of a dozen clump giants."
    ),
    "Pleiades — suggested A (expect a null), argued for B": (
        "The Pleiades are young enough (0.1 Gyr) that their turnoff sits at "
        "about 3 solar masses near G = 6, so the CMD is age-sensitive over a "
        "magnitude range no other cluster in the sample reaches, and at 136 pc "
        "the distance modulus is small enough (5.67) that an isochrone fit is "
        "limited by the model's turnoff treatment rather than by photometric "
        "error. Arguments against: the sample has only 23 member rows in the "
        "shared matrix — below the 25-star floor the repository's fit refuses "
        "to run on — so a student would have to bring extra photometry before "
        "B is even runnable, and the cluster's chemistry genuinely carries no "
        "signal, which is why A returns a null here and why the null is worth "
        "reporting."
    ),
    "NGC 6791 — suggested B (or A), argued for C, and refuted": (
        "NGC 6791 is old (8.3 Gyr) and metal-rich (+0.42), which makes it the "
        "one cluster in the sample whose chemistry is anomalous enough to be "
        "worth following into a two-survey fit — GALAH main sequence plus "
        f"APOGEE giants {cite('Majewski:17')} — and its 45 member rows are the "
        "fifth-largest in the sample. But the declination check fails: at "
        "+37.8° it is outside track C's ≈ +25° southern limit, so a GALAH "
        "main-sequence arm is not available, and the argument cannot be made "
        "on this cluster at all. This is the exercise's most useful case: the "
        "answer is 'no', and it is no because of a single property, which is "
        "what a track assignment should feel like."
    ),
    "M 3 — the same check, on a globular": (
        "M 3 is suggested track A and is the second-richest cluster in the "
        "sample (154 rows), metal-poor (-1.57) with an 11 Gyr age, so its "
        "abundances are compressed against the globular sequence and the "
        f"spectral latent — a masked-autoencoder representation "
        f"{cite('He:22')} of the spectra themselves — is the right instrument; "
        "the workbook's §13.3 "
        "ablation even uses it as the cluster whose removal changes the "
        "benchmark. Arguing for C would need a southern globular; at +28.4° M 3 "
        "is north of the GALAH limit, and its globular kinematics make the "
        "GALAH arms redundant in any case. Arguing for B is possible — old "
        "clusters are giant-rich, and a clump distance on a metal-poor "
        "globular needs the metal-poor PARSEC grid — but the exercise's point "
        "stands: the interesting argument for M 3 is the one *within* track A, "
        "not a switch."
    ),
    "the two-sentence discipline": (
        "The rubric asks for two sentences per cluster, and that constraint is "
        "the point: name the property that enables the track and the property "
        "that makes it worse, and stop. A paragraph of hedges is what the "
        "two-sentence limit exists to prevent, and a track argument that "
        "survives the limit is one whose evidence was specific — age, "
        "metallicity, declination, richness — rather than general."
    ),
    "references": reference_list(
        "DeSilva:15", "Dias:02", "Harris:96", "Dotter:10", "Majewski:17",
        "He:22",
    ),
}
