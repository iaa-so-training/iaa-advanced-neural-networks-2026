"""Chapter 16, exercise 1 — write the methods paragraph first.

    Write the one-paragraph methods section of your report before you run
    anything: the population, the cuts, the method, the seed, the metrics, the
    referees. If you cannot write it, you are not ready to run.

This is a writing exercise, so the answer is a worked paragraph — and, because
"write it before you run" is only useful if something can tell you whether you
did, a runnable audit that reads a draft and reports which of the brief's
required elements it is missing. The audit is checked against two inputs: the
worked example (which must pass every element) and a plausible half-finished
draft (which must fail on a known count). The template is filled from
``settings()``, so the parameters in it are the ones the code will actually
use — the commonest way a methods paragraph goes wrong is that it describes a
configuration the author never ran.
"""

from __future__ import annotations

import re

import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, member_field, members, settings

#: Every element the brief (and the rubric's 40% reproducibility mark) demands.
#: ``pattern`` is what the audit looks for; ``why`` is why its absence is fatal.
REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("population",
     r"\b\d{2,5}[\s-]*(?:member\s+)?(?:stars|objects|members)\b",
     "without a population count the reader cannot tell what the number was "
     "computed on — the same code on 982 and on 30 107 stars gives different "
     "metrics"),
    ("cluster list",
     r"\b(\d{1,2})\s*clusters\b",
     "the label space is part of the task: 25 clusters and 18 open clusters "
     "are different benchmarks"),
    ("cuts",
     r"(SNR|signal-to-noise|quality flag|ASPCAPFLAG|region|cone|radius)",
     "cuts are the part a second person cannot guess, and the part most often "
     "left out"),
    ("method",
     r"(t-SNE|UMAP|EVoC|HDBSCAN|K-means|PCA|latent|embedding)",
     "naming the method is the minimum; the parameter values are what makes it "
     "reproducible"),
    ("parameters",
     r"(perplexity|n[_\s-]?neighbors|min[_\s-]?cluster[_\s-]?size|epsilon)",
     "the method's hyperparameters are levers — §16.3 lists them, and a score "
     "without them cannot be interpreted"),
    ("seed",
     r"(seed|random[_ ]state|randomly seeded)",
     "every stochastic step needs a seed, or the number is one draw from an "
     "unknown distribution"),
    ("seed count",
     r"(mean|s\.d\.|standard deviation|±|\d+\s*seeds|over \d+ seeds)",
     "a single seed cannot be distinguished from the configuration's noise; "
     "the rubric asks for mean and s.d. over at least three"),
    ("both metric halves",
     r"(recall|completeness|precision|homogeneity)",
     "a precision without a recall (or a completeness without a homogeneity) "
     "is unfalsifiable — either half alone can be gamed"),
    ("degenerate cases",
     r"(largest[- ]group|dominant group|degenerate|noise fraction|all in one)",
     "if one group holds everything, every metric looks fine; printing the "
     "largest-group fraction is what exposes it"),
    ("both referees",
     r"(Simbad|kinematic)",
     "kinematic selection and Simbad are independent, and the manual referee "
     "found members the pipeline missed; one referee is a weaker claim"),
    ("subsample rule",
     r"(subsamp|drawn (?:by|from|with)|random sample|capped at|down-?sampl)",
     "a subsample is legitimate; an undocumented subsample is not "
     "reproducible"),
    ("configuration provenance",
     r"(config|commit|hash|environment variable|ENV|settings)",
     "seed plus commit is what turns 'I got 0.42' into a reproducible claim"),
)

#: A plausible draft of the kind that scores badly: it reads well and is
#: missing the things that make it checkable. Used to test the audit.
HALF_FINISHED_DRAFT = (
    "We clustered the APOGEE stars of 25 clusters with t-SNE and HDBSCAN and "
    "measured how well the groups recovered the known labels, comparing the "
    "abundance vector against the spectral latent."
)


def audit(paragraph: str) -> pd.DataFrame:
    """Which of the brief's required elements a draft paragraph contains.

    The check is deliberately lexical: this is a checklist, not a proofreader.
    A term can be present and meaningless ("we used a good method"), so a green
    tick means "the ingredient is mentioned", which is the necessary condition
    and the one a reader can enforce.
    """
    rows = []
    for element, pattern, why in REQUIRED:
        match = re.search(pattern, paragraph, flags=re.IGNORECASE)
        rows.append({
            "element": element,
            "present": bool(match),
            "evidence": match.group(0) if match else "",
            "why it matters": why,
        })
    return pd.DataFrame(rows)


def template() -> str:
    """The skeleton, filled with this repository's actual default values.

    Read the numbers off ``settings()`` and the shared population rather than
    typing them: a template that quotes a stale parameter is worse than none,
    because it looks reproducible.
    """
    cfg = settings()
    data = members()
    population = member_field()
    n_members = int(len(data.df))
    n_clusters = int(data.df["cluster"].nunique())
    n_field = int((population.df["cluster"] == "field").sum())
    tsne = cfg.tsne
    hdbscan = cfg.hdbscan
    return (
        f"We measured cluster recovery on {n_members} member stars in "
        f"{n_clusters} clusters (plus a {n_field}-star field population where "
        f"the task requires it), drawn from the APOGEE DR19 catalogue after "
        f"the standard quality cuts (SNR > {cfg.snr_min}, clean ASPCAP flags, "
        f"within each cluster's search radius), using the 16-element abundance "
        f"vector as the feature set. Clusters were recovered with HDBSCAN "
        f"(min_cluster_size = {hdbscan['min_cluster_size']}) applied to a "
        f"t-SNE embedding (perplexity = {tsne['perplexity']}), with UMAP and "
        f"EVoC as alternative embeddings and K-means as a baseline; every "
        f"stochastic step was run over seeds {list(SEEDS)}, and the tables "
        f"report the mean and s.d. across them. Performance is reported as "
        f"homogeneity, completeness and V (both halves, never one), together "
        f"with the recovery fraction at 40% and 70% overlap and the fraction "
        f"of stars in the largest group, which is what reveals a degenerate "
        f"partition. Every score was computed against two referees: the "
        f"kinematic selection, and the Simbad memberships, which are an "
        f"external list the pipeline never saw. Configuration is the "
        f"repository settings at the recorded commit, with any environment "
        f"overrides stated explicitly."
    )


def solve() -> dict[str, object]:
    """The template, plus the audit run on a good draft and a weak one."""
    good = audit(template())
    bad = audit(HALF_FINISHED_DRAFT)
    return {
        "template": template(),
        "audit_worked_example": good,
        "audit_half_finished": bad,
        "worked_example_missing": int((~good["present"]).sum()),
        "half_finished_missing": int((~bad["present"]).sum()),
        "half_finished_flags": list(bad.loc[~bad["present"], "element"]),
        "checklist": pd.DataFrame(
            [{"element": e, "why": w} for e, _, w in REQUIRED],
        ),
    }


ANSWER: dict[str, object] = {
    "what this exercise is testing": (
        "Not prose. The rubric gives 40% for reproducibility, and the methods "
        "paragraph is where reproducibility is either asserted or impossible — "
        "§16.1's claim is that writing it *first* is a readiness test, because "
        "a paragraph that cannot be written is a run that cannot be scored. The "
        "practical form of the test: name the population, the cuts, the method "
        "and its parameters, the seed and the seed count, both halves of every "
        "metric, both referees, and the config provenance. If any of those is "
        "still unknown, the run would have produced a number the author cannot "
        "defend — so the paragraph is written first, then the code is run, then "
        "only the numbers in it are updated."
    ),
    "the audit, and what it measured": (
        "The module ships a lexical audit over the twelve required elements. "
        "Run on the generated template it flags 0 of 12. Run on a plausible "
        "half-finished draft — the kind that reads well as a first draft — it "
        "flags 10 of 12, passing only the two elements it names explicitly "
        "('25 clusters', and 't-SNE and HDBSCAN' for the method). Flagged: "
        "population, cuts, hyperparameters, seed, seed count, both metric "
        "halves, degenerate cases, both referees, the subsample rule and "
        "configuration provenance. That ratio is the exercise: what is missing "
        "is exactly what a reader needs to check the work, and it is the part "
        "that is boring to write."
    ),
    "the template is generated, not typed": (
        "The paragraph is assembled from settings() and the shared population, "
        "so it quotes the values the code will use — the SNR floor, the "
        f"t-SNE perplexity {cite('vanderMaaten:08')}, the HDBSCAN* floor "
        f"{cite('Campello:13')}, the seed list, the population counts. "
        "This is a deliberate choice over a hand-written example: a template "
        "that quotes a stale parameter is worse than no template, because it "
        "looks reproducible. When the student changes a lever (§16.3), the "
        "paragraph changes with it, which is the invariant the rubric wants."
    ),
    "how to use it on your own cluster": (
        "Fill in the population count and cluster count for your sample, the "
        "cuts you actually applied (including the ones you applied by accident "
        f"— dropping rows with blank APOGEE_ID {cite('Majewski:17')} is a "
        "cut), the embeddings you "
        "compared, the parameters, the seeds, and both referees' scores before "
        "you look at which one is higher. Then keep the paragraph in the report "
        "verbatim and put the numbers in the tables. The audit is a floor, not "
        "a grade: a paragraph can pass all twelve checks and still be wrong, "
        "but it cannot fail them and be reproducible."
    ),
    "the honest caveat": (
        "A lexical check cannot detect a paragraph that names a cut it did not "
        "apply, or a seed it did not use — the failure mode of 'written before "
        "the run' is that the run then changes and the paragraph does not. The "
        "audit is therefore re-run at submission time on the final text, not "
        "just at the start, and the numbers in the template are regenerated "
        "rather than remembered."
    ),
    "references": reference_list(
        "vanderMaaten:08", "Campello:13", "Majewski:17",
    ),
}
