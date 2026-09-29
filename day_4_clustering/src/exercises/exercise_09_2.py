"""Chapter 9, exercise 2 — the row-order check.

    Reproduce the row-order check: cluster the DR19 member matrix with
    t-SNE + HDBSCAN* nine times, once sorted and eight times with a random
    row order, fixing the seed. Report the spread of homogeneity and of the
    number of predicted groups. Now do the same with the masked-AE latent.
    Why is one stable and the other not?

This is \\S 9.3's failure mode 3, and the single most important measurement in
the validation chapter: permuting rows leaves every pairwise distance exactly
unchanged, so any change in the score is the optimiser talking, not the data.
The workbook reports 0.218-0.560 in homogeneity and 2-22 groups; this module
re-runs the experiment on the current DR19 frame and reports what it actually
measures, which is a weaker but unmistakable version of the same effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exercises.citations import cite, reference_list
from exercises.utils import SEEDS, settings

#: The fixed seed the row-order experiment holds constant (\\S 9.4 rule 2's
#: first seed). The point is that the *seed* is not the variable here.
FIXED_SEED: int = SEEDS[0]

#: Number of random row orders, plus the sorted one: nine runs in total.
N_RANDOM_ORDERS: int = 8

#: Base for the permutation seeds, kept separate from the model seed so the
#: two sources of randomness cannot be confused.
PERMUTATION_SEED_BASE: int = 1000


def row_orders(
    n: int, n_random: int = N_RANDOM_ORDERS,
) -> dict[str, np.ndarray]:
    """``{'sorted': identity, 'random_0': perm, ...}`` — the nine orders."""
    orders: dict[str, np.ndarray] = {"sorted": np.arange(n)}
    for i in range(n_random):
        rng = np.random.default_rng(PERMUTATION_SEED_BASE + i)
        orders[f"random_{i}"] = rng.permutation(n)
    return orders


def _fit_once(
    X: np.ndarray, labels: np.ndarray, method: str, seed: int, cfg: object,
) -> dict[str, object]:
    """One embed-then-cluster run, scored and degeneracy-checked."""
    from cluster.baseline import separation_scores
    from cluster.benchmark import cluster_embedding, fit_tsne, fit_umap
    from cluster.config import Settings
    from cluster.stability import degeneracy

    assert isinstance(cfg, Settings)
    if method == "t-SNE":
        params = {k: v for k, v in cfg.tsne.items() if k != "method"}
        embedding = fit_tsne(X, params, seed)
    elif method == "UMAP":
        embedding = fit_umap(X, cfg.umap, seed)
    else:
        raise ValueError(f"unknown method {method!r}")

    pred = cluster_embedding(embedding, cfg.hdbscan)
    scores = separation_scores(labels, pred)
    collapse = degeneracy(pred)
    return {
        "homogeneity": round(scores["homogeneity"], 4),
        "completeness": round(scores["completeness"], 4),
        "v_measure": round(scores["v_measure"], 4),
        "n_groups": collapse["n_clusters"],
        "largest_fraction": collapse["largest_fraction"],
        "n_noise": collapse["n_noise"],
    }


def row_order_sweep(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "t-SNE",
    seed: int = FIXED_SEED,
    n_random: int = N_RANDOM_ORDERS,
) -> pd.DataFrame:
    """Nine runs on the same data under nine row orders, one row each."""
    cfg = settings()
    rows = []
    for name, index in row_orders(len(X), n_random).items():
        result = _fit_once(X[index], labels[index], method, seed, cfg)
        rows.append({"order": name, "method": method, **result})
    return pd.DataFrame(rows)


def masked_latent(
    df: pd.DataFrame, name: str = "masked_latent_members.parquet",
) -> tuple[np.ndarray, np.ndarray]:
    """The 256-D masked-AE latent, joined to the member frame on APOGEE_ID."""
    from exercises.utils import embedding_path

    latent = pd.read_parquet(embedding_path(name))
    merged = df[["APOGEE_ID", "cluster"]].merge(
        latent, on="APOGEE_ID", how="inner",
    )
    columns = [c for c in latent.columns if c.startswith("z")]
    return merged[columns].to_numpy(dtype=float), merged["cluster"].to_numpy()


def _summary(table: pd.DataFrame, label: str) -> dict[str, object]:
    """Sorted-vs-random summary of one sweep."""
    random = table[table["order"] != "sorted"]
    homogeneity = table["homogeneity"].to_numpy(dtype=float)
    return {
        "arm": label,
        "sorted": round(float(table.iloc[0]["homogeneity"]), 4),
        "random_mean": round(float(random["homogeneity"].mean()), 4),
        "random_std": round(float(random["homogeneity"].std(ddof=0)), 4),
        "min": round(float(homogeneity.min()), 4),
        "max": round(float(homogeneity.max()), 4),
        "range": round(float(homogeneity.max() - homogeneity.min()), 4),
        "groups_min": int(table["n_groups"].min()),
        "groups_max": int(table["n_groups"].max()),
    }


def solve(
    seed: int = FIXED_SEED, n_random: int = N_RANDOM_ORDERS,
) -> dict[str, object]:
    """Run all four sweeps: {abundances, masked-AE latent} x {t-SNE, UMAP}."""
    from exercises.utils import DataNotAvailable, members

    data = members()
    sweeps: dict[str, pd.DataFrame] = {
        "abundances / t-SNE": row_order_sweep(
            data.X, data.labels, "t-SNE", seed, n_random),
        "abundances / UMAP": row_order_sweep(
            data.X, data.labels, "UMAP", seed, n_random),
    }
    note = ""
    try:
        latent_X, latent_y = masked_latent(data.df)
        sweeps["masked-AE 256d / t-SNE"] = row_order_sweep(
            latent_X, latent_y, "t-SNE", seed, n_random)
        sweeps["masked-AE 256d / UMAP"] = row_order_sweep(
            latent_X, latent_y, "UMAP", seed, n_random)
        n_latent = int(len(latent_X))
    except DataNotAvailable as exc:
        note = f"masked-AE latent unavailable: {exc}"
        n_latent = 0

    summary = pd.DataFrame([_summary(t, k) for k, t in sweeps.items()])
    return {
        "seed": seed,
        "n_orders": n_random + 1,
        "n_stars_abundances": int(len(data.X)),
        "n_stars_latent": n_latent,
        "summary": summary,
        "sweeps": sweeps,
        "note": note,
    }


def plot(result: dict[str, object] | None = None):  # pragma: no cover — figure
    """Strip plot of homogeneity under nine row orders, per arm."""
    import matplotlib.pyplot as plt

    result = result if result is not None else solve()
    sweeps = result["sweeps"]
    assert isinstance(sweeps, dict)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (_name, table) in enumerate(sweeps.items()):
        values = table["homogeneity"].to_numpy(dtype=float)
        ax.scatter(np.full(len(values), i) + np.random.default_rng(0).normal(
            0, 0.04, len(values)), values, s=26, alpha=0.8)
        ax.scatter([i], [values[0]], s=90, facecolors="none",
                   edgecolors="crimson", lw=1.5,
                   label="sorted order" if i == 0 else None)
    ax.set_xticks(range(len(sweeps)), list(sweeps), rotation=12, fontsize=8)
    ax.set_ylabel("homogeneity")
    ax.set_title("Same distances, nine row orders, one fixed seed")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


ANSWER: dict[str, object] = {
    "what the experiment isolates": (
        "Permuting the rows of X is an isometry of the point cloud: every "
        "pairwise distance, every neighbourhood, every density is identical in "
        "all nine runs, and the model seed is pinned at 42 throughout. So any "
        "spread in the score is produced entirely by the optimiser's path — "
        "the order in which the Barnes-Hut approximation "
        f"{cite('vanderMaaten:14')} accumulates interactions, and the "
        "arbitrary sign/order conventions of the PCA initialisation. It is the "
        "variation that a seed loop structurally cannot see, which is why the "
        "t-SNE rows of Table 'honest' carry std = 0.000 and are still unstable."
    ),
    "what I measured (abundances)": (
        "Nine runs, seed 42, perplexity 30, init='pca', HDBSCAN "
        f"{cite('Campello:13')} min_cluster_size=5 on the 1 002-row DR19 "
        f"member matrix. t-SNE {cite('vanderMaaten:08')} "
        "homogeneity: sorted 0.520, eight random orders 0.470 +- 0.043, full "
        "range 0.394-0.520; predicted groups range 18-35; largest-group "
        f"fraction 0.115-0.352. UMAP {cite('McInnes:18')} on the same "
        "matrix: sorted 0.545, random "
        "0.529 +- 0.018, range 0.503-0.562, but groups swing 28-50 and the "
        "largest-group fraction jumps between 0.042 and 0.350 — two distinct "
        "solution families, not a continuum."
    ),
    "how that compares with the workbook": (
        "The workbook quotes 0.218-0.560 in homogeneity and 2-22 groups for "
        "the same check. My re-run on the current frame gives a smaller but "
        "unambiguous 0.394-0.520 and 18-35 groups. The direction and the "
        "lesson are identical; the magnitude is not, and the honest reading is "
        "that the exact range is itself unstable — it depends on which eight "
        "permutations you draw, on the sklearn/openTSNE backend, and on the "
        "frame version (the published range was measured on an earlier "
        "preparation of this matrix). Quote a range from your own run, not "
        "this one: that is the whole point of the exercise."
    ),
    "the masked-AE latent, and why it is stable": (
        "Joining the 256-D masked-autoencoder "
        f"{cite('He:22')} latent to the member frame on APOGEE_ID "
        "leaves 966 of the 1 002 rows (the 34 rows with no identifier cannot "
        "join, and a handful of stars are absent from the latent), still "
        "spanning all 25 clusters. Under the same nine orders: t-SNE homogeneity "
        "0.6515-0.6626 (random 0.6599 +- 0.0032, 25-29 groups) and UMAP "
        "0.7687-0.7987 (random 0.7830 +- 0.0090, 64-73 groups). That is a "
        "spread roughly ten times smaller than the abundances' on t-SNE, and "
        "the largest-group fraction barely moves (0.242-0.244). The reason is "
        "geometric, not algorithmic: in the latent the clusters are genuinely "
        "separated, so the KL objective has one deep basin and every optimiser "
        "path falls into it. In the 16-D abundance space the clusters overlap, "
        "the objective surface is nearly flat with many comparable local "
        "minima, and which one you land in is decided by accumulation order."
    ),
    "why one is stable and the other is not": (
        "Instability is a symptom of the data, not a defect of t-SNE. A "
        "well-separated representation makes the embedding problem convex "
        "enough that initialisation and ordering do not matter; a representation "
        "in which the true groups are not separated leaves the optimiser "
        "choosing between near-degenerate solutions. So the row-order spread is "
        "a *diagnostic*: a large spread says 'the structure you are claiming is "
        "not in this representation', which is exactly the conclusion §10.5 "
        "draws about abundances-only t-SNE. A small spread does not prove the "
        "structure is real — it proves the optimiser agrees with itself."
    ),
    "what to report": (
        "Three things, next to every embedding number. (1) The row-order "
        "spread, not just the seed spread — with init='pca' the seed spread of "
        "t-SNE is identically zero and therefore meaningless. (2) The number of "
        "predicted groups and the largest-group fraction, which here move far "
        "more than the score does and reveal the two UMAP solution families "
        "that the homogeneity alone hides. (3) The permutations themselves, or "
        "their generator seed, so the range is reproducible. If the row-order "
        "spread is comparable to the gap you are claiming between two methods, "
        "you do not have a result."
    ),
    "references": reference_list(
        "vanderMaaten:08", "vanderMaaten:14", "McInnes:18", "Campello:13", "He:22",
    ),
}
