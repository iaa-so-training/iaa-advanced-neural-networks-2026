import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tuning lab — move one knob, watch the score

    Pick your cluster, then change **one** lever at a time. Every run re-scores
    **t-SNE / UMAP / EVoC** against the kinematic ground truth (Gaia parallax +
    proper motion + radial velocity). Your goal: beat the baseline in
    `docs/region_sweep_results.md` — and be able to say *which* knob moved the
    number and *why*.

    The three frontier tracks live in `docs/student_activities.md`
    (spectral latent · isochrone+red-clump · GALAH cross-match).

    ### Running it

    Inside the workshop container, like `chemical_tagging.py`:

    ```bash
    export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
    export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

    docker run --rm -it $DAY4 $IMG uv run cluster download --all     # once
    docker run --rm -it -p 2718:2718 $DAY4 $IMG \
      uv run marimo edit notebooks/tuning_template.py --host 0.0.0.0 --no-token
    ```
    """)
    return


@app.cell
def _(mo):
    from cluster.clusters import CLUSTERS

    cluster_dropdown = mo.ui.dropdown(
        options=sorted(c.name for c in CLUSTERS),
        value="M 67",
        label="cluster",
    )
    region = mo.ui.number(value=30.0, start=1.0, stop=60.0, step=1.0, label="region (deg)")
    use_weights = mo.ui.checkbox(value=False, label="element 1/σ weights")
    normalize = mo.ui.checkbox(value=True, label="row-normalise (L2)")
    perplexity = mo.ui.number(value=30, start=5, stop=100, step=5, label="t-SNE perplexity")
    min_cluster = mo.ui.number(value=5, start=2, stop=50, step=1, label="HDBSCAN min_cluster_size")
    mo.vstack([
        cluster_dropdown,
        mo.hstack([region, perplexity, min_cluster]),
        mo.hstack([use_weights, normalize]),
    ])
    return CLUSTERS, cluster_dropdown, region, use_weights, normalize, perplexity, min_cluster


@app.cell
def _(CLUSTERS, cluster_dropdown, region, use_weights, normalize, perplexity, min_cluster, mo):
    from pathlib import Path

    import pandas as pd

    from cluster import config
    from cluster.benchmark import run_benchmark
    from cluster.data import prepare

    allstar = Path("data/astraAllStarASPCAP-0.6.0.fits.gz")
    mo.stop(
        not allstar.exists(),
        mo.md(
            "⚠️ `data/astraAllStarASPCAP-0.6.0.fits.gz` not found — from your checkout run "
            "`docker run --rm -it $DAY4 $IMG uv run cluster download --all`, "
            "or `uv run cluster download --all` inside the container."
        ),
    )

    settings = config.Settings()
    settings.cluster_names = [cluster_dropdown.value]
    settings.region_radius_deg = region.value
    settings.use_element_weights = use_weights.value
    settings.normalize_rows = normalize.value
    settings.tsne["perplexity"] = perplexity.value
    settings.hdbscan["min_cluster_size"] = min_cluster.value

    clusters = [c for c in CLUSTERS if c.name == cluster_dropdown.value]
    prepared = prepare(
        allstar, settings, clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )
    result = run_benchmark(prepared, settings)

    frames = []
    for name, r in result.results.items():
        f = r.scores.copy()
        f.insert(0, "method", name)
        frames.append(f)
    scores = pd.concat(frames, ignore_index=True)

    mo.md(
        f"### {cluster_dropdown.value} — region {region.value}° · "
        f"weights={use_weights.value} · normalize={normalize.value} · "
        f"perplexity={perplexity.value} · min_cluster={min_cluster.value}"
    )
    scores
    return settings, prepared, result, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to read it

    - **recall** — fraction of the cluster's true members the best-overlapping
      predicted cluster recovers.
    - **precision** — purity of that predicted cluster (field contamination pulls
      it down).
    - A recall jump with a precision collapse means your change swallowed the
      field into one blob — that's a lesson, not a win.

    Log what you tried; your deliverable is *one* change and why it moved the
    numbers.
    """)
    return
