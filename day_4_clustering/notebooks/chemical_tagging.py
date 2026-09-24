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
    # Chemical Tagging of Star Clusters — t-SNE vs UMAP vs EVoC

    Reproduces and extends Kos et al. (2017). The original paper tagged clusters
    with **t-SNE** on GALAH abundances. Here we benchmark **t-SNE vs UMAP vs EVoC**
    on **SDSS-V DR19 (APOGEE) + Gaia DR3**, over the clusters of Garcia-Dias et al. (2019)
    plus the Pleiades.

    **Key idea** — stars born together share a chemical fingerprint. Chemical
    tagging searches for clustering in the high-dimensional abundance space
    (C-space) and matches the groups to known clusters. Kinematics (parallax,
    proper motion, radial velocity) are *independent* of abundances, so they
    provide a clean ground truth.

    This notebook runs a **fast demo**: it restricts the sky to a 30° region
    around **M 67** and caps the field sample. The first cells (data prep and the
    benchmark) take a few minutes; the whole notebook, including the isochrone
    grids and the Gaia cross-match, runs in about ten.

    ### Running it

    You are inside the workshop container
    (`ghcr.io/iaa-so-training/day4-clustering`), so the environment is pinned and
    there is nothing to install. From your checkout:

    ```bash
    export IMG=ghcr.io/iaa-so-training/day4-clustering:latest
    export DAY4="-v $PWD/data:/app/data -v $PWD/results:/app/results -v $PWD/notebooks:/app/notebooks"

    docker run --rm -it $DAY4 $IMG uv run cluster download --all     # catalogue + embeddings, once
    docker run --rm -it -p 2718:2718 $DAY4 $IMG \
      uv run marimo edit notebooks/chemical_tagging.py --host 0.0.0.0 --no-token
    ```

    That is how this notebook is being served to you (http://localhost:2718).
    Everything it reads and writes stays on your machine, under `./data`,
    `./results` and `./notebooks` — the mounts above are the only bridge.

    ### Why it can feel slow, even on an idle machine

    The heavy cells are **serial by design**, so nothing saturates: reading the
    gzipped 1.17 GB catalogue is ~20 s on *one* core, UMAP drops to `n_jobs=1`
    whenever a `random_state` is set (it warns about this on every run), and EVoC
    builds its tree on one core — only t-SNE uses several. Measured on this
    16-core host: prepare 20.6 s at 1.0 core, UMAP 50 s, EVoC 75 s, t-SNE 43 s
    at 6.1 cores.

    Two things keep iteration cheap: the prepared sample is **cached on disk**
    (first call ~20 s, later ~0.04 s — same bytes, same settings, same seed; the
    key is printed), and the benchmark cells are **memoised per argument set**,
    so re-running them without changing knobs returns instantly. Need it faster
    still? `CLUSTER_TSNE_N_ITER=250` cuts t-SNE to ~14 s, `CLUSTER_NO_CACHE=1`
    (or `cluster run --no-cache`) forces a fresh read, and
    `CLUSTER_CACHE_DIR=<dir>` moves the cache.
    """)
    return


@app.cell
def _(mo):
    from cluster import config, seeding
    from cluster.baseline import (
        baseline_labels as _baseline_labels,
        confusion_matrix_frame,
        plot_confusion,
        separation_scores,
    )
    from cluster.benchmark import run_benchmark as _run_benchmark
    from cluster.catalog import attach_referee, cluster_panels_cell, hr_cell
    from cluster.clusters import CLUSTERS
    from cluster.data import prepare as _prepare
    from cluster.isochrone import gaia_age_cell, isochrone_cell
    from cluster.literature import literature_table
    from cluster.plots import (
        abundance_violins,
        embedding_interactive,
        method_comparison_bar,
    )

    # Memoise the expensive entry points at the marimo level: re-running a cell
    # with the same arguments reuses the previous result instead of recomputing
    # minutes of work. `prepare` additionally hits a disk cache
    # (cluster.data.prepare_cache_key), so even a fresh kernel skips the 20 s
    # single-core catalogue read. Nothing about the numbers changes — the cache
    # key covers the data, the settings and the seed, and tests pin the result
    # as bit-identical to a cold computation.
    prepare = mo.cache(_prepare)
    run_benchmark = mo.cache(_run_benchmark)
    baseline_labels = mo.cache(_baseline_labels)

    settings = config.Settings()
    settings.region_radius_deg = 30.0
    settings.cluster_names = ["M 67"]
    settings.max_stars = 5_000
    seeding.seed_everything(settings.random_state)

    print(
        f"FAST={settings.fast}  MAX_STARS={settings.max_stars}  "
        f"SNR_MIN={settings.snr_min}  SEED={settings.random_state}"
    )
    print(f"REGION={settings.region_radius_deg}°  CLUSTERS={settings.cluster_names}")
    print(f"ELEMENTS ({len(settings.elements)}): {settings.elements}")
    return (
        CLUSTERS,
        abundance_violins,
        attach_referee,
        baseline_labels,
        cluster_panels_cell,
        config,
        confusion_matrix_frame,
        embedding_interactive,
        gaia_age_cell,
        hr_cell,
        isochrone_cell,
        literature_table,
        method_comparison_bar,
        mo,
        plot_confusion,
        prepare,
        run_benchmark,
        separation_scores,
        settings,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 0. Locate the allStar catalogue

    The pipeline needs the APOGEE DR19 allStar FITS file (1.17 GB). This cell
    checks that it is present and raises a helpful error if it is not. Fetch it
    from your checkout with
    `docker run --rm -it $DAY4 $IMG uv run cluster download --all` (adds the
    embeddings + checkpoints used in §0c), or `uv run cluster download` for the
    catalogue alone if you are already in a shell inside the container.
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    allstar = Path("data/astraAllStarASPCAP-0.6.0.fits.gz")
    mo.stop(
        not allstar.exists(),
        mo.md(
            f"""
            ⚠️ `{allstar}` not found.

            From your checkout run:
            `docker run --rm -it $DAY4 $IMG uv run cluster download --all`
            (or `uv run cluster download --all` inside the container),
            then rerun this notebook.
            """
        ),
    )
    print(f"✓ allStar found: {allstar} ({allstar.stat().st_size / 1e9:.2f} GB)")
    # Path travels with allstar: marimo allows a name in one cell only, and §0c
    # needs it too.
    return allstar, Path


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0b. Paper baseline — re-create Garcia-Dias et al. (2019)

    Before pulling M 67 out of the field, the 2019 question: take **only the
    known cluster members** (all 23 clusters, no field), cluster them, and ask
    how well each star returns to its own cluster — scored with the paper's
    own metrics (homogeneity / v-measure / accuracy).

    Two feature sets: **abundances alone** (the 2019 setup) and
    **abundances + kinematics**. This re-runs the pipeline all-sky (a couple
    of minutes), independent of the M 67 demo below.
    """)
    return


@app.cell
def _(
    CLUSTERS,
    allstar,
    baseline_labels,
    config,
    confusion_matrix_frame,
    prepare,
    separation_scores,
):
    import pandas as pd

    baseline_settings = config.Settings()
    baseline_settings.max_stars = 0  # keep cluster members only
    baseline_clusters = [
        c for c in CLUSTERS
        if c.name in baseline_settings.resolve_cluster_names([c.name for c in CLUSTERS])
    ]
    baseline_prepared = prepare(
        allstar, baseline_settings, baseline_clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )

    rows = []
    cm_frames = {}
    for kin, tag in ((False, "chem"), (True, "kin")):
        true, labels = baseline_labels(
            baseline_prepared, baseline_settings,
            use_kinematics=kin, min_members=5,
        )
        for name, pred in labels.items():
            scores = separation_scores(true, pred)
            rows.append({
                "features": "chem+kin" if kin else "chem",
                "method": name,
                "n_stars": int(true.size),
                **scores,
            })
        cm_frames[tag] = {
            "true": true,
            "t-SNE": confusion_matrix_frame(true, labels["t-SNE"]),
        }
    baseline_table = pd.DataFrame(rows)
    print(f"baseline: {len(baseline_clusters)} clusters, all-sky")
    return (baseline_table, baseline_prepared, baseline_settings, cm_frames)


@app.cell
def _(baseline_table):
    baseline_table.round(3)
    return


@app.cell
def _(cm_frames, mo, plot_confusion):
    mo.mpl.interactive(
        plot_confusion(cm_frames["chem"]["t-SNE"], None, title="t-SNE — abundances only")
    )
    return


@app.cell
def _(cm_frames, mo, plot_confusion):
    mo.mpl.interactive(
        plot_confusion(cm_frames["kin"]["t-SNE"], None, title="t-SNE — abundances + kinematics")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 0c. The published spectral latent — same stars, same clusterers

    The asset bundle ships a **masked spectral autoencoder**: a network trained on
    the raw APOGEE spectra (no abundance labels) whose 256-D latent layer is
    exported for every star it covers
    (`data/embeddings/masked_latent.parquet`, 40 879 stars). That is a genuinely
    different view of a star — it never sees the ASPCAP abundances, so a
    metal-poor globular where the abundances collapse still has a spectrum.

    Comparing two feature sets is only fair on **the same stars**. `head_to_head`
    intersects the arms' `APOGEE_ID` sets first, applies the ≥5-member rule to the
    intersection, then scores every arm on exactly those stars with the same
    clusterers and the same seeds — and hands back `n_stars`, `n_clusters` and the
    per-arm losses so you can verify it did. (The trap it exists to avoid: quoting
    a 25-cluster abundance score next to a 5-cluster spectral score.)
    """)
    return


@app.cell
def _(Path, baseline_prepared, baseline_settings, mo):
    from cluster.headtohead import Arm, head_to_head as _head_to_head, pivot_scores, pivot_with_errors

    head_to_head = mo.cache(_head_to_head)  # seeds x methods x arms: the slowest cell in the notebook

    latent_path = Path("data/embeddings/masked_latent.parquet")
    mo.stop(
        not latent_path.exists(),
        mo.md(
            f"""
            ⚠️ `{latent_path}` not found.

            It is part of the asset bundle — from your checkout run
            `docker run --rm -it $DAY4 $IMG uv run cluster download --assets`
            (or `--all` for the catalogue as well), then rerun this notebook.
            """
        ),
    )

    arms = [
        Arm(label="abundances (16-d)", embedding_path=None),
        Arm(label="masked AE (256-d)", embedding_path=latent_path,
            notes="self-supervised, from spectra"),
    ]
    h2h = head_to_head(
        baseline_prepared, arms, baseline_settings, min_members=5, seeds=(42, 0, 1),
    )
    print(f"{h2h.n_stars} stars × {h2h.n_clusters} clusters, on the shared APOGEE_IDs")
    print(f"stars lost per arm: {h2h.dropped}")
    return (h2h,)


@app.cell
def _(h2h):
    pivot_scores(h2h, "homogeneity")
    return


@app.cell
def _(h2h):
    pivot_with_errors(h2h)  # mean ± std over the three seeds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **How to read it.** Homogeneity per clusterer, same 800 stars and 24 clusters
    for both rows:

    | method | abundances (16-d) | masked AE (256-d) |
    |---|---|---|
    | t-SNE | 0.23 | **0.73** |
    | UMAP | 0.54 | **0.77** |
    | EVoC | 0.46 | **0.69** |

    The latent wins on all three — the spectra keep chemical information the 16
    abundances do not. Two things worth saying out loud:

    - These same-population numbers are *lower* than the ones in
      `docs/spectral_benchmark_results.md`, where each arm was scored on its own
      coverage. Both are in the repo; only this table is apples-to-apples.
    - Cluster-only separation (here) and field retrieval (§2, members against the
      Simbad referee) are different questions. The field-retrieval version of the
      spectral arm is Track A in `docs/student_activities.md`.

    The same comparison from a shell, with seed error bars and a CSV:

    ```bash
    docker run --rm -it $DAY4 $IMG uv run cluster head-to-head \
        --arm "abundances (16-d)=abundances" \
        --arm "masked AE 256-d=data/embeddings/masked_latent.parquet" \
        --out results/head_to_head.csv
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Prepare the data

    Load allStar → quality cuts → kinematic membership labels → complete-case
    16-D abundance matrix (standardised). Region mode keeps only stars within
    30° of M 67, mirroring the target paper's per-region approach.
    """)
    return


@app.cell
def _(CLUSTERS, settings):
    clusters = [
        c
        for c in CLUSTERS
        if c.name in settings.resolve_cluster_names([c.name for c in CLUSTERS])
    ]
    return (clusters,)


@app.cell
def _(allstar, attach_referee, clusters, config, prepare, settings):
    prepared = prepare(
        allstar,
        settings,
        clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )
    # external referee (Simbad catalogue) for the benchmark scores
    prepared.df = attach_referee(prepared.df, clusters, settings)
    print(f"{prepared.X.shape[0]} stars × {prepared.X.shape[1]} abundances")
    print("referee counts:")
    print(prepared.df.groupby("referee").size())
    return (prepared,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Run the benchmark

    - **t-SNE** → 2-D → **HDBSCAN**
    - **UMAP** → 2-D → **HDBSCAN**
    - **EVoC** → clusters the 16-D vectors directly (embedding + density
      clustering fused)

    Scores are measured against the **Simbad catalogue** (the external
    referee), not the kinematic labels — see §6. §0c runs the same comparison
    with the published spectral latent swapped in for the abundances.
    """)
    return


@app.cell
def _(prepared, run_benchmark, settings):
    benchmark = run_benchmark(prepared, settings)
    return (benchmark,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Macro metrics

    Macro-averaged recall and precision per method (clusters with ≥ 1 true
    member). **Recall** = fraction of a cluster's true members recovered.
    **Precision** = fraction of the predicted group that are true members
    (purity).
    """)
    return


@app.cell
def _(benchmark):
    benchmark.macro()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Visualise — interactive embedding

    Pick a method, then zoom and hover the 2-D embedding coloured by referee
    (Simbad) membership. Hovering shows the star id, Teff/logg and its cluster
    label. EVoC has no public 2-D projection, so it is drawn on the UMAP
    canvas.
    """)
    return


@app.cell
def _(mo):
    embed_method = mo.ui.dropdown(
        options=["t-SNE", "UMAP", "EVoC"], value="UMAP", label="method",
    )
    return (embed_method,)


@app.cell
def _(benchmark, embed_method, embedding_interactive):
    embedding_interactive(benchmark, embed_method.value)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Enriched diagnostics

    **Left**: abundance violins comparing M 67 members against field stars —
    members should be tighter and chemically distinct.

    **Right**: grouped bars of macro recall / precision / kNN purity per
    method. kNN purity is the parameter-free chemical-cohesion score (do
    known members sit together in the embedding?); EVoC exposes no 2-D
    projection, so it has no purity bar.
    """)
    return


@app.cell
def _(abundance_violins, mo, prepared, settings):
    mo.mpl.interactive(
        abundance_violins(
            prepared.df, settings.elements, "M 67",
            random_state=settings.random_state,
        )
    )
    return


@app.cell
def _(benchmark, method_comparison_bar, mo):
    mo.mpl.interactive(method_comparison_bar(benchmark))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Interactive HR diagrams

    Pick a cluster and a membership source, then zoom and hover the Gaia CMD
    (absolute G vs BP−RP) and the Kiel diagram (logg vs Teff). Hovering a star
    shows its id, Teff/logg, and which of the three sources — **catalogue**
    (Simbad), **kinematic**, **combined** — flag it, so you can see where the
    methods agree and disagree along the sequence.
    """)
    return


@app.cell
def _(mo):
    from cluster.clusters import CLUSTER_BY_NAME

    cluster_picker = mo.ui.dropdown(
        options=sorted(CLUSTER_BY_NAME), value="M 67", label="cluster",
    )
    method_picker = mo.ui.radio(
        options=["catalog", "kinematic", "combined"],
        value="combined", label="highlight",
    )
    return (cluster_picker, method_picker)


@app.cell
def _(allstar, settings):
    # load the full catalogue once (heavy; marimo caches the result)
    from cluster.data import apply_quality_cuts, load_allstar

    df_hr = load_allstar(allstar, settings.elements)
    df_hr = apply_quality_cuts(df_hr, settings)
    print(f"{len(df_hr):,} stars after quality cuts")
    return (df_hr,)


@app.cell
def _(cluster_picker, df_hr, hr_cell, method_picker, settings):
    hr_cell(df_hr, cluster_picker.value, method_picker.value, settings)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Isochrone fit — a membership-quality proxy

    Fit a PARSEC isochrone (ASteCA) to the selected membership source's CMD
    using **two colours** — Gaia BP−RP *and* 2MASS J−Ks — which breaks the
    age-metallicity degeneracy. Grid chosen per cluster type (solar /
    metal-poor); 32-walker × 1000-step emcee chain.
    """)
    return


@app.cell
def _(cluster_picker, df_hr, isochrone_cell, method_picker, mo, settings):
    isochrone_cell(df_hr, cluster_picker.value, method_picker.value, settings, mo)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Gaia-only age fit

    APOGEE only sees bright giants, so §7 cannot pin globular ages. This cell
    queries **Gaia DR3** directly (full depth to the main sequence), selects
    members by proper motion, drops the bright giants + horizontal branch
    (G ≳ 16 for globulars), and fits the main-sequence turnoff — which fixes
    the age. (metallicity/distance stay degenerate in a single colour.)
    """)
    return


@app.cell
def _(cluster_picker, gaia_age_cell, settings):
    gaia_age_cell(cluster_picker.value, settings)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Three views of the cluster

    Kiel diagram (logg vs Teff, APOGEE spectroscopy), the 2MASS CMD (K vs
    J−Ks), and the Gaia CMD (G vs BP−RP, full depth). Together they show why
    the fit works: the Kiel/2MASS views separate giants from dwarfs, and the
    Gaia view supplies the main sequence that pins the age.
    """)
    return


@app.cell
def _(cluster_panels_cell, cluster_picker, df_hr, method_picker, settings):
    cluster_panels_cell(df_hr, cluster_picker.value, method_picker.value, settings)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Literature comparison

    Accepted parameters (VizieR: Dias for open clusters, Harris 1996/2010 for
    globulars; globular ages from Dotter 2010 / Marin-Franch 2009). Compare
    with the fitted values in §7/§8 to see where the pipeline is furthest
    from the literature.
    """)
    return


@app.cell
def _(CLUSTERS, literature_table):
    literature_table(CLUSTERS)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 11. What to look for

    - **Recall vs precision**: small `min_cluster_size` → high recall, low
      precision (field merges in). Tune `config.HDBSCAN` and watch the trade-off.
    - **Region mode matters**: the paper ran t-SNE on 30–45° regions, not the
      whole sky. Rerun with a different `settings.region_radius_deg` or
      `settings.cluster_names` and watch recall change.
    - **Chemical cohesion (kNN purity)** is the cleaner, parameter-free score:
      after embedding, do known members sit together?
    - **Scores are against Simbad** (§2): the catalogue is the referee, so
      recall/precision measure *recovery of literature members*, independent
      of the abundances and kinematics used to find them.
    - **Spectra beat abundances where abundances are weak** (§0c): on the same
      stars and the same clusterers, the published masked-AE latent adds
      +0.2–0.5 homogeneity. `cluster head-to-head` reproduces the table for any
      pair of feature arms.
    - **The combination wins** (§6): `MEMBERSHIP_METHOD="combined"` adds
      chemically-consistent stars to the kinematic core and lifts catalogue
      recall. Flip `settings.membership_method` and watch.
    - **Isochrone fit** (§7): two colours (BP−RP + J−Ks) pin the age for open
      clusters (M 67 → 4.4 Gyr ✓). Globulars are different: APOGEE's SNR cut
      sees only red giants (logg ≲ 2.5) — the main sequence/turnoff is too
      faint — and the RGB constrains metallicity/distance but **not age**
      (age std ≳ 0.9 dex). A real survey limitation, not a bug: you need
      deeper photometry for globular ages.
    - The Pleiades group is small in APOGEE (~10 clean members) — the hard,
      honest case. M 67 / NGC 6819 / M 3 are rich — the easy showcase.
    """)
    return


if __name__ == "__main__":
    app.run()
