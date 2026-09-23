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
    on **APOGEE DR17 + Gaia EDR3**, over the clusters of Garcia-Dias et al. (2019)
    plus the Pleiades.

    **Key idea** — stars born together share a chemical fingerprint. Chemical
    tagging searches for clustering in the high-dimensional abundance space
    (C-space) and matches the groups to known clusters. Kinematics (parallax,
    proper motion, radial velocity) are *independent* of abundances, so they
    provide a clean ground truth.

    This notebook runs a **fast demo**: it restricts the sky to a 30° region
    around **M 67** and caps the field sample, so the whole pipeline finishes
    in about a minute.
    """)
    return


@app.cell
def _():
    from cluster import config
    from cluster.baseline import (
        baseline_labels,
        confusion_matrix_frame,
        plot_confusion,
        separation_scores,
    )
    from cluster.benchmark import run_benchmark
    from cluster.catalog import attach_referee, cluster_panels_cell, hr_cell
    from cluster.clusters import CLUSTERS
    from cluster.data import prepare
    from cluster.isochrone import gaia_age_cell, isochrone_cell
    from cluster.literature import literature_table
    from cluster.plots import (
        abundance_violins,
        embedding_interactive,
        method_comparison_bar,
    )

    settings = config.Settings()
    settings.region_radius_deg = 30.0
    settings.cluster_names = ["M 67"]
    settings.max_stars = 5_000

    print(f"FAST={settings.fast}  MAX_STARS={settings.max_stars}  SNR_MIN={settings.snr_min}")
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

    The pipeline needs the APOGEE DR17 allStar FITS file (~3.7 GB). This cell
    checks that it is present and raises a helpful error if it is not.
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

            Run `uv run cluster download` first, then rerun this notebook.
            """
        ),
    )
    print(f"✓ allStar found: {allstar} ({allstar.stat().st_size / 1e9:.2f} GB)")
    return (allstar,)


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
    return (baseline_table, cm_frames)


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
    referee), not the kinematic labels — see §6.
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
    mo.mpl.interactive(abundance_violins(prepared.df, settings.elements, "M 67"))
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
