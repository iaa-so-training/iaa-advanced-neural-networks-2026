"""Command-line interface: ``cluster download`` / ``cluster run``."""

from __future__ import annotations

from pathlib import Path

import click

from . import config, tracking


@click.group()
def main() -> None:
    """Chemical tagging of star clusters — t-SNE vs UMAP vs EVoC."""


@main.command()
@click.option(
    "--destination",
    default="data/astraAllStarASPCAP-0.6.0.fits.gz",
    show_default=True,
    help="Where to save the catalog (catalogue download only).",
)
@click.option(
    "--assets",
    is_flag=True,
    help="Fetch the embeddings + checkpoint bundle from Hugging Face instead.",
)
@click.option(
    "--all",
    "everything",
    is_flag=True,
    help="Fetch both the catalogue and the asset bundle.",
)
@click.option(
    "--with-optional",
    is_flag=True,
    help="Also fetch the optional raw-spectra archive (~240 MB).",
)
@click.option("--list", "list_only", is_flag=True, help="List the bundle and exit.")
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    help="Verify the local bundle against the manifest and exit.",
)
@click.option(
    "--only",
    default="",
    help="Comma-separated subset, e.g. 'embeddings/attention_broad_merged.parquet'.",
)
@click.option("--assets-dir", default="data", show_default=True)
@click.option(
    "--hf-repo",
    default=None,
    help="Override the bundle repo (default: config.HF_REPO_ID).",
)
def download(
    destination: str,
    assets: bool,
    everything: bool,
    with_optional: bool,
    list_only: bool,
    check_only: bool,
    only: str,
    assets_dir: str,
    hf_repo: str | None,
) -> None:
    """Fetch the SDSS-V DR19 catalogue (~1.17 GB) and/or the embedding bundle.

    \b
      cluster download              # catalogue only
      cluster download --assets     # embeddings + checkpoints (Hugging Face)
      cluster download --all        # both
      cluster download --assets --list
      cluster download --assets --check
    """
    from . import download as download_module
    from .download import download_allstar

    want_catalogue = everything or not (assets or list_only or check_only)
    if want_catalogue:
        download_allstar(destination)

    if assets or everything or list_only or check_only:
        repo = hf_repo or config.HF_REPO_ID
        wanted = tuple(p.strip() for p in only.split(",") if p.strip())
        download_module.download_assets(
            root=assets_dir,
            repo_id=repo,
            include_optional=with_optional,
            only=wanted,
            list_only=list_only,
            check_only=check_only,
        )


@main.command()
@click.option("--fast/--full", default=None, help="Override config.FAST.")
@click.option("--max-stars", type=int, default=None, help="Override config.MAX_STARS.")
@click.option(
    "--region", type=float, default=None,
    help="Restrict to stars within this many degrees of the selected cluster(s) — the paper's per-region approach.",
)
@click.option(
    "--region-scaled", is_flag=True,
    help="Scale the region per cluster (max(3 deg, 10 x angular diameter)).",
)
@click.option(
    "--cluster", "cluster_names", multiple=True, default=None,
    help="Restrict to a cluster by name (repeatable; default: all).",
)
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
@click.option("--outdir", default="results", show_default=True)
@click.option(
    "--spectral", "spectral_path", default=None, type=click.Path(exists=True),
    help="Use spectral embeddings (parquet) instead of abundances.",
)
def run(
    fast: bool | None,
    max_stars: int | None,
    region: float | None,
    cluster_names: tuple[str, ...] | None,
    allstar: str,
    outdir: str,
    spectral_path: str | None,
    region_scaled: bool = False,
) -> None:
    """Prepare the data, run the benchmark, print the score table."""
    from .benchmark import knn_purity, run_benchmark
    from .clusters import CLUSTERS
    from .data import prepare
    from .plots import plot_method_grid

    settings = config.Settings()
    if fast is not None:
        settings.fast = fast
    if spectral_path is not None:
        # embeddings come from the raw spectrum; ASPCAP flag issues don't apply
        settings.require_aspcap_flag_clean = False
    if region_scaled:
        settings.region_scaled = True
    if max_stars is not None:
        settings.max_stars = max_stars
    elif fast is not None:
        # the FAST toggle implies its default cap (25k / None)
        settings.max_stars = 25_000 if fast else None
    if region is not None:
        settings.region_radius_deg = region
    if cluster_names:
        settings.cluster_names = list(cluster_names)

    allstar_path = Path(allstar)
    if not allstar_path.exists():
        raise click.ClickException(
            f"{allstar_path} not found. Run `cluster download` first."
        )

    clusters = [c for c in CLUSTERS if c.name in settings.resolve_cluster_names(
        [c.name for c in CLUSTERS]
    )]

    try:
        tracking.start_run("cluster-benchmark")
        tracking.log_params({
            "FAST": str(settings.fast),
            "MAX_STARS": str(settings.max_stars),
            "SNR_MIN": str(settings.snr_min),
            "n_elements": str(len(settings.elements)),
            "region": str(settings.region_radius_deg),
            "cluster_count": str(len(clusters)),
        })

        click.echo(
            f"🧪 FAST={settings.fast}  MAX_STARS={settings.max_stars}  "
            f"SNR_MIN={settings.snr_min}  ELEMENTS={len(settings.elements)}"
            f"  REGION={settings.region_radius_deg}"
        )
        click.echo("📦 Preparing data (loading allStar, quality cuts, membership)...")
        prepared = prepare(
            allstar_path, settings, clusters,
            seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
            seed_parallax_frac=config.SEED_PARALLAX_FRAC,
            seed_pm_tol=config.SEED_PM_TOL,
            seed_rv_tol=config.SEED_RV_TOL,
            n_refine_passes=config.N_REFINE_PASSES,
            refine_sigma=config.REFINE_SIGMA,
        )

        if spectral_path is not None:
            from .spectral import spectral_prepared

            prepared = spectral_prepared(prepared, spectral_path, settings)
            click.echo(
                f"   → spectral embeddings: {prepared.X.shape[0]} stars "
                f"× {prepared.X.shape[1]} latent dims"
            )

        # external referee (Simbad catalogue) for the benchmark scores
        from .catalog import attach_referee

        prepared.df = attach_referee(prepared.df, clusters, settings)

        click.echo(f"   → {prepared.X.shape[0]} stars × {prepared.X.shape[1]} abundances")
        per_cluster = prepared.df.groupby("cluster").size()
        for name, n in per_cluster.items():
            click.echo(f"     member({name}): {n}")
        ref_per = prepared.df.groupby("referee").size()
        for name, n in ref_per.items():
            click.echo(f"     referee({name}): {n}")

        click.echo("🏃 Running benchmark (t-SNE / UMAP / EVoC)...")
        result = run_benchmark(prepared, settings)

        click.echo("\n=== macro recall/precision ===")
        click.echo(result.macro().to_string(index=False))

        click.echo("\n=== chemical cohesion (kNN purity, members sit together?) ===")
        true_labels = prepared.df["cluster"].to_numpy()
        for name in ("t-SNE", "UMAP"):
            r = result.results.get(name)
            if r is not None and r.embedding is not None:
                purity = knn_purity(r.embedding, true_labels)
                macro = sum(purity.values()) / len(purity) if purity else float("nan")
                click.echo(f"  {name}: macro={macro:.3f}  ({len(purity)} clusters)")

        click.echo("\n=== per-cluster recall (rows) × method (cols) ===")
        summary = result.summary()
        click.echo(summary["recall"].round(2).to_string())

        Path(outdir).mkdir(parents=True, exist_ok=True)
        plot_path = Path(outdir) / "benchmark_grid.png"
        plot_method_grid(result, plot_path)
        click.echo(f"\n🖼  Plot saved to {plot_path}")

        # benchmark.run_benchmark already logs per-method macro metrics + kNN
        # purity; here we only record the run's parameters and the plot.
        tracking.log_artifact(plot_path)
    finally:
        tracking.end_run()


@main.command()
@click.option("--kinematics", is_flag=True, help="Append standardised kinematics (parallax / PM / RV) to the abundance features.")
@click.option("--min-members", type=int, default=5, show_default=True, help="Drop clusters with fewer members (paper's >=5 rule).")
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
@click.option("--outdir", default=None, help="If set, save a confusion-matrix figure per method here.")
@click.option(
    "--spectral", "spectral_path", default=None, type=click.Path(exists=True),
    help="Use spectral embeddings (parquet) instead of abundances.",
)
def baseline(
    kinematics: bool, min_members: int, allstar: str, outdir: str | None,
    spectral_path: str | None,
) -> None:
    """Paper baseline: cluster-only multiclass separation (Garcia-Dias 2019)."""
    import numpy as np
    import pandas as pd

    from .baseline import (
        baseline_labels,
        confusion_matrix_frame,
        plot_confusion,
        separation_scores,
    )
    from .clusters import CLUSTERS
    from .data import prepare

    settings = config.Settings()
    allstar_path = Path(allstar)
    if not allstar_path.exists():
        raise click.ClickException(
            f"{allstar_path} not found. Run `cluster download` first."
        )
    if spectral_path is not None:
        # embeddings come from the raw spectrum; ASPCAP flag issues don't apply
        settings.require_aspcap_flag_clean = False

    clusters = [c for c in CLUSTERS if c.name in settings.resolve_cluster_names(
        [c.name for c in CLUSTERS]
    )]

    click.echo(
        f"🧪 paper baseline  KINEMATICS={'on' if kinematics else 'off'}  "
        f"MIN_MEMBERS={min_members}  ELEMENTS={len(settings.elements)}"
        f"  FEATURES={'spectral' if spectral_path else 'abundances'}"
    )
    click.echo("📦 Preparing data (cluster members only)...")
    prepared = prepare(
        allstar_path, settings, clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )
    if spectral_path is not None:
        from .spectral import spectral_prepared

        prepared = spectral_prepared(prepared, spectral_path, settings)
        click.echo(f"   → spectral embeddings: {prepared.X.shape[0]} stars × {prepared.X.shape[1]} latent dims")
    true_labels, labels = baseline_labels(
        prepared, settings, use_kinematics=kinematics, min_members=min_members,
    )

    rows = [
        {
            "method": name,
            "n_stars": int(true_labels.size),
            "n_clusters": int(np.unique(true_labels).size) if true_labels.size else 0,
            **separation_scores(true_labels, pred),
        }
        for name, pred in labels.items()
    ]
    table = pd.DataFrame(rows)
    click.echo(
        "\n=== cluster-only separation "
        "(homogeneity / completeness / v-measure / accuracy) ==="
    )
    click.echo(table.round(3).to_string(index=False))

    if outdir is not None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        tag = "kin" if kinematics else "chem"
        for name, pred in labels.items():
            cm = confusion_matrix_frame(true_labels, pred)
            path = out / f"baseline_confusion_{name.replace('-', '').lower()}_{tag}.png"
            plot_confusion(
                cm, path,
                title=f"{name} — "
                f"{'abundances + kinematics' if kinematics else 'abundances only'}",
            )
            click.echo(f"🖼  Confusion matrix saved to {path}")


@main.command()
@click.option("--cluster", "cluster_name", required=True, help="Cluster name (e.g. 'M 67').")
@click.option("--region", type=float, default=30.0, show_default=True)
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
@click.option("--outdir", default="results", show_default=True)
@click.option("--force-simbad", is_flag=True, help="Re-query Simbad, ignoring the CSV cache.")
def hr(
    cluster_name: str,
    region: float,
    allstar: str,
    outdir: str,
    force_simbad: bool,
) -> None:
    """HR-diagram comparison: Simbad catalogue vs kinematic vs combined membership."""
    from .catalog import membership_masks_for
    from .clusters import CLUSTER_BY_NAME
    from .data import apply_quality_cuts, complete_case, load_allstar, make_matrix
    from .membership import angular_separation
    from .plots import hr_comparison

    settings = config.Settings()
    cluster = CLUSTER_BY_NAME.get(cluster_name)
    if cluster is None:
        raise click.ClickException(
            f"unknown cluster {cluster_name!r}; choose from "
            f"{sorted(CLUSTER_BY_NAME)}"
        )

    allstar_path = Path(allstar)
    if not allstar_path.exists():
        raise click.ClickException(
            f"{allstar_path} not found. Run `cluster download` first."
        )

    click.echo(f"📦 loading + cutting {cluster.name} (region {region}°)...")
    df = load_allstar(allstar_path, settings.elements)
    df = apply_quality_cuts(df, settings)
    sep = angular_separation(
        df["RA"].to_numpy(dtype=float), df["DEC"].to_numpy(dtype=float),
        cluster.ra_deg, cluster.dec_deg,
    )
    df = df[sep <= region]
    df = complete_case(df, settings)
    X = make_matrix(df, settings)

    click.echo("🔭 querying Simbad catalogue + computing memberships...")
    masks = membership_masks_for(df, cluster, X, settings, force=force_simbad)

    cat = masks["catalog"]
    kin = masks["kinematic"]
    comb = masks["combined"]
    click.echo(f"\n{cluster.name}: {len(df)} region stars")
    click.echo(f"  catalog (Simbad ≥{settings.simbad_membership_min}%): {cat.sum()}")
    click.echo(f"  kinematic: {kin.sum()}   combined: {comb.sum()}")
    click.echo(
        f"  recall vs catalog:    kin={ (kin & cat).sum() / max(int(cat.sum()), 1):.3f}"
        f"   combined={(comb & cat).sum() / max(int(cat.sum()), 1):.3f}"
    )
    click.echo(
        f"  precision vs catalog: kin={(kin & cat).sum() / max(int(kin.sum()), 1):.3f}"
        f"   combined={(comb & cat).sum() / max(int(comb.sum()), 1):.3f}"
    )

    Path(outdir).mkdir(parents=True, exist_ok=True)
    out = Path(outdir) / f"hr_{cluster.name.replace(' ', '_')}.png"
    hr_comparison(df, masks, cluster.name, out)
    click.echo(f"\n🖼  HR diagram saved to {out}")


def _prepared_for(allstar: str, settings: config.Settings):
    """Shared loader for the audit commands."""
    from .clusters import CLUSTERS
    from .data import prepare

    allstar_path = Path(allstar)
    if not allstar_path.exists():
        raise click.ClickException(
            f"{allstar_path} not found. Run `cluster download` first.",
        )
    names = settings.resolve_cluster_names([c.name for c in CLUSTERS])
    clusters = [c for c in CLUSTERS if c.name in names]
    return prepare(
        allstar_path, settings, clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )


@main.command()
@click.option(
    "--spectral", "spectral_path",
    default="data/embeddings/masked_latent_all.parquet", show_default=True,
    help="Embedding artifact to audit for a data-release batch effect.",
)
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
def provenance(spectral_path: str, allstar: str) -> None:
    """Batch-effect check: does the latent encode DR17 vs DR19?

    ``masked_latent_all.parquet`` merges two reductions, and the merge is not
    random — nearly all field stars are DR19 while most cluster members are
    the DR17 backfill. This command measures how much of a member-vs-field
    score could be explained by that split alone.
    """
    from .provenance import attach_source, format_report, provenance_report
    from .spectral import spectral_prepared

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False

    click.echo("📦 Preparing data...")
    prepared = spectral_prepared(
        _prepared_for(allstar, settings), spectral_path, settings,
    )
    click.echo(
        f"   → {prepared.X.shape[0]} stars × {prepared.X.shape[1]} latent dims",
    )

    df = attach_source(prepared.df)
    click.echo("")
    click.echo(format_report(provenance_report(df, prepared.X)))


@main.command("head-to-head")
@click.option(
    "--arm", "arm_specs", multiple=True, metavar="LABEL=PATH",
    help="Feature arm, e.g. 'masked AE 256-d=data/embeddings/masked_latent.parquet'. "
         "Use PATH='abundances' for the ASPCAP baseline. Repeatable.",
)
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
@click.option("--min-members", type=int, default=5, show_default=True)
@click.option("--seeds", default="42,0,1,2,7,13,99", show_default=True)
@click.option("--out", "out_path", default=None, help="Write the score table as CSV.")
def head_to_head_cmd(
    arm_specs: tuple[str, ...], allstar: str, min_members: int,
    seeds: str, out_path: str | None,
) -> None:
    """Compare feature sets on ONE common population, with seed error bars.

    Every arm is scored on the intersection of their ``APOGEE_ID`` sets, so
    the rows are actually comparable. Quoting a 25-cluster abundance score
    next to a 5-cluster spectral score is the mistake this command prevents.
    """
    from .headtohead import Arm, head_to_head, pivot_with_errors

    if not arm_specs:
        arm_specs = (
            "abundances (16-d)=abundances",
            "PCA 64-d=data/embeddings/pca_64.parquet",
            "PCA 256-d=data/embeddings/pca_256.parquet",
            "masked AE 256-d=data/embeddings/masked_latent.parquet",
            "supervised CNN 64-d=data/embeddings/attention_dr19_convpool.parquet",
        )

    arms = []
    for spec in arm_specs:
        if "=" not in spec:
            raise click.ClickException(f"bad --arm {spec!r}; expected LABEL=PATH")
        label, path = spec.split("=", 1)
        arms.append(
            Arm(label.strip(), None if path.strip() == "abundances" else path.strip()),
        )

    settings = config.Settings()
    settings.require_aspcap_flag_clean = False
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    click.echo("📦 Preparing data...")
    prepared = _prepared_for(allstar, settings)

    click.echo(f"⚖  Intersecting {len(arms)} arms onto a common population...")
    result = head_to_head(
        prepared, arms, settings, min_members=min_members, seeds=seed_list,
    )

    click.echo(
        f"\n=== common population: {result.n_stars} stars, "
        f"{result.n_clusters} clusters ===",
    )
    click.echo(f"clusters: {', '.join(result.clusters)}")
    lost = {k: v for k, v in result.dropped.items() if v}
    if lost:
        click.echo(f"members dropped to reach the intersection: {lost}")

    click.echo(f"\n=== homogeneity, mean ± std over {len(seed_list)} seeds ===")
    click.echo(pivot_with_errors(result).to_string())

    degen = result.degeneracy
    collapsed = degen[degen["degenerate"].to_numpy(dtype=bool)]
    if not collapsed.empty:
        click.echo("\n⚠ degenerate partitions (clusterer collapsed, score is a floor):")
        click.echo(
            collapsed[["features", "method", "n_clusters", "largest_fraction"]]
            .to_string(index=False),
        )
        click.echo(
            "  Ratios quoted against these numbers compare against a failure "
            "mode, not against the baseline.",
        )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        result.scores.to_csv(out_path, index=False)
        stab = str(out_path).replace(".csv", "_stability.csv")
        result.stability.to_csv(stab, index=False)
        click.echo(f"\n💾 {out_path}\n💾 {stab}")


@main.command()
@click.option("--spectral", "spectral_path", default=None, help="Embedding artifact.")
@click.option("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz", show_default=True)
@click.option("--exclude", multiple=True, help="Cluster to drop, e.g. --exclude 'M 3'. Repeatable.")
@click.option("--min-members", type=int, default=5, show_default=True)
@click.option("--seeds", default="42,0,1,2,7,13,99", show_default=True)
def ablate(
    spectral_path: str | None, allstar: str, exclude: tuple[str, ...],
    min_members: int, seeds: str,
) -> None:
    """Re-score with clusters removed — e.g. drop the globular M 3.

    The flagship sample is 44% one globular cluster, so "we separate clusters"
    could just mean "we separate a globular from open clusters". Dropping it
    tests whether the signal is real chemical tagging.
    """
    from .baseline import baseline_matrix, cluster_only
    from .spectral import spectral_prepared
    from .stability import format_stability, stability

    import pandas as pd

    settings = config.Settings()
    if spectral_path is not None:
        settings.require_aspcap_flag_clean = False
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    click.echo("📦 Preparing data...")
    prepared = _prepared_for(allstar, settings)
    if spectral_path is not None:
        prepared = spectral_prepared(prepared, spectral_path, settings)
        columns = list(prepared.elements)
    else:
        columns = list(settings.elements)

    sub = cluster_only(prepared.df)
    counts = sub["cluster"].value_counts()
    keep = [c for c in counts.index if int(counts[c]) >= min_members]
    sub = sub[pd.Series(sub["cluster"]).isin(keep).to_numpy()]

    for name in exclude:
        before = len(sub)
        sub = sub[(sub["cluster"] != name).to_numpy()]
        click.echo(f"   − dropped {name}: {before - len(sub)} stars")

    if sub.empty:
        raise click.ClickException("no stars left after exclusions")

    true_labels = sub["cluster"].to_numpy()
    X = baseline_matrix(sub, settings, False, elements=columns)
    click.echo(
        f"\n=== {len(true_labels)} stars, "
        f"{len(set(true_labels))} clusters: {', '.join(sorted(set(true_labels)))} ===",
    )
    click.echo(format_stability(stability(X, true_labels, settings, seeds=seed_list)))


if __name__ == "__main__":
    main()
