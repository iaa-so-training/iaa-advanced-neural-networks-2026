"""Reproducible benchmark experiment: run one config, log everything to MLflow.

Every tweak experiment goes through this harness so that all runs are
comparable and bit-reproducible:

* fixed ``random_state=42`` (forced, regardless of env)
* config overrides passed as ``--set KEY=VALUE`` (mapped to ``CLUSTER_*`` env
  before ``cluster`` is imported, so module-level defaults pick them up)
* the full config snapshot, git commit, data-file size and key package
  versions are logged as MLflow params
* recall / precision / kNN-purity (vs the Simbad referee) are logged as
  metrics; the embedding grid and macro table are logged as artifacts

Usage (from the repo root):

    uv run python scripts/experiment.py --name baseline --cluster "M 67"
    uv run python scripts/experiment.py --name hdbscan10 --cluster "M 67" \\
        --set HDBSCAN_MIN_CLUSTER_SIZE=10

All runs land in the ``chemical-tagging-tweaks`` MLflow experiment (override
with ``MLFLOW_EXPERIMENT_NAME``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _package_versions() -> dict[str, str]:
    import importlib.metadata

    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "scikit-learn", "umap-learn", "hdbscan", "evoc", "astropy", "mlflow"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "n/a"
    return versions


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="run name (logged to MLflow)")
    ap.add_argument("--cluster", default="M 67", help="cluster to evaluate")
    ap.add_argument("--region", type=float, default=30.0, help="sky cut in degrees")
    ap.add_argument("--max-stars", type=int, default=5000, help="field-star cap")
    ap.add_argument("--allstar", default="data/astraAllStarASPCAP-0.6.0.fits.gz")
    ap.add_argument("--outdir", default="results/experiments")
    ap.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="config override (repeatable), e.g. --set HDBSCAN_MIN_CLUSTER_SIZE=10",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    return ap.parse_args()


def _apply_overrides(overrides: list[str]) -> dict[str, str]:
    """Set CLUSTER_* env vars before cluster is imported; return them."""
    applied: dict[str, str] = {}
    for item in overrides:
        key, _, value = item.partition("=")
        if not key:
            continue
        applied[key] = value
        os.environ[f"CLUSTER_{key}"] = value
    return applied


def _flatten_params(settings: Any) -> dict[str, str]:
    """Flatten the Settings model into string params (dicts -> JSON)."""
    flat: dict[str, str] = {}
    for key, value in settings.model_dump().items():
        if isinstance(value, dict):
            flat[f"cfg.{key}"] = json.dumps(value, sort_keys=True)
        else:
            flat[f"cfg.{key}"] = str(value)
    return flat


def main() -> None:
    args = _parse()
    applied = _apply_overrides(args.set)
    # isolate the tweak experiments from the plain benchmark runs
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "chemical-tagging-tweaks")

    # import cluster only after the env overrides are in place
    from cluster import config, tracking
    from cluster.benchmark import knn_purity, run_benchmark
    from cluster.catalog import attach_referee
    from cluster.clusters import CLUSTERS
    from cluster.config import Settings
    from cluster.data import prepare
    from cluster.plots import plot_method_grid

    settings = Settings()
    settings.region_radius_deg = args.region
    settings.cluster_names = [args.cluster]
    settings.max_stars = args.max_stars
    settings.random_state = args.seed  # forced fixed seed

    clusters = [
        c for c in CLUSTERS
        if c.name in settings.resolve_cluster_names([c.name for c in CLUSTERS])
    ]

    allstar_path = Path(args.allstar)
    if not allstar_path.exists():
        raise SystemExit(f"{allstar_path} not found. Run `cluster download` first.")

    params = {
        "experiment.name": args.name,
        "git.commit": _git_commit(),
        "allstar": str(allstar_path),
        "allstar_bytes": str(allstar_path.stat().st_size),
        "overrides": json.dumps(applied, sort_keys=True) if applied else "{}",
        "seed": str(args.seed),
        **{f"ver.{k}": v for k, v in _package_versions().items()},
        **_flatten_params(settings),
    }

    tracking.start_run(args.name)
    try:
        tracking.log_params(params)

        print(f"🧪 {args.name}: cluster={args.cluster} region={args.region}° "
              f"max_stars={args.max_stars} seed={args.seed}")
        prepared = prepare(
            allstar_path, settings, clusters,
            seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
            seed_parallax_frac=config.SEED_PARALLAX_FRAC,
            seed_pm_tol=config.SEED_PM_TOL,
            seed_rv_tol=config.SEED_RV_TOL,
            n_refine_passes=config.N_REFINE_PASSES,
            refine_sigma=config.REFINE_SIGMA,
        )
        prepared.df = attach_referee(prepared.df, clusters, settings)
        result = run_benchmark(prepared, settings)

        # metrics vs the Simbad referee
        true = prepared.df["referee"].to_numpy()
        macro = result.macro()
        metrics: dict[str, float] = {}
        for _, row in macro.iterrows():
            metrics[f"{row['method']}_recall"] = float(row["recall"])
            metrics[f"{row['method']}_precision"] = float(row["precision"])
        for name in ("t-SNE", "UMAP"):
            r = result.results.get(name)
            if r is not None and r.embedding is not None:
                purity = knn_purity(r.embedding, true)
                if purity:
                    metrics[f"{name}_knn_purity"] = float(sum(purity.values()) / len(purity))
        metrics["n_referee_members"] = float(int((true != "field").sum()))
        metrics["n_stars"] = float(len(prepared.df))
        tracking.log_metrics(metrics)

        # artifacts
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plot_path = outdir / f"{args.name}_grid.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_method_grid(result, plot_path)
        tracking.log_artifact(plot_path)

        macro_path = outdir / f"{args.name}_macro.csv"
        macro.to_csv(macro_path, index=False)
        tracking.log_artifact(macro_path)

        print(f"   → {metrics}\n   → artifacts: {plot_path.name}, {macro_path.name}")
    finally:
        tracking.end_run()


if __name__ == "__main__":
    main()
