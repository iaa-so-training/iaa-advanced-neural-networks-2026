#!/usr/bin/env python
"""Write ``docs/reference_runs/<name>_<date>.json`` — the readings docs quote.

Student docs quote *ranges*, because the scores move by ~±0.02 across machines
(see ``docs/reproducibility.md``). This script is the other half of that policy:
it records the exact reading behind a quoted range, together with the
environment that produced it — a score without its machine, thread setting and
image digest is not a measurement.

It makes the same calls ``cluster run`` makes (prepare → referee → benchmark),
minus the plot and the MLflow logging, so a reference file is a faithful
snapshot of the documented command:

    uv run python scripts/reference_run.py --fast
    uv run python scripts/reference_run.py --fast --spectral \\
        data/embeddings/masked_latent.parquet --out docs/reference_runs/spectral_<date>.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from cluster import config
from cluster.benchmark import run_benchmark
from cluster.catalog import attach_referee
from cluster.cli import _prepared_for
from cluster.clusters import CLUSTERS
from cluster.doctor import fingerprint
from cluster.seeding import seed_everything


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", dest="fast", action="store_true", default=True,
                    help="25 000-star sample (the documented student command).")
    ap.add_argument("--full", dest="fast", action="store_false",
                    help="No star cap (slow: 10–20 min).")
    ap.add_argument("--allstar", default=config.ASTRA_ASPCAP_PATH)
    ap.add_argument("--spectral", default=None,
                    help="Score a spectral embedding instead of the abundances.")
    ap.add_argument("--region", type=float, default=None,
                    help="Restrict to a cone of this radius (degrees) around each cluster.")
    ap.add_argument("--clusters", default=None,
                    help="Comma-separated cluster names (default: all).")
    ap.add_argument("--out", default=None, help="Output path (default: docs/reference_runs/…).")
    return ap.parse_args()


def main() -> None:
    args = _parse()

    settings = config.Settings()
    settings.fast = args.fast
    settings.max_stars = 25_000 if args.fast else None
    if args.region is not None:
        settings.region_radius_deg = args.region
    if args.clusters:
        settings.cluster_names = [c.strip() for c in args.clusters.split(",") if c.strip()]
    if args.spectral:
        settings.require_aspcap_flag_clean = False
    seed_everything(settings.random_state)

    names = settings.resolve_cluster_names([c.name for c in CLUSTERS])
    clusters = [c for c in CLUSTERS if c.name in names]

    print(f"📦 preparing ({len(clusters)} clusters, seed={settings.random_state})…")
    prepared = _prepared_for(args.allstar, settings)
    if args.spectral:
        from cluster.spectral import spectral_prepared

        prepared = spectral_prepared(prepared, args.spectral, settings)
    prepared.df = attach_referee(prepared.df, clusters, settings)

    print("🏃 benchmark (t-SNE / UMAP / EVoC)…")
    result = run_benchmark(prepared, settings)

    macro = result.macro()
    summary = result.summary()
    name = Path(args.spectral).stem if args.spectral else ("fast" if args.fast else "full")
    payload = {
        "name": name,
        "measured_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": (
            "uv run python scripts/reference_run.py "
            + ("--fast" if args.fast else "--full")
            + (f" --spectral {args.spectral}" if args.spectral else "")
            + (f" --region {args.region}" if args.region is not None else "")
            + (f" --clusters {args.clusters}" if args.clusters else "")
        ),
        "environment": fingerprint(deep=False),
        "population": {
            "stars": int(prepared.X.shape[0]),
            "features": int(prepared.X.shape[1]),
            "members": int((prepared.df["cluster"] != "field").sum()),
            "field": int((prepared.df["cluster"] == "field").sum()),
            "clusters_scored": int(macro.shape[0]),
        },
        "macro": macro.to_dict(orient="records"),
        "per_cluster_recall": summary["recall"].round(6).to_dict(),
    }

    out = Path(
        args.out
        or f"docs/reference_runs/{name}_{dt.date.today().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(macro.to_string(index=False))
    print(f"💾 {out}")


if __name__ == "__main__":
    main()
