#!/usr/bin/env python
"""Where does the wall clock go, and how many cores does it actually use?

Answers the question the notebook provokes every year — *"it is slow, but my CPU
is idle"* — with wall time, process CPU time and the ratio between them, per
stage. A ratio near 1.0 means the machine is idle while one core works: that
stage is serial by construction and no amount of extra hardware helps it.

It also shows what the prepared-sample cache buys: the same `prepare` call with
`no_cache=True` (a real read), on a miss (compute + write) and on a hit (from
disk).

Run it inside the published image (the environment the students use):

    docker run --rm -it $DAY4 $IMG uv run python scripts/phase_profile.py
    docker run --rm -it $DAY4 $IMG uv run python scripts/phase_profile.py --full

The 2026-09-24 reference readings (16 cores, `--fast`) are quoted in
`docs/docker.md#why-it-is-slow-even-though-the-machine-looks-idle`.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable

os.environ.setdefault("PYTHONHASHSEED", "42")


def cpu_seconds_all_threads() -> float:
    """CPU time summed over every thread of this process (across all cores on Linux)."""
    total = 0.0
    for task in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{task}/stat") as fh:
                fields = fh.read().rsplit(") ", 1)[1].split()
            utime, stime = int(fields[11]), int(fields[12])
            total += (utime + stime) / os.sysconf("SC_CLK_TCK")
        except OSError:
            pass
    return total


def phase(label: str, fn: Callable[[], Any]) -> Any:
    wall0, cpu0 = time.perf_counter(), cpu_seconds_all_threads()
    out = fn()
    wall, cpu = time.perf_counter() - wall0, cpu_seconds_all_threads() - cpu0
    cores = cpu / max(wall, 1e-9)
    note = "serial" if cores < 1.4 else ("partly parallel" if cores < 3 else "parallel")
    print(
        f"PHASE {label:56s} wall={wall:7.2f}s cpu={cpu:7.2f}s "
        f"cores_used={cores:5.2f} of {os.cpu_count()}  ({note})",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="Use the full all-sky sample instead of --fast.")
    parser.add_argument(
        "--skip-parallel-umap",
        action="store_true",
        help="Skip the unseeded UMAP contrast (fast, but not reproducible).",
    )
    args = parser.parse_args()

    print(f"cpu_count={os.cpu_count()}  pid={os.getpid()}")
    t0 = time.perf_counter()

    import umap as umap_lib

    from cluster import config
    from cluster.benchmark import cluster_embedding, fit_evoc, fit_tsne, fit_umap, run_benchmark
    from cluster.cli import _prepared_for
    from cluster.seeding import seed_everything

    print(f"PHASE {'imports (astropy/sklearn/umap/evoc/numba)':56s} wall={time.perf_counter() - t0:7.2f}s")

    settings = config.Settings()
    if args.full:
        settings.fast = False
    seed_everything(settings.random_state)
    print(
        f"settings: fast={settings.fast} max_stars={settings.max_stars} "
        f"tsne={settings.tsne.get('backend', 'sklearn')} max_iter={settings.tsne.get('max_iter')} "
        f"umap={settings.umap} hdbscan={settings.hdbscan}"
    )

    # --- the catalogue read: the one every edit used to pay again -------------
    phase(
        "prepare --no-cache: read 1.17 GB .fits.gz + cuts + membership",
        lambda: _prepared_for(config.ASTRA_ASPCAP_PATH, settings, no_cache=True),
    )
    prepared = phase(
        "prepare, cache miss (compute + write)",
        lambda: _prepared_for(config.ASTRA_ASPCAP_PATH, settings),
    )
    hit = phase("prepare, cache hit (restore from disk)", lambda: _prepared_for(config.ASTRA_ASPCAP_PATH, settings))
    print(f"      → {hit.X.shape[0]} stars × {hit.X.shape[1]} abundances")
    if prepared.X.shape != hit.X.shape:
        raise SystemExit("cache mismatch: restored sample differs in shape")

    X = prepared.X
    seed = settings.random_state

    # --- the clusterers --------------------------------------------------------
    phase("t-SNE (sklearn BH, 2-D)", lambda: fit_tsne(X, dict(settings.tsne), seed))
    phase(
        "t-SNE again, max_iter=250 (CLUSTER_TSNE_N_ITER, interactive cost)",
        lambda: fit_tsne(X, {**settings.tsne, "max_iter": 250}, seed),
    )
    z_umap = phase("UMAP (random_state set → n_jobs forced to 1)", lambda: fit_umap(X, dict(settings.umap), seed))
    phase("HDBSCAN on the UMAP embedding", lambda: cluster_embedding(z_umap, dict(settings.hdbscan)))
    phase("EVoC (high-D, single-threaded)", lambda: fit_evoc(X, dict(settings.evoc), seed))

    if not args.skip_parallel_umap:
        phase(
            "UMAP contrast: n_jobs=-1, random_state=None (NOT reproducible)",
            lambda: umap_lib.UMAP(n_neighbors=15, random_state=None, n_jobs=-1).fit_transform(X),
        )

    print("(run_benchmark as a whole, for reference — the notebook memoises this call:)")
    phase("run_benchmark(prepared) — the full §3 cell", lambda: run_benchmark(prepared, settings))

    print(
        "\nStages at ~1 core are serial by construction: the on-disk cache and the\n"
        "notebook's memo exist so that editing a plot does not pay for them again."
    )


if __name__ == "__main__":
    main()
