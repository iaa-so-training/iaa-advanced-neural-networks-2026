"""Generate the didactic plots + GIFs for the IAA-SO chemical-tagging deck.

Run from the repo root:

    uv run python scripts/make_teaching_assets.py

Outputs land in results/teaching/ and are then copied into the deck's asset
folder (public/presentations/iaa-so-chemical-tagging-2026/).

Each asset is a self-contained teaching figure: a title, an iteration counter
where relevant, a legend, and a footnote reference to the paper behind the
algorithm.
"""

from __future__ import annotations

import os
import sys

# Allow `from teaching_assets import ...` regardless of the CWD.
sys.path.insert(0, os.path.dirname(__file__))

from teaching_assets import common, embeddings, iterative, steps  # noqa: E402


def main() -> None:
    common.set_out("results/teaching")
    os.makedirs(common.OUT, exist_ok=True)

    jobs = [
        # K-means
        iterative.make_kmeans_gif,
        iterative.make_kmeans_init,
        iterative.make_kmeans_fail,
        # KNN
        iterative.make_knn_probe,
        # DBSCAN
        iterative.make_dbscan_gif,
        iterative.make_dbscan_static,
        # HDBSCAN*
        iterative.make_mutual_reachability,
        iterative.make_hdbscan_sweep,
        # PLSCAN
        iterative.make_plscan_gif,
        iterative.make_plscan_barcode,
        # Embeddings
        embeddings.make_tsne_gif,
        embeddings.make_umap_gif,
        # EVoC + C-space
        embeddings.make_evoc_panel,
        embeddings.make_cspace_corner,
        # Per-step walkthrough stills (one PNG per logical step). These back the
        # step slides in the deck, so they have to rebuild with everything else -
        # they were previously generated out-of-band, which meant this script
        # silently produced an incomplete asset set.
        steps.make_kmeans_steps,
        steps.make_dbscan_steps,
        steps.make_hdbscan_steps,
        steps.make_plscan_steps,
        steps.make_tsne_steps,
        steps.make_umap_steps,
    ]

    for job in jobs:
        name = getattr(job, "__name__", str(job))
        try:
            path = job()
            print(f"  ok  {path}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\nDone. Assets in {os.path.abspath(common.OUT)}/")


if __name__ == "__main__":
    main()
