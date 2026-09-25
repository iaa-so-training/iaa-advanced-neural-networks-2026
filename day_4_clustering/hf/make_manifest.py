#!/usr/bin/env python
"""Build ``hf/MANIFEST.json`` — the single source of truth for the HF bundle.

Walks the curated file list, hashes each local file, records parquet
shape, and writes the manifest the downloader (`cluster download --assets`)
verifies against. Run from the repo root:

    .venv/bin/python hf/make_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "hf" / "MANIFEST.json"

# path_in_repo -> (local path, consumer)
FILES: dict[str, tuple[str, str]] = {
    # --- Track A (docs/student_activities.md §3, README spectral arm) ----------
    "embeddings/attention_broad_merged.parquet": (
        "data/embeddings/attention_broad_merged.parquet",
        "Track A: cluster run --spectral  (student_activities.md:87-99)",
    ),
    # --- headline masked-AE arm (docs/spectral_benchmark_results.md) ----------
    "embeddings/masked_latent.parquet": (
        "data/embeddings/masked_latent.parquet",
        "uniform DR19 masked-AE arm: head-to-head, ablate, provenance",
    ),
    "embeddings/masked_latent_all.parquet": (
        "data/embeddings/masked_latent_all.parquet",
        "same rows as masked_latent.parquet (alias kept for provenance)",
    ),
    "embeddings/masked_latent_combined.parquet": (
        "data/embeddings/masked_latent_combined.parquet",
        "member+field combined set (72 410 stars)",
    ),
    "embeddings/masked_latent_field.parquet": (
        "data/embeddings/masked_latent_field.parquet",
        "field-only intermediate arm",
    ),
    "embeddings/masked_latent_members.parquet": (
        "data/embeddings/masked_latent_members.parquet",
        "member-only intermediate arm",
    ),
    "embeddings/masked_latent_all_mixed_v1.parquet": (
        "data/embeddings/masked_latent_all_mixed_v1.parquet",
        "mixed-product arm (superseded by the uniform re-run)",
    ),
    "embeddings/masked_latent_dr19_apstar_v1.parquet": (
        "data/embeddings/masked_latent_dr19_apstar_v1.parquet",
        "pre-uniform DR19 arm (superseded; the product-mismatch artifact)",
    ),
    "embeddings/masked_latent_dr17.parquet": (
        "data/embeddings/masked_latent_dr17.parquet",
        "DR17 arm: cluster provenance (DR17-vs-DR19 batch effect)",
    ),
    "embeddings/masked_latent_dr17_missing.parquet": (
        "data/embeddings/masked_latent_dr17_missing.parquet",
        "738 DR17 backfill latents: cluster provenance",
    ),
    # --- supervised arms ------------------------------------------------------
    "embeddings/attention_dr19.parquet": (
        "data/embeddings/attention_dr19.parquet",
        "supervised DR19 attention latent (31 506 stars)",
    ),
    "embeddings/attention_dr19_long.parquet": (
        "data/embeddings/attention_dr19_long.parquet",
        "supervised DR19 long re-train: scripts/spectral_dr19_analysis.py",
    ),
    "embeddings/attention_dr19_convpool.parquet": (
        "data/embeddings/attention_dr19_convpool.parquet",
        "supervised CNN 64-d arm in cluster head-to-head",
    ),
    "embeddings/attention_merged.parquet": (
        "data/embeddings/attention_merged.parquet",
        "DR17 supervised arm",
    ),
    "embeddings/attention_all.parquet": (
        "data/embeddings/attention_all.parquet",
        "DR17 supervised arm (all)",
    ),
    "embeddings/attention_norm_merged.parquet": (
        "data/embeddings/attention_norm_merged.parquet",
        "normalised-input ablation arm",
    ),
    # --- linear + disentangled arms ------------------------------------------
    "embeddings/pca_64.parquet": (
        "data/embeddings/pca_64.parquet",
        "PCA 64-d linear baseline (competes with the AE — key result)",
    ),
    "embeddings/pca_256.parquet": (
        "data/embeddings/pca_256.parquet",
        "PCA 256-d linear baseline",
    ),
    "embeddings/disentangled_merged.parquet": (
        "data/embeddings/disentangled_merged.parquet",
        "Phase-B disentangled AE arm (16-d)",
    ),
    "embeddings/disentangled_64_merged.parquet": (
        "data/embeddings/disentangled_64_merged.parquet",
        "Phase-B disentangled AE arm (64-d)",
    ),
    # --- raw-flux control + joins --------------------------------------------
    "embeddings/members_rawflux.parquet": (
        "data/embeddings/members_rawflux.parquet",
        "raw 8 575-px flux control arm (1 034 member spectra)",
    ),
    "embeddings/members_dense_0.parquet": (
        "data/embeddings/members_dense_0.parquet",
        "supervised dense-tap arm (20-d)",
    ),
    "embeddings/members_dense_1.parquet": (
        "data/embeddings/members_dense_1.parquet",
        "supervised dense-tap arm (8-d)",
    ),
    "embeddings/cluster_members.csv": (
        "data/embeddings/cluster_members.csv",
        "member list used by the embedding exports",
    ),
    "embeddings/broad_star_list.csv": (
        "data/embeddings/broad_star_list.csv",
        "star list behind the DR17 broad spectra download",
    ),
    # --- checkpoints ----------------------------------------------------------
    "models/masked_ae_rerun.pt": (
        "data/embeddings/masked_ae_rerun.pt",
        "masked autoencoder used to produce masked_latent.parquet: "
        "scripts/embed_dr19_rerun.py (--model default), deck walkthrough figures",
    ),
    "models/masked_ae.pt": (
        "data/embeddings/masked_ae.pt",
        "released masked-AE checkpoint (same architecture as masked_ae_rerun.pt)",
    ),
    "models/model_dr19.pt": (
        "data/embeddings/model_dr19.pt",
        "supervised CNN-LSTM-attention checkpoint (DR19) — load with "
        "cluster.models.CnnLstmAttention, NOT the masked AE",
    ),
    "models/model.pt": (
        "data/embeddings/model.pt",
        "supervised attention model (DR19, 31 506 stars)",
    ),
    "models/model_long.pt": (
        "data/embeddings/model_long.pt",
        "supervised attention model, long re-train",
    ),
    "models/model_convpool.pt": (
        "data/embeddings/model_convpool.pt",
        "supervised conv-pool model (64-d arm)",
    ),
    # --- optional: raw DR19 spectra used by the re-embed + deck figures -------
    "optional/mwmstar.tar": (
        "data/mwmstar.tar",
        "736 DR19 mwmStar FITS (re-embed / masked-AE figure regeneration)",
    ),
}

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def parquet_shape(path: Path) -> dict[str, int] | None:
    if path.suffix != ".parquet":
        return None
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        return {"rows": pf.metadata.num_rows, "columns": len(pf.schema_arrow.names)}
    except Exception:  # pragma: no cover - manifest is still useful without it
        return None


def main() -> None:
    files: list[dict[str, object]] = []
    missing: list[str] = []
    total = 0
    for path_in_repo, (local, consumer) in sorted(FILES.items()):
        p = REPO_ROOT / local
        if not p.exists():
            missing.append(local)
            continue
        size = p.stat().st_size
        total += size
        entry: dict[str, object] = {
            "path": path_in_repo,
            "source": local,
            "bytes": size,
            "sha256": sha256(p),
            "consumer": consumer,
        }
        shape = parquet_shape(p)
        if shape is not None:
            entry.update(shape)
        files.append(entry)

    manifest = {
        "dataset": "chemical tagging of star clusters — IAA-SO school 2026",
        "generated_from": "iaa-advanced-neural-networks-2026-draft",
        "total_bytes": total,
        "n_files": len(files),
        "files": files,
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {OUT} — {len(files)} files, {total / 1e6:.0f} MB")
    for m in missing:
        print(f"  MISSING (not staged): {m}")


if __name__ == "__main__":
    main()
