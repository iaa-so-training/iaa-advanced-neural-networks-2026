"""Spectral-embedding feature source for chemical tagging.

The lightsurf repo (the spectrum → abundance RNN) exports the trained model's
latent layer as a data artifact — a parquet/CSV with an identifier column plus
one column per latent dimension. This module reads that artifact, aligns it to
the allStar rows, and post-processes it into a clustering-ready matrix with
the same treatment as the abundance matrix (standardise + L2-normalise).

Keeping TensorFlow out of this repo is deliberate: embedding *extraction* is a
forward pass done once in lightsurf (on the GPU machine), and this repo only
*consumes* the exported matrix — so the clustering benchmark stays TF-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data import PreparedData

ID_COLUMN = "APOGEE_ID"


def load_embedding_frame(path: str | Path) -> pd.DataFrame:
    """Read an embeddings artifact (parquet or CSV).

    Expected shape: one identifier column (``APOGEE_ID``) + N latent columns.
    """
    path = Path(path)
    if path.suffix in {".parquet", ".pq", ".pqt"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    """Return the latent-dimension columns (everything except the identifier)."""
    return [c for c in frame.columns if c != ID_COLUMN]


def align_embeddings(
    df: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Inner-join embeddings onto ``df`` on the identifier column.

    Only stars present in *both* frames are kept (a star without a spectrum,
    or without an embedding, cannot be clustered in spectral space).
    """
    if id_column not in df.columns:
        raise ValueError(f"allStar frame has no column {id_column!r}")
    if id_column not in embedding_frame.columns:
        raise ValueError(f"embedding frame has no column {id_column!r}")
    return df.merge(embedding_frame, on=id_column, how="inner")


def embedding_matrix(
    df: pd.DataFrame, embed_columns: list[str], settings: Settings,
) -> np.ndarray:
    """Standardise (zero median, unit std) + L2-normalise the embeddings.

    Mirrors the abundance path in ``data.make_matrix`` (without imputation —
    embeddings are complete by construction). L2-normalisation keeps Euclidean
    distance equal to cosine, so t-SNE/UMAP compare fairly with EVoC.
    """
    X = df[embed_columns].to_numpy(dtype=float)
    if settings.standardize:
        med = np.nanmedian(X, axis=0)
        scale = np.nanstd(X, axis=0)
        scale[scale == 0] = 1.0
        X = (X - med) / scale
    if settings.normalize_rows:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    return X


def spectral_prepared(
    prepared: PreparedData, embedding_path: str | Path, settings: Settings,
) -> PreparedData:
    """Swap the abundance matrix for spectral embeddings (same pipeline).

    Loads the embedding artifact, inner-joins it onto ``prepared.df`` on
    ``APOGEE_ID`` (stars without an embedding are dropped), and rebuilds the
    clustering matrix from the latent columns. The result feeds the same
    ``run_benchmark`` / ``run_baseline`` entry points unchanged.
    """
    frame = load_embedding_frame(embedding_path)
    cols = embedding_columns(frame)
    # the allStar frame has one row per visit; embeddings are one per unique
    # star (combined spectrum). Dedup before the join so a star isn't counted
    # once per visit.
    df = prepared.df.drop_duplicates(subset=[ID_COLUMN], keep="first")
    merged = align_embeddings(df, frame)
    X = embedding_matrix(merged, cols, settings)
    return PreparedData(df=merged, X=X, elements=cols)
