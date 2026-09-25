"""Data-product provenance tracking for spectral embeddings.

``masked_latent_all.parquet`` is a *union* of two **data products**: SDSS-V
DR19 ``apStar`` spectra (raw flux, median ~5.8e3) and DR17 ``aspcapStar``
spectra (continuum-normalised, median ~1.01) for the members that were absent
from the APO-North star list used at the time. Three orders of magnitude
apart, and ``apStar`` is 2-D (row 0 = combined) where ``aspcapStar`` is
already 1-D — so the masked autoencoder sees a systematic offset between them.

That offset is a *batch effect*. It matters because the split is not random:
essentially every field star comes from ``apStar`` while ~91% of cluster
members come from the ``aspcapStar`` backfill. Any member-vs-field score
computed on the union therefore partly measures "which pipeline reduced this
spectrum" rather than "is this star chemically a cluster member".

**This is a product mismatch, not a data-release one.** DR19 reanalyses and
*includes* DR17 (``astraAllStarASPCAP-0.6.0`` carries 717,689 rows with
``release='dr17'``), and DR19 serves a uniform ``mwmStar`` spectrum for every
backfilled member. The real fix is to re-download and re-embed from one
product — see ``scripts/build_dr19_rerun_list.py`` — and to verify the offset
with the paired control in ``scripts/diagnose_product_mismatch.py``. The
helpers here quantify the damage while that re-run is pending; the ``DR17`` /
``DR19`` labels are retained as the shorthand for "which product".

This module makes the confound measurable rather than invisible:

* :func:`attach_source` labels every row ``DR17`` or ``DR19``.
* :func:`provenance_auc` asks a linear probe to recover the label from the
  latent. ~0.5 means the embedding is provenance-blind; ~1.0 means the
  product is written all over it.
* :func:`provenance_crosstab` exposes the member/field imbalance that turns
  the batch effect into a confound.

Use :func:`uniform_provenance_mask` to score on a single-product subsample
when a metric would otherwise be contaminated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE

SOURCE_COLUMN = "source"
ID_COLUMN = "APOGEE_ID"

#: The DR19 1.3-redux embedding (APO-North coverage only).
DR19_EMBEDDING = "data/embeddings/masked_latent.parquet"
#: DR17 ``aspcapStar`` backfill for members DR19 misses.
DR17_EMBEDDING = "data/embeddings/masked_latent_dr17_missing.parquet"

#: Above this, a member-vs-field score on the mixed population is not
#: interpretable: the latent knows the release better than it knows chemistry.
PROVENANCE_AUC_CEILING = 0.65


def load_source_map(
    dr19_path: str | Path = DR19_EMBEDDING,
    dr17_path: str | Path = DR17_EMBEDDING,
) -> pd.Series:
    """Map ``APOGEE_ID -> {"DR17", "DR19"}`` from the two embedding files.

    Reads only the identifier column, so this stays cheap on the ~57 MB
    latent artifacts.
    """
    frames = []
    for path, label in ((dr19_path, "DR19"), (dr17_path, "DR17")):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; cannot establish embedding provenance. "
                "Both the DR19 and the DR17-backfill artifacts are required."
            )
        ids = pd.read_parquet(path, columns=[ID_COLUMN])[ID_COLUMN].astype(str)
        frames.append(pd.Series(label, index=ids.to_numpy(), dtype=object))

    merged = pd.concat(frames)
    overlap = int(merged.index.duplicated().sum())
    if overlap:
        raise ValueError(
            f"{overlap} APOGEE_IDs appear in both the DR19 and DR17 artifacts; "
            "provenance is ambiguous."
        )
    merged.index.name = ID_COLUMN
    return merged


def attach_source(
    df: pd.DataFrame,
    source_map: pd.Series | None = None,
    *,
    dr19_path: str | Path = DR19_EMBEDDING,
    dr17_path: str | Path = DR17_EMBEDDING,
) -> pd.DataFrame:
    """Return ``df`` with a ``source`` column (``DR17``/``DR19``/``unknown``)."""
    if source_map is None:
        source_map = load_source_map(dr19_path, dr17_path)
    out = df.copy()
    out[SOURCE_COLUMN] = (
        out[ID_COLUMN].astype(str).map(source_map).fillna("unknown")
    )
    return out


def provenance_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Member/field counts by data release — the shape of the confound."""
    if SOURCE_COLUMN not in df.columns:
        raise ValueError("call attach_source() first")
    role = np.where(df["cluster"] != "field", "member", "field")
    table = pd.crosstab(pd.Series(role, name="role"), df[SOURCE_COLUMN])
    for col in ("DR17", "DR19"):
        if col not in table.columns:
            table[col] = 0
    table["n"] = table.sum(axis=1)
    table["frac_DR19"] = (table["DR19"] / table["n"]).round(3)
    return table


def provenance_auc(
    X: np.ndarray,
    source: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Cross-validated AUC of a linear probe predicting the data release.

    Returns ``(mean, std)``. 0.5 = the latent carries no release signature;
    1.0 = the release is perfectly recoverable and every downstream score on
    a release-imbalanced population is suspect.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = (np.asarray(source) == "DR19").astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")

    minority = int(min(np.bincount(y)))
    n_splits = max(2, min(n_splits, minority))

    scores = cross_val_score(
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        X,
        y,
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=random_state),
        scoring="roc_auc",
    )
    return float(scores.mean()), float(scores.std())


def separating_dimensions(
    X: np.ndarray,
    source: np.ndarray | pd.Series,
    *,
    alpha: float = 1e-6,
) -> int:
    """Count latent dimensions whose mean differs by release (Welch t-test)."""
    from scipy import stats

    mask = np.asarray(source) == "DR19"
    if mask.all() or not mask.any():
        return 0
    _, p = stats.ttest_ind(X[mask], X[~mask], axis=0, equal_var=False)
    return int(np.sum(np.asarray(p) < alpha))


def uniform_provenance_mask(
    df: pd.DataFrame, release: str = "DR19",
) -> np.ndarray:
    """Boolean mask selecting a single data release (confound-free subsample)."""
    if SOURCE_COLUMN not in df.columns:
        raise ValueError("call attach_source() first")
    return (df[SOURCE_COLUMN] == release).to_numpy()


def provenance_report(
    df: pd.DataFrame, X: np.ndarray, *, random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Full batch-effect diagnostic for a prepared spectral population."""
    if SOURCE_COLUMN not in df.columns:
        df = attach_source(df)

    source = df[SOURCE_COLUMN].to_numpy()
    member = (df["cluster"] != "field").to_numpy()

    auc_all, std_all = provenance_auc(X, source, random_state=random_state)
    # Members only: strips the member/field imbalance, so a high AUC here is
    # unambiguously a batch effect rather than a chemical difference.
    if member.sum() > 10:
        auc_mem, std_mem = provenance_auc(
            X[member], source[member], random_state=random_state,
        )
    else:
        auc_mem = std_mem = float("nan")

    return {
        "crosstab": provenance_crosstab(df),
        "auc_all": auc_all,
        "auc_all_std": std_all,
        "auc_members_only": auc_mem,
        "auc_members_only_std": std_mem,
        "n_separating_dims": separating_dimensions(X, source),
        "n_dims": int(X.shape[1]),
        "confounded": bool(
            np.isfinite(auc_mem) and auc_mem > PROVENANCE_AUC_CEILING
        ),
    }


def format_report(report: dict[str, object]) -> str:
    """Human-readable rendering of :func:`provenance_report`."""
    lines = [
        "=== data-release provenance diagnostic ===",
        "",
        str(report["crosstab"]),
        "",
        f"latent -> release AUC (all rows)     : "
        f"{report['auc_all']:.4f} ± {report['auc_all_std']:.4f}",
        f"latent -> release AUC (members only) : "
        f"{report['auc_members_only']:.4f} ± {report['auc_members_only_std']:.4f}",
        f"latent dims separating release       : "
        f"{report['n_separating_dims']} / {report['n_dims']}  (p < 1e-6)",
        "",
    ]
    if report["confounded"]:
        lines += [
            "⚠ CONFOUNDED: the embedding encodes the data release "
            f"(members-only AUC > {PROVENANCE_AUC_CEILING}).",
            "  Member-vs-field scores on this mixed population are not",
            "  interpretable. Score on a single release, or report both.",
        ]
    else:
        lines.append("✓ no material release signature in the latent.")
    return "\n".join(lines)
