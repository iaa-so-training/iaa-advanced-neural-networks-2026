"""Shared helpers for the workbook exercises.

Three jobs:

1. **Find and load the data once.** :func:`members` and :func:`member_field`
   go through ``cluster.data.prepare``, which costs ~20 s on the 1.17 GB DR19
   catalogue. Results are memoised in-process and cached to
   ``results/exercise_cache/`` as parquet, so the second notebook to ask for
   the member matrix pays milliseconds instead of twenty seconds.
2. **Fail loudly and usefully.** Without the catalogue the exercises raise
   :class:`DataNotAvailable` naming the exact ``cluster download`` command,
   rather than a ``FileNotFoundError`` from three frames down.
3. **Small numerical utilities** shared by several exercises: the answer
   formatter, the seed list, kNN purity on raw features, and the
   degeneracy-aware scoring wrapper.

Nothing here is exercise-specific; anything used by exactly one exercise
belongs in that exercise's own module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cluster import config
from cluster.clusters import CLUSTER_BY_NAME, CLUSTERS, Cluster
from cluster.config import Settings
from cluster.data import PreparedData, prepare

__all__ = [
    "CACHE_DIR",
    "DataNotAvailable",
    "MemberData",
    "SEEDS",
    "abundance_matrix",
    "catalogue_path",
    "cluster_named",
    "embedding_path",
    "heading",
    "knn_purity_raw",
    "member_field",
    "members",
    "project_root",
    "settings",
    "show",
]

#: The seven seeds every stochastic number in the workbook is averaged over
#: (\S 9.4, rule 2). Never quote a single run.
SEEDS: tuple[int, ...] = (42, 0, 1, 2, 7, 13, 99)

#: Where the memoised member/field frames are parked between sessions.
CACHE_DIR = Path("results/exercise_cache")


class DataNotAvailable(RuntimeError):
    """Raised when a required artifact is missing, with the fix in the text."""


def project_root() -> Path:
    """Repository root, located from this file rather than the cwd.

    Notebooks run from ``notebooks/`` and scripts from the repo root; both
    must resolve ``data/`` to the same place.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def _resolve(path: str | Path) -> Path:
    """Interpret a relative path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def catalogue_path() -> Path:
    """Path to the DR19 Astra catalogue, or raise with the download command."""
    path = _resolve(config.ASTRA_ASPCAP_PATH)
    if not path.is_file():
        raise DataNotAvailable(
            f"the SDSS-V DR19 catalogue is not on disk:\n    {path}\n\n"
            "It is ~1.17 GB and is not shipped with the repository. Fetch it "
            "with\n\n    uv run cluster download\n\n"
            "or, with the workshop container,\n\n"
            "    docker run --rm -it $DAY4 $IMG uv run cluster download\n\n"
            "Set CLUSTER_ASTRA_ASPCAP_PATH if you keep it elsewhere.",
        )
    return path


def embedding_path(name: str) -> Path:
    """Path to an embedding artifact under ``data/embeddings/``.

    ``name`` may be a bare file name (``masked_latent.parquet``) or a path
    relative to the repository root.
    """
    candidate = _resolve(name)
    if not candidate.is_file():
        candidate = _resolve(Path("data/embeddings") / name)
    if not candidate.is_file():
        raise DataNotAvailable(
            f"embedding artifact {name!r} is not on disk:\n    {candidate}\n\n"
            "The embeddings ship as a Hugging Face bundle, not in git. "
            "Fetch them with\n\n    uv run cluster download --assets\n\n"
            "and verify with `uv run cluster download --assets --check`.",
        )
    return candidate


def settings(**overrides: Any) -> Settings:
    """A :class:`Settings` snapshot with the workbook defaults, plus overrides.

    ``settings(max_stars=5000, normalize_rows=False)`` is the idiom used by the
    exercises that sweep a lever.
    """
    base = Settings()
    for key, value in overrides.items():
        if not hasattr(base, key):
            raise AttributeError(f"Settings has no field {key!r}")
        setattr(base, key, value)
    return base


def cluster_named(name: str) -> Cluster:
    """Look up one target cluster by name, with a helpful error."""
    found = CLUSTER_BY_NAME.get(name)
    if found is None:
        raise KeyError(
            f"unknown cluster {name!r}; the workbook's 25 are "
            f"{sorted(CLUSTER_BY_NAME)}",
        )
    return found


@dataclass(frozen=True)
class MemberData:
    """The member-plus-field population and its abundance matrix.

    ``df`` and ``X`` share a row order, so ``df['cluster']`` labels ``X``.
    ``elements`` names the 16 abundance columns in ``X``'s column order.
    """

    df: pd.DataFrame
    X: np.ndarray
    elements: list[str]

    @property
    def labels(self) -> np.ndarray:
        """True cluster label per row (``'field'`` for non-members)."""
        return self.df["cluster"].to_numpy()

    @property
    def is_member(self) -> np.ndarray:
        return self.df["is_member"].to_numpy(dtype=bool)

    def members_only(self) -> MemberData:
        """Drop the field: the cluster-only population of \\S 9.2."""
        mask = self.is_member
        sub = pd.DataFrame(self.df[mask]).reset_index(drop=True)
        return MemberData(df=sub, X=self.X[mask], elements=list(self.elements))

    def min_members(self, n: int = 5) -> MemberData:
        """Keep clusters with at least ``n`` members (the paper's rule)."""
        counts = self.df["cluster"].value_counts()
        keep = sorted(str(c) for c in counts.index if int(counts[c]) >= n)
        mask = self.df["cluster"].isin(keep).to_numpy(dtype=bool)
        sub = pd.DataFrame(self.df[mask]).reset_index(drop=True)
        return MemberData(df=sub, X=self.X[mask], elements=list(self.elements))

    def __repr__(self) -> str:  # pragma: no cover — display only
        labels = self.labels
        n_clusters = len({str(c) for c in labels if str(c) != "field"})
        return (
            f"MemberData({len(self.df)} stars, {self.X.shape[1]} features, "
            f"{n_clusters} clusters)"
        )


def _cache_key(max_stars: int | None, region_scaled: bool, **flags: Any) -> str:
    parts = [f"n{max_stars if max_stars is not None else 'all'}"]
    if region_scaled:
        parts.append("scaled")
    parts += [f"{k}{int(bool(v))}" for k, v in sorted(flags.items())]
    return "_".join(parts)


def _prepare(cfg: Settings, clusters: list[Cluster]) -> PreparedData:
    return prepare(
        catalogue_path(), cfg, clusters,
        seed_position_radius_deg=config.SEED_POSITION_RADIUS_DEG,
        seed_parallax_frac=config.SEED_PARALLAX_FRAC,
        seed_pm_tol=config.SEED_PM_TOL,
        seed_rv_tol=config.SEED_RV_TOL,
        n_refine_passes=config.N_REFINE_PASSES,
        refine_sigma=config.REFINE_SIGMA,
    )


@lru_cache(maxsize=8)
def _cached_population(
    max_stars: int | None,
    normalize_rows: bool,
    use_element_weights: bool,
    region_scaled: bool,
) -> MemberData:
    """Prepare (and disk-cache) one member+field population.

    The parquet cache holds the *frame*; the matrix is rebuilt from it, which
    is cheap and keeps the cache valid when only a matrix-level flag changes.
    """
    from cluster.data import make_matrix

    cfg = settings(
        max_stars=max_stars,
        normalize_rows=normalize_rows,
        use_element_weights=use_element_weights,
        region_scaled=region_scaled,
    )
    key = _cache_key(
        max_stars, region_scaled,
    )
    cache = _resolve(CACHE_DIR) / f"population_{key}.parquet"

    if cache.is_file() and not os.environ.get("EXERCISES_NO_CACHE"):
        df = pd.read_parquet(cache)
    else:
        df = _prepare(cfg, list(CLUSTERS)).df
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)

    X = make_matrix(df, cfg)
    return MemberData(df=df, X=X, elements=list(cfg.elements))


def member_field(
    *,
    max_stars: int | None = 25_000,
    normalize_rows: bool = True,
    use_element_weights: bool = False,
    region_scaled: bool = False,
) -> MemberData:
    """The member-plus-field population: the field-retrieval task of \\S 9.2.

    ``max_stars`` caps the *field* (members are always kept). The default
    25 000 matches ``CLUSTER_FAST=1`` and takes ~20 s the first time, then
    milliseconds from the cache.
    """
    return _cached_population(
        max_stars, normalize_rows, use_element_weights, region_scaled,
    )


def members(
    *,
    min_members: int = 5,
    normalize_rows: bool = True,
    use_element_weights: bool = False,
) -> MemberData:
    """The cluster-only population: the separation task of \\S 9.2.

    This is the 1 002-row / 25-cluster matrix the workbook's baseline tables
    are computed on (``min_members=5`` drops the clusters the 2019 paper's
    sample selection also dropped).
    """
    population = member_field(
        max_stars=25_000,
        normalize_rows=normalize_rows,
        use_element_weights=use_element_weights,
    )
    return population.members_only().min_members(min_members)


def abundance_matrix(
    df: pd.DataFrame, cfg: Settings | None = None,
) -> np.ndarray:
    """Rebuild the C-space matrix for an arbitrary frame (same recipe)."""
    from cluster.data import make_matrix

    return make_matrix(df, cfg if cfg is not None else settings())


def knn_purity_raw(
    X: np.ndarray, labels: np.ndarray, k: int = 15, min_members: int = 5,
) -> dict[str, float]:
    """kNN purity computed in whatever space ``X`` is given in.

    ``cluster.benchmark.knn_purity`` is the same measure on a 2-D embedding;
    this wrapper simply makes the intent explicit when the space is the raw
    16-D C-space or a 256-D latent.
    """
    from cluster.benchmark import knn_purity

    return knn_purity(X, labels, k=k, min_members=min_members)


def heading(text: str, rule: str = "=") -> str:
    """A underlined heading for notebook output."""
    return f"{text}\n{rule * len(text)}"


def show(answer: dict[str, Any], key: str | None = None) -> None:
    """Print one entry of an ``ANSWER`` dict, or all of them in order.

    The notebooks call ``show(ANSWER, 'concentration')`` so a student reveals
    one finding at a time instead of the whole solution at once.
    """
    if key is not None:
        if key not in answer:
            raise KeyError(
                f"no answer entry {key!r}; available: {sorted(answer)}",
            )
        _print_entry(key, answer[key])
        return
    for name, value in answer.items():
        _print_entry(name, value)
        print()


def _print_entry(key: str, value: Any) -> None:
    print(heading(key, "-"))
    if isinstance(value, pd.DataFrame):
        print(value.to_string())
    elif isinstance(value, dict):
        width = max((len(str(k)) for k in value), default=0)
        for k, v in value.items():
            print(f"  {str(k):<{width}}  {_fmt(v)}")
    elif isinstance(value, (list, tuple)) and len(value) > 8:  # noqa: UP038 — the tuple form keeps the element type legible to pyrefly
        print(f"  {type(value).__name__} of {len(value)}: "
              f"{', '.join(_fmt(v) for v in list(value)[:6])}, …")
    elif isinstance(value, str):
        print(_wrap(value))
    else:
        print(f"  {_fmt(value)}")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _wrap(text: str, width: int = 78) -> str:
    import textwrap

    paragraphs = [p.strip() for p in text.strip().split("\n\n")]
    return "\n\n".join(
        textwrap.fill(p, width=width, initial_indent="  ",
                      subsequent_indent="  ")
        for p in paragraphs
    )
