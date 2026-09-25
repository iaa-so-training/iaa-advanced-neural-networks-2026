"""Central configuration for the chemical-tagging workshop.

Every knob in the pipeline lives here. Flip a flag to change behaviour.
All values can also be overridden with environment variables
(prefix ``CLUSTER_``), e.g. ``CLUSTER_FAST=0`` or ``CLUSTER_SNR_MIN=50``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _parse(value: str) -> Any:
    """Best-effort parse of an env-var string."""
    value = value.strip()
    low = value.lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    if low in {"none", "null", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") or "," in value:
        return [part.strip() for part in value.strip("[]").split(",") if part.strip()]
    return value


def env(name: str, default: Any) -> Any:
    return _parse(os.environ[f"CLUSTER_{name}"]) if f"CLUSTER_{name}" in os.environ else default


# --------------------------------------------------------------------------- #
# Data acquisition / location
# --------------------------------------------------------------------------- #
# SDSS-V DR19: the single DR17 allStar (ASPCAP synspec) is replaced by the
# Astra ASPCAP catalog — stellar params + [X/H] abundances + Gaia DR3
# astrometry/photometry + quality flags, all in one file.
ASTRA_ASPCAP_URL = (
    "https://dr19.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/"
    "astraAllStarASPCAP-0.6.0.fits.gz"
)
ASTRA_ASPCAP_BYTES = 1_171_102_556
ASTRA_ASPCAP_PATH: str = env("ASTRA_ASPCAP_PATH", "data/astraAllStarASPCAP-0.6.0.fits.gz")

# Legacy aliases (the catalog used to be called "allStar").
ALLSTAR_URL = ASTRA_ASPCAP_URL
ALLSTAR_BYTES = ASTRA_ASPCAP_BYTES


# --------------------------------------------------------------------------- #
# Asset bundle (embeddings + model checkpoints)
# --------------------------------------------------------------------------- #
# The parquets and checkpoints have no public upstream — they are published as
# a Hugging Face dataset and fetched with ``cluster download --assets``.
# ``hf/MANIFEST.json`` in this repo is the source of truth (path, bytes,
# sha256, consumer); the published copy must be regenerated with
# ``python hf/make_manifest.py`` after any change to the artifacts.
HF_REPO_ID: str = env("HF_REPO", "RafaelDias/iaa-chemical-tagging-2026")
HF_REPO_TYPE: str = env("HF_REPO_TYPE", "dataset")


def _default_manifest() -> str:
    """Locate ``hf/MANIFEST.json`` relative to the *code*, not the cwd.

    The manifest ships with the checkout and with the container image
    (``/app/hf/MANIFEST.json``), so ``download --list`` and ``--check`` work
    offline from any working directory. Falls back to a cwd-relative name,
    which is fetched from the Hub if it is not on disk.
    """
    for parent in Path(__file__).resolve().parents:
        shipped = parent / "hf" / "MANIFEST.json"
        if shipped.is_file():
            return str(shipped)
    return "MANIFEST.json"


ASSETS_MANIFEST: str = env("HF_MANIFEST", _default_manifest())


def astra_h_col(element: str) -> str:
    """Map an internal element name to the Astra [X/H] column.

    e.g. ``C_FE -> c_h``, ``NA_FE -> na_h``, ``FE_H -> fe_h``.
    """
    if element == "FE_H":
        return "fe_h"
    return element.split("_")[0].lower() + "_h"


# --------------------------------------------------------------------------- #
# FAST tag — cut the dataset short to accelerate the run.
# Remove it (set FAST=False) to run the full all-sky experiment.
# --------------------------------------------------------------------------- #
FAST: bool = env("FAST", True)

# Hard cap on the number of *field* stars embedded. Cluster members are
# always kept (stratified sampling), so the benchmark is unaffected.
# FAST default: 25 000 stars → t-SNE/UMAP/EVoC finish in ~1 minute on a laptop.
# FULL default: None → all ~183 000 clean stars (~10-20 min).
MAX_STARS: int | None = env("MAX_STARS", 25_000 if FAST else None)

# Optional single-cluster sky restriction (Pleiades-only faithful repro).
# None = all-sky. Set to a degree value to keep only stars within that
# angular radius of the first cluster in CLUSTER_NAMES.
REGION_RADIUS_DEG: float | None = env("REGION_RADIUS_DEG", None)

# When True, the region radius is per-cluster, scaled to the cluster angular
# diameter: max(3 deg, 10 x diam_deg) — see Cluster.region_deg. Overrides
# REGION_RADIUS_DEG.
REGION_SCALED: bool = env("REGION_SCALED", False)


# --------------------------------------------------------------------------- #
# Quality cuts
# --------------------------------------------------------------------------- #
SNR_MIN: float = env("SNR_MIN", 100.0)
REQUIRE_CLEAN_FLAGS: bool = env("REQUIRE_CLEAN_FLAGS", True)  # STARFLAG==0 (spectrum is clean)
# ASPCAPFLAG==0 gate, separate from STARFLAG: the abundance pipeline needs
# clean ASPCAP processing, but spectral embeddings read the raw spectrum, so
# they are valid even when ASPCAP flags abundance-derivation issues.
REQUIRE_ASPCAP_FLAG_CLEAN: bool = env("REQUIRE_ASPCAP_FLAG_CLEAN", True)
DWARF_ONLY: bool = env("DWARF_ONLY", False)  # logg >= 3.5 filter
DWARF_LOGG_MIN: float = env("DWARF_LOGG_MIN", 3.5)


# --------------------------------------------------------------------------- #
# Chemical space (C-space)
# --------------------------------------------------------------------------- #
ELEMENTS: list[str] = env(
    "ELEMENTS",
    [
        "C_FE", "N_FE", "O_FE", "NA_FE", "MG_FE", "AL_FE", "SI_FE", "S_FE",
        "K_FE", "CA_FE", "TI_FE", "V_FE", "CR_FE", "MN_FE", "NI_FE", "FE_H",
    ],
)
STANDARDIZE: bool = env("STANDARDIZE", True)  # zero median, unit std
# Weight each element by 1/sigma (its median *_ERR). Applied AFTER
# standardisation so the per-column unit-std rescale does not undo it.
USE_ELEMENT_WEIGHTS: bool = env("USE_ELEMENT_WEIGHTS", False)  # weight by 1/σ

# Element flag cut. When True, additionally drop stars whose per-element
# *_FLAG != 0. (Default False: global ASPCAPFLAG/STARFLAG already gate
# quality; per-element flags are over-strict and cull most cluster members.)
REQUIRE_ELEMENT_FLAG_CLEAN: bool = env("REQUIRE_ELEMENT_FLAG_CLEAN", False)

# Missing-value handling in C-space. Metal-poor globulars (M 15, M 92) have
# undetected weak lines -> NaN in a few elements, so the strict complete-case
# filter drops them entirely. When IMPUTE_MISSING is True, stars with at least
# MIN_FINITE_ELEMENTS finite abundances are kept and the remaining NaN are
# filled with the column median (a naive but transparent imputation).
IMPUTE_MISSING: bool = env("IMPUTE_MISSING", True)
MIN_FINITE_ELEMENTS: int = env("MIN_FINITE_ELEMENTS", 8)

# L2-normalise each row of X after standardisation. This makes Euclidean
# distance equal to angular (cosine) distance, so t-SNE/UMAP (Euclidean)
# compare fairly with EVoC (which uses cosine internally). Abundance
# *patterns* are the chemical-tagging signal, so this is also the more
# physical choice (scale-invariant).
#
# This is the single biggest precision lever: without it HDBSCAN merges the
# whole field into one blob (recall 1.0, precision 0.03); with it clusters
# are real but smaller (recall 0.34, precision 0.12 on M 67).
NORMALIZE_ROWS: bool = env("NORMALIZE_ROWS", True)


# --------------------------------------------------------------------------- #
# Cluster membership (kinematic ground-truth labels)
# --------------------------------------------------------------------------- #
# Method: "kinematic" (self-contained Gaia + RV box) is the only implemented
# method; "catalog" is a stub for a future external membership catalogue.
MEMBERSHIP_METHOD: str = env("MEMBERSHIP_METHOD", "kinematic")

# "combined" method: expand the kinematic core by stars that agree both
# chemically (within COMBINED_CHEM_SIGMA robust std of the core in C-space)
# and kinematically (within COMBINED_KIN_SIGMA robust std of the core in
# parallax/PM/RV).
COMBINED_CHEM_SIGMA: float = env("COMBINED_CHEM_SIGMA", 3.0)
COMBINED_KIN_SIGMA: float = env("COMBINED_KIN_SIGMA", 3.0)

# External referee catalogue (Simbad). SIMBAD_MEMBERSHIP_MIN is the minimum
# literature membership probability (0-100, max across papers) for a star to
# count as a catalogue member; SIMBAD_CROSSMATCH_ARCSEC is the sky cross-match
# tolerance against allStar. SIMBAD_CACHE_DIR holds per-cluster CSV caches so
# repeat runs skip the network.
SIMBAD_MEMBERSHIP_MIN: int = env("SIMBAD_MEMBERSHIP_MIN", 90)
SIMBAD_CROSSMATCH_ARCSEC: float = env("SIMBAD_CROSSMATCH_ARCSEC", 2.0)
SIMBAD_CACHE_DIR: str = env("SIMBAD_CACHE_DIR", "data/simbad")

# Seed box tolerances (then refined by 3-pass sigma clipping).
SEED_POSITION_RADIUS_DEG: float = env("SEED_POSITION_RADIUS_DEG", 0.6)
SEED_PARALLAX_FRAC: float = env("SEED_PARALLAX_FRAC", 0.30)   # +/-30%
SEED_PM_TOL: float = env("SEED_PM_TOL", 5.0)                  # mas/yr
SEED_RV_TOL: float = env("SEED_RV_TOL", 25.0)                 # km/s
N_REFINE_PASSES: int = env("N_REFINE_PASSES", 3)
REFINE_SIGMA: float = env("REFINE_SIGMA", 2.5)


# --------------------------------------------------------------------------- #
# Benchmark methods + hyperparameters
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = env("RANDOM_STATE", 42)

# Isochrone fitting (ASteCA + PARSEC). Longer chains give sharper posterior
# uncertainties (the membership-quality proxy); 32x500 is ~1-2 min per fit.
ISOFIT_N_WALKERS: int = env("ISOFIT_N_WALKERS", 32)
ISOFIT_N_STEPS: int = env("ISOFIT_N_STEPS", 1000)
ISOFIT_SEED: int = env("ISOFIT_SEED", 42)

TSNE: dict[str, Any] = {
    "perplexity": env("TSNE_PERPLEXITY", 30),
    "max_iter": env("TSNE_N_ITER", 1_000 if FAST else 2_000),
    "metric": env("TSNE_METRIC", "euclidean"),
    "init": "pca",
    "learning_rate": "auto",
    # backend: "sklearn" (default; fast BH, single-threaded) or "opentsne"
    # (multi-core). openTSNE pays a numba JIT + extra-iteration overhead that
    # only pays off on the full ~163k-star run; at demo sizes sklearn is faster.
    "backend": env("TSNE_BACKEND", "sklearn"),
}

UMAP: dict[str, Any] = {
    "n_neighbors": env("UMAP_N_NEIGHBORS", 15),
    "min_dist": env("UMAP_MIN_DIST", 0.1),
    "n_epochs": env("UMAP_N_EPOCHS", 500),
    "metric": env("UMAP_METRIC", "euclidean"),
}

EVOC: dict[str, Any] = {
    "noise_level": env("EVOC_NOISE_LEVEL", 0.5),
    "base_min_cluster_size": env("EVOC_MIN_CLUSTER_SIZE", 5),
    "n_neighbors": env("EVOC_N_NEIGHBORS", 15),
    "n_epochs": env("EVOC_N_EPOCHS", 50),
}

# Clustering applied on top of the t-SNE / UMAP 2-D embeddings.
# (EVoC clusters internally, so it is not combined with HDBSCAN.)
#
# min_cluster_size=5 is the smallest sensible value: it still catches small
# clusters (Pleiades ~10, M 92 ~2 members in APOGEE). Raising it (e.g. 20)
# cuts field fragmentation and lifts recall for rich clusters (M 67, M 3,
# M 5) but drops the small ones — a precision/recall trade-off.
# min_samples=None falls back to min_cluster_size (identical to =5 here).
#
# NOTE: the dominant precision lever is NOT these params — it is
# config.NORMALIZE_ROWS (L2-normalisation, which stops HDBSCAN from merging
# the whole field into one blob). Measured on M 67: normalize_rows=False
# gives recall 1.00 / precision 0.03 (blob); True gives recall 0.34 /
# precision 0.12.
HDBSCAN: dict[str, Any] = {
    "min_cluster_size": env("HDBSCAN_MIN_CLUSTER_SIZE", 5),
    "min_samples": env("HDBSCAN_MIN_SAMPLES", None),
    "cluster_selection_epsilon": env("HDBSCAN_CLUSTER_SELECTION_EPSILON", 0.0),
}


# --------------------------------------------------------------------------- #
# Which clusters to label + score
# --------------------------------------------------------------------------- #
CLUSTER_NAMES: list[str] = env(
    "CLUSTER_NAMES",
    "all",  # "all" resolves to every cluster in clusters.CLUSTERS
)


def _default_elements() -> list[str]:
    return list(ELEMENTS)


def _default_tsne() -> dict[str, Any]:
    return dict(TSNE)


def _default_umap() -> dict[str, Any]:
    return dict(UMAP)


def _default_evoc() -> dict[str, Any]:
    return dict(EVOC)


def _default_hdbscan() -> dict[str, Any]:
    return dict(HDBSCAN)


def _default_cluster_names() -> list[str] | str:
    return CLUSTER_NAMES


class Settings(BaseModel):
    """Snapshot of the configuration, handed to pipeline stages."""

    fast: bool = FAST
    max_stars: int | None = MAX_STARS
    region_radius_deg: float | None = REGION_RADIUS_DEG
    region_scaled: bool = REGION_SCALED
    snr_min: float = SNR_MIN
    require_clean_flags: bool = REQUIRE_CLEAN_FLAGS
    require_aspcap_flag_clean: bool = REQUIRE_ASPCAP_FLAG_CLEAN
    dwarf_only: bool = DWARF_ONLY
    dwarf_logg_min: float = DWARF_LOGG_MIN
    elements: list[str] = Field(default_factory=_default_elements)
    standardize: bool = STANDARDIZE
    use_element_weights: bool = USE_ELEMENT_WEIGHTS
    require_element_flag_clean: bool = REQUIRE_ELEMENT_FLAG_CLEAN
    impute_missing: bool = IMPUTE_MISSING
    min_finite_elements: int = MIN_FINITE_ELEMENTS
    normalize_rows: bool = NORMALIZE_ROWS
    membership_method: str = MEMBERSHIP_METHOD
    combined_chem_sigma: float = COMBINED_CHEM_SIGMA
    combined_kin_sigma: float = COMBINED_KIN_SIGMA
    simbad_membership_min: int = SIMBAD_MEMBERSHIP_MIN
    simbad_crossmatch_arcsec: float = SIMBAD_CROSSMATCH_ARCSEC
    simbad_cache_dir: str = SIMBAD_CACHE_DIR
    random_state: int = RANDOM_STATE
    isofit_n_walkers: int = ISOFIT_N_WALKERS
    isofit_n_steps: int = ISOFIT_N_STEPS
    isofit_seed: int = ISOFIT_SEED
    tsne: dict[str, Any] = Field(default_factory=_default_tsne)
    umap: dict[str, Any] = Field(default_factory=_default_umap)
    evoc: dict[str, Any] = Field(default_factory=_default_evoc)
    hdbscan: dict[str, Any] = Field(default_factory=_default_hdbscan)
    cluster_names: list[str] | str = Field(default_factory=_default_cluster_names)

    @classmethod
    def default(cls) -> Settings:
        """Build a Settings instance from the module-level env-var defaults."""
        return cls()

    def resolve_cluster_names(self, available: list[str]) -> list[str]:
        names = self.cluster_names
        if names == "all" or names == ["all"]:
            return list(available)
        if isinstance(names, str):
            names = [names]
        return [c for c in names if c in available]
