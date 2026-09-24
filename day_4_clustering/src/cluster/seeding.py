"""One place to seed everything stochastic in a run.

Reproducibility in this workshop has three tiers (full detail in
``docs/reproducibility.md``):

* **exact** — data identity, the sampled population, run-to-run determinism on
  one machine, native <-> container parity on one machine;
* **tolerance** — method scores *across* machines: sklearn's Barnes-Hut t-SNE
  (OpenMP), numba (UMAP, EVoC, HDBSCAN) and BLAS all reduce in
  environment-dependent orders, so identical code gives slightly different
  floats on different CPUs;
* **illustrative** — exact per-cluster tables on other hardware.

:func:`seed_everything` is the tier-one half; :func:`thread_report` records the
environment the tier-two tolerance is measured against, so a quoted number can
always be traced back to the machine and thread setting that produced it.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed the global RNGs the pipeline draws from.

    ``PYTHONHASHSEED`` is only read while the interpreter starts, so setting it
    here cannot affect the current process — it is set (when unset) for child
    processes, and it is exported before Python starts in the container and in
    ``run.sh``. Where it matters, the run-to-run guarantee comes from the two
    library RNGs below, and every model call also takes an explicit
    ``random_state``.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def thread_report() -> dict[str, Any]:
    """Effective thread configuration: env knobs plus what the libraries use.

    ``unset`` means the library default was in force (usually every core), which
    is the recommended student setting — it is faster, and pinning threads only
    buys cross-machine stability when the machines are otherwise identical.
    """
    report: dict[str, Any] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "unset"),
        "cpu_count": os.cpu_count(),
    }
    try:
        from numba import get_num_threads

        report["numba_effective"] = int(get_num_threads())
    except Exception:
        report["numba_effective"] = None
    try:  # the torch extra is optional
        import torch

        report["torch_threads"] = int(torch.get_num_threads())
    except Exception:
        report["torch_threads"] = None
    return report
