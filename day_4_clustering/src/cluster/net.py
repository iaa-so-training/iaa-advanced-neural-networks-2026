"""Bounded waits around the archive calls the pipeline makes.

A workshop runs on hotel wifi, and an archive that accepts the connection and
then goes quiet would otherwise hang a notebook cell forever: the socket sits in
``CLOSE_WAIT``, no exception is raised, and the cell spins with no end. Every
network call in the pipeline goes through :func:`call_with_timeout`, which gives
up after ``CLUSTER_NET_TIMEOUT`` seconds (default 30) and raises
:class:`NetworkTimeout`, so callers can degrade instead of freezing — the
literature table reports the gap and the notebooks print a note.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, TypeVar

T = TypeVar("T")

#: seconds a single archive call may take before we give up on it
NETWORK_TIMEOUT_S = float(os.environ.get("CLUSTER_NET_TIMEOUT", "30"))


class NetworkTimeout(RuntimeError):
    """A remote archive did not answer inside the time budget."""


def call_with_timeout(
    fn: Callable[..., T],
    *args: Any,
    what: str = "the archive",
    timeout_s: float | None = None,
    **kwargs: Any,
) -> T:
    """Run ``fn(*args, **kwargs)`` with a bounded wait.

    The worker thread cannot be cancelled — the underlying library owns the
    socket — so it is abandoned and the pool is shut down *without* waiting
    (a plain ``with ThreadPoolExecutor(...)`` would block on the stuck thread
    and defeat the timeout). The caller gets :class:`NetworkTimeout` instead.
    """
    limit = NETWORK_TIMEOUT_S if timeout_s is None else timeout_s
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=limit)
    except FutureTimeout as exc:
        raise NetworkTimeout(
            f"{what} did not answer within {limit:.0f} s — check your connection, "
            f"raise the budget with CLUSTER_NET_TIMEOUT=<seconds>, or work from the "
            f"cached files under data/"
        ) from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
