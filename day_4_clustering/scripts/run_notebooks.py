"""Execute the exercise decks end to end and time each one.

The contract tests check the *shape* of the notebooks (structure, references,
valid cell syntax); this runs them, which is the only way to know the decks a
student opens actually execute. Same reasoning as ``run_all_exercises.py``
sitting alongside the module-level tests.

The shipped decks are never written to: each is executed as a copy under
``results/notebook_runs/`` (gitignored), so ``make_exercise_notebooks.py
--check`` keeps passing afterwards.

    uv run python scripts/run_notebooks.py                  # every deck
    uv run python scripts/run_notebooks.py chapter_11       # just one
    uv run python scripts/run_notebooks.py --keep-going     # don't stop early

Exits non-zero if any deck fails, so it can gate a release.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast, override

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

ROOT = Path(__file__).resolve().parents[1]
MASTER_DECK = ROOT / "notebooks" / "workbook_exercises.ipynb"
CHAPTER_DIR = ROOT / "notebooks" / "exercises"
RUN_DIR = ROOT / "results" / "notebook_runs"
REPORT = RUN_DIR / "report.json"

#: The slowest exercise observed is ~180 s on its own (15.1, the isochrone
#: fit); a whole deck shares one kernel, so leave generous headroom.
CELL_TIMEOUT = 2400

#: One notebook's outcome: timings, and where it went wrong if it did.
Result = dict[str, Any]


class TimingClient(NotebookClient):
    """A ``NotebookClient`` that records how long each cell took."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cell_seconds: dict[int, float] = {}

    @override
    async def async_execute_cell(
        self,
        cell: NotebookNode,
        cell_index: int,
        execution_count: int | None = None,
        store_history: bool = True,
    ) -> NotebookNode:
        # This, not the synchronous ``execute_cell`` wrapper, is what the
        # parent's run loop awaits for each cell.
        start = time.monotonic()
        try:
            return await super().async_execute_cell(
                cell, cell_index,
                execution_count=execution_count,
                store_history=store_history,
            )
        finally:
            self.cell_seconds[cell_index] = time.monotonic() - start


def _read_deck(path: Path) -> NotebookNode:
    """Read a deck as a v4 notebook.

    ``as_version=4`` is what makes the result a ``NotebookNode``: the nbformat
    stub also covers the v1-v3 layouts, which read as bare lists.
    """
    return cast(NotebookNode, nbformat.read(path, as_version=4))


def _by_seconds(result: Result) -> float:
    """Sort key: slowest notebook first."""
    seconds = result["seconds"]
    return -seconds if isinstance(seconds, int | float) else 0.0


def _by_cell_seconds(item: tuple[int, float]) -> float:
    """Sort key: slowest cell first."""
    return -item[1]


def decks() -> list[Path]:
    """Every deck the generator ships: the 16 chapters plus the master."""
    found = sorted(CHAPTER_DIR.glob("chapter_*.ipynb"))
    if MASTER_DECK.is_file():
        found.append(MASTER_DECK)
    return found


def _crashed_result(deck: Path, crashed: str) -> Result:
    """A result for a deck that could not run at all."""
    failures: list[dict[str, Any]] = []
    slowest: list[dict[str, Any]] = []
    return {
        "notebook": deck.name,
        "seconds": None,
        "code_cells": None,
        "crashed": crashed,
        "failures": failures,
        "slowest_cells": slowest,
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_one(deck: Path) -> Result:
    """Execute one deck and report what happened."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    copy = RUN_DIR / deck.name
    shutil.copy2(deck, copy)

    notebook = _read_deck(copy)
    client = TimingClient(
        notebook,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        allow_errors=True,          # collect every bad cell, not just the first
        interrupt_on_timeout=True,
        resources={"metadata": {"path": str(ROOT)}},
    )

    print(f"  running {deck.name} ...", flush=True)
    start = time.monotonic()
    crashed: str | None = None
    try:
        client.execute()
    except Exception as exc:  # dead kernel, timeout, ...
        crashed = f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - start

    failures: list[dict[str, Any]] = []
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                failures.append({
                    "cell": index,
                    "name": output.get("ename"),
                    "value": output.get("evalue"),
                    "traceback_tail": "\n".join(output.get("traceback", []))[-1200:],
                })

    executed = RUN_DIR / "executed"
    executed.mkdir(exist_ok=True)
    nbformat.write(notebook, executed / deck.name)

    return {
        "notebook": deck.name,
        "seconds": round(seconds, 1),
        "code_cells": sum(1 for c in notebook.cells if c.get("cell_type") == "code"),
        "crashed": crashed,
        "failures": failures,
        "slowest_cells": [
            {"cell": index, "seconds": round(value, 1)}
            for index, value in sorted(
                client.cell_seconds.items(), key=_by_cell_seconds,
            )[:3]
        ],
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _stored_notebooks() -> dict[str, Any]:
    """Previous results, keyed by notebook name."""
    if not REPORT.is_file():
        return {}
    try:
        stored = json.loads(REPORT.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(stored, dict):
        return {}
    notebooks = stored.get("notebooks")
    return cast("dict[str, Any]", notebooks) if isinstance(notebooks, dict) else {}


def merge_report(results: list[Result]) -> None:
    """Fold this run's results into the stored report, newest result wins."""
    by_name = _stored_notebooks()
    for result in results:
        by_name[result["notebook"]] = result
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"notebooks": by_name, "last_run": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2,
    ) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "only", nargs="*",
        help="deck names (or fragments) to run; default is every deck",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="run every selected deck even after one fails",
    )
    args = parser.parse_args()

    selected = decks()
    if args.only:
        selected = [
            deck for deck in selected
            if any(wanted in deck.name for wanted in args.only)
        ]
    if not selected:
        print("no decks matched")
        return 1

    results: list[Result] = []
    start = time.monotonic()
    for deck in selected:
        try:
            result = run_one(deck)
        except Exception as exc:
            result = _crashed_result(deck, f"{type(exc).__name__}: {exc}")
        results.append(result)

        ok = not result["failures"] and not result["crashed"]
        seconds = result["seconds"]
        shown = f"{seconds:.1f} s" if isinstance(seconds, int | float) else "n/a"
        print(
            f"  {'OK' if ok else 'FAIL':>4}  {result['notebook']:<34}{shown:>12}"
            f"  failures={len(result['failures'])}"
            + (f"  crashed={result['crashed']}" if result["crashed"] else ""),
            flush=True,
        )
        if not ok and not args.keep_going:
            print("  stopping after the first failure (--keep-going to continue)")
            break

    total = time.monotonic() - start
    merge_report(results)

    print("\n" + "=" * 72)
    print(f"{'notebook':<34}{'seconds':>12}{'cells':>7}  result")
    for result in sorted(results, key=_by_seconds):
        seconds = result["seconds"]
        shown = f"{seconds:.1f}" if isinstance(seconds, int | float) else "n/a"
        ok = not result["failures"] and not result["crashed"]
        print(f"{result['notebook']:<34}{shown:>12}"
              f"{str(result['code_cells']):>7}  {'OK' if ok else 'FAIL'}")
    print("=" * 72)
    print(f"TOTAL {total:.1f} s over {len(results)} notebook(s)")
    print(f"report: {REPORT.relative_to(ROOT)}")

    failed = [r["notebook"] for r in results if r["failures"] or r["crashed"]]
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
