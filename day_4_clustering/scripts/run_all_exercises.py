"""Run every exercise module's solve() end to end and report what breaks.

The honest check: each module must import, and its solve() must return a
non-empty dict. Nothing here is mocked — the modules hit the real catalogue.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback

sys.path.insert(0, "src")

from exercises import EXERCISES  # noqa: E402

rows = []
for chapter in sorted(EXERCISES):
    for number in range(1, EXERCISES[chapter] + 1):
        name = f"exercises.exercise_{chapter:02d}_{number}"
        label = f"{chapter}.{number}"
        started = time.time()
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            rows.append((label, "IMPORT-FAIL", 0.0, f"{type(exc).__name__}: {exc}"))
            continue
        solve = getattr(module, "solve", None)
        if solve is None:
            rows.append((label, "no-solve", time.time() - started, "ANSWER-only"))
            continue
        try:
            result = solve()
            ok = isinstance(result, dict) and bool(result)
            rows.append((
                label,
                "ok" if ok else "EMPTY",
                time.time() - started,
                f"{len(result)} keys",
            ))
        except Exception as exc:  # noqa: BLE001
            rows.append((
                label,
                "SOLVE-FAIL",
                time.time() - started,
                f"{type(exc).__name__}: {exc}",
            ))
            traceback.print_exc(limit=3)

print("\n" + "=" * 72)
print(f"{'ex':>6}  {'status':<11} {'secs':>7}  detail")
print("-" * 72)
for label, status, secs, detail in rows:
    print(f"{label:>6}  {status:<11} {secs:7.1f}  {detail[:70]}")

bad = [r for r in rows if r[1] in ("IMPORT-FAIL", "SOLVE-FAIL", "EMPTY")]
print("-" * 72)
print(f"total {len(rows)}  ok {sum(1 for r in rows if r[1] == 'ok')}  "
      f"answer-only {sum(1 for r in rows if r[1] == 'no-solve')}  "
      f"failed {len(bad)}")
print(f"TOTAL RUNTIME {sum(r[2] for r in rows):.0f} s")
if bad:
    print("\nFAILURES:")
    for label, status, _, detail in bad:
        print(f"  {label} {status}: {detail}")
    sys.exit(1)
print("ALL MODULES RAN CLEAN")
