# AGENTS.md — tests

One pytest suite over both packages: `src/cluster/` (the pipeline) and
`src/exercises/` (the workbook exercises). ~812 tests, ~5 minutes.

```bash
uv run pytest -q                              # everything
uv run pytest tests/test_exercises.py -q      # the exercise contracts
uv run pytest -m needs_data                   # only the catalogue-backed tests
uv run pytest tests/test_data.py -q -k cuts   # one area
```

## Layout

One `test_<module>.py` per module in `src/cluster/`, plus:

| File | Covers |
|---|---|
| `conftest.py` | Shared synthetic fixtures — `make_allstar_frame()` and friends |
| `test_exercises.py` | All 56 exercise modules: the contract, citations, figures, deck sync |
| `test_reproducibility.py` | That a seeded run reproduces |
| `test_models.py` | Torch-dependent; skipped unless `uv sync --extra torch` |

## The suite is hermetic by default

**No test downloads or reads the real 1.17 GB catalogue.** Fixtures build
minimal schema-valid frames instead. An autouse fixture sets `CLUSTER_NO_CACHE=1`
so the prepared-sample cache cannot leak state between tests or quietly satisfy
a "did `prepare()` actually run?" assertion — tests that exercise the cache opt
back in explicitly.

This is not a stylistic preference. CI runs with **no catalogue at all**, so a
test that reads it fails on a machine that can never have it.

The few tests that genuinely need real data carry the guard:

```python
needs_data = pytest.mark.skipif(...)   # defined in test_exercises.py

@needs_data
def test_something_that_reads_the_catalogue(): ...
```

Verify a change keeps this property by pointing the path at nothing:

```bash
CLUSTER_ASTRA_ASPCAP_PATH=/nonexistent/no-catalogue.fits.gz uv run pytest -q
```

Everything must pass or skip. A failure here means a new test reaches for data
without the marker, and it will go red in CI.

## What `test_exercises.py` enforces

Beyond ordinary assertions, it holds the structural contracts that keep 56
modules and 17 generated decks in agreement:

- every workbook exercise has a module, and no module invents an exercise the workbook does not set;
- each `ANSWER` is a non-empty dict of real prose, with no LaTeX left in it;
- `solve()` and `plot()` take no *required* arguments — the decks call them bare;
- no module touches data at import time (deck generation imports every module);
- every citation key resolves against `article/references.bib`;
- every notebook cell is valid Python;
- the shipped decks match the modules (`make_exercise_notebooks.py --check`);
- **every module defining `plot()` has a calling cell in both the master deck and its chapter deck**;
- every solve cell resets `result` before computing, so a raising `solve()` cannot leak the previous exercise's result into the plot cell;
- every script an exercise tells the student to run actually exists in this repository.

The deck-cell guard exists because 52 `plot()` functions were once written, committed,
and called by nothing — no cell, no test, all `# pragma: no cover` — and five
had rotted undetected, two of them badly enough to break their `solve()` as
well. A test that only asked "does it return a figure?" would not have caught
it, and neither would a test that accepted a hit in *either* deck.

## Mutation-test your guards

A guard is worth what it catches, not what it claims. Both of the above were
verified by deliberately breaking the thing they watch and confirming a red
test — and the first version of the deck guard **passed** a mutation it should
have caught (a plot removed from its chapter deck, still present in the master),
which is how the both-decks requirement came about.

If you add a guard, break it on purpose once before trusting it. The same
applies to `--check`: mutation testing showed it does *not* catch edits to a
module's internals, only to what gets rendered into a deck.

Three guards here failed their first mutation test and had to be rewritten,
which is the argument for the practice: the deck-cell guard accepted a plot
present in *either* deck; the script-existence guard used a regex anchored on a
quote when the path actually sits mid-string after a runner (`uv run python
scripts/foo.py`); and both reported green against the very bug they were
written for.

## Migration gaps are invisible until something runs

This work was migrated from another repository, and three times the exercises
came across while what they depend on did not:

- `cluster.baseline.recovery_fraction` was never copied, which broke `solve()`
  for two exercises — not just their plots;
- `scripts/casamiquela_comparison.py` was never copied, so exercise 13.4's
  error told the student to run a script that did not exist;
- that script, once copied, still carried `REPO = Path("/home/<user>/git/…")`
  from the machine it was written on. It imported cleanly *here*, because that
  checkout still exists on this machine, and would have failed for every
  student.

None of these show up in a type check or an import sweep: the first is reached
only when `solve()` runs, the second only appears in a message produced on a
machine that *lacks* the data, and the third is invisible to the author by
construction. `test_scripts_an_exercise_names_actually_exist` and
`test_no_source_file_hardcodes_an_absolute_home_path` close the last two; the
first is why `scripts/run_all_exercises.py` is worth running after any
cross-repository move.

## Figures in tests

Figure tests force the `Agg` backend and must not require a network. The
SkyView-backed figure is tested through its offline fallback by monkeypatching
`figures.NO_NETWORK` — never by making a real request, which would make the
suite flaky and slow.
