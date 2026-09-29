# AGENTS.md — src/exercises

One module per workbook exercise. The workbook (`../../article/workbook.tex`)
sets 56 exercises across 16 chapters; each has a module here, and the Jupyter
decks in `../../notebooks/` are **generated from these modules**, never written
by hand.

## The contract

```python
"""Chapter N, exercise M — <the exercise statement, quoted from the workbook>.

<what it teaches, and which section of the workbook it belongs to>
"""

from __future__ import annotations

from exercises.utils import members


def solve() -> dict[str, object]:
    """Recompute the result. No required arguments — notebooks call it bare."""
    ...


def plot(result: dict[str, object] | None = None):
    """Draw the result. Also callable bare; pass solve()'s output to reuse it."""
    ...


ANSWER: dict[str, object] = {
    "the finding": "...",
    "references": [...],
}
```

Four rules, each enforced by a test in `../../tests/test_exercises.py`:

1. **`solve()` and `plot()` take no *required* arguments.** The deck calls them bare or with the result already in hand; a function needing a hand-built argument cannot be presented by a notebook.
2. **Nothing touches data at import time** — except the bibliography. A deck imports every module it renders; a module that loads the catalogue on import makes deck *generation* need the data.
3. **No LaTeX reaches a student's cell.** `ANSWER` is prose, already de-TeXed.
4. **Every claim is cited** by key into `../../article/references.bib`.

## Naming

`exercise_<chapter>_<number>.py`, chapter zero-padded to two digits:
`exercise_01_1.py`, `exercise_15_4.py`. Flat, chapter-scoped, no subpackages —
the generator maps a workbook exercise number straight onto a module name.

## Shared helpers

| File | Purpose |
|---|---|
| `utils.py` | `members()`, `member_field`, `show()`, `catalogue_path()`, `SEEDS`, `settings`, `DataNotAvailable` |
| `citations.py` | `cite()`, `reference_list()`, `bibliography()`, `BIB_PATH` |
| `figures.py` | The three figure families every plot is built from |

### Citations

```python
from exercises.citations import cite, reference_list

cite('Campello:13')                   # "(Campello et al. 2013)"
cite('Kos:17', parenthetical=False)   # "Kos et al. (2017)"
cite('Ester:96', bare=True)           # "Ester et al. 1996"
reference_list('Ester:96', ...)       # key -> "Author (year), title, venue, doi:…"
```

Keys are resolved against `article/references.bib` at import. **An unknown key
raises** — a hand-typed author-year string is exactly the failure mode this
exists to prevent, so never work around a `KeyError` by inlining the text.

This is also why the bibliography must be inside the Docker image: these calls
run at *import* time, so a missing bib fails every deck at its first cell.

### Figures

`figures.py` holds what modules draw with, so a module says *what* to plot and
never *how*:

| Helper | Draws |
|---|---|
| `embedding_scatter` | A 2-D embedding coloured by cluster; `highlight=` dims all but one |
| `cmd_diagram` | A Gaia colour-magnitude diagram, optionally with a fitted isochrone |
| `sky_cutout` | A real survey image with members circled, registered through astropy WCS |

All return a matplotlib `Figure`. Nothing interactive: decks execute headless in
CI and ship as static artifacts, so a plotly widget would be an empty div.

`sky_cutout` fetches from SkyView and caches the FITS under
`results/exercise_cache/skyview/`. With no network it falls back to a labelled
RA/Dec scatter built from catalogue astrometry, so the cell still renders.
`EXERCISES_NO_NETWORK=1` forces that path; `EXERCISES_SKYVIEW_TIMEOUT` (30 s)
bounds the fetch — astroquery's `SkyView` has **no timeout of its own**, so the
bound is applied through astropy's `remote_timeout`.

`cmd_diagram` refuses `absolute=True` together with a `curve_*` argument: a
fitted isochrone is in apparent magnitude, and drawing it over absolute
magnitudes would be a quietly wrong figure rather than an error.

## Two traps this package has already fallen into

**A `plot()` nothing calls is a `plot()` nobody notices is broken.** 52 of these
functions were once written, committed, and wired to nothing — no deck cell
imported them, no test ran them, all marked `# pragma: no cover`. Five had
rotted; two called a `cluster.baseline` symbol that did not exist, which meant
their `solve()` was dead too. The guard is
`test_every_plot_reaches_a_notebook_cell`, which requires a calling cell in
**both** the master deck and the chapter deck. If you add a `plot()`, re-run the
generator; if the test fails, that is it working.

**`plot()` must reuse what `solve()` computed.** The deck binds
`result = solve()` and then calls `plot(result)`. A bare `plot()` recomputes
everything — with all 52 plots doing that, the master deck went from 14 minutes
to over ten hours. The generator decides which form to emit by inspecting the
first parameter: it passes the result only when that parameter is **named
`result` and annotated as a dict**. Name alone is not enough (a tuning
parameter like `plot(min_pts: int)` must not receive a dict) and the annotation
alone is not enough either — `exercise_03_1.plot` takes
`curve: dict[str, np.ndarray]`, a *different* dict from its own
`concentration_curve()`, and raises `KeyError` if handed `solve()`'s output. So
**annotate the first parameter**, and call it `result` only when it really is
one.

**A raising `solve()` must not poison the plot below it.** The decks execute
top to bottom in a single kernel, so when `solve()` raises — routinely, for
exercises whose data is not on disk — `result` would otherwise still hold the
*previous* exercise's dict, and `plot(result)` would draw the wrong exercise's
data and fail with `KeyError: 'scatter'` instead of the `DataNotAvailable` that
names the missing file. The generator emits `result = None` at the top of every
solve cell for exactly this reason; `plot(None)` then falls through to the
function's own default path and reports the real problem.

## Numbers

Every number in an `ANSWER` came out of a real run of that module's `solve()`.
The workbook's whole subject is that published numbers are hard to reproduce —
inventing one here, or copying one from the paper without re-running, defeats
the exercise. Where a module's result *disagrees* with the published value, the
`ANSWER` says so and explains the discrepancy; that is teaching material, not a
bug to be papered over.
