# Workbook exercise decks

Jupyter decks for the 56 exercises in the companion workbook
(`article/workbook.tex`), plus the solution package they present.

```
notebooks/workbook_exercises.ipynb        every chapter, one file
notebooks/exercises/chapter_NN_<label>.ipynb   one deck per chapter
src/exercises/exercise_<chapter>_<n>.py   the worked solution for each exercise
src/exercises/utils.py                    shared data loading and helpers
```

## The rule these follow

**The notebooks compute nothing.** Every number a student sees is produced by a
module under `src/exercises/`, which they can open, read, and re-run from a
plain Python prompt. A notebook cell only ever states a question, offers a
clean cell to work in, or prints an answer that came from a module. That keeps
the decks, the workbook and the code from drifting apart, and it means a
student who prefers a terminal loses nothing.

Each exercise gets three or four cells:

1. the question, exactly as the workbook states it;
2. a **clean cell** to work in — pre-seeded with the `cluster` imports the
   solution uses, which is the hint that the repository already contains the
   machinery;
3. the **answer**, printed from `ANSWER` in the module;
4. for computational exercises, a **recompute** cell that calls `solve()`;
5. where seeing the result matters, a **figure** cell that calls `plot()`.

The scratch-cell imports are read out of the module that solves the exercise,
parsed rather than copied: a module that aliases what it imports (say
`settings as _settings`) would otherwise put the alias machinery into the
student's cell, and `import as` is not even valid Python. The public name is
what a student wants anyway. `tests/test_exercises.py` compiles every cell of
every shipped deck, so the generator cannot emit a cell nobody can run.

## Running them

The decks run in a container built from this repository's own `Dockerfile` —
nothing is installed on your laptop, and the environment is the pinned one in
`uv.lock` rather than whatever happens to be on the host:

```bash
mkdir -p data results notebooks      # once: docker would create them as root
docker compose up                    # then open http://localhost:9999
```

`docker compose up` builds the image if it is not built yet and mounts three
folders from this directory, which is the whole reason the compose file exists:

| host | in the container | holds |
|---|---|---|
| `./data` | `/app/data` | the SDSS DR19 catalogue and the embeddings bundle |
| `./results` | `/app/results` | the parquet cache, figures, `mlruns/` |
| `./notebooks` | `/app/notebooks` | the decks, so the cells you edit are saved |

Open `notebooks/workbook_exercises.ipynb` (or one of the chapter decks under
`notebooks/exercises/`) and work through it. The container writes into the
mounts as *your* uid, so nothing it produces needs `sudo` to delete.

The server carries no token and is published on `127.0.0.1` only, which is the
same trust level as running it natively. To serve it beyond this machine, set a
token as well:

```bash
JUPYTER_BIND=0.0.0.0 docker compose up     # and add a token to docker-compose.yml
```

Any other command runs in the same image — the data downloads, the sweeps, the
tests — by replacing the server with the command:

```bash
docker compose run --rm jupyter uv run cluster download --all
```

### Working natively instead

Everything below is written in the bare `uv run …` form, which is what those
commands are *inside* the container. To run them on the host you need Python
≥ 3.13 and `uv`:

```bash
uv sync                       # installs jupyterlab + ipykernel
uv run jupyter lab notebooks/workbook_exercises.ipynb --port 9999
```

Prefer this when you are changing the code and want the edit-run loop without a
rebuild; prefer the container when you want the environment we tested.

### Data

The exercises that touch real data need the DR19 catalogue; the spectral
chapters (13–14) also need the embedding bundle:

```bash
uv run cluster download            # ~1.17 GB catalogue
uv run cluster download --assets   # embeddings + checkpoints
```

In the container the same two commands are one prefix longer, and land in the
same `./data` you mounted:

```bash
docker compose run --rm jupyter uv run cluster download --all
```

Without the data the modules raise `exercises.utils.DataNotAvailable` naming the
command to run, rather than failing deep inside a loader. Preparing the
population costs ~20 s the first time and is then cached as parquet under
`results/exercise_cache/`, so every later call is instant. Set
`EXERCISES_NO_CACHE=1` to force a rebuild.

## Rebuilding the notebooks

The decks are generated — edit the modules or the workbook, never the `.ipynb`:

```bash
uv run python scripts/make_exercise_notebooks.py           # rebuild
uv run python scripts/make_exercise_notebooks.py --check   # CI: fail on drift
```

The build is deterministic (cell ids are positional), so an unchanged rebuild
produces a byte-identical file and `--check` is a meaningful gate.

### Cross-references are resolved, not copied

Exercise prose that says `Table~\ref{tab:clusters}` has to reach the student as
a number they can actually find. The generator walks the workbook in `\input`
order and numbers floats by *environment*, which is what LaTeX counts — not by
label, because chapter 5 contains two numbered equations that carry no label
and counting labels would shift every later equation number by two. The class
(`ar-1col.cls`) redefines `\thetable`, `\thefigure` and `\theequation` to plain
arabic counters, so there is one sequence per kind and no section prefix:
the cluster table is Table 1, the honest table is Table 4, mutual reachability
is Equation 4. `tests/test_exercises.py` pins those numbers.

## The contract, and why it is tested

`tests/test_exercises.py` enforces that every workbook exercise has a module,
that no module invents an exercise the workbook does not set, that each
`ANSWER` is a non-empty dict of real prose, that `solve()` takes no required
arguments, that no module touches data at import time, that no LaTeX reaches a
student's cell, and that the notebooks are up to date. The data-backed tests
are marked `needs_data` and skip automatically on a clean checkout.

Figures carry the same contract. `plot()` must be callable bare, and a module
that defines one must have a cell calling it in **both** the master deck and
its chapter deck. That guard exists because 52 `plot()` functions were once
written, committed, and never wired to anything: no cell imported them, no
test ran them, and five had quietly rotted — two of them calling a
`cluster.baseline` symbol that did not exist, which also broke their `solve()`.
Nothing noticed, because nothing ever called them.

```bash
uv run pytest tests/test_exercises.py -q
```

The contract tests check *shape*. To check that every module actually runs —
importing, hitting the real catalogue, and returning results — use the sweep:

```bash
uv run python scripts/run_all_exercises.py     # runs all 56 solve()s, ~20 min
```

It reports per-exercise status and runtime and exits non-zero if anything
fails to import, raises, or returns an empty result.

The decks themselves are checked the same way — the sweep above covers the
modules, this covers what a student actually opens:

```bash
uv run python scripts/run_notebooks.py            # all 17 decks, ~20 min
uv run python scripts/run_notebooks.py chapter_11 # one deck
```

Every cell executes in a real kernel, so a deck that cannot run is caught
before it ships. Each deck is executed as a *copy* under
`results/notebook_runs/` — the shipped `.ipynb` files are never written to, so
`--check` still passes afterwards. Per-deck and slowest-cell timings land in
`results/notebook_runs/report.json`; the module is imported once per deck, so
deck runtime tracks the exercises it contains and not much else.

## Citations

Answers that lean on the literature say so, and the citation is a **key into
`article/references.bib`** — the same bibliography the chapters cite, never a
free-typed author-year string:

```python
from exercises.citations import cite, reference_list

f"...the construction {cite('Campello:13')} introduced..."   # (Campello et al. 2013)
f"...{cite('Kos:17', parenthetical=False)} ran t-SNE..."     # Kos et al. (2017)
f"...({cite('Ester:96', bare=True)}, §6.3)..."               # (Ester et al. 1996, §6.3)

ANSWER = {
    ...,
    "references": reference_list("Campello:13", "Ester:96"),
}
```

`cite()` raises on a key the bibliography does not contain, so a typo fails a
test run instead of shipping a reference the student cannot follow. Three
tests hold the line: the bib must parse with an author and year for every
entry, no rendered citation may leak LaTeX (`\url`, brace-protected corporate
authors, accent macros), and no module may hand-type an author-year string
outside a `cite()` call.

What gets a citation: the origin of a method the answer leans on, an astronomy
claim taken from the literature, or software whose behaviour the answer
depends on. What does not: the repository's own measured numbers. Those are
the module's results, and a citation there would misattribute them.

## Writing a new exercise module

```python
"""Chapter 7, exercise 2 — <the question, in a few words>.

    <the exercise statement, quoted from the workbook>

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
}
```

### Figures

`src/exercises/figures.py` holds the three families the exercises draw, so a
module says *what* to plot and never *how*:

| helper | what it draws |
|---|---|
| `embedding_scatter` | a 2-D embedding (t-SNE / UMAP / EVoC) coloured by cluster, with an optional `highlight=` to dim everything but one |
| `cmd_diagram` | a Gaia colour-magnitude diagram, optionally with a fitted isochrone over it |
| `sky_cutout` | a real survey image of the field with the members circled, through astropy's WCS |

They return matplotlib figures. Nothing is interactive: the decks are executed
headless in CI and shipped as static artifacts, so a plotly widget would render
as an empty div for anyone reading the committed notebook.

`embedding_scatter` reuses the workshop's own palette (`cluster.plots`), so an
exercise figure and the corresponding workbook figure colour M 67 the same.

`sky_cutout` fetches from SkyView and caches the FITS under
`results/exercise_cache/skyview/`, because the decks are re-run constantly and
SkyView is a shared public service. With no connection — CI, or a student on a
train — it falls back to a plainly-labelled RA/Dec scatter built from the
astrometry already in the catalogue, so the cell still renders. Force that path
with `EXERCISES_NO_NETWORK=1`; the fetch timeout is `EXERCISES_SKYVIEW_TIMEOUT`
(30 s by default), because a deck must never hang on a remote service.

One rule above all others, and the workbook's own subject: **never write a
number into `ANSWER` that you have not run.** Compute it, read the output, then
write the prose around the real figure. Where something genuinely cannot be
computed here — the masked autoencoder needs a GPU and a training set that is
not in this repository — the answer says so and explains what would be needed,
because that is also what §9 asks of a student's report.
