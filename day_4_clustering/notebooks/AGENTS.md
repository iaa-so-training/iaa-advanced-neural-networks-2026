# AGENTS.md — notebooks

The Jupyter decks students work through. **Every deck here is generated** by
`../scripts/make_exercise_notebooks.py` from the modules in `../src/exercises/`.
Editing a `.ipynb` by hand is always the wrong move: the next generator run
overwrites it, and CI fails in between.

| Path | What it is |
|---|---|
| `workbook_exercises.ipynb` | The master deck — all 56 exercises in workbook order |
| `exercises/chapter_NN_*.ipynb` | One deck per chapter (16), same cells, chapter-scoped |
| `chemical_tagging.ipynb` | The narrative walkthrough of the pipeline (hand-written, not generated) |
| `tuning_template.ipynb` | A starting point for parameter exploration (hand-written) |
| `README.md` | The student-facing instructions |

## Regenerating

```bash
uv run python ../scripts/make_exercise_notebooks.py            # write the decks
uv run python ../scripts/make_exercise_notebooks.py --check    # verify, change nothing
```

`--check` is what CI runs. It fails when a shipped deck has drifted from the
modules, which is the signal that someone edited a notebook by hand or changed
a module without regenerating.

**What `--check` actually guards** (established by mutation test, so do not
over-trust it): content *rendered into* a deck — the workbook prose, the
scratch-cell imports, the emitted call cells. Editing a module's internals
without changing what the deck renders does **not** fail `--check`. Changing a
module's `exercises.utils` import line does.

## The cell pattern

Each exercise contributes:

1. the question, as markdown, quoted exactly from the workbook;
2. a **clean cell** to work in, pre-seeded with the `cluster` imports the solution uses — the hint that the machinery already exists in the repo;
3. the **answer**, printed from the module's `ANSWER`;
4. for computational exercises, a **recompute** cell binding `result = solve()`;
5. where seeing the result matters, a **figure** cell calling `plot(result)`.

## The one rule: present, don't compute

A deck cell never computes anything that isn't also computed in a module. The
notebook imports `solve()` and `plot()` and displays what they return. This is
what makes the exercises testable at all — logic in a notebook cell is reachable
only by executing the notebook, so it cannot be unit-tested, type-checked, or
reused.

The corollary that costs real time if you forget it: the plot cell is emitted as
`plot(result)`, reusing what the `solve()` cell already computed. A bare
`plot()` recomputes everything — that mistake took the master deck from 14
minutes to over ten hours. The generator picks the form from the first
parameter's annotation (see `../src/exercises/AGENTS.md`).

The second-order trap, which cost a full deck run to find: **the decks run top
to bottom in one kernel**, so a `solve()` that raises leaves `result` bound to
the *previous* exercise's dict, and `plot(result)` then draws the wrong
exercise's data and dies with a meaningless `KeyError: 'scatter'` instead of
the `DataNotAvailable` that names the missing file. Every solve cell therefore
emits `result = None` before computing, and
`test_solve_cells_reset_result_before_computing` enforces it.

## Running them

Containers are the supported path:

```bash
cd ..
mkdir -p data results notebooks
docker compose up                      # JupyterLab on http://localhost:9999
JUPYTER_PORT=9998 docker compose up    # when 9999 is already taken
```

`data/`, `results/` and `notebooks/` are bind-mounted, so edits and downloads
survive the container.

Natively instead:

```bash
uv run jupyter lab notebooks/workbook_exercises.ipynb
```

## Executing them headlessly

```bash
uv run python ../scripts/run_notebooks.py                 # all decks
uv run python ../scripts/run_notebooks.py chapter_07_hdbscan
uv run python ../scripts/run_notebooks.py --keep-going    # don't stop at the first failure
```

The runner executes **copies** and writes `results/notebook_runs/`, so the
shipped decks stay byte-identical and `--check` keeps passing. A report lands in
`results/notebook_runs/report.json` with per-notebook timings and the slowest
cells — written *per notebook as it finishes*, so a partial report means a run
that did not complete.

Expect roughly 14 minutes for the master deck and about as long for the 16
chapter decks with the catalogue present. Substantially longer than that means
something is recomputing that shouldn't be.

## Outputs are not committed

Shipped decks carry **no** execution outputs — they must be byte-identical to
what the generator produces. A JupyterLab session left open will autosave
outputs into `workbook_exercises.ipynb` and turn `--check` red; regenerate
before committing rather than hand-stripping the outputs.
