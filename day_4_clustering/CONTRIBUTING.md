# Contributing to Day 4 — Clustering

This guide covers `day_4_clustering/` only. The other days in this repository
are maintained separately; nothing here applies to them.

Read it before your first pull request: it says what CI will check, where the
detailed documentation lives, and which of the four moving parts your change
belongs in.

---

## 1. The workflow: fork → branch → draft PR → review

You almost certainly do not have write access to
`iaa-so-training/iaa-advanced-neural-networks-2026`, and you do not need it.
The repository is public and forking is enabled.

**1. Fork it** on GitHub (top-right *Fork* button), then clone your fork:

```bash
git clone https://github.com/<your-username>/iaa-advanced-neural-networks-2026.git
cd iaa-advanced-neural-networks-2026/day_4_clustering
```

Add the original as a second remote so you can stay current:

```bash
git remote add upstream https://github.com/iaa-so-training/iaa-advanced-neural-networks-2026.git
git fetch upstream
```

**2. Branch.** Never work on `main` — it makes your fork painful to sync and
your PR impossible to review cleanly.

```bash
git switch -c day4/short-description upstream/main
```

**3. Open the PR as a draft, early.** Push your branch and open the pull
request against `iaa-so-training/…:main` while the work is still unfinished:

```bash
git push -u origin day4/short-description
gh pr create --draft --title "day4: what you changed" --body "..."
```

A draft PR runs CI on every push, so you find out within minutes whether the
tests, the deck check, the linter and the type-checker are happy — but it
**cannot be merged by accident**, and it does not ask anyone to review work you
are not finished with. This is the single most useful habit in this repository.

**4. Mark it ready and assign a reviewer** when you want eyes on it:

```bash
gh pr ready                          # draft -> ready for review
gh pr edit --add-reviewer <username> # ask a specific person
```

Do both. A PR that is ready but has no reviewer assigned is invisible; a PR
that is assigned but still a draft reads as "not yet, please wait". Current
maintainers you can assign: `garciadias`, `cosmosz5`.

**5. Respond to review by pushing more commits.** Do not force-push a rewritten
branch mid-review — it destroys the reviewer's place in the diff. Squashing
happens at merge time.

> **First-time contributors:** GitHub Actions will not run your PR's workflows
> until a maintainer approves them. If your checks say *"waiting for approval"*,
> nothing is wrong — ask in the PR.

---

## 2. What CI enforces

One workflow, `.github/workflows/day4-tests.yml`, gated to run only when
something under `day_4_clustering/**` changes. Every step below **blocks the
merge** if it fails. Run them locally before you push — they take about five
minutes in total and save a round-trip.

| Step | Command | What it means |
|---|---|---|
| tests | `uv run pytest -q` | The full suite (~890 tests, ~4 min). Hermetic: no test needs the 1.17 GB catalogue or a network. |
| deck drift | `uv run python scripts/make_exercise_notebooks.py --check` | The shipped notebooks still match the exercise modules they are generated from. |
| lint | `uv run ruff check src scripts tests` | Ruff, configured in `pyproject.toml`. Currently clean — keep it that way. |
| type check | `uv run pyrefly check` | A **ratchet**, not a pass/fail gate (see below). |

All four at once:

```bash
uv sync
uv run pytest -q
uv run python scripts/make_exercise_notebooks.py --check
uv run ruff check src scripts tests
uv run pyrefly check
```

### The type-check ratchet

Pyrefly runs in strict mode and the codebase is **not clean yet** — there are
59 known errors, mostly missing annotations in third-party-facing code. A plain
pass/fail gate would be red on every PR and would be ignored within a week.

So the step compares the error count against a ceiling in the workflow:

- **more errors than the ceiling → the build fails.** Your PR may not make the
  type situation worse.
- **fewer → the build passes** and prints a notice asking you to lower
  `PYREFLY_CEILING` in `day4-tests.yml`.

If you drive the number down, lower the ceiling in the same PR. That is the
entire mechanism, and it only works if the ceiling follows the floor.

### Not enforced by CI, but expected

- **Never state a number you have not run.** Every figure in the workbook,
  every value in an exercise answer, and every claim in `docs/` comes from a
  recorded run. A plausible-looking estimate is worse than an admitted gap, in
  a project whose subject is that published numbers are hard to trust.
- **Never hand-edit a recorded number** in `docs/reference_runs/*.json`. Re-run
  and add a new dated file.
- **Do not commit** anything under `data/`, `results/`, or `ref/`.

---

## 3. The four parts, and which one your change belongs in

The work is split across four places on purpose. Putting a change in the wrong
one is the most common review comment.

### a. The pipeline — `src/cluster/`

The library: loading DR19, quality cuts, embeddings, clustering, scoring,
benchmarks. It knows nothing about exercises or teaching. If you are fixing how
a number is *computed*, it belongs here.

Every configuration knob is an environment variable named `CLUSTER_<NAME>`
(see `src/cluster/config.py`). `CLUSTER_FAST` defaults to `True`, which is the
usual reason a number does not match the documented one.

### b. The workbook — `article/`

The LaTeX text students read: 17 chapters, 64 figure images, and
`article/references.bib` as the **single source of truth for citations**.
Exercise statements live here; the modules answer what this text asks.

Citations anywhere in the exercises are *keys into this file*, never typed
author-year strings — an unknown key raises rather than printing a dead
reference.

### c. The exercises and notebooks — `src/exercises/` and `notebooks/`

One module per exercise, `exercise_<chapter>_<n>.py`, exposing `ANSWER`,
`solve()`, and optionally `plot()`.

**The notebook presents; the module computes.** Nothing is calculated in a cell
that is not also calculated in a module. That single rule is what keeps the
text, the code and the decks from drifting apart.

**The decks in `notebooks/` are generated. Never hand-edit them.** Change the
module or the workbook, then rebuild:

```bash
uv run python scripts/make_exercise_notebooks.py
```

A hand-edit is silently destroyed by the next regeneration, and CI's drift
check will reject it first.

### d. The presentation — a separate repository

The slide deck is **not in this repository**. It lives in
[`garciadias/garciadias.github.io`](https://github.com/garciadias/garciadias.github.io)
as a Vue component (`src/decks/IaaSoChemicalTaggingDeck.vue`), because it is
part of a personal site with its own build and deploy.

The split matters in one direction: the deck quotes numbers and commands that
this repository produces. If you change a headline result, a CLI command, or
the setup instructions, **the deck can go stale and this repository's CI will
not notice**. Say so in your PR so the deck is updated too.

---

## 4. Agent instructions: `AGENTS.md`

If you use an AI coding assistant — Claude Code, Cursor, Copilot, Codex — this
tree is already documented for it, following <https://agents.md/>. Nine files,
one per area, loaded automatically by tools that support the convention:

| File | Covers |
|---|---|
| `AGENTS.md` | Project overview, the golden rule, layout, CLI, container |
| `src/cluster/AGENTS.md` | The 23 pipeline modules, the `CLUSTER_*` knobs |
| `src/exercises/AGENTS.md` | The module contract, citations, figures, traps |
| `notebooks/AGENTS.md` | Generated-not-edited, what `--check` really guards |
| `tests/AGENTS.md` | Hermetic tests, `needs_data`, mutation-testing guards |
| `scripts/AGENTS.md` | The four kinds of script, the DR19 chain order |
| `article/AGENTS.md` | Build targets, the bibliography contract |
| `docs/AGENTS.md` | Reference runs as evidence |
| `hf/AGENTS.md` | The manifest contract, publishing, data tiers |

They are worth reading yourself even if you never use an assistant: they are
denser than the prose documentation and they record the traps that actually
cost time here, not generic advice.

**Keep them true.** If you change something an `AGENTS.md` describes — a
command, a count, a default — update it in the same PR. A stale instruction
file is worse than none, because it is confidently wrong.

---

## 5. Getting set up

Two supported paths. Docker is the one students are given, because it needs
nothing but Docker.

```bash
./run.sh download --all   # DR19 catalogue + embeddings, ~2.2 GB, once
./run.sh lab              # JupyterLab -> http://localhost:8889
docker compose up         # alternative: JupyterLab -> http://localhost:9999
```

Native, for development:

```bash
uv sync
uv run cluster --help     # 8 pipeline commands
uv run pytest -q
```

Windows uses `.\run.ps1` in place of `./run.sh`. More detail in
[`README.md`](README.md) and [`docs/docker.md`](docs/docker.md).

### Data tiers

Not everything arrives with one command, and an exercise that fails will tell
you which tier it needs:

| Tier | Command |
|---|---|
| Catalogue (~1.17 GB) | `uv run cluster download` |
| Assets (32 files, ~1.29 GB) | `uv run cluster download --assets` |
| Optional spectra (~240 MB) | `uv run cluster download --assets --with-optional`, then `tar -xf data/mwmstar.tar -C data/` |

Some artifacts are **computed, not downloaded** — `results/casamiquela_comparison.csv`
is a derived scoring table built by `scripts/casamiquela_comparison.py`, and is
deliberately not published. The error message always names the right remedy.

---

## 6. Commits and licence

Write commit messages that say *why*, not *what the diff shows*. Subject in the
imperative, a blank line, then the reasoning.

Contributions are accepted under the repository's licence — see
[`LICENSE`](LICENSE).

## Questions

Open an issue, or ask in the pull request. A draft PR with a specific question
attached to specific lines gets a better answer than an abstract one.
