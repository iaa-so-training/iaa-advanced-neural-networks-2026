"""Contract tests for the workbook exercise package.

These run without the 1.17 GB DR19 catalogue: anything that needs real data
is marked ``needs_data`` and skipped when the catalogue is absent, so CI on a
clean checkout still enforces the structural contract.

    pytest tests/test_exercises.py                  # structure only
    pytest tests/test_exercises.py -m needs_data    # with the catalogue
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from exercises import CHAPTERS, EXERCISES, iter_exercises, load, module_name
from exercises.utils import DataNotAvailable, catalogue_path

ALL_EXERCISES = [
    (chapter, number)
    for chapter, count in sorted(EXERCISES.items())
    for number in range(1, count + 1)
]


def _has_catalogue() -> bool:
    try:
        catalogue_path()
    except DataNotAvailable:
        return False
    return True


needs_data = pytest.mark.skipif(
    not _has_catalogue(),
    reason="DR19 catalogue absent; run `uv run cluster download`",
)


def test_every_workbook_exercise_has_a_module() -> None:
    """The workbook's 56 exercises each map to a module on disk."""
    on_disk = {(chapter, number) for chapter, number, _ in iter_exercises()}
    missing = sorted(set(ALL_EXERCISES) - on_disk)
    assert not missing, f"no module for exercises {missing}"


def test_no_stray_exercise_modules() -> None:
    """Nothing on disk claims to be an exercise the workbook does not set."""
    on_disk = {(chapter, number) for chapter, number, _ in iter_exercises()}
    extra = sorted(on_disk - set(ALL_EXERCISES))
    assert not extra, f"modules with no workbook exercise: {extra}"


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_module_imports(chapter: int, number: int) -> None:
    """Every exercise module imports cleanly — no data access at import time."""
    module = load(chapter, number)
    assert module.__doc__, f"{module.__name__} has no docstring"


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_answer_shape(chapter: int, number: int) -> None:
    """``ANSWER`` is a non-empty dict of readable prose, keyed by topic."""
    module = load(chapter, number)
    answer = getattr(module, "ANSWER", None)
    assert isinstance(answer, dict), f"{module.__name__}.ANSWER is not a dict"
    assert answer, f"{module.__name__}.ANSWER is empty"
    for key, value in answer.items():
        assert isinstance(key, str) and key, f"bad ANSWER key in {module.__name__}"
        assert value is not None, f"{module.__name__}.ANSWER[{key!r}] is None"
        if isinstance(value, str):
            assert len(value) > 40, (
                f"{module.__name__}.ANSWER[{key!r}] is too short to be an answer"
            )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_docstring_quotes_the_exercise(chapter: int, number: int) -> None:
    """The module docstring names its chapter and restates the question."""
    module = load(chapter, number)
    doc = module.__doc__ or ""
    assert f"Chapter {chapter}" in doc, (
        f"{module.__name__} docstring does not name its chapter"
    )
    assert f"exercise {number}" in doc.lower(), (
        f"{module.__name__} docstring does not name its exercise number"
    )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_solve_is_callable_without_required_arguments(
    chapter: int, number: int,
) -> None:
    """``solve()``, where present, runs with no arguments."""
    module = load(chapter, number)
    solve = getattr(module, "solve", None)
    if solve is None:
        return
    signature = inspect.signature(solve)
    required = [
        name for name, p in signature.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"{module.__name__}.solve() requires arguments {required}; "
        "notebooks call it bare"
    )


def test_chapter_table_matches_the_workbook() -> None:
    """``CHAPTERS``/``EXERCISES`` agree with the .tex files they describe."""
    import re
    from pathlib import Path

    chapters = Path(__file__).resolve().parents[1] / "article" / "chapters"
    for chapter, (stem, _, label) in CHAPTERS.items():
        tex = (chapters / f"{stem}.tex").read_text()
        assert f"\\label{{sec:{label}}}" in tex, f"{stem}: label mismatch"
        block = re.search(
            r"\\begin\{issues\}\[EXERCISES\](.*?)\\end\{issues\}", tex, re.S,
        )
        assert block is not None, f"{stem}: no EXERCISES block"
        n_items = len(re.findall(r"\n\s*\\item\s", block.group(1)))
        assert n_items == EXERCISES[chapter], (
            f"{stem}: workbook has {n_items} exercises, "
            f"EXERCISES says {EXERCISES[chapter]}"
        )


def test_notebooks_are_up_to_date() -> None:
    """``make_exercise_notebooks.py --check`` passes (no drift)."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/make_exercise_notebooks.py", "--check"],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _shipped_decks() -> list[Path]:
    """The decks the generator writes.

    Jupyter's own ``*-checkpoint.ipynb`` autosaves are deliberately excluded:
    they are not generated output, and a stale copy should not be able to fail
    the suite.
    """
    root = Path(__file__).resolve().parents[1]
    decks = [root / "notebooks" / "workbook_exercises.ipynb"]
    decks += sorted((root / "notebooks" / "exercises").glob("chapter_*.ipynb"))
    return [deck for deck in decks if deck.is_file()]


@pytest.mark.parametrize("deck", _shipped_decks(), ids=lambda path: path.name)
def test_every_notebook_cell_is_valid_python(deck: Path) -> None:
    """Every code cell in a shipped deck compiles.

    The decks are *generated* source. A scratch cell is assembled from another
    module's import statements, and that assembly can produce nonsense even
    when the module it came from is perfectly good — which is how an exercise
    once shipped telling students to ``import as``. Nothing else in this suite
    looks at the text a student is asked to run, so this compiles it.
    """
    from typing import cast

    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    # The setup cell uses ``%matplotlib inline``, so the cell text goes through
    # IPython's transformer before Python sees it.
    transform = TransformerManager().transform_cell
    # ``as_version=4`` is what makes this a NotebookNode: the stub also covers
    # the v1-v3 layouts, which read as bare lists of cells.
    notebook = cast(
        nbformat.NotebookNode, nbformat.read(deck, as_version=4),
    )

    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        try:
            compile(transform(source), f"{deck.name}:cell{index}", "exec")
        except SyntaxError as exc:
            first = (source.strip().splitlines() or [""])[0]
            pytest.fail(
                f"{deck.name} cell {index} is not valid Python: {exc}\n"
                f"    first line: {first}"
            )


def _generator():
    """Import ``scripts/make_exercise_notebooks.py`` by path.

    Loaded from its file rather than by module name: ``scripts/`` is not a
    package, and naming it as an import would make the static checker chase a
    module that only exists relative to the repository root.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_make_exercise_notebooks", root / "scripts" / "make_exercise_notebooks.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exercise_statements() -> list[tuple[int, int, str]]:
    """Every exercise statement as the notebooks will render it."""
    generator = _generator()

    return [
        (chapter, number, text)
        for chapter in sorted(EXERCISES)
        for number, text in enumerate(generator.exercise_texts(chapter), start=1)
    ]


#: LaTeX commands that are *not* maths: if one of these reaches a student's
#: cell, the converter missed it. Maths (``\sigma``, ``\kappa``, ``\mathbf``…)
#: is expected — the notebooks render ``$…$`` through MathJax.
NON_MATH_COMMANDS = re.compile(
    r"\\[a-zA-Z]+",
)


def test_no_latex_escapes_leak_into_statements() -> None:
    """Structural LaTeX in the exercise prose is converted, not shown raw.

    A student should never see ``\\ref{tab:clusters}`` or ``\\S 3.3``. Maths is
    exempt: it stays inside ``$…$`` and renders.
    """
    offenders: dict[tuple[int, int], list[str]] = {}
    for chapter, number, text in _exercise_statements():
        # strip inline maths, which is legitimately full of backslashes
        stripped = re.sub(r"\$[^$]*\$", " ", text)
        leftovers = sorted(set(NON_MATH_COMMANDS.findall(stripped)))
        if leftovers:
            offenders[(chapter, number)] = leftovers
    assert not offenders, f"LaTeX leaked into exercise prose: {offenders}"


def test_references_resolve_to_workbook_numbers() -> None:
    """``\\ref`` targets become the number the compiled workbook prints.

    The table the exercises point at is the cluster table, which is Table 1
    in the workbook — not a number the converter may invent.
    """
    numbers = _generator().reference_numbers()
    assert numbers.get("sec:fundamentals") == 3
    assert numbers.get("sec:kmeans") == 4
    assert numbers.get("tab:clusters") == 1
    # The workbook numbers floats with a single arabic counter per kind (the
    # class redefines \thetable/\thefigure/\theequation), and chapter 5 holds
    # two *unlabelled* numbered equations — which is why these are counted
    # from environments and not from labels.
    assert numbers.get("tab:honest") == 4
    assert numbers.get("fig:dbscansteps") == 11
    assert numbers.get("eq:mreach") == 4
    assert numbers.get("eq:stability") == 5
    assert numbers.get("eq:perplexity") == 10
    # every reference the exercises actually make must resolve
    referenced = set()
    for _, _, text in _exercise_statements():
        referenced |= set(re.findall(r"§(\d+)", text))
    assert referenced, "no resolved section references found at all"


# --------------------------------------------------------------------------- #
# Citations: keys must resolve, and rendered text must not leak LaTeX.
# --------------------------------------------------------------------------- #

def test_bibliography_parses() -> None:
    """``article/references.bib`` is readable and every entry has a year."""
    from exercises.citations import bibliography

    bib = bibliography()
    assert len(bib) >= 50, f"only {len(bib)} bib entries parsed"
    missing_year = sorted(k for k, ref in bib.items() if not ref.year)
    assert not missing_year, f"bib entries with no year: {missing_year}"
    missing_author = sorted(k for k, ref in bib.items() if not ref.authors)
    assert not missing_author, f"bib entries with no author: {missing_author}"


def test_unknown_citation_key_raises() -> None:
    """A typo'd key fails loudly instead of printing a dead reference."""
    from exercises.citations import cite

    with pytest.raises(KeyError, match="references.bib"):
        cite("Campelo:13")


def test_citations_render_without_latex() -> None:
    """No brace-protection, ``\\url`` or accent macro reaches a student.

    The bib is written for LaTeX; the answers are printed to a terminal, so
    every rendered citation has to be plain text.
    """
    from exercises.citations import bibliography, cite, reference_list

    for key in bibliography():
        rendered = cite(key) + " " + reference_list(key)[key]
        assert "\\" not in rendered, f"LaTeX leaked from {key}: {rendered}"
        assert "{" not in rendered and "}" not in rendered, (
            f"unstripped braces from {key}: {rendered}"
        )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_answer_references_are_real_bib_keys(chapter: int, number: int) -> None:
    """Any ``references`` entry names keys that exist in the bibliography.

    ``reference_list`` already raises on an unknown key, so this catches the
    other direction: a hand-written ``references`` dict that bypassed it.
    """
    from exercises.citations import bibliography

    answer = load(chapter, number).ANSWER
    references = answer.get("references")
    if references is None:
        return
    assert isinstance(references, dict), (
        f"exercise {chapter}.{number}: 'references' must be a dict of "
        f"key -> rendered reference, got {type(references).__name__}"
    )
    bib = bibliography()
    unknown = sorted(set(references) - set(bib))
    assert not unknown, (
        f"exercise {chapter}.{number} cites keys absent from references.bib: "
        f"{unknown}"
    )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_answers_do_not_hand_type_citations(chapter: int, number: int) -> None:
    """Author-year citations come from ``cite()``, not from typing.

    A hand-typed ``(Campello et al. 2013)`` cannot be checked against the
    bibliography and silently rots when the bib changes. The source must go
    through ``cite``/``reference_list``; this test fails on a literal
    author-year string in the module source that is not inside a ``cite`` call.
    """
    source = inspect.getsource(load(chapter, number))
    # strip the calls that legitimately produce author-year text
    without_calls = re.sub(r"cite\([^)]*\)|reference_list\([^)]*\)", " ", source,
                           flags=re.S)
    hand_typed = re.findall(
        r"\(\s*[A-Z][A-Za-z'\-]+(?:\s+(?:et al\.?|and\s+[A-Z][A-Za-z'\-]+))?"
        r"\s+(?:19|20)\d\d[a-z]?\s*\)",
        without_calls,
    )
    assert not hand_typed, (
        f"exercise {chapter}.{number} hand-types citations {hand_typed}; "
        f"use cite('Key:yy') so the key is checked against references.bib"
    )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_citation_import_is_actually_used(chapter: int, number: int) -> None:
    """A module that imports the citation API must use it.

    This catches a half-finished edit: an interrupted worker can leave the
    import in place with no ``cite()`` call and no ``references`` key, which
    every other test happily accepts because the module still runs.
    """
    source = inspect.getsource(load(chapter, number))
    if "from exercises.citations import" not in source:
        return
    imported = re.search(
        r"from exercises\.citations import ([^\n]+)", source,
    )
    assert imported is not None
    names = imported.group(1)
    if "cite" in names:
        assert re.search(r"\bcite\(", source), (
            f"exercise {chapter}.{number} imports cite but never calls it — "
            f"an unfinished citation edit"
        )
    if "reference_list" in names:
        assert re.search(r"\breference_list\(", source), (
            f"exercise {chapter}.{number} imports reference_list but never "
            f"calls it — an unfinished citation edit"
        )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_cited_exercises_carry_a_reference_list(chapter: int, number: int) -> None:
    """Whatever an answer cites inline, it also lists under ``references``.

    The inline ``(Author year)`` is the pointer; the ``references`` entry is
    where a student finds the journal and DOI to actually reach the paper.
    One without the other is a dead end.
    """
    module = load(chapter, number)
    source = inspect.getsource(module)
    if not re.search(r"\bcite\(", source):
        return
    answer = module.ANSWER
    assert "references" in answer, (
        f"exercise {chapter}.{number} cites inline but has no 'references' "
        f"entry; add reference_list(...) as the last ANSWER key"
    )
    assert answer["references"], (
        f"exercise {chapter}.{number} has an empty 'references' entry"
    )


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _modules_with_plot() -> list[tuple[int, int]]:
    """Every exercise whose module defines ``plot()``."""
    found = []
    for chapter, number, _ in iter_exercises():
        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "exercises" / f"{module_name(chapter, number)}.py"
        ).read_text()
        if re.search(r"^def plot\(", source, re.M):
            found.append((chapter, number))
    return found


@pytest.mark.parametrize(("chapter", "number"), _modules_with_plot())
def test_plot_is_callable_with_no_arguments(chapter: int, number: int) -> None:
    """``plot()`` takes no *required* argument.

    The generated deck cell calls it bare, exactly as it calls ``solve()``. A
    plot that needs a hand-built argument cannot be presented by a notebook,
    which is the whole contract of this package.
    """
    module = load(chapter, number)
    plot = module.plot
    parameters = inspect.signature(plot).parameters
    required = [
        name for name, p in parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not required, (
        f"exercise {chapter}.{number} plot() requires {required}; "
        f"give them defaults so the notebook can call plot()"
    )


@pytest.mark.parametrize(("chapter", "number"), _modules_with_plot())
def test_every_plot_reaches_a_notebook_cell(chapter: int, number: int) -> None:
    """A module that defines ``plot()`` has a cell calling it.

    52 plot functions were once written, committed, and never wired to
    anything: no deck cell imported them, no test ran them, and five had
    rotted (two calling a `cluster.baseline` symbol that did not exist) with
    nothing to notice. This is the guard that a plot reaches a student.
    """
    name = module_name(chapter, number)
    wanted = f"from exercises.{name} import plot"
    decks = _shipped_decks()
    assert decks, "no shipped decks found"
    hits = {
        deck.name for deck in decks
        if wanted in deck.read_text(encoding="utf-8")
    }
    assert hits, (
        f"exercise {chapter}.{number} defines plot() but no deck calls it — "
        f"re-run scripts/make_exercise_notebooks.py"
    )
    # Both decks, not either: a student working through one chapter must get
    # the same figures as one working through the master deck. Checking "any
    # deck" hides a plot that dropped out of its chapter deck alone.
    master = "workbook_exercises.ipynb"
    chapter_decks = {deck for deck in hits if deck != master}
    assert master in hits, (
        f"exercise {chapter}.{number} plot() is missing from the master deck"
    )
    assert chapter_decks, (
        f"exercise {chapter}.{number} plot() reaches only the master deck, "
        f"not its chapter deck — re-run scripts/make_exercise_notebooks.py"
    )


@pytest.mark.parametrize(("chapter", "number"), ALL_EXERCISES)
def test_scripts_an_exercise_names_actually_exist(chapter: int, number: int) -> None:
    """A module that tells you to run a script must name one that is here.

    Exercise error messages double as instructions — ``DataNotAvailable``
    tells the student which command regenerates the missing file. Two such
    scripts were left behind when this work was migrated between repositories,
    so the advice pointed at nothing: the exercise failed, named a fix, and the
    fix did not exist. Nothing caught it, because the message is only produced
    on the machine that lacks the data.
    """
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "src" / "exercises" / f"{module_name(chapter, number)}.py"
    ).read_text()
    # The path usually sits mid-string, after a runner: "uv run python
    # scripts/foo.py" or ".venv/bin/python scripts/foo.py" — so anchor on the
    # directory, not on a quote.
    named = set(re.findall(r"((?:article/)?scripts/[a-z0-9_]+\.py)", source))
    missing = sorted(path for path in named if not (root / path).is_file())
    assert not missing, (
        f"exercise {chapter}.{number} points the student at {missing}, "
        f"which is not in this repository"
    )


def test_no_source_file_hardcodes_an_absolute_home_path() -> None:
    """Nothing may hardcode a path from the machine it was written on.

    ``scripts/casamiquela_comparison.py`` arrived from another repository with
    ``REPO = Path("/home/<user>/git/…-draft")`` baked in. It imported fine on
    the machine that still had that checkout and would have failed for every
    student — the worst kind of defect, because it is invisible to the author.
    Resolve paths from ``Path(__file__)`` instead.
    """
    root = Path(__file__).resolve().parents[1]
    trees = [root / "src", root / "scripts", root / "tests"]
    offenders: list[str] = []
    for tree in trees:
        for path in tree.rglob("*.py"):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1,
            ):
                if re.search(r"""["']/home/\w+""", line):
                    offenders.append(f"{path.relative_to(root)}:{number}")
    assert not offenders, (
        f"absolute home paths are not portable: {offenders}"
    )


@pytest.mark.parametrize("deck", _shipped_decks(), ids=lambda p: p.name)
def test_solve_cells_reset_result_before_computing(deck: Path) -> None:
    """A failed ``solve()`` must not leave the previous exercise's result bound.

    The decks run top to bottom in one kernel. When ``solve()`` raises —
    routinely, for exercises whose data is not on disk — a bare
    ``result = solve()`` leaves ``result`` holding the *previous* exercise's
    dict, and the plot cell below then draws the wrong exercise's data and dies
    with a meaningless ``KeyError: 'scatter'`` instead of the
    ``DataNotAvailable`` that names the missing file. Each solve cell therefore
    resets ``result`` first.
    """
    notebook = json.loads(deck.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    solve_cells = [source for source in sources if "import solve" in source]
    assert solve_cells, f"{deck.name} has no solve cells"
    missing = [
        source.splitlines()[2] for source in solve_cells
        if "result = None" not in source
    ]
    assert not missing, (
        f"{deck.name}: {len(missing)} solve cell(s) bind result without "
        f"resetting it first — a raising solve() would leak the previous "
        f"exercise's result into the plot cell: {missing[:3]}"
    )


def test_figures_module_exposes_the_three_families() -> None:
    """The shared figure helpers the exercises build their plots from."""
    from exercises import figures

    for name in ("embedding_scatter", "cmd_diagram", "sky_cutout"):
        assert callable(getattr(figures, name)), f"figures.{name} missing"


def test_sky_cutout_survives_having_no_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline, ``sky_cutout`` draws the RA/Dec fallback instead of raising.

    The decks run in CI and on student laptops with no connection; a figure
    that needs SkyView must degrade to the astrometry the catalogue already
    carries rather than fail the cell.
    """
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    from exercises import figures

    monkeypatch.setattr(figures, "NO_NETWORK", True)
    frame = pd.DataFrame({
        "RA": [10.0, 10.1, 10.2, 99.0],
        "DEC": [-5.0, -5.1, -5.2, 42.0],
    })
    mask = [True, True, True, False]
    fig = figures.sky_cutout(frame, "NGC test", member_mask=mask)
    axes = fig.get_axes()[0]
    assert axes.get_xlabel() == "RA (deg)"
    # RA increases eastwards, i.e. leftwards on the sky
    assert axes.get_xlim()[0] > axes.get_xlim()[1]


def test_cmd_diagram_refuses_an_isochrone_on_absolute_axes() -> None:
    """A fit is in apparent magnitude; overlaying it on M_G would be wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd

    from exercises import figures

    frame = pd.DataFrame({
        "GAIAEDR3_PHOT_G_MEAN_MAG": [12.0, 13.0],
        "GAIAEDR3_PHOT_BP_MEAN_MAG": [12.5, 13.6],
        "GAIAEDR3_PHOT_RP_MEAN_MAG": [11.4, 12.3],
        "GAIAEDR3_PARALLAX": [1.0, 1.2],
    })
    with pytest.raises(ValueError, match="apparent magnitude"):
        figures.cmd_diagram(
            frame, absolute=True, curve_color=[1.0], curve_mag=[12.0],
        )


# --------------------------------------------------------------------------- #
# The data-backed half: only runs where the catalogue is present.
# --------------------------------------------------------------------------- #

@needs_data
def test_shared_population_is_the_workbook_sample() -> None:
    """``utils.members()`` is the 1 002-row / 25-cluster matrix of §2."""
    from exercises.utils import members

    data = members()
    assert data.X.shape == (len(data.df), 16)
    labels = {str(c) for c in data.labels}
    assert "field" not in labels
    assert len(labels) == 25, f"expected 25 clusters, got {len(labels)}"
    assert 950 <= len(data.df) <= 1050, f"unexpected member count {len(data.df)}"


@needs_data
def test_member_field_keeps_every_member() -> None:
    """The field cap never drops a cluster member (stratified sampling)."""
    from exercises.utils import member_field, members

    assert int(member_field().is_member.sum()) >= len(members().df)
