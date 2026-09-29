"""Build the Jupyter exercise decks from the workbook and the solution modules.

The notebooks are *presentation only*: every cell either states an exercise,
offers the student a clean scratch cell, or reveals an answer that lives in
``src/exercises/exercise_<chapter>_<n>.py``. No result is computed in a
notebook that is not computed in a module, so the notebooks and the workbook
can never drift from the code that backs them.

    uv run python scripts/make_exercise_notebooks.py            # build all
    uv run python scripts/make_exercise_notebooks.py --check    # CI: no drift

Outputs:
    notebooks/workbook_exercises.ipynb          master deck (all 16 chapters)
    notebooks/exercises/chapter_NN_<slug>.ipynb one deck per chapter
"""

from __future__ import annotations

import argparse
import ast
import keyword
import re
import sys
from functools import lru_cache
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exercises import CHAPTERS, EXERCISES, module_name  # noqa: E402

CHAPTER_DIR = ROOT / "article" / "chapters"
NOTEBOOK_DIR = ROOT / "notebooks"
PER_CHAPTER_DIR = NOTEBOOK_DIR / "exercises"

#: Chapters whose exercises are answered by running code against the data,
#: rather than by reasoning alone. Chapter-level, so the notebook can say
#: which kind of cell a student is looking at.
KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}


#: ``\S\ref{sec:kmeans}`` -> ``§4``: the workbook cross-references chapters by
#: label, but a reader of the notebook wants the chapter number.
LABEL_TO_CHAPTER = {label: number for number, (_, _, label) in CHAPTERS.items()}


def _section_ref(match: re.Match[str]) -> str:
    """``sec:kmeans`` -> ``§4``, falling back to the raw label."""
    label = match.group(1)
    return f"§{LABEL_TO_CHAPTER.get(label, label)}"


def _document_tex() -> str:
    """The whole workbook, chapters spliced in ``\\input`` order.

    LaTeX numbers tables, figures and equations in document order, so the
    concatenation is what lets :func:`reference_numbers` resolve ``\\ref`` to
    the same number the PDF shows.
    """
    main = (ROOT / "article" / "workbook.tex").read_text()
    order = re.findall(r"\\input\{([^}]*)\}", main)
    parts = []
    for name in order:
        path = ROOT / "article" / f"{name}.tex"
        if path.is_file():
            parts.append(path.read_text())
    return "\n".join(parts)


#: LaTeX environments that consume one equation number.
NUMBERED_MATH = ("equation", "align", "gather", "multline", "eqnarray")

#: Matches the *unstarred* form of those environments (starred = unnumbered).
_NUMBERED_MATH_RE = re.compile(
    r"\\begin\{(" + "|".join(NUMBERED_MATH) + r")\}",
)


def _labels_in_environments(
    text: str, begin: str, prefix: str, *, numbered: bool,
) -> dict[str, int]:
    """Number floats by counting *environments*, which is what LaTeX does.

    Counting ``\\label``s instead would be wrong the moment a table, figure or
    equation appears without one: the workbook has two unlabelled numbered
    equations in chapter 5, and counting labels silently shifted every later
    equation number by two.
    """
    pattern = _NUMBERED_MATH_RE if numbered else re.compile(
        r"\\begin\{" + begin + r"\*?\}",
    )
    close = begin
    numbers: dict[str, int] = {}
    counter = 0
    for match in pattern.finditer(text):
        end = text.find(f"\\end{{{close}}}", match.end())
        body = text[match.end(): end if end >= 0 else match.end()]
        counter += 1
        found = re.search(r"\\label\{" + prefix + r":([^}]*)\}", body)
        if found:
            numbers[f"{prefix}:{found.group(1)}"] = counter
    return numbers


@lru_cache(maxsize=1)
def reference_numbers() -> dict[str, int]:
    """Map every ``\\label`` in the workbook to the number the PDF prints.

    Counters are per kind and walk the document in order, which is the same
    rule LaTeX applies. The class redefines ``\\thetable``, ``\\thefigure`` and
    ``\\theequation`` to plain arabic counters, so there is a single sequence
    per kind and no section prefix.
    """
    text = _document_tex()
    numbers: dict[str, int] = {}
    for match in re.finditer(r"\\label\{sec:([^}]*)\}", text):
        numbers[f"sec:{match.group(1)}"] = LABEL_TO_CHAPTER.get(
            match.group(1), 0,
        )
    numbers.update(_labels_in_environments(text, "table", "tab", numbered=False))
    numbers.update(_labels_in_environments(text, "figure", "fig", numbered=False))
    numbers.update(_labels_in_environments(text, "equation", "eq", numbered=True))
    return numbers


def _section_reference(match: re.Match[str]) -> str:
    """``sec:kmeans`` -> ``§4``, using the workbook's chapter numbering."""
    resolved = reference_numbers().get(f"sec:{match.group(1)}")
    return f"§{resolved}" if resolved else match.group(1)


def _numbered_reference(match: re.Match[str]) -> str:
    """``tab:clusters`` -> ``1``.

    Just the number: the workbook's prose always names the kind itself
    (``Table~\\ref{tab:clusters}``), so adding a prefix here would print
    "Table Table 1".
    """
    resolved = reference_numbers().get(f"{match.group(1)}:{match.group(2)}")
    return str(resolved) if resolved else match.group(2)


def latex_to_text(tex: str) -> str:
    """Render the workbook's LaTeX exercise prose as readable markdown.

    Not a general LaTeX converter — it handles exactly the constructs the
    exercise blocks use, and leaves anything else alone. Math is deliberately
    left in ``$...$``: the notebooks render it, so ``$\\sigma$`` stays.

    Inner constructs are resolved *before* the wrapping ones, because the
    workbook writes references inside emphasis (``\\textbf{Table~\\ref{...}}``)
    and a non-nesting regex cannot match those in one pass.
    """
    text = tex
    # 1. innermost: citations and cross-references. Section refs carry their
    #    own §, whether or not the source wrote a \S before them; numbered
    #    refs resolve to the bare number.
    text = re.sub(r"\\citet\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\citep\{([^{}]*)\}", r"(\1)", text)
    text = re.sub(r"\\citealp\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\S\s*\\ref\{sec:([^{}]*)\}", _section_reference, text)
    text = re.sub(r"\\ref\{sec:([^{}]*)\}", _section_reference, text)
    text = re.sub(
        r"\\ref\{(tab|fig|eq):([^{}]*)\}", _numbered_reference, text,
    )
    # 2. the emphasis that may have wrapped them
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", text)
    text = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{[^{}]*\}", r"\1", text)
    # 3. a leftover \S before a section name
    text = re.sub(r"\\S\s*\\ref\{sec:[^{}]*\}", "§", text)
    text = re.sub(r"\\S\s*", "§", text)
    # 4. spacing, dashes and escapes
    text = text.replace("\\,", " ").replace("\\ ", " ").replace("\\%", "%")
    text = text.replace("~", " ").replace("---", "—").replace("--", "–")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("\\_", "_").replace("\\&", "&").replace("\\#", "#")
    # 5. `$[$Fe/H$]$` is the workbook's "[Fe/H]"; markdown would render the
    #    two brackets as separate fragments of inline math.
    text = text.replace("$[$", "[").replace("$]$", "]")
    return " ".join(text.split())


def exercise_texts(chapter: int) -> list[str]:
    """The exercise statements of one chapter, in workbook order."""
    stem = CHAPTERS[chapter][0]
    tex = (CHAPTER_DIR / f"{stem}.tex").read_text()
    blocks = re.findall(
        r"\\begin\{issues\}\[EXERCISES\](.*?)\\end\{issues\}", tex, re.S,
    )
    if not blocks:
        return []
    body = re.sub(r"\\begin\{enumerate\}|\\end\{enumerate\}", "", blocks[0])
    parts = re.split(r"\n\s*\\item\s", "\n" + body)
    return [latex_to_text(p) for p in parts if p.strip()]


def slug(title: str) -> str:
    """``'K-means: the default tool' -> 'kmeans'`` for the file name."""
    return CHAPTERS_SLUG[title]


CHAPTERS_SLUG = {title: label for _, title, label in CHAPTERS.values()}


def intro_cells(chapter: int | None) -> list[nbformat.NotebookNode]:
    """Title + setup cells shared by the master and per-chapter decks."""
    if chapter is None:
        title = "# Companion workbook — exercise deck\n\n"
        scope = (
            "Every exercise from `article/workbook.tex`, all 16 chapters.\n\n"
        )
    else:
        stem, name, _label = CHAPTERS[chapter]
        title = f"# Chapter {chapter} — {name}\n\n"
        scope = (
            f"The exercises of §{chapter} "
            f"(`article/chapters/{stem}.tex`).\n\n"
        )

    how_it_works = (
        "**How this deck works.** Each exercise gets three cells:\n\n"
        "1. the question, exactly as the workbook states it;\n"
        "2. a **clean cell** for you to work in — the imports you need are "
        "already there, the solution is not;\n"
        "3. the **answer**, which is printed from "
        "`src/exercises/exercise_<chapter>_<n>.py`.\n\n"
        "The notebook computes nothing itself. Every number comes from a "
        "module you can open, read and re-run, so anything you see here you "
        "can also reproduce from a plain Python prompt.\n\n"
        "Work the clean cell *before* running the answer cell. The answers "
        "are not hidden — they are just one cell further down, and the "
        "exercise only works if you resist for a few minutes.\n\n"
        "**Data.** The exercises that touch real data need the SDSS-V DR19 "
        "catalogue (~1.17 GB) and, for the spectral chapters, the embedding "
        "bundle:\n\n"
        "```bash\n"
        "uv run cluster download          # the catalogue\n"
        "uv run cluster download --assets # the embeddings\n"
        "```\n\n"
        "Loading the catalogue takes ~20 s the first time; afterwards the "
        "prepared population is cached under `results/exercise_cache/` and "
        "every later call is instant."
    )

    setup = (
        "# Run me first: puts the repository root on the path and imports\n"
        "# the shared helpers the exercises use.\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path.cwd()\n"
        "while not (ROOT / 'pyproject.toml').is_file() and ROOT != ROOT.parent:\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n"
        "\n"
        "%matplotlib inline\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from exercises import utils\n"
        "from exercises.utils import show\n"
        "\n"
        "print('exercise helpers ready —', ROOT)"
    )

    return [
        new_markdown_cell(title + scope + how_it_works),
        new_code_cell(setup),
    ]


def exercise_cells(
    chapter: int, number: int, statement: str,
) -> list[nbformat.NotebookNode]:
    """The three cells of one exercise."""
    name = module_name(chapter, number)
    module_path = f"src/exercises/{name}.py"
    has_module = (ROOT / module_path).is_file()

    question = (
        f"### Exercise {chapter}.{number}\n\n"
        f"{statement}\n\n"
        f"<sub>Solution module: `{module_path}`</sub>"
    )

    # The clean cell imports the exercise's own module plus whatever of the
    # `cluster` package it is built on, which is the hint: the import list
    # tells the student which part of the codebase already solves this.
    hints = module_hints(chapter, number) if has_module else []
    hint_lines = "\n".join(hints)
    scratch = (
        f"# --- Exercise {chapter}.{number} — your workings ---------------\n"
        f"{hint_lines}\n"
        "\n"
        "# Your code here.\n"
    )

    if has_module:
        cells = [
            new_markdown_cell(question),
            new_code_cell(scratch),
            new_code_cell(
                f"# --- Exercise {chapter}.{number} — the answer "
                f"------------------\n"
                f"from exercises.{name} import ANSWER\n"
                "\n"
                "show(ANSWER)",
            ),
        ]
        if module_has_solve(chapter, number):
            cells.append(new_code_cell(
                f"# Recompute it: solve() is where the real work happens, and\n"
                f"# it reads the same data you have on disk.\n"
                f"from exercises.{name} import solve\n"
                "\n"
                "result = None  # a failure below must not leak into the plot\n"
                "result = solve()\n"
                "for key, value in result.items():\n"
                "    show({key: value})",
            ))
        if module_has_figure(chapter, number):
            reuse = (
                module_has_solve(chapter, number)
                and plot_takes_result(chapter, number)
            )
            if reuse:
                body = (
                    f"# --- Exercise {chapter}.{number} — the picture "
                    f"-----------------\n"
                    f"# plot() draws what solve() just computed. The result is\n"
                    f"# handed straight over, so nothing is computed twice.\n"
                    f"from exercises.{name} import plot\n"
                    "\n"
                    "plot(result)"
                )
            else:
                body = (
                    f"# --- Exercise {chapter}.{number} — the picture "
                    f"-----------------\n"
                    f"# plot() draws the result. Like solve(), it computes in\n"
                    f"# the module, not here.\n"
                    f"from exercises.{name} import plot\n"
                    "\n"
                    "plot()"
                )
            cells.append(new_code_cell(body))
        return cells

    # pragma: no cover — every exercise ships a module
    return [
        new_markdown_cell(question),
        new_code_cell(scratch),
        new_code_cell(f"# {name} is not implemented yet."),
    ]


def module_source(chapter: int, number: int) -> str:
    path = ROOT / "src" / "exercises" / f"{module_name(chapter, number)}.py"
    return path.read_text() if path.is_file() else ""


def module_has_solve(chapter: int, number: int) -> bool:
    return "def solve(" in module_source(chapter, number)


def plot_takes_result(chapter: int, number: int) -> bool:
    """Whether ``plot()`` can be handed the result ``solve()`` already computed.

    Requires *both* that the first parameter is named ``result`` and that it is
    annotated as a dict. Neither test alone is enough:

    - arity alone is wrong, because some plots take a tuning parameter
      (``plot(min_pts: int)``) that is not a result at all;
    - the annotation alone is wrong, because ``exercise_03_1.plot`` takes
      ``curve: dict[str, np.ndarray]`` — a *different* dict, built by its own
      ``concentration_curve()``, which raises ``KeyError`` if handed
      ``solve()``'s output.

    This matters for more than tidiness: with a bare ``plot()`` every exercise
    computes twice — once in the ``solve()`` cell and again in the plot cell —
    which took the master deck from 14 minutes to over ten hours.
    """
    source = module_source(chapter, number)
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - a module that will not parse
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "plot":
            parameters = node.args.posonlyargs + node.args.args
            if not parameters or parameters[0].arg != "result":
                return False
            annotation = parameters[0].annotation
            if annotation is None:
                return False
            return "dict" in ast.unparse(annotation)
    return False


def module_has_figure(chapter: int, number: int) -> bool:
    """Whether this exercise draws its result.

    Figures are opt-in and use the name the modules already use: ``plot()``.
    Detected the same way as ``solve()`` — by source inspection, so generating
    the decks never imports an exercise or touches the catalogue.
    """
    return re.search(r"^def plot\(", module_source(chapter, number), re.M) is not None


#: Helpers in :mod:`exercises.utils` that actually load the catalogue or the
#: population. Importing only, say, ``SEEDS`` is not a data dependency, and
#: claiming otherwise would send students looking for a download they do not
#: need.
DATA_HELPERS = frozenset({
    "DataNotAvailable", "MemberData", "abundance_matrix", "catalogue_path",
    "cluster_named", "embedding_path", "knn_purity_raw", "member_field",
    "members", "settings",
})

PENCIL_AND_PAPER = [
    "# This one is pencil-and-paper (or a few lines of numpy) —",
    "# no data or repository machinery is needed, beyond the imports above.",
]


def _utils_import_names(source: str) -> set[str]:
    """The names a module imports from :mod:`exercises.utils`.

    Parsed as Python rather than split on whitespace, because a module may
    alias what it imports (``from exercises.utils import settings as _settings``).
    Re-importing that text verbatim would put the alias machinery — and the
    keyword ``as`` — into a scratch cell as if it were a name, which is not
    even valid syntax. The original name is kept here: the alias exists to
    suit that module's own namespace, while a scratch cell starts empty and is
    better served by the public name a student can call.
    """
    names: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # a module that will not parse offers no hint
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "exercises.utils":
            names.update(
                alias.name
                for alias in node.names
                if not keyword.iskeyword(alias.name)
            )
    return names


def module_hints(chapter: int, number: int) -> list[str]:
    """Import lines and a one-line steer for the clean cell.

    When the exercise's own module imports something from ``cluster``, the
    scratch cell imports it too: that is the hint that the repository already
    contains the machinery for this exercise, and the student still has to
    work out how to put it together.

    Exercises that reach the data through :mod:`exercises.utils` — itself a
    thin wrapper over ``cluster.data.prepare`` — get that import instead, so
    the steer is never a false claim about what the codebase offers.
    """
    source = module_source(chapter, number)

    cluster_imports: list[str] = []
    for match in re.finditer(
        r"^\s*from (cluster[\w.]*) import ([^\n(]+)$", source, re.M,
    ):
        names = " ".join(match.group(2).split())
        cluster_imports.append(f"from {match.group(1)} import {names}")
    for match in re.finditer(r"^\s*import (cluster[\w.]*)$", source, re.M):
        cluster_imports.append(f"import {match.group(1)}")
    seen: set[str] = set()
    cluster_imports = [
        h for h in cluster_imports if not (h in seen or seen.add(h))
    ]

    from_utils = _utils_import_names(source)
    utils_names: list[str] = sorted(from_utils)
    utils_line = (
        f"from exercises.utils import {', '.join(utils_names)}"
        if utils_names else ""
    )
    needs_data = bool(DATA_HELPERS.intersection(utils_names))

    if cluster_imports:
        header = [
            "# The `cluster` package already implements the pieces this",
            "# exercise needs — these are the ones the solution uses:",
        ]
        return header + ([utils_line] if utils_line else []) + sorted(cluster_imports)

    if needs_data:
        return [
            "# The shared helpers load and cache the data this exercise needs",
            "# (`exercises.utils` wraps `cluster.data.prepare`):",
            utils_line,
        ]

    # handy constants only — the work is still the student's
    return ([utils_line] if utils_line else []) + PENCIL_AND_PAPER


def build_chapter(chapter: int, standalone: bool) -> nbformat.NotebookNode:
    """One chapter's notebook."""
    cells = intro_cells(chapter) if standalone else []
    statements = exercise_texts(chapter)
    if not standalone:
        stem, name, _label = CHAPTERS[chapter]
        cells.append(new_markdown_cell(
            f"---\n\n## Chapter {chapter} — {name}\n\n"
            f"<sub>§{chapter} · `article/chapters/{stem}.tex`</sub>",
        ))
    for number, statement in enumerate(statements, start=1):
        cells.extend(exercise_cells(chapter, number, statement))

    nb = new_notebook(cells=cells, metadata=KERNEL_META)
    return nb


def build_master() -> nbformat.NotebookNode:
    """The single deck covering every chapter."""
    cells = intro_cells(None)
    toc = ["\n**Contents**\n"]
    for chapter in sorted(CHAPTERS):
        _, name, label = CHAPTERS[chapter]
        toc.append(f"{chapter}. {name} — {EXERCISES[chapter]} exercises")
    cells.append(new_markdown_cell("\n".join(toc)))

    for chapter in sorted(CHAPTERS):
        cells.extend(build_chapter(chapter, standalone=False).cells)

    return new_notebook(cells=cells, metadata=KERNEL_META)


def write(nb: nbformat.NotebookNode, path: Path) -> bool:
    """Write ``nb``; return True when the file changed."""
    nbformat.validate(nb)
    text = nbformat.writes(nb, version=4) + "\n"
    if path.is_file() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def _stamp_ids(nb: nbformat.NotebookNode, prefix: str) -> nbformat.NotebookNode:
    """Give every cell a deterministic id.

    ``nbformat.v4.new_*_cell`` mints a random uuid per call, so an unmodified
    rebuild would differ from the file on disk in every cell and ``--check``
    could never pass. Numbering the cells by position makes the build
    reproducible, which is what lets CI diff it.
    """
    for index, cell in enumerate(nb.cells):
        cell["id"] = f"{prefix}-{index:03d}"
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Exit non-zero if a notebook is out of date (for CI).",
    )
    args = parser.parse_args()

    targets: list[tuple[nbformat.NotebookNode, Path]] = [
        (_stamp_ids(build_master(), "master"),
         NOTEBOOK_DIR / "workbook_exercises.ipynb"),
    ]
    for chapter in sorted(CHAPTERS):
        _, _, label = CHAPTERS[chapter]
        targets.append((
            _stamp_ids(
                build_chapter(chapter, standalone=True), f"ch{chapter:02d}",
            ),
            PER_CHAPTER_DIR / f"chapter_{chapter:02d}_{label}.ipynb",
        ))

    changed: list[Path] = []
    for nb, path in targets:
        if args.check:
            text = nbformat.writes(nb, version=4) + "\n"
            if not path.is_file() or path.read_text() != text:
                changed.append(path)
        elif write(nb, path):
            changed.append(path)

    if args.check:
        if changed:
            print("out of date — re-run scripts/make_exercise_notebooks.py:")
            for path in changed:
                print(f"  {path.relative_to(ROOT)}")
            return 1
        print(f"{len(targets)} notebooks up to date")
        return 0

    for path in changed:
        print(f"wrote {path.relative_to(ROOT)}")
    n_cells = sum(len(nb.cells) for nb, _ in targets)
    print(f"{len(targets)} notebooks, {n_cells} cells total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
