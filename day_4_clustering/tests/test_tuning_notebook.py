"""Regression checks for the hand-written interactive tuning notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "tuning_template.ipynb"


def _cells() -> list[dict[str, Any]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return notebook["cells"]


def test_tuning_notebook_code_cells_compile() -> None:
    """Keep the hand-written notebook executable after direct edits."""
    for index, cell in enumerate(_cells()):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell{index}", "exec")


def test_tuning_callback_captures_widget_output() -> None:
    """Widget callbacks must render in the notebook instead of Jupyter's log."""
    sources = [
        "".join(cell["source"])
        for cell in _cells()
        if cell["cell_type"] == "code"
    ]
    controls = next(source for source in sources if "cluster_dropdown =" in source)
    callback = next(source for source in sources if "def _render(" in source)

    assert "result_output = widgets.Output()" in controls
    assert "history_output = widgets.Output()" in controls
    assert "with result_output:" in callback
    assert "with history_output:" in callback
    assert "history_rows[column] = value" in callback
    assert callback.index("with result_output:") < callback.index("clear_output(wait=True)")


def test_tuning_notebook_ships_without_results() -> None:
    """Do not commit local measurements or execution artefacts."""
    for cell in _cells():
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
