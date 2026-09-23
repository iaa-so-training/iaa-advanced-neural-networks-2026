"""Tests for plotting helpers using the non-interactive Agg backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cluster import plots
from cluster.benchmark import BenchmarkResult, MethodResult
from cluster.plots import (
    _color_for,
    _field_last,
    abundance_violins,
    cluster_panels,
    embedding_interactive,
    hr_comparison,
    hr_interactive,
    method_comparison_bar,
    plot_method_grid,
    scatter_embedding,
)


def _scores(cluster: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster": [cluster],
            "n_true": [2],
            "n_pred": [2],
            "overlap": [2],
            "precision": [1.0],
            "recall": [1.0],
        }
    )


def _result() -> BenchmarkResult:
    n = 8
    df = pd.DataFrame({"cluster": ["A"] * 3 + ["B"] * 3 + ["field"] * 2})
    return BenchmarkResult(
        results={
            "t-SNE": MethodResult(
                name="t-SNE",
                embedding=np.column_stack([np.arange(n, dtype=float), np.zeros(n)]),
                labels=np.array([0, 0, 0, 1, 1, 1, -1, -1]),
                scores=pd.concat([_scores("A"), _scores("B")], ignore_index=True),
            ),
            "UMAP": MethodResult(
                name="UMAP",
                embedding=np.column_stack([np.arange(n, dtype=float), np.ones(n)]),
                labels=np.array([0, 0, 0, 1, 1, 1, -1, -1]),
                scores=pd.concat([_scores("A"), _scores("B")], ignore_index=True),
            ),
        },
        df=df,
        X=np.zeros((n, 2)),
    )


def _result_with_evoc() -> BenchmarkResult:
    result = _result()
    result.results["EVoC"] = MethodResult(
        name="EVoC",
        embedding=None,
        labels=np.array([0, 0, 0, 1, 1, 1, -1, -1]),
        scores=pd.concat([_scores("A"), _scores("B")], ignore_index=True),
    )
    return result


def test_color_for_known_and_unknown() -> None:
    assert _color_for("Pleiades") == "#ee1c2e"
    assert _color_for("unknown") == "#555555"


def test_field_last_sort_key() -> None:
    assert _field_last("field") == (False, "field")
    assert _field_last("Pleiades") == (True, "Pleiades")


def test_scatter_embedding_draws_field_first() -> None:
    df = pd.DataFrame({"cluster": ["field", "A", "A"]})
    Z = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    ax = scatter_embedding(Z, df, "cluster", title="scatter")
    assert ax is not None
    assert ax.get_title() == "scatter"
    plt.close(ax.figure)


def test_plot_method_grid_saves_png(tmp_path: Path) -> None:
    out = tmp_path / "grid.png"
    fig = plot_method_grid(_result(), out)
    assert out.exists()
    plt.close(fig)


def test_plot_method_grid_evoc_uses_umap_canvas(tmp_path: Path) -> None:
    out = tmp_path / "grid_evoc.png"
    fig = plot_method_grid(_result_with_evoc(), out)
    assert out.exists()
    plt.close(fig)


def test_label_colors_distinct_for_unnamed_labels() -> None:
    # EVoC's c0, c1, ... used to fall through _color_for to one grey, so the
    # EVoC panel rendered as a single dark mass.
    colors = plots._label_colors(["c0", "c1", "c2", "noise", "field", "Pleiades"])
    assert len({colors[c] for c in ("c0", "c1", "c2")}) == 3
    assert "#555555" not in colors.values()
    assert colors["Pleiades"] == "#ee1c2e"
    assert colors["field"] == "#b0b0b0"
    assert colors["noise"] != colors["c0"]


def test_plot_method_grid_evoc_panel_has_one_colour_per_label() -> None:
    fig = plot_method_grid(_result_with_evoc())
    evoc_ax = fig.axes[-1]
    fills = {tuple(np.round(c.get_facecolor()[0], 3)) for c in evoc_ax.collections}
    assert len(fills) == 3  # c0, c1, noise
    plt.close(fig)


def test_plot_method_grid_single_method() -> None:
    result = _result()
    result.results = {"t-SNE": result.results["t-SNE"]}
    fig = plot_method_grid(result)
    assert fig is not None
    plt.close(fig)


def test_abundance_violins_raises_for_missing_cluster() -> None:
    df = pd.DataFrame({"cluster": ["field", "field"], "C_FE": [0.0, 1.0]})
    with pytest.raises(ValueError, match="not found"):
        abundance_violins(df, ["C_FE"], "Pleiades")


def test_abundance_violins_creates_figure(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "cluster": ["Pleiades", "Pleiades", "field", "field"],
            "C_FE": [-0.2, 0.2, 1.0, 2.0],
            "N_FE": [0.0, 0.1, -1.0, 3.0],
        }
    )
    out = tmp_path / "violins.png"
    fig = abundance_violins(df, ["C_FE", "N_FE"], "Pleiades", out_path=out)
    assert out.exists()
    plt.close(fig)


def test_abundance_violins_no_data_branch(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "cluster": ["Pleiades", "field"],
            "C_FE": [0.0, np.nan],
        }
    )
    out = tmp_path / "violins_empty.png"
    fig = abundance_violins(df, ["C_FE"], "Pleiades", out_path=out)
    assert out.exists()
    plt.close(fig)


def test_method_comparison_bar_saves_png(tmp_path: Path) -> None:
    out = tmp_path / "bars.png"
    fig = method_comparison_bar(_result(), out)
    assert out.exists()
    plt.close(fig)


def test_method_comparison_bar_raises_when_no_methods() -> None:
    result = _result()
    result.results = {}
    with pytest.raises(ValueError, match="no scored methods"):
        method_comparison_bar(result)


def test_hr_comparison_has_six_panels(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "GAIAEDR3_PHOT_G_MEAN_MAG": [10.0, 11.0, 12.0, 13.0],
        "GAIAEDR3_PHOT_BP_MEAN_MAG": [10.8, 11.9, 13.0, 14.2],
        "GAIAEDR3_PHOT_RP_MEAN_MAG": [9.2, 10.1, 11.0, 12.0],
        "GAIAEDR3_PARALLAX": [7.0, 7.2, 7.1, 7.3],
        "TEFF": [4800.0, 4700.0, 4600.0, 4500.0],
        "LOGG": [4.2, 4.3, 4.1, 4.4],
    })
    masks = {
        "catalog": np.array([True, False, True, False]),
        "kinematic": np.array([True, False, False, False]),
        "combined": np.array([True, True, False, False]),
    }
    out = tmp_path / "hr.png"
    fig = hr_comparison(df, masks, "T", out_path=out)
    assert out.exists()
    assert len(fig.axes) == 6
    plt.close(fig)


def test_hr_interactive_returns_four_traces() -> None:
    df = pd.DataFrame({
        "APOGEE_ID": ["a", "b", "c", "d"],
        "GAIAEDR3_PHOT_G_MEAN_MAG": [10.0, 11.0, 12.0, 13.0],
        "GAIAEDR3_PHOT_BP_MEAN_MAG": [10.8, 11.9, 13.0, 14.2],
        "GAIAEDR3_PHOT_RP_MEAN_MAG": [9.2, 10.1, 11.0, 12.0],
        "GAIAEDR3_PARALLAX": [7.0, 7.2, 7.1, 7.3],
        "TEFF": [4800.0, 4700.0, 4600.0, 4500.0],
        "LOGG": [4.2, 4.3, 4.1, 4.4],
    })
    masks = {
        "catalog": np.array([True, False, True, False]),
        "kinematic": np.array([True, False, False, False]),
        "combined": np.array([True, True, False, False]),
    }
    fig = hr_interactive(df, masks, "T", highlight="combined")
    # two panels (CMD + Kiel) x two traces (field + members)
    assert len(fig.data) == 4


def test_embedding_interactive_two_traces() -> None:
    df = pd.DataFrame({
        "APOGEE_ID": ["a", "b", "c", "d"],
        "TEFF": [4800.0, 4700.0, 4600.0, 4500.0],
        "LOGG": [4.2, 4.3, 4.1, 4.4],
        "cluster": ["T", "field", "T", "field"],
        "referee": ["T", "field", "T", "field"],
    })
    result = BenchmarkResult(
        results={
            "UMAP": MethodResult(
                name="UMAP",
                embedding=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
                labels=np.array([0, -1, 0, -1]),
                scores=pd.DataFrame(),
            ),
        },
        df=df,
    )
    fig = embedding_interactive(result, "UMAP")
    # field + members
    assert len(fig.data) == 2


def test_cluster_panels_three_panels() -> None:
    df_a = pd.DataFrame({
        "TEFF": [4800.0, 4700.0, 4600.0, 4500.0],
        "LOGG": [4.2, 4.3, 4.1, 4.4],
        "J": [8.5, 8.6, 8.7, 8.8],
        "K": [7.8, 7.9, 8.0, 8.1],
    })
    df_g = pd.DataFrame({
        "phot_g_mean_mag": [10.0, 11.0, 12.0, 13.0],
        "phot_bp_mean_mag": [10.8, 11.9, 13.0, 14.2],
        "phot_rp_mean_mag": [9.2, 10.1, 11.0, 12.0],
    })
    m_a = np.array([True, False, True, False])
    m_g = np.array([True, True, False, False])
    fig = cluster_panels(df_a, df_g, "T", m_a, m_g)
    # 3 panels x 2 traces (field + member)
    assert len(fig.data) == 6
