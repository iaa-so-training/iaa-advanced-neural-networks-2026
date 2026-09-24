"""Visualisation helpers for the benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import BenchmarkResult, knn_purity


_CLUSTER_COLORS = {
    "Pleiades": "#ee1c2e",
    "M 67": "#00539f",
    "NGC 6819": "#008800",
    "NGC 6791": "#f97b02",
    "NGC 7789": "#991384",
    "M 5": "#007eb3",
    "M 3": "#fcd700",
    "field": "#b0b0b0",
}


# labels drawn as a pale bottom layer: non-members and EVoC/HDBSCAN noise
_BACKGROUND = {"field": "#b0b0b0", "noise": "#d9d9d9"}


def _color_for(name: str) -> str:
    return _CLUSTER_COLORS.get(name, "#555555")


def _field_last(v: str) -> tuple[bool, str]:
    """Sort key: background labels ('field', 'noise') first, so members draw on top."""
    return (v not in _BACKGROUND, v)


def _label_colors(labels: list[str]) -> dict[str, str]:
    """One colour per label.

    Background labels get their pale grey, named clusters keep their fixed
    colour, and every other label (the rest of the 25 clusters, or EVoC's
    ``c0``, ``c1``, ...) cycles through tab20 so no two neighbours share the
    single ``_color_for`` fallback grey.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cycle = [c for i, c in enumerate(colormaps["tab20"].colors) if i not in (14, 15)]  # drop the greys
    colors: dict[str, str] = {}
    i = 0
    for v in sorted(set(labels), key=_field_last):
        if v in _BACKGROUND:
            colors[v] = _BACKGROUND[v]
        elif v in _CLUSTER_COLORS:
            colors[v] = _CLUSTER_COLORS[v]
        else:
            colors[v] = to_hex(cycle[i % len(cycle)])
            i += 1
    return colors


def scatter_embedding(
    Z: np.ndarray,
    df: pd.DataFrame,
    color_by: str,
    ax: Any = None,
    *,
    title: str = "",
    legend: bool = True,
    s: float = 4,
) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    values = df[color_by].astype(str).to_numpy()
    # draw field first (bottom layer), then members
    order = np.argsort(values == "field")
    Z = Z[order]
    values = values[order]

    labels: list[str] = [str(v) for v in values]
    unique = sorted(set(labels), key=_field_last)
    colors = _label_colors(labels)
    for v in unique:
        m = values == v
        ax.scatter(
            Z[m, 0], Z[m, 1], s=s, c=colors[v], label=v, alpha=0.7,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend and len(unique) <= 12:
        ax.legend(markerscale=4, fontsize=7, frameon=False, loc="best")
    return ax


def plot_method_grid(
    benchmark: BenchmarkResult, out_path: str | Path | None = None,
) -> Any:
    """One panel per method: 2-D embedding coloured by true membership.

    EVoC has no public 2-D projection, so it is shown on the UMAP canvas
    coloured by its own cluster labels.
    """
    import matplotlib.pyplot as plt

    from .benchmark import BenchmarkResult

    assert benchmark.df is not None
    df = benchmark.df
    methods = list(benchmark.results)
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6))
    if len(methods) == 1:
        axes = [axes]

    umap_result = benchmark.results.get("UMAP")
    umap_z = umap_result.embedding if umap_result is not None else None
    color_by = "referee" if df is not None and "referee" in df.columns else "cluster"
    for ax, name in zip(axes, methods):
        r = benchmark.results[name]
        if name == "EVoC" and r.embedding is None:
            assert umap_z is not None
            # colour EVoC's labels on the shared UMAP canvas
            tmp = df.copy()
            tmp["_evoc"] = [f"c{int(l)}" if l >= 0 else "noise" for l in r.labels]
            scatter_embedding(
                umap_z, tmp, "_evoc", ax=ax,
                title=f"{name} (labels on UMAP canvas)", legend=False,
            )
        else:
            assert r.embedding is not None
            scatter_embedding(
                r.embedding, df, color_by, ax=ax,
                title=f"{name}\ncoloured by true membership",
            )
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def abundance_violins(
    df: pd.DataFrame,
    elements: list[str],
    cluster: str,
    out_path: str | Path | None = None,
    random_state: int = 42,
) -> Any:
    """Violin plot of each abundance: ``cluster`` members vs field stars.

    Field stars are subsampled for speed (violin shapes are stable with a few
    thousand points); cluster members are always kept. Returns the figure.
    """
    import matplotlib.pyplot as plt

    members = df[df["cluster"] == cluster]
    if members.empty:
        raise ValueError(f"cluster {cluster!r} not found in df['cluster']")
    field = df[df["cluster"] == "field"]
    if field.empty:
        field = df[df["cluster"] != cluster]
    if len(field) > 3000:
        field = field.sample(n=3000, random_state=random_state)

    n = len(elements)
    fig, axes = plt.subplots(
        1, n, figsize=(3.2 * n, 4.0), sharey=False, squeeze=False,
    )
    axes = axes[0]
    member_color = _color_for(cluster)
    field_color = _CLUSTER_COLORS["field"]

    for ax, element in zip(axes, elements):
        member_vals = members[element].to_numpy(dtype=float)
        field_vals = field[element].to_numpy(dtype=float)
        member_vals = member_vals[np.isfinite(member_vals)]
        field_vals = field_vals[np.isfinite(field_vals)]

        if len(member_vals) == 0 or len(field_vals) == 0:
            ax.set_title(element, fontsize=8)
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=7,
            )
            ax.set_xticks([])
            continue

        parts = ax.violinplot(
            [member_vals, field_vals], positions=[0, 1],
            showmeans=False, showmedians=True, showextrema=False,
        )
        for body, color in zip(parts["bodies"], (member_color, field_color)):
            body.set_facecolor(color)
            body.set_alpha(0.75)
            body.set_edgecolor("none")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

        ax.set_title(element, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["member", "field"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle(
        f"{cluster}: member vs field abundance distributions",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def method_comparison_bar(
    benchmark: BenchmarkResult, out_path: str | Path | None = None,
) -> Any:
    """Grouped bar chart of macro recall / precision / kNN purity per method.

    Recall and precision come from ``BenchmarkResult.macro()``. Purity is the
    parameter-free chemical-cohesion score (``knn_purity``): the mean fraction
    of a member's 10 nearest neighbours that belong to the same cluster. It is
    only available for methods that expose a 2-D embedding, so EVoC has no bar
    in that group.
    """
    import matplotlib.pyplot as plt

    assert benchmark.df is not None
    macro = benchmark.macro()
    if macro.empty:
        raise ValueError("benchmark has no scored methods")

    methods: list[str] = [str(m) for m in macro["method"]]
    recall = macro["recall"].to_numpy(dtype=float)
    precision = macro["precision"].to_numpy(dtype=float)

    true_labels = benchmark.df["cluster"].to_numpy()
    purity: list[float] = []
    for name in methods:
        result = benchmark.results.get(name)
        if result is not None and result.embedding is not None:
            per_cluster = knn_purity(result.embedding, true_labels)
            purity.append(
                float(np.mean(list(per_cluster.values())))
                if per_cluster else float("nan")
            )
        else:
            purity.append(float("nan"))

    metrics = ["recall", "precision", "purity (kNN)"]
    values = np.asarray([recall, precision, np.asarray(purity)])
    colors = ["#00539f", "#ee1c2e", "#008800"]
    x = np.arange(len(methods))
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 * len(methods)), 4.2))
    for i, (metric, vals, color) in enumerate(zip(metrics, values, colors)):
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("macro score", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Macro recall / precision / kNN purity per method", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    for patch in ax.patches:
        bar: Any = patch
        height = bar.get_height()
        if np.isfinite(height):
            ax.annotate(
                f"{height:.2f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig


def hr_interactive(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cluster_name: str,
    highlight: str = "combined",
) -> Any:
    """Interactive (plotly) HR diagram for the notebook.

    One row, two panels: Gaia CMD (absolute G vs BP−RP) and Kiel diagram
    (logg vs Teff). Members of ``highlight`` (one of "catalog", "kinematic",
    "combined") are red, everything else grey. Hover shows the star id,
    Teff/logg and which of the three membership sources flag it — so the
    students can zoom into a sequence and see where the methods agree.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plx = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    g = df["GAIAEDR3_PHOT_G_MEAN_MAG"].to_numpy(dtype=float)
    bp = df["GAIAEDR3_PHOT_BP_MEAN_MAG"].to_numpy(dtype=float)
    rp = df["GAIAEDR3_PHOT_RP_MEAN_MAG"].to_numpy(dtype=float)
    teff = df["TEFF"].to_numpy(dtype=float)
    logg = df["LOGG"].to_numpy(dtype=float)
    apid = df["APOGEE_ID"].astype(str).to_numpy()

    bp_rp = bp - rp
    abs_g = _abs_g(plx, g)

    cat = np.asarray(masks["catalog"], dtype=bool)
    kin = np.asarray(masks["kinematic"], dtype=bool)
    comb = np.asarray(masks["combined"], dtype=bool)
    hl = np.asarray(masks.get(highlight, comb), dtype=bool)

    def _yn(b: bool) -> str:
        return "yes" if b else "no"

    hover = np.array([
        f"{apid[i]}<br>Teff {teff[i]:.0f} K &nbsp; logg {logg[i]:.2f}"
        f"<br>catalog {_yn(cat[i])} &nbsp; kinematic {_yn(kin[i])} &nbsp; combined {_yn(comb[i])}"
        for i in range(len(df))
    ], dtype=object)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Gaia CMD", "Kiel diagram"),
        horizontal_spacing=0.12,
    )

    ok_cmd = np.isfinite(bp_rp) & np.isfinite(abs_g) & (plx > 0)
    ok_kiel = np.isfinite(teff) & np.isfinite(logg)

    def _add(panel_x: np.ndarray, panel_y: np.ndarray, mask: np.ndarray, row: int, col: int, name: str) -> None:
        fig.add_trace(
            go.Scatter(
                x=panel_x[mask & ~hl], y=panel_y[mask & ~hl],
                mode="markers", name=f"{name}: field",
                marker=dict(size=3, color="#c9c9c9", opacity=0.35),
                hoverinfo="skip",
            ), row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=panel_x[mask & hl], y=panel_y[mask & hl],
                mode="markers", name=f"{name}: {highlight}",
                marker=dict(size=7, color="#ee1c2e", opacity=0.9),
                text=hover[mask & hl], hovertemplate="%{text}<extra></extra>",
            ), row=row, col=col,
        )

    _add(bp_rp, abs_g, ok_cmd, 1, 1, "CMD")
    _add(teff, logg, ok_kiel, 1, 2, "Kiel")

    fig.update_yaxes(autorange="reversed", row=1, col=1, title_text="absolute G")
    fig.update_yaxes(autorange="reversed", row=1, col=2, title_text="log g")
    fig.update_xaxes(title_text="BP − RP", row=1, col=1)
    fig.update_xaxes(autorange="reversed", title_text="Teff (K)", row=1, col=2)
    fig.update_layout(
        title=f"{cluster_name}: {highlight} members (hover to inspect)",
        height=420, legend=dict(font=dict(size=11)), margin=dict(t=60),
    )
    return fig


def cluster_panels(
    df_apogee: pd.DataFrame,
    df_gaia: pd.DataFrame,
    cluster_name: str,
    apogee_mask: np.ndarray | None = None,
    gaia_mask: np.ndarray | None = None,
    fit: Any = None,
) -> Any:
    """Three-panel plotly diagnostic: Kiel, 2MASS CMD, Gaia CMD.

    Panel 1: logg vs Teff (APOGEE spectroscopy).
    Panel 2: K vs J-K (2MASS photometry) — the second colour.
    Panel 3: G vs BP-RP (Gaia, full depth) — the main sequence.
    Members (of the selected source) are red, field grey; the fitted
    isochrone (if given) is overlaid on the Gaia panel.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    apogee_mask = np.asarray(apogee_mask, dtype=bool) if apogee_mask is not None else np.zeros(len(df_apogee), dtype=bool)
    gaia_mask = np.asarray(gaia_mask, dtype=bool) if gaia_mask is not None else np.zeros(len(df_gaia), dtype=bool)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Kiel: logg vs Teff", "2MASS: K vs J−Ks", "Gaia: G vs BP−RP"),
        horizontal_spacing=0.08,
    )

    # --- panel 1: Kiel ---
    teff = df_apogee["TEFF"].to_numpy(dtype=float)
    logg = df_apogee["LOGG"].to_numpy(dtype=float)
    ok = np.isfinite(teff) & np.isfinite(logg)
    fig.add_trace(go.Scatter(x=teff[ok & ~apogee_mask], y=logg[ok & ~apogee_mask],
                             mode="markers", marker=dict(size=3, color="#c9c9c9", opacity=0.3),
                             name="field", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=teff[ok & apogee_mask], y=logg[ok & apogee_mask],
                             mode="markers", marker=dict(size=6, color="#ee1c2e", opacity=0.9),
                             name="member", showlegend=False), row=1, col=1)

    # --- panel 2: 2MASS ---
    j = df_apogee["J"].to_numpy(dtype=float)
    k = df_apogee["K"].to_numpy(dtype=float)
    jk = j - k
    ok2 = np.isfinite(jk) & np.isfinite(k)
    fig.add_trace(go.Scatter(x=jk[ok2 & ~apogee_mask], y=k[ok2 & ~apogee_mask],
                             mode="markers", marker=dict(size=3, color="#c9c9c9", opacity=0.3),
                             name="field", hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=jk[ok2 & apogee_mask], y=k[ok2 & apogee_mask],
                             mode="markers", marker=dict(size=6, color="#ee1c2e", opacity=0.9),
                             name="member", showlegend=False), row=1, col=2)

    # --- panel 3: Gaia ---
    g = df_gaia["phot_g_mean_mag"].to_numpy(dtype=float)
    bp = df_gaia["phot_bp_mean_mag"].to_numpy(dtype=float)
    rp = df_gaia["phot_rp_mean_mag"].to_numpy(dtype=float)
    bprp = bp - rp
    ok3 = np.isfinite(bprp) & np.isfinite(g)
    fig.add_trace(go.Scatter(x=bprp[ok3 & ~gaia_mask], y=g[ok3 & ~gaia_mask],
                             mode="markers", marker=dict(size=3, color="#c9c9c9", opacity=0.3),
                             name="field", hoverinfo="skip", showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=bprp[ok3 & gaia_mask], y=g[ok3 & gaia_mask],
                             mode="markers", marker=dict(size=5, color="#ee1c2e", opacity=0.9),
                             name="member", showlegend=False), row=1, col=3)
    if fit is not None and getattr(fit, "curve_color", None) is not None:
        fig.add_trace(go.Scatter(x=fit.curve_color, y=fit.curve_mag, mode="lines",
                                 name="isochrone", line=dict(color="#00539f", width=2)),
                      row=1, col=3)

    fig.update_xaxes(autorange="reversed", row=1, col=1, title_text="Teff (K)")
    fig.update_yaxes(autorange="reversed", row=1, col=1, title_text="log g")
    fig.update_xaxes(title_text="J − Ks", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2, title_text="K")
    fig.update_xaxes(title_text="BP − RP", row=1, col=3)
    fig.update_yaxes(autorange="reversed", row=1, col=3, title_text="G")
    fig.update_layout(
        title=f"{cluster_name}: three views of the cluster",
        height=430, legend=dict(font=dict(size=11)), margin=dict(t=70),
    )
    return fig


def embedding_interactive(benchmark: BenchmarkResult, method: str) -> Any:
    """Interactive (plotly) 2-D embedding for one method.

    Colours by the referee (Simbad) membership; hover shows the star id,
    Teff/logg, and the in-pipeline cluster label. EVoC has no public 2-D
    projection, so it is drawn on the UMAP canvas.
    """
    import plotly.graph_objects as go

    assert benchmark.df is not None
    df = benchmark.df
    result = benchmark.results[method]

    if method == "EVoC" and result.embedding is None:
        umap = benchmark.results.get("UMAP")
        assert umap is not None and umap.embedding is not None
        Z = umap.embedding
        note = "EVoC (labels on UMAP canvas)"
    else:
        assert result.embedding is not None
        Z = result.embedding
        note = method

    color_by = "referee" if "referee" in df.columns else "cluster"
    member = (df[color_by].to_numpy() != "field")
    apid = df["APOGEE_ID"].astype(str).to_numpy()
    teff = df["TEFF"].to_numpy(dtype=float)
    logg = df["LOGG"].to_numpy(dtype=float)
    labels = df["cluster"].astype(str).to_numpy()

    hover = np.array([
        f"{apid[i]}<br>Teff {teff[i]:.0f} K &nbsp; logg {logg[i]:.2f}"
        f"<br>cluster={labels[i]} &nbsp; referee={color_by}={df[color_by].to_numpy()[i]}"
        for i in range(len(df))
    ], dtype=object)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Z[~member, 0], y=Z[~member, 1], mode="markers", name="field",
        marker=dict(size=3, color="#c9c9c9", opacity=0.3), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=Z[member, 0], y=Z[member, 1], mode="markers", name="member",
        marker=dict(size=6, color="#ee1c2e", opacity=0.9),
        text=hover[member], hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{note} — coloured by {color_by} membership",
        height=520, legend=dict(font=dict(size=11)),
        xaxis_title="dim 1", yaxis_title="dim 2",
    )
    return fig


def _abs_g(plx_mas: np.ndarray, g_mag: np.ndarray) -> np.ndarray:
    """Absolute Gaia G magnitude from parallax (mas) and apparent G."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return g_mag - 5.0 * np.log10(1000.0 / plx_mas) + 5.0


def hr_comparison(
    df: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cluster_name: str,
    out_path: str | Path | None = None,
) -> Any:
    """HR-diagram comparison of three membership sources.

    Top row — Gaia colour-magnitude diagram (absolute G vs BP−RP).
    Bottom row — Kiel diagram (logg vs Teff).
    Columns — catalogue (Simbad) / kinematic / combined members.
    Field stars are grey; members of that source are red. A cleaner
    main sequence / giant branch under the members = better membership.
    """
    import matplotlib.pyplot as plt

    sources = ["catalog", "kinematic", "combined"]
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9), sharex="col", sharey="row",
    )

    g = df["GAIAEDR3_PHOT_G_MEAN_MAG"].to_numpy(dtype=float)
    bp = df["GAIAEDR3_PHOT_BP_MEAN_MAG"].to_numpy(dtype=float)
    rp = df["GAIAEDR3_PHOT_RP_MEAN_MAG"].to_numpy(dtype=float)
    plx = df["GAIAEDR3_PARALLAX"].to_numpy(dtype=float)
    teff = df["TEFF"].to_numpy(dtype=float)
    logg = df["LOGG"].to_numpy(dtype=float)

    cmd_x = bp - rp
    cmd_y = _abs_g(plx, g)
    finite_cmd = np.isfinite(cmd_x) & np.isfinite(cmd_y) & (plx > 0)
    finite_kiel = np.isfinite(teff) & np.isfinite(logg)

    for col, source in enumerate(sources):
        mask = np.asarray(masks[source], dtype=bool)

        ax = axes[0, col]
        m = finite_cmd
        ax.scatter(
            cmd_x[m & ~mask], cmd_y[m & ~mask], s=1.0, c="#c9c9c9",
            alpha=0.35, linewidths=0,
        )
        ax.scatter(
            cmd_x[m & mask], cmd_y[m & mask], s=8.0, c="#ee1c2e",
            alpha=0.9, linewidths=0,
        )
        ax.set_title(f"{source}\n({int(mask.sum())} members)", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("BP − RP (mag)", fontsize=9)

        ax = axes[1, col]
        k = finite_kiel
        ax.scatter(
            teff[k & ~mask], logg[k & ~mask], s=1.0, c="#c9c9c9",
            alpha=0.35, linewidths=0,
        )
        ax.scatter(
            teff[k & mask], logg[k & mask], s=8.0, c="#ee1c2e",
            alpha=0.9, linewidths=0,
        )
        ax.invert_xaxis()  # hot stars left
        ax.invert_yaxis()  # giants up
        ax.set_xlabel("Teff (K)", fontsize=9)

    axes[0, 0].set_ylabel("absolute G (mag)", fontsize=9)
    axes[1, 0].set_ylabel("log g (dex)", fontsize=9)
    fig.suptitle(
        f"{cluster_name}: HR diagrams — Simbad catalogue vs kinematic vs combined",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return fig
