"""Plotly figures for the desk. The map style needs no token."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from lte_analyzer.flags import (
    FLAG_INTERFERENCE,
    FLAG_MARGINAL,
    FLAG_OK,
    FLAG_ORDER,
    FLAG_WEAK,
    POOR_RSRQ_DB,
    POOR_SINR_DB,
    USABLE_RSRP_DBM,
    WEAK_RSRP_DBM,
)

FLAG_COLORS = {
    FLAG_OK: "#94a3b8",
    FLAG_MARGINAL: "#6366f1",
    FLAG_INTERFERENCE: "#f59e0b",
    FLAG_WEAK: "#ef4444",
}

# Low values (poor radio) start at red. RSRP and RSRQ are negative, so the
# low end of the scale is the bad end, which matches the field habit.
KPI_SCALES = {
    "rsrp": ["#b91c1c", "#f97316", "#facc15", "#84cc16", "#15803d"],
    "rsrq": ["#b91c1c", "#f97316", "#facc15", "#84cc16", "#15803d"],
    "sinr": ["#b91c1c", "#f97316", "#facc15", "#84cc16", "#15803d"],
}

KPI_TITLES = {
    "rsrp": "RSRP (dBm)",
    "rsrq": "RSRQ (dB)",
    "sinr": "SINR (dB)",
}


def _center(frame: pd.DataFrame) -> dict[str, float]:
    return {"lat": float(frame["lat"].median()), "lon": float(frame["lon"].median())}


def _hover(frame: pd.DataFrame) -> list[str]:
    lines = []
    has_pci = "pci" in frame.columns
    for row in frame.itertuples(index=False):
        pci = f"<br>PCI {int(row.pci)}" if has_pci and pd.notna(row.pci) else ""
        lines.append(
            f"{row.route}<br>{row.timestamp:%H:%M:%S}"
            f"{pci}<br>RSRP {row.rsrp:.0f} dBm · RSRQ {row.rsrq:.1f} dB · SINR {row.sinr:.1f} dB"
            f"<br>DL {row.dl_mbps:.1f} Mbps · UL {row.ul_mbps:.1f} Mbps"
            f"<br>{row.flag_label}"
        )
    return lines


def coverage_map(frame: pd.DataFrame, kpi: str) -> go.Figure:
    fig = go.Figure(
        go.Scattermap(
            lat=frame["lat"],
            lon=frame["lon"],
            mode="markers",
            marker=dict(
                size=9,
                color=frame[kpi],
                colorscale=KPI_SCALES[kpi],
                colorbar=dict(title=KPI_TITLES[kpi], thickness=14),
                opacity=0.9,
            ),
            text=_hover(frame),
            hoverinfo="text",
            name=KPI_TITLES[kpi],
        )
    )
    center = _center(frame)
    fig.update_layout(
        map_style="open-street-map",
        map_zoom=11,
        map_center=center,
        margin=dict(l=0, r=0, t=8, b=0),
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", size=12),
    )
    return fig


def flag_map(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    hover = _hover(frame)
    frame = frame.copy()
    frame["_hover"] = hover
    sizes = {FLAG_OK: 6, FLAG_MARGINAL: 9, FLAG_INTERFERENCE: 11, FLAG_WEAK: 12}
    opacity = {FLAG_OK: 0.35, FLAG_MARGINAL: 0.85, FLAG_INTERFERENCE: 0.92, FLAG_WEAK: 0.95}
    for flag in FLAG_ORDER:
        chunk = frame[frame["flag"] == flag]
        if chunk.empty:
            continue
        fig.add_trace(
            go.Scattermap(
                lat=chunk["lat"],
                lon=chunk["lon"],
                mode="markers",
                name=chunk["flag_label"].iloc[0],
                marker=dict(size=sizes[flag], color=FLAG_COLORS[flag], opacity=opacity[flag]),
                text=chunk["_hover"],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        map_style="open-street-map",
        map_zoom=11,
        map_center=_center(frame),
        margin=dict(l=0, r=0, t=8, b=0),
        height=560,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", size=12),
    )
    return fig


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=8, r=8, t=48, b=8),
        height=height,
        font=dict(color="#e7eefc"),
        legend=dict(bgcolor="rgba(15,23,42,0.4)"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


def histogram(frame: pd.DataFrame, kpi: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=frame[kpi], marker_color="#38bdf8", nbinsx=28, name=KPI_TITLES[kpi]))
    lines = {
        "rsrp": [(WEAK_RSRP_DBM, "Weak"), (USABLE_RSRP_DBM, "Usable")],
        "rsrq": [(POOR_RSRQ_DB, "Poor")],
        "sinr": [(POOR_SINR_DB, "Poor")],
    }
    for value, label in lines[kpi]:
        fig.add_vline(x=value, line_dash="dash", line_color="#f8fafc", annotation_text=label, annotation_position="top")
    return _style(fig, KPI_TITLES[kpi], height=280)


def scatter_kpi(frame: pd.DataFrame, x: str, color: str, title: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=frame[x],
            y=frame["dl_mbps"],
            mode="markers",
            marker=dict(
                size=7,
                color=frame[color],
                colorscale=KPI_SCALES[color],
                opacity=0.75,
                colorbar=dict(title=KPI_TITLES[color], thickness=12),
            ),
            text=_hover(frame),
            hoverinfo="text",
            name="Samples",
        )
    )
    edges = {
        "rsrp": [-140, -120, -110, -100, -90, -80, -55],
        "sinr": [-15, -5, 0, 3, 8, 15, 20, 30],
    }[x]
    bins = pd.cut(frame[x], bins=edges, include_lowest=True)
    grouped = frame.groupby(bins, observed=True)["dl_mbps"].median()
    centers = [interval.mid for interval in grouped.index]
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=grouped.to_numpy(),
            mode="lines+markers",
            name="Bin median",
            line=dict(color="#f8fafc", width=2),
            marker=dict(size=8, color="#f8fafc"),
        )
    )
    return _style(fig, title)


def model_bars(baseline_mae: float, model_mae: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Guess the training average", "Linear regression"],
            y=[baseline_mae, model_mae],
            marker_color=["#64748b", "#38bdf8"],
            text=[f"{baseline_mae:.1f}", f"{model_mae:.1f}"],
            textposition="outside",
        )
    )
    fig.update_yaxes(title="Test MAE (Mbps)")
    return _style(fig, "Typical error on minutes the fit did not see", height=360)


def pred_vs_actual(y_true: list[float], y_pred: list[float]) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(size=7, color="#38bdf8", opacity=0.75),
            name="Held-out samples",
        )
    )
    hi = max(max(y_true), max(y_pred), 1)
    fig.add_trace(
        go.Scatter(
            x=[0, hi],
            y=[0, hi],
            mode="lines",
            name="Perfect line",
            line=dict(color="#f8fafc", dash="dash", width=1),
        )
    )
    fig.update_xaxes(title="Actual downlink (Mbps)")
    fig.update_yaxes(title="Predicted downlink (Mbps)")
    return _style(fig, "Held-out prediction against the speed the phone recorded", height=420)


def coefficient_bars(coefficients: list[dict]) -> go.Figure:
    names = [f"{item['feature']} (per 1 SD)" for item in coefficients]
    values = [float(item["per_std"]) for item in coefficients]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color="#38bdf8"))
    fig.update_yaxes(title="Mbps associated with a typical swing")
    return _style(fig, "Which KPI moves with speed, inside this fit", height=360)
