"""Plotly chart builders for the dashboard.

The palette follows the validated light-mode reference palette:
blue / orange / aqua / yellow / magenta / green / violet / red.
Red and green are reserved for polarity (up/down, positive/negative) in
contexts where the meaning is explicit; categorical identity uses the first
six slots in fixed order.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Reference palette (light surface) -------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#6250d6"
RED = "#e34948"

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

FACTOR_COMPONENTS = [
    ("Size", "size", BLUE),
    ("Value", "value", ORANGE),
    ("Momentum", "momentum", AQUA),
    ("Volatility", "volatility", YELLOW),
    ("Specific", "specific", MAGENTA),
    ("Other", None, GREEN),
]


def apply_base_layout(
    fig: go.Figure,
    *,
    height: int = 420,
    title: Optional[str] = None,
    x_title: Optional[str] = None,
    y_title: Optional[str] = None,
    hovermode: str = "x unified",
    showlegend: bool = True,
    legend_orientation: str = "h",
    legend_y: float = 1.02,
    margin_bottom: int = 20,
) -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=13, color=TEXT_PRIMARY),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(size=17, color=TEXT_PRIMARY),
            )
            if title
            else None
        ),
        margin=dict(l=8, r=8, t=52 if title else 16, b=margin_bottom),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=GRID,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        ),
        showlegend=showlegend,
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=legend_y,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_SECONDARY),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        linecolor=BASELINE,
        zeroline=False,
        tickfont=dict(family=FONT_FAMILY, size=11, color=MUTED),
        title=dict(
            text=x_title,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_SECONDARY),
        ),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=BASELINE,
        zeroline=False,
        tickfont=dict(family=FONT_FAMILY, size=11, color=MUTED),
        title=dict(
            text=y_title,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_SECONDARY),
        ),
    )
    return fig


def period_shading(fig: go.Figure, periods: Iterable[Dict[str, Any]]) -> None:
    """Alternate period bands and rebalance markers on a date axis."""
    periods = list(periods)
    for index, period in enumerate(periods):
        start = period.get("start")
        end = period.get("end")
        if not start or not end:
            continue
        if index % 2 == 1:
            fig.add_vrect(
                x0=str(start)[:10],
                x1=str(end)[:10],
                fillcolor="rgba(137,135,129,0.07)",
                line_width=0,
                layer="below",
            )
        if index > 0:
            fig.add_vline(
                x=str(start)[:10],
                line_width=1,
                line_color=BASELINE,
                line_dash="dot",
                layer="below",
            )


def nav_chart(
    nav_rows: List[Dict[str, Any]],
    benchmark_rows: List[Dict[str, Any]],
    periods: List[Dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    nav_dates = [str(row["date"])[:10] for row in nav_rows]
    nav_values = [row.get("nav") for row in nav_rows]

    fig.add_trace(
        go.Scatter(
            x=nav_dates,
            y=nav_values,
            mode="lines",
            name="Portfolio (net)",
            line=dict(color=BLUE, width=2.2),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Portfolio NAV: %{y:.4f}<extra></extra>"
            ),
        )
    )

    benchmark_values: List[Optional[float]] = []
    first_valid: Optional[float] = None
    for row in benchmark_rows:
        value = row.get("nav")
        if value is not None and first_valid is None:
            first_valid = float(value)
        benchmark_values.append(
            float(value) / first_valid
            if value is not None and first_valid
            else None
        )
    if benchmark_rows and first_valid:
        fig.add_trace(
            go.Scatter(
                x=[str(row["date"])[:10] for row in benchmark_rows],
                y=benchmark_values,
                mode="lines",
                name="CSI 300 ETF (normalized)",
                line=dict(color=ORANGE, width=1.8),
                connectgaps=False,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Benchmark NAV: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=1.0,
        line_width=1,
        line_color=BASELINE,
        layer="below",
    )
    period_shading(fig, periods)
    apply_base_layout(
        fig,
        height=460,
        y_title="Cumulative NAV (start = 1.0)",
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".2f")
    return fig


def drawdown_chart(drawdown_rows: List[Dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[str(row["date"])[:10] for row in drawdown_rows],
            y=[row.get("drawdown") for row in drawdown_rows],
            mode="lines",
            line=dict(color=RED, width=1.6),
            fill="tozeroy",
            fillcolor="rgba(227,73,72,0.16)",
            name="Drawdown",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE, layer="below")
    apply_base_layout(
        fig,
        height=320,
        title="Portfolio drawdown",
        y_title="Drawdown",
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def period_returns_chart(periods: List[Dict[str, Any]]) -> go.Figure:
    names = [period.get("name", "") for period in periods]
    returns = [
        (period.get("return") or 0.0) * 100.0 for period in periods
    ]
    colors = [BLUE if value >= 0 else RED for value in returns]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=returns,
            marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x}<br>Return: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE, layer="below")
    apply_base_layout(
        fig,
        height=320,
        title="Period returns (net of costs)",
        y_title="Return",
        showlegend=False,
        hovermode="closest",
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def factor_contribution_chart(rows: List[Dict[str, Any]]) -> go.Figure:
    periods = [str(row.get("period", "")) for row in rows]
    fig = go.Figure()

    for label, key, color in FACTOR_COMPONENTS:
        if key is None:
            values = []
            for row in rows:
                first = row.get("bench_resid") or 0.0
                second = row.get("unattributed") or 0.0
                values.append((float(first) + float(second)) * 100.0)
        else:
            values = [
                (row.get(key) or 0.0) * 100.0
                for row in rows
            ]
        fig.add_trace(
            go.Bar(
                name=label,
                x=periods,
                y=values,
                marker=dict(
                    color=color,
                    line=dict(color=SURFACE, width=2),
                ),
                hovertemplate=(
                    f"{label}<br>%{{x}}<br>%{{y:+.2f}} pp<extra></extra>"
                ),
            )
        )

    totals = [(row.get("total") or 0.0) * 100.0 for row in rows]
    fig.add_trace(
        go.Scatter(
            name="Total active return",
            x=periods,
            y=totals,
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=10,
                color=TEXT_PRIMARY,
                line=dict(color=SURFACE, width=2),
            ),
            hovertemplate=(
                "Total<br>%{x}<br>%{y:+.2f} pp<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0, line_width=1, line_color=BASELINE, layer="below")
    apply_base_layout(
        fig,
        height=540,
        title="Factor and specific contributions by period",
        y_title="Contribution to active return (pp)",
        x_title="Period",
        hovermode="x unified",
        legend_orientation="h",
        legend_y=-0.24,
        margin_bottom=105,
    )
    fig.update_layout(barmode="relative", bargap=0.28)
    return fig


def group_spread_chart(rows: List[Dict[str, Any]]) -> go.Figure:
    periods = [str(row.get("period", "")) for row in rows]
    spreads = [(row.get("spread") or 0.0) * 100.0 for row in rows]
    colors = [BLUE if value >= 0 else RED for value in spreads]

    upper: List[float] = []
    lower: List[float] = []
    for row, spread in zip(rows, spreads):
        high = row.get("boot_hi")
        low = row.get("boot_lo")
        upper.append(
            max(0.0, float(high) * 100.0 - spread)
            if high is not None
            else 0.0
        )
        lower.append(
            max(0.0, spread - float(low) * 100.0)
            if low is not None
            else 0.0
        )

    fig = go.Figure(
        go.Bar(
            x=periods,
            y=spreads,
            marker=dict(color=colors, line=dict(color=SURFACE, width=2)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper,
                arrayminus=lower,
                color=BASELINE,
                thickness=1.2,
                width=4,
            ),
            hovertemplate=(
                "%{x}<br>Spread: %{y:+.2f} pp<br>"
                "Bootstrap CI shown as error bars<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE, layer="below")

    finite_spreads = [value for value in spreads if np.isfinite(value)]
    if finite_spreads:
        mean_spread = float(np.mean(finite_spreads))
        fig.add_hline(
            y=mean_spread,
            line_width=1,
            line_color=TEXT_SECONDARY,
            layer="below",
        )
        fig.add_annotation(
            xref="paper",
            x=0.995,
            y=mean_spread,
            text=f"Mean {mean_spread:+.1f} pp",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(
                family=FONT_FAMILY,
                size=11,
                color=TEXT_SECONDARY,
            ),
        )

    apply_base_layout(
        fig,
        height=400,
        title="AI minus non-AI equal-weight return",
        y_title="Spread (pp)",
        x_title="Period",
        showlegend=False,
        hovermode="closest",
    )
    fig.update_yaxes(tickformat="+.0f")
    return fig


def group_returns_chart(rows: List[Dict[str, Any]]) -> go.Figure:
    periods = [str(row.get("period", "")) for row in rows]
    ai_values = [(row.get("mean_ai") or 0.0) * 100.0 for row in rows]
    non_values = [(row.get("mean_non") or 0.0) * 100.0 for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="AI group",
            x=periods,
            y=ai_values,
            marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
            hovertemplate="AI<br>%{x}<br>%{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Non-AI group",
            x=periods,
            y=non_values,
            marker=dict(color=ORANGE, line=dict(color=SURFACE, width=2)),
            hovertemplate="Non-AI<br>%{x}<br>%{y:+.2f}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=BASELINE, layer="below")
    apply_base_layout(
        fig,
        height=400,
        title="Equal-weight period returns by group",
        y_title="Return",
        x_title="Period",
        hovermode="closest",
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(ticksuffix="%")
    return fig


def holdings_contribution_chart(
    holdings: List[Dict[str, Any]],
    limit: int = 25,
) -> go.Figure:
    ordered = sorted(
        holdings,
        key=lambda item: item.get("contrib") or 0.0,
    )
    truncated = len(ordered) > limit
    if truncated:
        largest = sorted(
            ordered,
            key=lambda item: abs(item.get("contrib") or 0.0),
            reverse=True,
        )[:limit]
        ordered = sorted(largest, key=lambda item: item.get("contrib") or 0.0)

    labels = [
        f"{item.get('code6', '')} {item.get('display_name') or item.get('name', '')}".strip()[:26]
        for item in ordered
    ]
    values = [(item.get("contrib") or 0.0) * 100.0 for item in ordered]
    colors = [BLUE if value >= 0 else RED for value in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color=SURFACE, width=1)),
            hovertemplate=(
                "%{y}<br>Contribution: %{x:+.2f} pp<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=BASELINE, layer="below")
    suffix = f" (top {limit} by absolute contribution)" if truncated else ""
    apply_base_layout(
        fig,
        height=max(360, 20 * len(ordered) + 90),
        title=f"Contribution to period return{suffix}",
        x_title="Contribution (pp)",
        showlegend=False,
        hovermode="closest",
    )
    fig.update_yaxes(automargin=True)
    return fig


def selection_count_chart(rows: List[Dict[str, Any]]) -> go.Figure:
    dates = [str(row.get("query_date", ""))[:10] for row in rows]
    counts = [row.get("count", 0) for row in rows]

    fig = go.Figure(
        go.Bar(
            x=dates,
            y=counts,
            marker=dict(color=BLUE, line=dict(color=SURFACE, width=2)),
            hovertemplate="%{x}<br>Selected: %{y}<extra></extra>",
        )
    )
    apply_base_layout(
        fig,
        height=300,
        title="Selected stocks by rebalance date",
        y_title="Number of stocks",
        showlegend=False,
        hovermode="closest",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def candlestick_chart(
    df: pd.DataFrame,
    entry_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing=dict(
                line=dict(color=STATUS_CRITICAL, width=1),
                fillcolor=STATUS_CRITICAL,
            ),
            decreasing=dict(
                line=dict(color=STATUS_GOOD, width=1),
                fillcolor=STATUS_GOOD,
            ),
            name="OHLC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if "volume" in df.columns:
        colors = [
            STATUS_CRITICAL if close >= open_ else STATUS_GOOD
            for open_, close in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                marker=dict(color=colors, opacity=0.55),
                name="Volume",
                showlegend=False,
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    if entry_date and end_date:
        for row in (1, 2):
            fig.add_vrect(
                x0=str(entry_date)[:10],
                x1=str(end_date)[:10],
                fillcolor="rgba(42,120,214,0.08)",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )

    fig.update_layout(
        height=520,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        margin=dict(l=8, r=8, t=16, b=16),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=GRID,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT_PRIMARY),
        ),
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    for axis in ("xaxis", "xaxis2"):
        fig.update_layout({
            axis: dict(
                gridcolor=GRID,
                linecolor=BASELINE,
                zeroline=False,
                tickfont=dict(family=FONT_FAMILY, size=11, color=MUTED),
            )
        })
    for axis in ("yaxis", "yaxis2"):
        fig.update_layout({
            axis: dict(
                gridcolor=GRID,
                linecolor=BASELINE,
                zeroline=False,
                tickfont=dict(family=FONT_FAMILY, size=11, color=MUTED),
            )
        })
    fig.update_yaxes(title_text="Price (CNY, adj.)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig
