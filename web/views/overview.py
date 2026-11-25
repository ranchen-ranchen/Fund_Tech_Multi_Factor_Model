"""Portfolio overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, format, ui


def _daily_frame(
    portfolio: dict,
) -> pd.DataFrame:
    nav_rows = portfolio.get("nav", [])
    drawdown_rows = portfolio.get("drawdown", [])
    benchmark_rows = portfolio.get("benchmark_aligned", [])

    drawdown_map = {
        str(row.get("date", ""))[:10]: row.get("drawdown")
        for row in drawdown_rows
    }
    benchmark_map = {
        str(row.get("date", ""))[:10]: row.get("nav")
        for row in benchmark_rows
    }
    first_benchmark = next(
        (value for value in benchmark_map.values() if value is not None),
        None,
    )

    rows = []
    for row in nav_rows:
        date_text = str(row.get("date", ""))[:10]
        benchmark_value = benchmark_map.get(date_text)
        drawdown = drawdown_map.get(date_text)
        rows.append({
            "Date": date_text,
            "Portfolio NAV": row.get("nav"),
            "Benchmark NAV": (
                benchmark_value / first_benchmark
                if benchmark_value is not None and first_benchmark
                else None
            ),
            "Drawdown (%)": (
                drawdown * 100.0 if drawdown is not None else None
            ),
        })
    return pd.DataFrame(rows)


def render() -> None:
    portfolio = data.load_portfolio()
    if not portfolio:
        ui.missing_data(
            "portfolio",
            "python target_code_pipeline.py --skip-prep --no-update-k "
            "--export-details web/data/backtest_details.json\n"
            "python web/export_data.py",
        )
        return

    summary = (portfolio.get("summary") or [{}])[0]
    periods = portfolio.get("periods", [])
    nav_rows = portfolio.get("nav", [])
    benchmark_rows = portfolio.get("benchmark_aligned", [])
    drawdown_rows = portfolio.get("drawdown", [])
    meta = data.load_meta()

    if not nav_rows:
        st.warning("Portfolio NAV series is empty.")
        return

    start = str(nav_rows[0].get("date", ""))[:10]
    end = str(nav_rows[-1].get("date", ""))[:10]

    ui.page_header(
        "Portfolio overview",
        f"CSI 300 AI-compute-chain portfolio · {start} to {end} · net of trading costs",
    )

    cost_rate = summary.get("cost_rate") or 0.005
    risk_free = summary.get("risk_free_rate") or 0.02

    ui.metric_row([
        (
            "Final NAV",
            format.num(summary.get("final_nav"), 4),
            "Initial capital = 1.0.",
        ),
        (
            "Total return",
            format.pct(summary.get("total_return"), signed=True),
            "Net of round-trip costs.",
        ),
        (
            "CAGR",
            format.pct(summary.get("annual_return"), signed=True),
            "Geometric annualized return, 252 trading days.",
        ),
        (
            "Sharpe (standard)",
            format.num(summary.get("sharpe"), 2),
            "Arithmetic annualized excess return / annualized volatility.",
        ),
    ])
    ui.metric_row([
        (
            "Information ratio",
            format.num(summary.get("information_ratio"), 2),
            "Arithmetic annualized active return / tracking error.",
        ),
        (
            "Tracking error",
            format.pct(summary.get("tracking_error")),
            "Annualized standard deviation of daily active returns.",
        ),
        (
            "Max drawdown",
            format.pct(summary.get("max_drawdown"), signed=True),
            "Minimum of NAV / running maximum − 1.",
        ),
        (
            "Cumulative cost",
            format.pct(summary.get("total_cost")),
            "Total trading costs in initial-capital units.",
        ),
    ])

    st.caption(
        f"One-way cost {cost_rate:.2%} · round-trip {2 * cost_rate:.2%} per period · "
        f"risk-free rate {risk_free:.2%}. Sharpe and IR use arithmetic returns; "
        "CAGR is geometric. Benchmark: CSI 300 ETF (510300), normalized to its "
        "first common date."
    )

    ui.chart(charts.nav_chart(nav_rows, benchmark_rows, periods))
    if not benchmark_rows:
        st.info("Benchmark series is unavailable; only the portfolio NAV is shown.")

    left, right = st.columns(2)
    with left:
        ui.chart(charts.drawdown_chart(drawdown_rows))
    with right:
        ui.chart(charts.period_returns_chart(periods))

    period_df = pd.DataFrame([
        {
            "Period": period.get("name"),
            "Start": str(period.get("start") or "")[:10],
            "End": str(period.get("end") or "")[:10],
            "Trading days": period.get("trading_days"),
            "Start NAV": period.get("start_nav"),
            "End NAV": period.get("end_nav"),
            "Return (%)": (period.get("return") or 0.0) * 100.0,
        }
        for period in periods
    ])

    st.subheader("Period summary")
    st.dataframe(
        period_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Start NAV": st.column_config.NumberColumn(format="%.4f"),
            "End NAV": st.column_config.NumberColumn(format="%.4f"),
            "Return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Trading days": st.column_config.NumberColumn(format="%d"),
        },
    )
    ui.download_df(period_df, filename="period_summary.csv", key="overview_periods")

    daily_df = _daily_frame(portfolio)
    with st.expander("Daily NAV table view"):
        st.dataframe(
            daily_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Portfolio NAV": st.column_config.NumberColumn(format="%.4f"),
                "Benchmark NAV": st.column_config.NumberColumn(format="%.4f"),
                "Drawdown (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        ui.download_df(daily_df, filename="daily_nav.csv", key="overview_daily")

    if meta.get("data_as_of"):
        st.caption(
            f"Data as of {meta['data_as_of']} · dashboard export "
            f"{str(meta.get('generated_at', ''))[:19].replace('T', ' ')}"
        )
