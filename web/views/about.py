"""Data provenance and methodology page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import data, format, ui


BENCHMARK_PCT_KEYS = {
    "Total Return",
    "Annualized Return",
    "Annualized Return (Arithmetic)",
    "Annualized Volatility",
    "Max Drawdown",
    "Daily Win Rate",
}
BENCHMARK_RATIO_KEYS = {
    "Sharpe Ratio",
    "Sharpe Ratio (CAGR/vol)",
    "Calmar Ratio",
}
BENCHMARK_PRICE_KEYS = {"Initial Price", "Final Price"}


def _benchmark_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for key, value in summary.items():
        if key in BENCHMARK_PCT_KEYS and isinstance(value, (int, float)):
            display = format.pct(value)
        elif key in BENCHMARK_RATIO_KEYS and isinstance(value, (int, float)):
            display = format.num(value, 2)
        elif key in BENCHMARK_PRICE_KEYS and isinstance(value, (int, float)):
            display = format.num(value, 4)
        elif key == "Trading Days":
            display = format.integer(value)
        else:
            display = str(value)
        rows.append({"Metric": key, "Value": display})
    return pd.DataFrame(rows)


def _health_frame(health: dict) -> pd.DataFrame:
    labels = [
        ("Required stock codes", "required_code_count"),
        ("HFQ K-line files", "hfq_file_count"),
        ("BFQ K-line files", "bfq_file_count"),
        ("Missing HFQ files", None),
        ("NAV last date", "nav_last_date"),
        ("Benchmark last date", "benchmark_last_date"),
        ("K-line first date (min)", "kline_first_date_min"),
        ("K-line last date (min)", "kline_last_date_min"),
        ("K-line last date (max)", "kline_last_date_max"),
        ("CSI 300 weight files", "weight_file_count"),
        ("Valuation rows", "valuation_rows"),
        ("Business-description rows", "main_business_rows"),
        ("Factor attribution periods", "factor_periods"),
        ("Group analysis periods", "group_periods"),
        ("Stock index rows", "stock_count"),
    ]
    rows = []
    for label, key in labels:
        if key is None:
            value = len(health.get("missing_hfq_codes", []))
        else:
            value = health.get(key)
        if isinstance(value, int):
            display = format.integer(value)
        elif value is None:
            display = "—"
        else:
            display = str(value)
        rows.append({"Check": label, "Value": display})
    return pd.DataFrame(rows)


def render() -> None:
    meta = data.load_meta()
    health = data.load_health()
    benchmark = data.load_benchmark() or {}
    portfolio = data.load_portfolio() or {}
    summary = (portfolio.get("summary") or [{}])[0]

    ui.page_header(
        "Data & methodology",
        "Provenance, coverage and the conventions behind every number in this dashboard.",
    )

    warnings = meta.get("warnings") or []
    ui.metric_row([
        (
            "Data as of",
            meta.get("data_as_of") or "—",
            "Latest date across NAV, benchmark and K-line caches.",
        ),
        (
            "Dashboard export",
            str(meta.get("generated_at") or "—")[:19].replace("T", " "),
            "Time this JSON dataset was generated.",
        ),
        (
            "Backtest generated",
            str(meta.get("backtest_generated_at") or "—")[:19].replace("T", " "),
            "Time the structured backtest details were written.",
        ),
        (
            "Warnings",
            format.integer(len(warnings)),
            "Missing or stale artifacts detected by the exporter.",
        ),
    ])

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No data warnings reported by the exporter.")

    st.subheader("Source files")
    sources = meta.get("sources") or []
    if sources:
        source_frame = pd.DataFrame([
            {
                "Artifact": item.get("name"),
                "Path": item.get("path"),
                "Modified": format.date_only(item.get("modified")),
                "Size": format.human_size(item.get("size_bytes")),
            }
            for item in sources
        ])
        st.dataframe(source_frame, width="stretch", hide_index=True)
    else:
        st.info("No source files recorded; run web/export_data.py.")

    st.subheader("Data coverage")
    if health:
        st.dataframe(
            _health_frame(health),
            width="stretch",
            hide_index=True,
        )
        missing = health.get("missing_hfq_codes", [])
        if missing:
            with st.expander(f"Missing HFQ codes ({len(missing)})"):
                st.write(", ".join(missing))
    else:
        st.info("Coverage report not found; run web/export_data.py.")

    st.subheader("Benchmark summary")
    if benchmark.get("summary_parsed"):
        st.dataframe(
            _benchmark_frame(benchmark["summary_parsed"]),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Source: {benchmark.get('source', '—')}")
    else:
        st.info("Benchmark summary not available.")

    st.subheader("Metric conventions")
    metric_rows = [
        ("Final NAV", "Ending net asset value; initial capital = 1.0."),
        ("Total return", "Final NAV − 1."),
        ("CAGR", "Geometric annualized return using 252 trading days."),
        ("Annualized volatility", "Daily return standard deviation × √252."),
        (
            "Sharpe (standard)",
            "(Arithmetic annualized return − risk-free rate) / annualized volatility.",
        ),
        (
            "Sharpe (legacy)",
            "CAGR / annualized volatility; retained for comparison only.",
        ),
        ("Max drawdown", "Minimum of NAV / running maximum − 1."),
        (
            "Active return",
            "Mean daily portfolio-minus-benchmark return × 252 (arithmetic).",
        ),
        (
            "Tracking error",
            "Standard deviation of daily active return × √252.",
        ),
        ("Information ratio", "Active return / tracking error."),
        (
            "Geometric excess return",
            "((P_end/P_start) / (B_end/B_start))^(1/years) − 1; display only.",
        ),
        (
            "Cost model",
            f"{(summary.get('cost_rate') or 0.005):.2%} one-way; "
            f"{2 * (summary.get('cost_rate') or 0.005):.2%} round-trip per period.",
        ),
        (
            "Risk-free rate",
            f"{(summary.get('risk_free_rate') or 0.02):.2%} annualized.",
        ),
        (
            "Attribution",
            "WLS cross-sectional regression on size, value, momentum and "
            "volatility; contributions use active weights.",
        ),
    ]
    st.dataframe(
        pd.DataFrame(metric_rows, columns=["Metric", "Definition"]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Point-in-time conventions")
    st.markdown(
        """
- The benchmark series is forward-filled only; backward fill is deliberately removed to avoid look-ahead bias.
- Valuation lookups use the latest snapshot on or before the rebalance date, then an exact match on that snapshot date.
- Each period's beginning universe comes from the most recent half-year weight file with period end on or before the rebalance date.
- LLM verdicts are cached by description hash, prompt version and model; changing any of them invalidates the cache.
- Index membership is modeled with half-year snapshots, not a daily point-in-time membership history.
"""
    )

    st.subheader("Known limitations")
    st.markdown(
        """
- The cost model is a flat 50 bp per side; there is no slippage, stamp duty, market impact or borrow cost.
- Portfolio weights are equal, not optimized.
- The risk model is compact: no market or industry factors, and size is approximated by amount / turnover.
- Vendor data (baostock / akshare) can lag, rate-limit, or lack delisted and suspended coverage.
- The AI-compute-chain labels use current business descriptions and were chosen with hindsight.
- This is research and backtesting only. It is not investment advice and does not place orders.
"""
    )

    st.subheader("Refresh workflow")
    st.markdown(
        "The dashboard is read-only. To update it, run the research scripts and "
        "the exporter in a terminal, then click **Reload data** in the sidebar."
    )
    st.code(
        """# 1) Rebuild structured backtest details (existing selection, no price update)
python target_code_pipeline.py --skip-prep --no-update-k \\
    --export-details web/data/backtest_details.json

# 2) Optional: refresh attribution and group analysis
python barra_model_analysis.py
python group_return_analysis.py

# 3) Rebuild dashboard JSON
python web/export_data.py

# 4) Start the dashboard (local only)
web/run.sh""",
        language="bash",
    )
    st.caption(
        "Remove --skip-prep to reselect via the LLM, or remove --no-update-k to "
        "refresh prices. Both steps can take minutes and may call external services."
    )

    st.subheader("Security posture")
    st.markdown(
        """
- The Streamlit server binds to `127.0.0.1` only; it is not reachable from other machines.
- The app never reads the DeepSeek API key or calls the LLM.
- The app never writes to research artifacts; all writes happen in the pipeline and exporter.
- Only allowlisted repository files are read and rendered.
"""
    )
