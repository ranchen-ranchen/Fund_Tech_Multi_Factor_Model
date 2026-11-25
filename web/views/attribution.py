"""Barra-style factor attribution page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, format, ui


PP_COLUMNS = {
    "Size (pp)": "size",
    "Value (pp)": "value",
    "Momentum (pp)": "momentum",
    "Volatility (pp)": "volatility",
    "Factor total (pp)": "factor_total",
    "Specific (pp)": "specific",
    "Benchmark residual (pp)": "bench_resid",
    "Unattributed (pp)": "unattributed",
    "Total (pp)": "total",
}


def _display_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {
            "Period": row.get("period"),
            "Start": str(row.get("start") or "")[:10],
            "End": str(row.get("end_effective") or "")[:10],
            "Stocks": row.get("n_stocks"),
            "Attributed": row.get("n_attr"),
        }
        for label, key in PP_COLUMNS.items():
            value = row.get(key)
            record[label] = value * 100.0 if value is not None else None
        record["Reconcile"] = row.get("reconcile_check")
        records.append(record)
    return pd.DataFrame(records)


def _coverage_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append({
            "Period": row.get("period"),
            "Stocks": row.get("n_stocks"),
            "Attributed": row.get("n_attr"),
            "Missing target weight (%)": (row.get("missing_target_weight") or 0.0) * 100.0,
            "Missing benchmark weight (%)": (row.get("missing_bench_weight") or 0.0) * 100.0,
            "Unmodeled target weight (%)": (row.get("unmodeled_target_weight") or 0.0) * 100.0,
            "Unmodeled benchmark weight (%)": (row.get("unmodeled_bench_weight") or 0.0) * 100.0,
            "Benchmark weight sum (%)": (row.get("bench_weight_sum") or 0.0) * 100.0,
            "Cash weight (%)": (row.get("cash_weight") or 0.0) * 100.0,
            "Target-only names": row.get("n_target_only"),
            "Target-only weight (%)": (row.get("target_only_weight") or 0.0) * 100.0,
            "Target-only codes": row.get("target_only_codes") or "",
        })
    return pd.DataFrame(records)


def render() -> None:
    payload = data.load_factor()
    if not payload or not payload.get("rows"):
        ui.missing_data(
            "factor attribution",
            "python barra_model_analysis.py\n"
            "python web/export_data.py",
        )
        return

    rows = payload["rows"]
    ui.page_header(
        "Factor attribution",
        "Arithmetic active-return decomposition versus the holdings-weighted CSI 300 benchmark.",
    )

    totals = [row.get("total") for row in rows if row.get("total") is not None]
    specific = [row.get("specific") for row in rows if row.get("specific") is not None]
    latest_total = rows[-1].get("total") if rows else None

    ui.metric_row([
        (
            "Periods",
            str(len(rows)),
            "Semi-annual attribution intervals.",
        ),
        (
            "Mean total active return",
            format.pp(format.mean(totals)),
            "Average arithmetic active return across periods.",
        ),
        (
            "Mean specific contribution",
            format.pp(format.mean(specific)),
            "Stock-specific component, averaged across periods.",
        ),
        (
            "Latest period total",
            format.pp(latest_total),
            f"{rows[-1].get('period', '')} active return.",
        ),
    ])

    ui.chart(charts.factor_contribution_chart(rows))
    st.caption(
        "Other = benchmark residual + unattributed. Contributions are shown in "
        "percentage points of active return and reconcile to the total marker."
    )

    frame = _display_frame(rows)
    pp_labels = list(PP_COLUMNS)
    st.subheader("Attribution detail")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            label: st.column_config.NumberColumn(format="%.2f")
            for label in pp_labels
        },
    )
    ui.download_df(
        frame,
        filename="factor_attribution.csv",
        key="attribution_download",
    )

    with st.expander("Coverage diagnostics"):
        coverage = _coverage_frame(rows)
        st.dataframe(
            coverage,
            width="stretch",
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in coverage.columns
                if "(%)" in column
            },
        )
        st.caption(
            "Target-only names are selected stocks absent from the benchmark "
            "weight file; cash weight is the benchmark shortfall."
        )

    st.caption(
        "Point-in-time exposures use only information available at each rebalance "
        "date. The model has size, value, momentum and volatility factors; there "
        "are no market or industry factors."
    )
