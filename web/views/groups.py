"""AI versus non-AI group-return diagnostic page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, format, ui


PERCENT_COLUMNS = {
    "AI mean (%)": "mean_ai",
    "Non-AI mean (%)": "mean_non",
    "Spread (pp)": "spread",
    "Bootstrap low (pp)": "boot_lo",
    "Bootstrap high (pp)": "boot_hi",
    "Cap-weighted spread (pp)": "cap_spread",
    "Equal-weight universe (%)": "full_ew",
    "ETF (%)": "etf_ret",
}


def _display_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {
            "Period": row.get("period"),
            "Start": str(row.get("start") or "")[:10],
            "End": str(row.get("end") or "")[:10],
            "AI names": row.get("n_ai"),
            "Non-AI names": row.get("n_non"),
        }
        for label, key in PERCENT_COLUMNS.items():
            value = row.get(key)
            record[label] = value * 100.0 if value is not None else None
        record["Welch t"] = row.get("welch_t")
        record["Welch p"] = row.get("welch_p_normal")
        record["Permutation p"] = row.get("perm_p")
        records.append(record)
    return pd.DataFrame(records)


def render() -> None:
    payload = data.load_group()
    if not payload or not payload.get("rows"):
        ui.missing_data(
            "AI vs non-AI group analysis",
            "python group_return_analysis.py\n"
            "python web/export_data.py",
        )
        return

    rows = payload["rows"]
    spreads = [row.get("spread") for row in rows if row.get("spread") is not None]
    ai_means = [row.get("mean_ai") for row in rows if row.get("mean_ai") is not None]
    hit_rate = (
        sum(1 for value in spreads if value > 0) / len(spreads)
        if spreads
        else None
    )

    ui.page_header(
        "AI vs non-AI",
        "Descriptive two-group diagnostic: LLM-selected names versus the rest of the CSI 300 universe.",
    )

    ui.metric_row([
        (
            "Periods",
            str(len(rows)),
            "Semi-annual intervals matching the backtest boundaries.",
        ),
        (
            "Mean spread",
            format.pp(format.mean(spreads)),
            "Equal-weight AI mean minus non-AI mean, averaged across periods.",
        ),
        (
            "Hit rate",
            format.pct(hit_rate),
            "Share of periods with a positive AI-minus-non-AI spread.",
        ),
        (
            "Mean AI return",
            format.pct(format.mean(ai_means), signed=True),
            "Equal-weight AI-group period return, averaged across periods.",
        ),
    ])

    tabs = st.tabs(["Spread & CI", "Group returns", "Generated report", "Table"])
    with tabs[0]:
        ui.chart(charts.group_spread_chart(rows))
        st.caption(
            "Error bars are bootstrap confidence intervals from the source "
            "analysis. The thin horizontal line is the mean period spread."
        )
    with tabs[1]:
        ui.chart(charts.group_returns_chart(rows))
        st.caption("Gross equal-weight period returns; costs are not applied here.")
    with tabs[2]:
        report_path = payload.get("report_path") or "group_return_analysis.md"
        report = data.load_markdown(report_path)
        if report:
            st.caption("Source report is in Chinese.")
            st.markdown(report)
        else:
            st.info("Generated group-return report not found.")
    with tabs[3]:
        frame = _display_frame(rows)
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.2f")
                for column in PERCENT_COLUMNS
            },
        )
        ui.download_df(
            frame,
            filename="group_analysis.csv",
            key="group_download",
        )

    st.caption(
        "The labels use current business descriptions rather than a point-in-time "
        "membership history, and the theme was chosen with hindsight. The long-short "
        "spread is a factor return, not an investable A-share strategy."
    )
