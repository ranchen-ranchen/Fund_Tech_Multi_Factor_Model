"""Holdings and period drill-down page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, format, ui


def _holdings_frame(period: dict, stocks: dict) -> pd.DataFrame:
    rows = []
    for holding in period.get("holdings", []):
        code = str(holding.get("code6") or "")
        stock = stocks.get(code, {})
        rows.append({
            "Code": code,
            "Name": stock.get("name", ""),
            "Weight (%)": (holding.get("weight") or 0.0) * 100.0,
            "Entry price": holding.get("entry_price"),
            "Shares": holding.get("shares"),
            "End price": holding.get("end_price"),
            "Return (%)": (holding.get("total_ret") or 0.0) * 100.0,
            "Contribution (pp)": (holding.get("contrib") or 0.0) * 100.0,
            "Max DD (%)": (holding.get("max_dd") or 0.0) * 100.0,
            "Vol (%)": (holding.get("ann_vol") or 0.0) * 100.0,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("Contribution (pp)", ascending=False)
    return frame


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

    periods = portfolio.get("periods", [])
    if not periods:
        st.warning("No completed backtest periods are available.")
        return

    stocks = data.load_stocks()
    verdicts = data.load_verdicts() or {}
    verdict_map = {row["code"]: row for row in verdicts.get("rows", [])}

    ui.page_header(
        "Holdings & periods",
        "Inspect each semi-annual holding period and the stocks inside it.",
    )

    period_names = [period.get("name", "") for period in periods]
    selected_name = st.selectbox(
        "Period",
        options=period_names,
        index=len(period_names) - 1,
        key="holdings_period",
    )
    period = next(
        (item for item in periods if item.get("name") == selected_name),
        periods[-1],
    )

    start = str(period.get("start") or "")[:10]
    end = str(period.get("end") or "")[:10]
    st.caption(f"{start} to {end} · {period.get('trading_days', 0)} trading days")

    ui.metric_row([
        (
            "Period return",
            format.pct(period.get("return"), signed=True),
            "Net of round-trip costs.",
        ),
        (
            "Holdings",
            str(len(period.get("holdings", []))),
            "Equal-weighted at the period start.",
        ),
        (
            "Start NAV",
            format.num(period.get("start_nav"), 4),
            "Cumulative NAV at the previous period end.",
        ),
        (
            "End NAV",
            format.num(period.get("end_nav"), 4),
            "Cumulative NAV after the period-end sale cost.",
        ),
    ])

    frame = _holdings_frame(period, stocks)
    st.subheader("Holdings")
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Weight (%)": st.column_config.NumberColumn(format="%.2f"),
            "Entry price": st.column_config.NumberColumn(format="%.4f"),
            "Shares": st.column_config.NumberColumn(format="%.8f"),
            "End price": st.column_config.NumberColumn(format="%.4f"),
            "Return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Contribution (pp)": st.column_config.NumberColumn(format="%.2f"),
            "Max DD (%)": st.column_config.NumberColumn(format="%.2f"),
            "Vol (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    ui.download_df(
        frame,
        filename=f"holdings_{selected_name}.csv",
        key="holdings_download",
    )

    if not frame.empty:
        enriched = []
        for holding in period.get("holdings", []):
            code = str(holding.get("code6") or "")
            enriched.append({
                **holding,
                "display_name": stocks.get(code, {}).get("name", ""),
            })
        ui.chart(charts.holdings_contribution_chart(enriched))
        if len(enriched) > 25:
            st.caption(
                "The chart shows the 25 largest absolute contributors; "
                "the table above lists every holding."
            )

    st.subheader("Stock inspector")
    if frame.empty:
        st.info("This period has no holdings.")
        return

    code_options = frame["Code"].tolist()
    name_by_code = dict(zip(frame["Code"], frame["Name"]))
    selected_code = st.selectbox(
        "Stock",
        options=code_options,
        format_func=lambda code: f"{code} · {name_by_code.get(code, '')}",
        key="holdings_stock",
    )
    holding = next(
        (
            item for item in period.get("holdings", [])
            if str(item.get("code6")) == str(selected_code)
        ),
        None,
    )
    stock = stocks.get(str(selected_code), {})
    verdict = verdict_map.get(str(selected_code))

    name = stock.get("name") or name_by_code.get(selected_code, "")
    st.markdown(f"**{selected_code} · {name or 'Unknown name'}**")

    if verdict:
        verdict_text = "In AI compute chain" if verdict.get("is_in_chain") else "Not in chain"
        if verdict.get("is_in_chain"):
            st.success(f"LLM verdict: {verdict_text}")
        else:
            st.info(f"LLM verdict: {verdict_text}")
        st.caption(verdict.get("reason", ""))
    else:
        st.caption("No cached LLM verdict for this stock.")

    held_periods = stock.get("held_periods") or []
    if held_periods:
        st.caption("Held in: " + ", ".join(held_periods))

    window = st.segmented_control(
        "Chart window",
        options=["1Y", "3Y", "All"],
        default="3Y",
        key="holdings_window",
    ) or "3Y"

    file_stem = stock.get("file_stem") or (holding or {}).get("name")
    kline = data.load_kline(file_stem) if file_stem else None
    if kline is None:
        st.warning("No back-adjusted K-line CSV found for this stock.")
        return

    last_date = kline["date"].max()
    if window == "1Y":
        first_date = last_date - pd.Timedelta(days=365)
    elif window == "3Y":
        first_date = last_date - pd.Timedelta(days=1095)
    else:
        first_date = kline["date"].min()
    view = kline[kline["date"] >= first_date].copy()

    entry_date = (holding or {}).get("entry_date")
    end_date = (holding or {}).get("end_date")
    ui.chart(charts.candlestick_chart(view, entry_date, end_date))
    st.caption(
        "Back-adjusted prices. A-share convention: red = up, green = down. "
        "The shaded band marks the selected holding period."
    )

    table = view[["date", "open", "high", "low", "close", "volume"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table = table.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    with st.expander("Daily price table view"):
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Open": st.column_config.NumberColumn(format="%.4f"),
                "High": st.column_config.NumberColumn(format="%.4f"),
                "Low": st.column_config.NumberColumn(format="%.4f"),
                "Close": st.column_config.NumberColumn(format="%.4f"),
                "Volume": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        ui.download_df(
            table,
            filename=f"kline_{file_stem}.csv",
            key="holdings_kline_download",
        )
