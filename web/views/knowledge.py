"""LLM verdicts, selections and knowledge-base pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lib import charts, data, format, ui


def _verdicts() -> None:
    payload = data.load_verdicts()
    rows = (payload or {}).get("rows", [])
    if not rows:
        ui.missing_data(
            "LLM verdict",
            "python target_code_pipeline.py\n"
            "python web/export_data.py",
        )
        return

    counts = (payload or {}).get("counts", {})
    total = counts.get("total", len(rows))
    in_chain = counts.get("in_chain", 0)
    share = in_chain / total if total else None

    ui.metric_row([
        ("Cached verdicts", format.integer(total), "One latest verdict per stock code."),
        ("In AI compute chain", format.integer(in_chain), "Model verdict is true."),
        ("Not in chain", format.integer(total - in_chain), "Model verdict is false."),
        ("In-chain share", format.pct(share), "Share of cached verdicts that are true."),
    ])

    frame = pd.DataFrame([
        {
            "Code": row.get("code"),
            "Name": row.get("name"),
            "In chain": bool(row.get("is_in_chain")),
            "Reason": row.get("reason"),
            "Model": row.get("model"),
            "Cached at": row.get("cached_at"),
        }
        for row in rows
    ])

    left, right = st.columns([2, 1])
    with left:
        search = st.text_input(
            "Search code, name, or reason",
            key="verdict_search",
            placeholder="e.g. 000063 or a Chinese company name",
        )
    with right:
        verdict_filter = st.segmented_control(
            "Verdict",
            options=["All", "In chain", "Not in chain"],
            default="All",
            key="verdict_filter",
        ) or "All"

    filtered = frame.copy()
    if search:
        needle = search.strip()
        mask = (
            filtered["Code"].astype(str).str.contains(needle, case=False, na=False)
            | filtered["Name"].astype(str).str.contains(needle, case=False, na=False)
            | filtered["Reason"].astype(str).str.contains(needle, case=False, na=False)
        )
        filtered = filtered[mask]
    if verdict_filter == "In chain":
        filtered = filtered[filtered["In chain"]]
    elif verdict_filter == "Not in chain":
        filtered = filtered[~filtered["In chain"]]

    st.caption(f"{len(filtered)} of {len(frame)} verdicts shown")
    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        column_config={
            "In chain": st.column_config.CheckboxColumn("In chain"),
            "Reason": st.column_config.TextColumn("Reason", width="large"),
        },
    )
    ui.download_df(
        filtered,
        filename="llm_verdicts.csv",
        key="verdict_download",
    )
    st.caption(
        "Reasons are model-generated and shown verbatim in Chinese. Verdicts are "
        "cached by description hash, prompt version and model."
    )


def _selections() -> None:
    payload = data.load_selections()
    rows = (payload or {}).get("rows", [])
    if not rows:
        ui.missing_data(
            "target selection",
            "python target_code_pipeline.py\n"
            "python web/export_data.py",
        )
        return

    unique_codes = {
        item["code"] for row in rows for item in row.get("codes", [])
    }
    latest = rows[-1]

    ui.metric_row([
        ("Rebalance dates", format.integer(len(rows)), "Semi-annual selection dates."),
        ("Latest selection", format.integer(latest.get("count")), latest.get("query_date", "")),
        ("Unique selected codes", format.integer(len(unique_codes)), "Across all dates."),
        (
            "Latest added / removed",
            f"{len(latest.get('added', []))} / {len(latest.get('removed', []))}",
            "Versus the previous rebalance date.",
        ),
    ])

    ui.chart(charts.selection_count_chart(rows))

    dates = [row.get("query_date", "") for row in rows]
    selected_date = st.selectbox(
        "Inspect a rebalance date",
        options=dates,
        index=len(dates) - 1,
        key="selection_date",
    )
    selected = next(
        (row for row in rows if row.get("query_date") == selected_date),
        latest,
    )

    code_frame = pd.DataFrame([
        {
            "Code": item.get("code"),
            "Name": item.get("name"),
            "bs_code": item.get("bs_code"),
            "file_stem": item.get("file_stem"),
        }
        for item in selected.get("codes", [])
    ])
    st.caption(
        f"{selected_date} · {len(code_frame)} names · "
        f"{len(selected.get('added', []))} added · "
        f"{len(selected.get('removed', []))} removed"
    )
    st.dataframe(code_frame, width="stretch", hide_index=True)
    ui.download_df(
        code_frame,
        filename=f"selection_{selected_date}.csv",
        key="selection_download",
    )

    summary = pd.DataFrame([
        {
            "Date": row.get("query_date"),
            "Count": row.get("count"),
            "Added": len(row.get("added", [])),
            "Removed": len(row.get("removed", [])),
            "Codes": ", ".join(item["code"] for item in row.get("codes", [])),
        }
        for row in rows
    ])
    with st.expander("All rebalance dates"):
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Codes": st.column_config.TextColumn("Codes", width="large"),
            },
        )
        ui.download_df(
            summary,
            filename="selections.csv",
            key="selections_all_download",
        )


def _report() -> None:
    knowledge = data.load_knowledge()
    report = (knowledge or {}).get("report")
    if not report:
        ui.missing_data(
            "knowledge-base report",
            "python knowledge_base/industry_eval.py\n"
            "python web/export_data.py",
        )
        return

    markdown = data.load_markdown(report.get("path", ""))
    if not markdown:
        st.info("Report file is missing from disk.")
        return

    as_of = report.get("as_of")
    st.caption(
        f"Evidence cutoff: {as_of or 'unknown'} · generated "
        f"{format.date_only(report.get('modified'))} · {format.human_size(report.get('size_bytes'))}"
    )
    st.caption("Source report is in Chinese.")
    st.download_button(
        "Download Markdown",
        data=markdown.encode("utf-8"),
        file_name="industry_eval_report.md",
        mime="text/markdown",
        key="report_download",
    )
    st.markdown(markdown)


def _dossiers() -> None:
    knowledge = data.load_knowledge()
    items = [
        item for item in (knowledge or {}).get("all", [])
        if item.get("kind") != "report"
    ]
    if not items:
        ui.missing_data(
            "event dossier",
            "python knowledge_base/news_collect.py --year 2021\n"
            "python web/export_data.py",
        )
        return

    labels = {
        item["path"]: (
            f"{item.get('title', item.get('path'))} · "
            f"{item.get('kind')}"
        )
        for item in items
    }
    selected_path = st.selectbox(
        "Dossier",
        options=list(labels),
        format_func=lambda path: labels[path],
        key="dossier_select",
    )
    selected = next(
        (item for item in items if item.get("path") == selected_path),
        items[0],
    )
    markdown = data.load_markdown(selected_path)
    if not markdown:
        st.info("Dossier file is missing from disk.")
        return

    st.caption(
        f"{selected.get('title', '')} · {format.human_size(selected.get('size_bytes'))} · "
        f"modified {format.date_only(selected.get('modified'))}"
    )
    st.caption("Source dossier is in Chinese.")
    st.download_button(
        "Download Markdown",
        data=markdown.encode("utf-8"),
        file_name=selected_path.split("/")[-1],
        mime="text/markdown",
        key="dossier_download",
    )
    st.markdown(markdown)


def render() -> None:
    ui.page_header(
        "Knowledge & selection",
        "Cached LLM verdicts, semi-annual selections and the thematic research library.",
    )
    tabs = st.tabs([
        "LLM verdicts",
        "Selections",
        "Industry report",
        "Event dossiers",
    ])
    with tabs[0]:
        _verdicts()
    with tabs[1]:
        _selections()
    with tabs[2]:
        _report()
    with tabs[3]:
        _dossiers()
