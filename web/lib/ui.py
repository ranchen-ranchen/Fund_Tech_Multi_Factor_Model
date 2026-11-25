"""Shared Streamlit UI helpers."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from lib import charts


def chart(fig: Any) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=charts.CHART_CONFIG,
    )


def missing_data(what: str, command: str) -> None:
    st.warning(f"No **{what}** data found.")
    st.markdown("Generate it in a terminal, then click **Reload data** in the sidebar:")
    st.code(command, language="bash")
    st.caption("This dashboard is read-only and never runs the research pipeline itself.")


def page_header(title: str, subtitle: Optional[str] = None) -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def table_view(
    df: pd.DataFrame,
    *,
    label: str = "Table view",
    key: Optional[str] = None,
) -> None:
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True, key=key)


def download_df(
    df: pd.DataFrame,
    *,
    filename: str,
    label: str = "Download CSV",
    key: Optional[str] = None,
) -> None:
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def metric_row(items: list[tuple[str, str, Optional[str]]]) -> None:
    """Render a row of bordered metric tiles.

    Each item is (label, value, help). Help text is optional.
    """
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        with column:
            st.metric(label, value, help=help_text, border=True)
