"""Read-only research dashboard.

Run from the web/ directory:  streamlit run app.py
The server is configured to bind to 127.0.0.1 by .streamlit/config.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from lib import data  # noqa: E402
from views import about, attribution, groups, holdings, knowledge, overview  # noqa: E402


st.set_page_config(
    page_title="Quant Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

meta = data.load_meta()
counts = meta.get("counts", {})

with st.sidebar:
    st.title("Quant Research")
    st.caption("Read-only · Local only")

    if meta.get("data_as_of"):
        st.caption(f"Data as of **{meta['data_as_of']}**")
        st.caption(
            f"{counts.get('periods', '—')} periods · "
            f"{counts.get('stocks', '—')} stocks · "
            f"{counts.get('verdicts', '—')} verdicts"
        )
    else:
        st.caption("No exported dashboard data yet.")

    page = st.radio(
        "Navigate",
        options=[
            "Overview",
            "Holdings & periods",
            "Factor attribution",
            "AI vs non-AI",
            "Knowledge & selection",
            "Data & methodology",
        ],
        key="nav_page",
    )

    st.divider()

    if st.button("Reload data", width="stretch"):
        data.clear_cache()
        st.rerun()
    st.caption("Re-reads `web/data`; it does not run the research pipeline.")

    with st.expander("How to refresh"):
        st.code(
            """python target_code_pipeline.py --skip-prep --no-update-k \\
    --export-details web/data/backtest_details.json
python web/export_data.py""",
            language="bash",
        )
        st.caption("Run from the project root, then reload this page.")

    st.divider()
    st.caption("Research only. Not investment advice.")

if page == "Overview":
    overview.render()
elif page == "Holdings & periods":
    holdings.render()
elif page == "Factor attribution":
    attribution.render()
elif page == "AI vs non-AI":
    groups.render()
elif page == "Knowledge & selection":
    knowledge.render()
else:
    about.render()
