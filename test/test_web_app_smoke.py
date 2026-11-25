"""Smoke test: every dashboard page renders without an exception.

Skipped until the exporter has produced web/data/portfolio.json.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_APP = PROJECT_ROOT / "web" / "app.py"
DATA_FILE = PROJECT_ROOT / "web" / "data" / "portfolio.json"

pytestmark = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="dashboard data has not been exported",
)


def test_all_pages_render():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(WEB_APP))
    app.run(timeout=60)
    assert not app.exception

    pages = [
        "Overview",
        "Holdings & periods",
        "Factor attribution",
        "AI vs non-AI",
        "Knowledge & selection",
        "Data & methodology",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, f"{page}: {app.exception}"
