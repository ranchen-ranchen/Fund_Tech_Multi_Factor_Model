"""Tests for the read-only dashboard exporter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web import export_data as export  # noqa: E402


def test_parse_benchmark_csv_handles_two_sections_and_bom(tmp_path):
    path = tmp_path / "benchmark.csv"
    path.write_text(
        "﻿# Performance Summary\n"
        "Metric,Value\n"
        "Symbol,510300\n"
        "Total Return,-0.100000\n"
        "Trading Days,2\n"
        "\n"
        "# Daily Series\n"
        ",Close,Cumulative NAV,Drawdown\n"
        "2021-01-04,5.0000,1.0,0.0\n"
        "2021-01-05,5.5000,1.1,-0.02\n",
        encoding="utf-8",
    )

    parsed = export.parse_benchmark_csv(path)

    assert parsed["summary"]["Symbol"] == "510300"
    assert "Metric" not in parsed["summary"]
    assert "Metric" not in parsed["summary_parsed"]
    assert parsed["summary_parsed"]["Total Return"] == pytest.approx(-0.1)
    assert parsed["summary_parsed"]["Trading Days"] == 2
    assert len(parsed["daily"]) == 2
    assert parsed["daily"][1]["nav"] == pytest.approx(1.1)
    assert parsed["daily"][1]["drawdown"] == pytest.approx(-0.02)


def test_json_safe_converts_nan_and_timestamps():
    payload = {
        "nan": float("nan"),
        "inf": np.inf,
        "timestamp": pd.Timestamp("2021-01-01"),
        "integer": np.int64(7),
        "array": np.array([1.0, np.nan]),
    }

    safe = export.json_safe(payload)

    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["timestamp"].startswith("2021-01-01")
    assert safe["integer"] == 7
    assert safe["array"] == [1.0, None]
    json.dumps(safe, allow_nan=False)


def test_code_mapping():
    assert export.to_bs_code("600000") == "sh.600000"
    assert export.to_bs_code("688981") == "sh.688981"
    assert export.to_bs_code("000063") == "sz.000063"
    assert export.to_bs_code("300408") == "sz.300408"
    assert export.to_bs_code("430047") == "bj.430047"
    assert export.file_stem_from_code6("600000") == "sh_600000"


def test_read_target_selections_computes_added_and_removed(tmp_path):
    path = tmp_path / "target.csv"
    path.write_text(
        "query_date,target_code\n"
        "2021-01-01,000063 000066\n"
        "2021-06-30,000063 000938\n",
        encoding="utf-8",
    )
    names = {"000063": "A", "000066": "B", "000938": "C"}

    rows = export.read_target_selections(path, names)

    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["added"] == ["000063", "000066"]
    assert rows[1]["added"] == ["000938"]
    assert rows[1]["removed"] == ["000066"]
    assert rows[1]["codes"][0]["name"] == "A"


def test_read_llm_verdicts_keeps_latest_record(tmp_path):
    path = tmp_path / "cache.jsonl"
    records = [
        {
            "code": "000063",
            "is_in_chain": False,
            "reason": "old",
            "cached_at": "2026-01-01T00:00:00",
            "model": "deepseek-chat",
            "prompt_version": "v1",
        },
        {
            "code": "000063",
            "is_in_chain": True,
            "reason": "new",
            "cached_at": "2026-02-01T00:00:00",
            "model": "deepseek-chat",
            "prompt_version": "v1",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    result = export.read_llm_verdicts(path, {"000063": "A"})

    assert result["counts"] == {"total": 1, "in_chain": 1, "not_in_chain": 0}
    assert result["rows"][0]["reason"] == "new"
    assert result["rows"][0]["is_in_chain"] is True


def test_first_last_dates(tmp_path):
    path = tmp_path / "sh_600000.csv"
    path.write_text(
        "﻿date,code,open,high,low,close\n"
        "2021-01-04,sh.600000,1,2,1,2\n"
        "2021-01-05,sh.600000,2,3,2,3\n"
        "2021-01-06,sh.600000,3,4,3,4\n",
        encoding="utf-8",
    )

    first, last = export.first_last_dates(path)

    assert first == "2021-01-04"
    assert last == "2021-01-06"
