"""Read-only data access for the dashboard.

All functions are cached by file path + modification time, so a local data
refresh is picked up by clearing the Streamlit cache. Nothing here writes to
the research repository.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


WEB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_DIR.parent
DATA_DIR = WEB_DIR / "data"
HFQ_DIR = PROJECT_ROOT / "stock_daily_k_hfq" / "individual"

_KLINE_STEM = re.compile(r"^(sh|sz|bj)_\d{6}$")


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _load_json_cached(path_str: str, mtime: float) -> Any:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(name: str) -> Optional[Any]:
    path = DATA_DIR / name
    if not path.exists():
        return None
    return _load_json_cached(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _load_text_cached(path_str: str, mtime: float) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="replace")


def load_markdown(rel_path: str) -> Optional[str]:
    if not rel_path:
        return None
    path = (PROJECT_ROOT / rel_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    if not path.exists() or not path.is_file():
        return None
    return _load_text_cached(str(path), _mtime(path))


@st.cache_data(show_spinner=False)
def _load_kline_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path_str, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"date", "open", "high", "low", "close"}.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "preclose",
                   "volume", "amount", "turn", "pctchg"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df.reset_index(drop=True)


def load_kline(file_stem: str) -> Optional[pd.DataFrame]:
    if not file_stem or not _KLINE_STEM.match(file_stem):
        return None
    path = HFQ_DIR / f"{file_stem}.csv"
    if not path.exists():
        return None
    df = _load_kline_cached(str(path), _mtime(path))
    return df if not df.empty else None


def load_stocks() -> Dict[str, Dict[str, Any]]:
    payload = load_json("stocks.json") or {}
    return {row["code"]: row for row in payload.get("rows", [])}


def load_portfolio() -> Optional[Dict[str, Any]]:
    return load_json("portfolio.json")


def load_benchmark() -> Optional[Dict[str, Any]]:
    return load_json("benchmark.json")


def load_factor() -> Optional[Dict[str, Any]]:
    return load_json("factor_attribution.json")


def load_group() -> Optional[Dict[str, Any]]:
    return load_json("group_analysis.json")


def load_selections() -> Optional[Dict[str, Any]]:
    return load_json("selections.json")


def load_verdicts() -> Optional[Dict[str, Any]]:
    return load_json("verdicts.json")


def load_knowledge() -> Optional[Dict[str, Any]]:
    return load_json("knowledge.json")


def load_meta() -> Dict[str, Any]:
    return load_json("meta.json") or {}


def load_health() -> Dict[str, Any]:
    return load_json("data_health.json") or {}


def clear_cache() -> None:
    st.cache_data.clear()
