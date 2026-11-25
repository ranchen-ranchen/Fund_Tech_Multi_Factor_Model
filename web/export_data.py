#!/usr/bin/env python3
"""Build the read-only dataset consumed by the Streamlit dashboard.

This script is deliberately separate from the research pipeline:

* it never calls the DeepSeek API;
* it never runs a backtest or refreshes market data;
* it never parses the free-form backtest log;
* it only reads research artifacts and writes JSON under ``web/data``.

The one structured input that does not exist by default is
``backtest_details.json``. It is produced by running the pipeline with
``--export-details web/data/backtest_details.json``.

Usage
-----
    python web/export_data.py
    python web/export_data.py --backtest-details web/data/backtest_details.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
WEB_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = WEB_DIR.parent
DEFAULT_DATA_DIR = WEB_DIR / "data"
DEFAULT_DETAILS = DEFAULT_DATA_DIR / "backtest_details.json"

BENCHMARK_PATTERNS = (
    "hs300etf_510300_*_performance.csv",
    "hs300etf_*_performance.csv",
)
FACTOR_CSV = "factor_contrib_detail.csv"
GROUP_CSV = "group_return_analysis.csv"
GROUP_MD = "group_return_analysis.md"
TARGET_CSV_CANDIDATES = ("target_code.csv", "tmp_target_code.csv")
LLM_CACHE = "target_code_llm_cache.jsonl"
MAIN_BUSINESS = "data/stock_main_business/main_business.csv"
HFQ_DIR = "stock_daily_k_hfq/individual"
BFQ_DIR = "stock_daily_k_bfq/individual"
KNOWLEDGE_REPORT = "knowledge_base/analysis/industry_eval_report.md"
KNOWLEDGE_DIR = "knowledge_base/output"

NUMERIC_SUMMARY_KEYS = {
    "Initial Price",
    "Final Price",
    "Trading Days",
    "Total Return",
    "Annualized Return",
    "Annualized Return (Arithmetic)",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Sharpe Ratio (CAGR/vol)",
    "Max Drawdown",
    "Calmar Ratio",
    "Daily Win Rate",
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    """Convert pandas/numpy objects to strict-JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(payload),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    os.replace(tmp_path, path)


def file_meta(path: Path, project_root: Path) -> Dict[str, Any]:
    stat = path.stat()
    try:
        rel = str(path.relative_to(project_root))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "size_bytes": stat.st_size,
    }


# ---------------------------------------------------------------------------
# Code helpers
# ---------------------------------------------------------------------------
def to_bs_code(code6: str) -> str:
    """Map a 6-digit A-share code to baostock's sh./sz./bj. form."""
    raw = str(code6).strip().lower()
    if "." in raw:
        return raw
    raw = raw.zfill(6)
    if raw.startswith(("60", "68", "51", "58", "11", "90", "50")):
        return f"sh.{raw}"
    if raw.startswith(("00", "30", "12", "15", "16", "18", "20", "39", "13")):
        return f"sz.{raw}"
    if raw.startswith(("43", "83", "87", "88", "92")):
        return f"bj.{raw}"
    return f"sh.{raw}" if raw[0] in ("5", "6", "9") else f"sz.{raw}"


def file_stem_from_code6(code6: str) -> str:
    return to_bs_code(code6).replace(".", "_")


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------
def as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_benchmark_csv(path: Path) -> Dict[str, Any]:
    """Parse the two-section benchmark CSV written by eval_hs300_perform.py."""
    summary: Dict[str, str] = {}
    daily: List[Dict[str, Any]] = []
    section: Optional[str] = None

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue
            if line.startswith("# Performance Summary"):
                section = "summary"
                continue
            if line.startswith("# Daily Series"):
                section = "daily"
                continue

            fields = next(csv.reader([line]))
            if section == "summary":
                if len(fields) < 2:
                    continue
                key = fields[0].strip()
                value = fields[1].strip()
                if key == "Metric" and value == "Value":
                    continue
                summary[key] = value
            elif section == "daily":
                if len(fields) < 4:
                    continue
                date_text = fields[0].strip()
                if not date_text or date_text.lower() == "close":
                    continue
                daily.append({
                    "date": date_text,
                    "close": as_float(fields[1]),
                    "nav": as_float(fields[2]),
                    "drawdown": as_float(fields[3]),
                })

    summary_parsed: Dict[str, Any] = {}
    for key, raw_value in summary.items():
        if key in NUMERIC_SUMMARY_KEYS:
            number = as_float(raw_value)
            if number is None:
                summary_parsed[key] = raw_value
            elif key == "Trading Days":
                summary_parsed[key] = int(number)
            else:
                summary_parsed[key] = number
        else:
            summary_parsed[key] = raw_value

    return {
        "summary": summary,
        "summary_parsed": summary_parsed,
        "daily": daily,
    }


def find_benchmark(project_root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = project_root / path
        return path if path.exists() else None
    candidates: List[Path] = []
    for pattern in BENCHMARK_PATTERNS:
        candidates.extend(project_root.glob(pattern))
    if not candidates:
        return None
    return max(set(candidates), key=lambda p: p.stat().st_mtime)


def resolve_target_csv(project_root: Path) -> Optional[Path]:
    for name in TARGET_CSV_CANDIDATES:
        path = project_root / name
        if path.exists():
            return path
    return None


def read_main_business(project_root: Path) -> Dict[str, str]:
    path = project_root / MAIN_BUSINESS
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception:
        return {}
    if "股票代码" not in df.columns or "股票名称" not in df.columns:
        return {}
    names: Dict[str, str] = {}
    for code, name in zip(df["股票代码"], df["股票名称"]):
        code_text = str(code).strip().zfill(6)
        if code_text:
            names[code_text] = str(name).strip()
    return names


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return []
    return df.to_dict(orient="records")


def read_target_selections(
    path: Path,
    name_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception:
        return []

    if "query_date" not in df.columns or "target_code" not in df.columns:
        return []

    rows: List[Dict[str, Any]] = []
    previous: set[str] = set()
    for _, record in df.iterrows():
        query_date = str(record["query_date"]).strip()
        if not query_date:
            continue
        codes: List[str] = []
        seen: set[str] = set()
        for token in str(record["target_code"]).split():
            token = token.strip()
            if not token.isdigit() or len(token) != 6:
                continue
            if token not in seen:
                seen.add(token)
                codes.append(token)
        current = set(codes)
        rows.append({
            "query_date": query_date,
            "count": len(codes),
            "codes": [
                {
                    "code": code,
                    "name": name_map.get(code, ""),
                    "bs_code": to_bs_code(code),
                    "file_stem": file_stem_from_code6(code),
                }
                for code in codes
            ],
            "added": sorted(current - previous),
            "removed": sorted(previous - current),
        })
        previous = current
    return rows


def read_llm_verdicts(
    path: Path,
    name_map: Dict[str, str],
) -> Dict[str, Any]:
    records: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code_raw = str(item.get("code", "")).strip()
                if not code_raw.isdigit() or len(code_raw) > 6:
                    continue
                code = code_raw.zfill(6)
                cached_at = str(item.get("cached_at", ""))
                previous = records.get(code)
                if previous is None or cached_at >= str(previous.get("cached_at", "")):
                    records[code] = item

    rows: List[Dict[str, Any]] = []
    for code in sorted(records):
        item = records[code]
        rows.append({
            "code": code,
            "name": name_map.get(code, ""),
            "bs_code": to_bs_code(code),
            "file_stem": file_stem_from_code6(code),
            "is_in_chain": bool(item.get("is_in_chain")),
            "reason": str(item.get("reason", "")),
            "model": str(item.get("model", "")),
            "prompt_version": str(item.get("prompt_version", "")),
            "cached_at": str(item.get("cached_at", "")),
        })

    in_chain = sum(1 for row in rows if row["is_in_chain"])
    return {
        "rows": rows,
        "counts": {
            "total": len(rows),
            "in_chain": in_chain,
            "not_in_chain": len(rows) - in_chain,
        },
    }


def read_knowledge_index(project_root: Path) -> Dict[str, Any]:
    report_path = project_root / KNOWLEDGE_REPORT
    report: Optional[Dict[str, Any]] = None
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="replace")
        as_of_match = re.search(r"证据截止[：:]\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
        report = {
            "title": "Industry Screening Report",
            "kind": "report",
            "path": str(report_path.relative_to(project_root)),
            "as_of": as_of_match.group(1) if as_of_match else None,
            **file_meta(report_path, project_root),
        }

    annual: List[Dict[str, Any]] = []
    quarterly: List[Dict[str, Any]] = []
    pattern = re.compile(r"^(\d{4})_events(?:_Q([1-4]))?\.md$")
    output_dir = project_root / KNOWLEDGE_DIR
    if output_dir.exists():
        for path in sorted(output_dir.glob("*_events*.md")):
            match = pattern.match(path.name)
            if not match:
                continue
            year = int(match.group(1))
            quarter = int(match.group(2)) if match.group(2) else None
            item = {
                "title": (
                    f"{year} Q{quarter} Dossier"
                    if quarter
                    else f"{year} Annual Dossier"
                ),
                "kind": "quarterly" if quarter else "annual",
                "year": year,
                "quarter": quarter,
                **file_meta(path, project_root),
            }
            if quarter:
                quarterly.append(item)
            else:
                annual.append(item)

    annual.sort(key=lambda item: (item["year"], item["path"]))
    quarterly.sort(key=lambda item: (item["year"], item["quarter"] or 0, item["path"]))
    return {
        "report": report,
        "annual": annual,
        "quarterly": quarterly,
        "all": ([report] if report else []) + annual + quarterly,
    }


# ---------------------------------------------------------------------------
# K-line index
# ---------------------------------------------------------------------------
def first_last_dates(path: Path) -> tuple[Optional[str], Optional[str]]:
    """Read only the first and last data dates of a daily K CSV."""
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            header = f.readline()
            if not header:
                return None, None
            for line in f:
                if not line.strip():
                    continue
                date_text = next(csv.reader([line]))[0].strip()
                if not date_text:
                    continue
                if first_date is None:
                    first_date = date_text
                last_date = date_text
    except (OSError, csv.Error):
        return None, None
    return first_date, last_date


def scan_kline_files(project_root: Path) -> Dict[str, Any]:
    hfq_dir = project_root / HFQ_DIR
    bfq_dir = project_root / BFQ_DIR
    files: Dict[str, Dict[str, Any]] = {}
    if hfq_dir.exists():
        for path in sorted(hfq_dir.glob("*.csv")):
            first_date, last_date = first_last_dates(path)
            files[path.stem] = {
                "first_date": first_date,
                "last_date": last_date,
                "path": str(path.relative_to(project_root)),
            }
    bfq_count = len(list(bfq_dir.glob("*.csv"))) if bfq_dir.exists() else 0
    return {
        "files": files,
        "hfq_count": len(files),
        "bfq_count": bfq_count,
    }


def build_stocks(
    required_codes: Iterable[str],
    name_map: Dict[str, str],
    kline_files: Dict[str, Dict[str, Any]],
    selections: List[Dict[str, Any]],
    verdict_rows: List[Dict[str, Any]],
    periods: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    held_periods: Dict[str, List[str]] = {}
    for period in periods:
        for holding in period.get("holdings", []):
            code_raw = holding.get("code6")
            if not code_raw:
                continue
            code = str(code_raw).zfill(6)
            held_periods.setdefault(code, []).append(period.get("name", ""))

    selected_dates: Dict[str, List[str]] = {}
    for row in selections:
        for item in row.get("codes", []):
            selected_dates.setdefault(item["code"], []).append(row["query_date"])

    verdict_map = {row["code"]: row for row in verdict_rows}

    rows: List[Dict[str, Any]] = []
    for code in sorted(set(required_codes)):
        code = str(code).zfill(6)
        file_stem = file_stem_from_code6(code)
        kline = kline_files.get(file_stem, {})
        verdict = verdict_map.get(code)
        rows.append({
            "code": code,
            "name": name_map.get(code, ""),
            "bs_code": to_bs_code(code),
            "file_stem": file_stem,
            "kline_path": kline.get("path"),
            "first_date": kline.get("first_date"),
            "last_date": kline.get("last_date"),
            "held_periods": held_periods.get(code, []),
            "selected_dates": selected_dates.get(code, []),
            "is_in_chain": verdict["is_in_chain"] if verdict else None,
            "verdict_reason": verdict["reason"] if verdict else None,
        })
    return rows


def build_health(
    project_root: Path,
    details: Dict[str, Any],
    benchmark: Optional[Dict[str, Any]],
    factor_rows: List[Dict[str, Any]],
    group_rows: List[Dict[str, Any]],
    selections: List[Dict[str, Any]],
    stocks: List[Dict[str, Any]],
    kline_scan: Dict[str, Any],
    name_count: int,
) -> Dict[str, Any]:
    required = {row["code"] for row in stocks}
    missing_hfq = sorted(
        row["code"] for row in stocks if not row.get("kline_path")
    )

    nav_dates = [str(row.get("date", ""))[:10] for row in details.get("nav", [])]
    benchmark_dates = [
        str(row.get("date", ""))[:10]
        for row in (benchmark or {}).get("daily", [])
    ]
    kline_last_dates = [
        item.get("last_date")
        for item in kline_scan.get("files", {}).values()
        if item.get("last_date")
    ]

    weight_files = sorted(project_root.glob("data/hs300_weight_*h*.csv"))
    valuation_path = project_root / "data/hs300_valuation_6m.csv"
    valuation_rows = 0
    if valuation_path.exists():
        try:
            valuation_rows = int(len(pd.read_csv(valuation_path, encoding="utf-8-sig")))
        except Exception:
            valuation_rows = 0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "required_code_count": len(required),
        "hfq_file_count": kline_scan.get("hfq_count", 0),
        "bfq_file_count": kline_scan.get("bfq_count", 0),
        "missing_hfq_codes": missing_hfq,
        "nav_last_date": max(nav_dates) if nav_dates else None,
        "benchmark_last_date": max(benchmark_dates) if benchmark_dates else None,
        "kline_first_date_min": (
            min(
                item["first_date"]
                for item in kline_scan.get("files", {}).values()
                if item.get("first_date")
            )
            if kline_scan.get("files")
            else None
        ),
        "kline_last_date_min": min(kline_last_dates) if kline_last_dates else None,
        "kline_last_date_max": max(kline_last_dates) if kline_last_dates else None,
        "weight_file_count": len(weight_files),
        "valuation_rows": valuation_rows,
        "main_business_rows": name_count,
        "factor_periods": len(factor_rows),
        "group_periods": len(group_rows),
        "selection_dates": [row["query_date"] for row in selections],
        "stock_count": len(stocks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export research artifacts to JSON for the read-only dashboard."
    )
    parser.add_argument(
        "--project-root",
        default=str(DEFAULT_PROJECT_ROOT),
        help="Project root (default: parent of web/).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Output directory for dashboard JSON (default: web/data).",
    )
    parser.add_argument(
        "--backtest-details",
        default=str(DEFAULT_DETAILS),
        help=(
            "Structured backtest JSON produced by "
            "target_code_pipeline.py --export-details (default: web/data/backtest_details.json)."
        ),
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark CSV path; default auto-discovers the newest 510300 file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    details_path = Path(args.backtest_details).resolve()

    if not details_path.exists():
        print("ERROR: structured backtest details not found:")
        print(f"  {details_path}")
        print("")
        print("Run the pipeline once with the export flag, for example:")
        print("  python target_code_pipeline.py --skip-prep --no-update-k \\")
        print("      --export-details web/data/backtest_details.json")
        print("")
        print("Then re-run: python web/export_data.py")
        return 2

    details = read_json(details_path)
    if not isinstance(details, dict) or "periods" not in details:
        print(f"ERROR: invalid backtest details file: {details_path}")
        return 2

    warnings: List[str] = []
    if details.get("schema_version") != SCHEMA_VERSION:
        warnings.append(
            "Backtest details use schema version "
            f"{details.get('schema_version')!r}; expected {SCHEMA_VERSION}. "
            "Regenerate with the current pipeline."
        )
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- names -------------------------------------------------------------
    name_map = read_main_business(project_root)
    if not name_map:
        warnings.append("Stock name table not found; names will be blank.")

    # --- benchmark ---------------------------------------------------------
    benchmark_path = find_benchmark(project_root, args.benchmark)
    benchmark = None
    if benchmark_path is not None:
        try:
            benchmark = parse_benchmark_csv(benchmark_path)
            benchmark["source"] = str(benchmark_path.relative_to(project_root))
        except Exception as exc:
            warnings.append(f"Could not parse benchmark CSV: {exc}")
    else:
        warnings.append("Benchmark CSV not found; benchmark sections are empty.")

    # --- attribution / group analysis -------------------------------------
    factor_path = project_root / FACTOR_CSV
    group_path = project_root / GROUP_CSV
    factor_rows = read_csv_records(factor_path)
    group_rows = read_csv_records(group_path)
    if not factor_rows:
        warnings.append("Factor attribution CSV not found; run barra_model_analysis.py.")
    if not group_rows:
        warnings.append("Group analysis CSV not found; run group_return_analysis.py.")

    # --- selections / LLM verdicts ----------------------------------------
    target_path = resolve_target_csv(project_root)
    selections: List[Dict[str, Any]] = []
    if target_path is not None:
        selections = read_target_selections(target_path, name_map)
    if not selections:
        warnings.append("Target selection CSV not found; run target_code_pipeline.py.")

    llm_path = project_root / LLM_CACHE
    verdict_data = read_llm_verdicts(llm_path, name_map)
    verdict_rows = verdict_data["rows"]
    if not verdict_rows:
        warnings.append("LLM verdict cache not found.")

    # --- knowledge base ----------------------------------------------------
    knowledge = read_knowledge_index(project_root)
    if not knowledge.get("report"):
        warnings.append("Knowledge-base report not found; run industry_eval.py.")

    # --- K-line index ------------------------------------------------------
    kline_scan = scan_kline_files(project_root)

    required_codes: set[str] = set()
    for period in details.get("periods", []):
        for holding in period.get("holdings", []):
            code_raw = holding.get("code6")
            if code_raw:
                required_codes.add(str(code_raw).zfill(6))
    for row in selections:
        for item in row.get("codes", []):
            required_codes.add(item["code"])
    for row in verdict_rows:
        required_codes.add(row["code"])

    stocks = build_stocks(
        required_codes=required_codes,
        name_map=name_map,
        kline_files=kline_scan["files"],
        selections=selections,
        verdict_rows=verdict_rows,
        periods=details.get("periods", []),
    )
    missing_codes = [row["code"] for row in stocks if not row.get("kline_path")]
    if missing_codes:
        warnings.append(
            f"{len(missing_codes)} required codes have no hfq K-line file: "
            + ", ".join(missing_codes[:10])
            + (" ..." if len(missing_codes) > 10 else "")
        )

    health = build_health(
        project_root=project_root,
        details=details,
        benchmark=benchmark,
        factor_rows=factor_rows,
        group_rows=group_rows,
        selections=selections,
        stocks=stocks,
        kline_scan=kline_scan,
        name_count=len(name_map),
    )

    # --- derived dates -----------------------------------------------------
    date_candidates: List[str] = []
    date_candidates.extend(
        str(row.get("date", ""))[:10] for row in details.get("nav", [])
    )
    if benchmark:
        date_candidates.extend(
            str(row.get("date", ""))[:10] for row in benchmark.get("daily", [])
        )
    date_candidates.extend(
        str(row.get("end_effective", ""))[:10] for row in factor_rows
        if row.get("end_effective")
    )
    date_candidates.extend(
        str(row.get("end", ""))[:10] for row in group_rows
        if row.get("end")
    )
    date_candidates.extend(
        item["last_date"] for item in kline_scan["files"].values()
        if item.get("last_date")
    )
    data_as_of = max(date_candidates) if date_candidates else None

    # --- sources -----------------------------------------------------------
    source_candidates = [
        ("backtest_details", details_path),
        ("benchmark", benchmark_path),
        ("factor_attribution", factor_path),
        ("group_analysis", group_path),
        ("group_report", project_root / GROUP_MD),
        ("target_selection", target_path),
        ("llm_cache", llm_path),
        ("main_business", project_root / MAIN_BUSINESS),
        ("knowledge_report", project_root / KNOWLEDGE_REPORT),
    ]
    sources = []
    for name, path in source_candidates:
        if path is not None and path.exists():
            sources.append({"name": name, **file_meta(path, project_root)})

    # --- write outputs -----------------------------------------------------
    portfolio_payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": details.get("meta", {}),
        "generated_at": details.get("generated_at"),
        "summary": details.get("summary", []),
        "nav": details.get("nav", []),
        "drawdown": details.get("drawdown", []),
        "periods": details.get("periods", []),
        "stock_overview": details.get("stock_overview", []),
        "benchmark_aligned": details.get("benchmark", {}).get("aligned", []),
        "benchmark_file": details.get("benchmark", {}).get("file"),
    }

    benchmark_payload = {
        "schema_version": SCHEMA_VERSION,
        "source": benchmark.get("source") if benchmark else None,
        **(benchmark or {"summary": {}, "summary_parsed": {}, "daily": []}),
    }

    outputs = {
        "portfolio.json": portfolio_payload,
        "benchmark.json": benchmark_payload,
        "factor_attribution.json": {
            "schema_version": SCHEMA_VERSION,
            "source": str(factor_path.relative_to(project_root)),
            "rows": factor_rows,
            "image_path": "factor_contrib_barplot.png",
        },
        "group_analysis.json": {
            "schema_version": SCHEMA_VERSION,
            "source": str(group_path.relative_to(project_root)),
            "rows": group_rows,
            "report_path": GROUP_MD,
            "image_path": "group_return_spread.png",
        },
        "selections.json": {
            "schema_version": SCHEMA_VERSION,
            "source": str(target_path.relative_to(project_root)) if target_path else None,
            "rows": selections,
        },
        "verdicts.json": {
            "schema_version": SCHEMA_VERSION,
            "source": str(llm_path.relative_to(project_root)),
            "rows": verdict_rows,
            "counts": verdict_data["counts"],
        },
        "stocks.json": {
            "schema_version": SCHEMA_VERSION,
            "rows": stocks,
        },
        "knowledge.json": {
            "schema_version": SCHEMA_VERSION,
            **knowledge,
        },
        "data_health.json": {
            "schema_version": SCHEMA_VERSION,
            **health,
        },
        "meta.json": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_as_of": data_as_of,
            "backtest_generated_at": details.get("generated_at"),
            "sources": sources,
            "counts": {
                "periods": len(details.get("periods", [])),
                "holdings": sum(
                    len(period.get("holdings", []))
                    for period in details.get("periods", [])
                ),
                "nav_rows": len(details.get("nav", [])),
                "benchmark_rows": len((benchmark or {}).get("daily", [])),
                "factor_periods": len(factor_rows),
                "group_periods": len(group_rows),
                "selection_dates": len(selections),
                "verdicts": len(verdict_rows),
                "in_chain": verdict_data["counts"]["in_chain"],
                "stocks": len(stocks),
                "knowledge_files": len(knowledge.get("all", [])),
            },
            "warnings": warnings,
        },
    }

    for name, payload in outputs.items():
        write_json(data_dir / name, payload)

    print(f"Wrote {len(outputs)} JSON files to {data_dir}")
    for name in outputs:
        size = (data_dir / name).stat().st_size
        print(f"  {name:<26s} {size:>8d} bytes")
    print("")
    print("Counts:")
    for key, value in outputs["meta.json"]["counts"].items():
        print(f"  {key:<18s} {value}")
    if warnings:
        print("")
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
