#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按 data/hs300_weight_*.csv 的股票池补抓 PE(TTM) / PB(MRQ)。

口径
----
- 股票池：所有 weight 文件的代码并集（含 2020h2），统一为 baostock 代码；
- 目标日期：各 weight 文件的期末（2020-12-31, 2021-06-30, ...），
  即每期“上一期期末”的估值快照；
- 默认只补抓现有 CSV 中缺失的 (code, target_date) 组合，已有数据保留；
- 每个组合只查询目标日期前 365 天，避免重复下载多年日频数据；
- 输出：data/hs300_valuation_6m.csv。

运行（项目根目录）：
    python data/fetch_hs300_valuation.py            # 增量补抓缺失组合
    python data/fetch_hs300_valuation.py --full     # 忽略现有文件，按股票全量重抓
"""

import argparse
import glob
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.text_utils import normalize_code, read_hs300_weight  # noqa: E402

WEIGHT_GLOB = str(ROOT / "data/hs300_weight_*h*.csv")
OUTPUT_FILE = str(ROOT / "data/hs300_valuation_6m.csv")
MAX_RETRY = 3
RETRY_SLEEP = 1.0
SLEEP_PER_QUERY = 0.2
LOOKBACK_DAYS = 90           # 每个目标日期查询前向窗口，覆盖短期停牌
REQUIRED_COLS = ["code", "target_date", "actual_date", "peTTM", "pbMRQ"]


def weight_files() -> list[tuple[pd.Timestamp, str]]:
    """返回 [(期末日期, weight 文件路径)]，按期末日期升序。"""
    files = []
    for path in glob.glob(WEIGHT_GLOB):
        m = re.search(r"hs300_weight_(\d{4})h([12])\.csv$", os.path.basename(path))
        if not m:
            print(f"[warn] 跳过无法解析文件名的权重文件: {path}")
            continue
        year, half = int(m.group(1)), int(m.group(2))
        asof = pd.Timestamp(f"{year}-06-30" if half == 1 else f"{year}-12-31")
        files.append((asof, path))
    if not files:
        raise FileNotFoundError(f"未找到任何权重文件: {WEIGHT_GLOB}")
    files.sort(key=lambda x: x[0])
    return files


def load_universe(files: list[tuple[pd.Timestamp, str]]) -> list[str]:
    """读取所有 weight 文件的代码并集，统一为 sh./sz./bj. 格式。"""
    codes = set()
    for _, path in files:
        weights = read_hs300_weight(path)
        for code in weights.index:
            codes.add(normalize_code(str(code)))
    if not codes:
        raise ValueError("weight 文件股票池为空")
    return sorted(codes)


def target_dates(files: list[tuple[pd.Timestamp, str]]) -> list[str]:
    """各 weight 文件的期末，即每期期初使用的估值快照日期。"""
    return sorted({asof.strftime("%Y-%m-%d") for asof, _ in files})


def codes_by_target(
    files: list[tuple[pd.Timestamp, str]],
) -> dict[str, list[str]]:
    """每个目标日期对应的 weight 文件股票池（该期期初基准持仓）。"""
    result: dict[str, list[str]] = {}
    for asof, path in files:
        target = asof.strftime("%Y-%m-%d")
        weights = read_hs300_weight(path)
        codes = sorted({normalize_code(str(c)) for c in weights.index})
        if target in result:
            result[target] = sorted(set(result[target]) | set(codes))
        else:
            result[target] = codes
    return result


def load_existing() -> pd.DataFrame:
    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame(columns=REQUIRED_COLS)
    df = pd.read_csv(
        OUTPUT_FILE, dtype={"code": str, "target_date": str},
        encoding="utf-8-sig",
    )
    if "code" in df.columns:
        df = df.dropna(subset=["code"])
        df["code"] = df["code"].astype(str).map(normalize_code)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[REQUIRED_COLS]


def coverage_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    """已有有效 PB 的 (code, target_date) 集合。"""
    if df.empty:
        return set()
    valid = df[
        df["pbMRQ"].notna()
        & (pd.to_numeric(df["pbMRQ"], errors="coerce").notna())
    ]
    return set(zip(valid["code"].astype(str), valid["target_date"].astype(str)))


def fetch_one(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取单只股票的日频 PE/PB；查询失败会重试，空数据返回空表。"""
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,peTTM,pbMRQ",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            if rs.error_code != "0":
                raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame(columns=["date", "peTTM", "pbMRQ"])

            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df.dropna(subset=["date"]).sort_values("date")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError(f"{code} 查询失败: {last_err}")


def make_record(code: str, target: str, df_stock: pd.DataFrame):
    """从日频数据中取 <= target 的最后一行；无数据返回 None。"""
    if df_stock is None or df_stock.empty:
        return None
    target_dt = pd.Timestamp(target)
    subset = df_stock[df_stock["date"] <= target_dt]
    if subset.empty:
        return None
    row = subset.iloc[-1]
    return {
        "code": code,
        "target_date": target,
        "actual_date": row["date"].strftime("%Y-%m-%d"),
        "peTTM": row["peTTM"],
        "pbMRQ": row["pbMRQ"],
    }


def fetch_supplement(universe_by_target: dict[str, list[str]],
                     targets: list[str],
                     existing: pd.DataFrame) -> list[dict]:
    """只抓缺失的 (code, target_date) 组合；每期只补该期 weight 文件股票池。"""
    covered = coverage_pairs(existing)
    todo = [
        (code, target)
        for target in targets
        for code in universe_by_target[target]
        if (code, target) not in covered
    ]
    print(f"待补抓组合: {len(todo)} 个（已有 {len(covered)} 个）")
    if not todo:
        return []

    records = []
    failed = []
    no_data = []
    for i, (code, target) in enumerate(todo, 1):
        start = (
            pd.Timestamp(target) - pd.Timedelta(days=LOOKBACK_DAYS)
        ).date().isoformat()
        try:
            df_stock = fetch_one(code, start, target)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code}@{target}: {exc}")
            print(f"[{i}/{len(todo)}] {code}@{target} 失败: {exc}", flush=True)
            continue

        record = make_record(code, target, df_stock)
        if record is None:
            no_data.append(f"{code}@{target}")
        else:
            records.append(record)

        if i % 20 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] 新增 {len(records)} 条，"
                  f"无数据 {len(no_data)}，失败 {len(failed)}", flush=True)
        time.sleep(SLEEP_PER_QUERY)

    if no_data:
        print(f"无数据组合 {len(no_data)} 个: {', '.join(no_data[:10])}"
              + (" ..." if len(no_data) > 10 else ""))
    if failed:
        print(f"失败组合 {len(failed)} 个:")
        for item in failed[:20]:
            print(f"  {item}")
        if len(failed) > 20:
            print(f"  ... 共 {len(failed)} 个")
    return records


def fetch_full(codes: list[str], targets: list[str]) -> list[dict]:
    """全量模式：每只股票查一次 [最早目标-365, 最晚目标]，再取各目标日期。"""
    start = (
        pd.Timestamp(targets[0]) - pd.Timedelta(days=LOOKBACK_DAYS)
    ).date().isoformat()
    end = targets[-1]
    print(f"全量抓取: {len(codes)} 只，查询区间 {start} ~ {end}")

    records = []
    failed = []
    no_data = []
    for i, code in enumerate(codes, 1):
        try:
            df_stock = fetch_one(code, start, end)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code}: {exc}")
            continue

        if df_stock.empty:
            no_data.append(code)
        else:
            for target in targets:
                record = make_record(code, target, df_stock)
                if record is not None:
                    records.append(record)

        if i % 20 == 0 or i == len(codes):
            print(f"[{i}/{len(codes)}] 新增 {len(records)} 条，"
                  f"无数据 {len(no_data)}，失败 {len(failed)}", flush=True)
        time.sleep(SLEEP_PER_QUERY)

    if no_data:
        print(f"无数据股票 {len(no_data)} 只: {', '.join(no_data[:10])}"
              + (" ..." if len(no_data) > 10 else ""))
    if failed:
        print(f"失败股票 {len(failed)} 只:")
        for item in failed[:20]:
            print(f"  {item}")
        if len(failed) > 20:
            print(f"  ... 共 {len(failed)} 只")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="按 weight 文件股票池补抓 PE/PB")
    parser.add_argument("--full", action="store_true",
                        help="忽略现有文件，按股票全量重抓")
    args = parser.parse_args()
    socket.setdefaulttimeout(60)

    files = weight_files()
    targets = target_dates(files)
    universe_by_target = codes_by_target(files)
    codes = sorted(set().union(*universe_by_target.values()))
    existing = load_existing()

    print(f"weight 文件 {len(files)} 个，股票池 {len(codes)} 只")
    print(f"目标日期: {targets}")
    print(f"现有数据: {len(existing)} 行，"
          f"{len(coverage_pairs(existing))} 个有效 (code, target_date)")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_code} - {lg.error_msg}")

    try:
        if args.full or existing.empty:
            if existing.empty and not args.full:
                print("[warn] 现有估值文件为空，自动切换为全量模式")
            records = fetch_full(codes, targets)
        else:
            records = fetch_supplement(universe_by_target, targets, existing)
    finally:
        bs.logout()

    if not records:
        print("没有新增记录，输出文件保持不变")
        return 0

    new_df = pd.DataFrame(records)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = (
        combined[REQUIRED_COLS]
        .dropna(subset=["code", "target_date"])
        .drop_duplicates(subset=["code", "target_date"], keep="last")
        .sort_values(["code", "target_date"])
        .reset_index(drop=True)
    )

    tmp_path = f"{OUTPUT_FILE}.tmp"
    combined.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    os.replace(tmp_path, OUTPUT_FILE)

    print(f"完成：输出 {OUTPUT_FILE}，共 {len(combined)} 行")
    print(f"  覆盖股票 {combined['code'].nunique()} 只，"
          f"目标日期 {combined['target_date'].nunique()} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
