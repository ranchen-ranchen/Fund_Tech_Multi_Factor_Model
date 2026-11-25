#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并流水线：LLM 选股 + 多期等权买入持有回测

流程
----
1. prep：对每个调仓日读取上一期期末的 data/hs300_weight_*.csv（作为本期期初的
   沪深300持仓/成分股），调用 DeepSeek 判断是否属于 AI 算力链，
   将选中的代码写入目标 CSV（默认 tmp_target_code.csv）。
2. eval：读取目标 CSV，按调仓日切分区间，等权买入持有，并与 510300 ETF 基准对比。

用法
----
python target_code_pipeline.py                  # 重新选股；默认回测 2021-01-01 ~ 今天
python target_code_pipeline.py --skip-prep      # 复用已有 tmp_target_code.csv，只跑回测
python target_code_pipeline.py --publish        # 全流程成功后，原子替换 target_code.csv
python target_code_pipeline.py --benchmark PATH # 指定基准文件（默认自动查找）
python target_code_pipeline.py --end-date DATE  # 指定回测终点（不晚于 2026-12-31）
python target_code_pipeline.py --no-update-k    # 跳过回测前的 K 线增量更新
python target_code_pipeline.py --no-cache       # 不读写缓存（本次运行内仍按唯一代码去重）
python target_code_pipeline.py --refresh-cache  # 忽略已有缓存，强制重新判定

回测区间
--------
起点固定为 2021-01-01（2021h1 使用 hs300_weight_2020h2.csv 作为期初持仓）；
终点默认取运行当天，且不晚于 2026-12-31。--start-date / --end-date 仍可覆盖，
区间参数只影响回测，选股仍覆盖 QUERY_DATES 中的全部调仓日。

- 每个调仓日使用“上一期期末”的 weight 文件作为本期期初持仓：
  2021-01-01 用 2020h2，2021-06-30 用 2021h1，2021-12-31 用 2021h2，依此类推；
- 起点落在两个调仓日之间时，沿用此前最近调仓日的选股，并从起点当天开始持有；
- 起点早于第一个调仓日时，实际从第一个调仓日开始；
- 终点早于最后一个调仓日时，其后的调仓日会被丢弃，最后一期在终点结束；
- 终点晚于最后一个调仓日 2026-06-30 时，最后一期沿用该日的持仓与选股结果。

行情数据
--------
回测前会检查持仓股票的 K 线是否覆盖各期终点；不足时通过 baostock 增量抓取
（后复权，与 stock_daily_k_hfq 一致）并原子写回个股 CSV。可用 --no-update-k
跳过（离线或数据已就绪时）。

缓存
----
LLM 判定结果保存在 target_code_llm_cache.jsonl（可用 --cache-file 修改）。
缓存键为 (代码, 业务描述哈希, prompt 版本, 模型)，任一变化都会自动失效。
12 个调仓日共有 3600 次持仓出现，但唯一代码仅 461 只，因此首次最多 461 次
调用，之后全部命中缓存；每次判定后立即落盘，进程中断后重跑可从断点继续。

注意：脚本使用相对路径，请在项目根目录（/home/omen/work/quant）下运行。
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from strategy.backtest_buy_hold import run_backtest
from strategy.industry_chain_classifier import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    classify_ai_compute,
)
from utils.security_utils import redact_secrets
from utils.text_utils import (
    load_business_descriptions,
    normalize_code,
    read_hs300_weight,
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
WEIGHT_GLOB = "data/hs300_weight_*h*.csv"
BUSINESS_CSV = "data/stock_main_business/main_business.csv"
K_DIR = "stock_daily_k_hfq/individual"

# prep 阶段的调仓日（单一事实来源）；eval 阶段的起点直接读取目标 CSV 的 query_date。
# 每个日期使用“上一期期末”的 weight 文件作为本期期初持仓：
# 2021-01-01 -> 2020h2，2021-06-30 -> 2021h1，依此类推。
QUERY_DATES = [
    "2021-01-01",
    "2021-06-30", "2021-12-31",
    "2022-06-30", "2022-12-31",
    "2023-06-30", "2023-12-31",
    "2024-06-30", "2024-12-31",
    "2025-06-30", "2025-12-31",
    "2026-06-30",
]

DEFAULT_TARGET_CSV = "tmp_target_code.csv"
CANONICAL_TARGET_CSV = "target_code.csv"
DEFAULT_CACHE_FILE = "target_code_llm_cache.jsonl"
DEFAULT_START_DATE = QUERY_DATES[0]          # 固定起点 2021-01-01
MAX_END_DATE = "2026-12-31"                  # 终点上限
K_ADJUST_FLAG = "1"                          # baostock 1=后复权，与 K_DIR 一致
DEFAULT_LOG_FILE = "portfolio_backtest.log"
DEFAULT_PLOT_FILE = "portfolio_nav.png"


def default_end_date() -> str:
    """默认回测终点：运行当天，但不晚于 MAX_END_DATE。"""
    today = datetime.now().date()
    max_end = datetime.strptime(MAX_END_DATE, "%Y-%m-%d").date()
    return min(today, max_end).isoformat()


# ---------------------------------------------------------------------------
# prep：LLM 选股（唯一代码预聚合 + 磁盘缓存）
# ---------------------------------------------------------------------------
def _description_hash(description: Optional[str]) -> str:
    return hashlib.sha1((description or "").encode("utf-8")).hexdigest()


def load_cache(cache_file: str) -> Dict[tuple, dict]:
    """读取 JSONL 缓存；损坏行会被跳过，后写入的同键记录覆盖先前的。"""
    cache: Dict[tuple, dict] = {}
    if not os.path.exists(cache_file):
        return cache

    with open(cache_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (
                    str(record["code"]),
                    str(record["description_sha1"]),
                    str(record["prompt_version"]),
                    str(record["model"]),
                )
                if not isinstance(record.get("is_in_chain"), bool):
                    raise ValueError("is_in_chain 不是布尔值")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(
                    f"[cache] 跳过损坏行 {cache_file}:{lineno}（{exc}）",
                    file=sys.stderr,
                )
                continue
            cache[key] = record

    return cache


def append_cache(cache_file: str, record: dict) -> None:
    """追加一条缓存并立即落盘，保证进程中断后可续跑。"""
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def classify_with_cache(
    code: str,
    description: Optional[str],
    *,
    model: str,
    cache: Dict[tuple, dict],
    cache_file: Optional[str],
    refresh: bool = False,
) -> tuple:
    """
    返回 (record, from_cache)。

    缓存键 = (code, 描述哈希, prompt 版本, 模型)。
    描述、提示词或模型任一变化都会使旧缓存自动失效。
    """
    key = (code, _description_hash(description), PROMPT_VERSION, model)

    if cache_file and not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit, True

    if not description or not description.strip():
        result = {"is_in_chain": False, "reason": "业务描述为空"}
    else:
        result = classify_ai_compute(description, model=model)

    record = {
        "code": code,
        "description_sha1": key[1],
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "is_in_chain": bool(result["is_in_chain"]),
        "reason": str(result.get("reason", "") or ""),
        "cached_at": datetime.now().isoformat(timespec="seconds"),
    }

    if cache_file:
        append_cache(cache_file, record)
        cache[key] = record

    return record, False


def weight_asof_files(weight_glob: str = WEIGHT_GLOB) -> List[Tuple[pd.Timestamp, str]]:
    """返回 [(期末日期, weight 文件路径)]，按期末日期升序。

    文件名约定为 hs300_weight_{year}h{1|2}.csv：
    h1 的期末为当年 6/30，h2 的期末为当年 12/31。
    """
    files: List[Tuple[pd.Timestamp, str]] = []
    for path in glob.glob(weight_glob):
        m = re.search(r"hs300_weight_(\d{4})h([12])\.csv$", os.path.basename(path))
        if not m:
            print(f"[prep] 跳过无法解析文件名的权重文件: {path}", flush=True)
            continue
        year, half = int(m.group(1)), int(m.group(2))
        asof = pd.Timestamp(f"{year}-06-30" if half == 1 else f"{year}-12-31")
        files.append((asof, path))

    if not files:
        raise FileNotFoundError(f"未找到任何权重文件: {weight_glob}")
    files.sort(key=lambda x: x[0])
    return files


def collect_date_codes(
    dates: List[str],
    weight_glob: str = WEIGHT_GLOB,
) -> Dict[str, List[str]]:
    """读取每个调仓日的期初成分股：取不晚于该日期的最近一期 weight 文件。

    即“上一期期末的持仓作为本期期初的数据”：
      2021-01-01 -> hs300_weight_2020h2.csv
      2021-06-30 -> hs300_weight_2021h1.csv
      2021-12-31 -> hs300_weight_2021h2.csv
      ...
    若某日期不是半年末（例如 2021-09-30），沿用此前最近一期 weight 文件。
    早于最早一期 weight 文件（2020-12-31）的日期会明确报错。
    """
    files = weight_asof_files(weight_glob)
    date_codes: Dict[str, List[str]] = {}

    for date in dates:
        ts = pd.Timestamp(date)
        candidates = [(asof, path) for asof, path in files if asof <= ts]
        if not candidates:
            raise ValueError(
                f"{date} 之前没有可用的 weight 文件（最早为 "
                f"{files[0][0].date()}），请检查 --weight-glob 或调仓日"
            )

        asof, path = candidates[-1]
        if asof != ts:
            print(
                f"[prep] {date} 使用上一期期末 {asof.date()} 的持仓: "
                f"{os.path.basename(path)}",
                flush=True,
            )

        weights = read_hs300_weight(path)
        codes = sorted(set(str(c) for c in weights.index))
        if not codes:
            raise ValueError(
                f"{date} 未从 {path} 读取到任何成分股，请检查文件内容"
            )
        date_codes[date] = codes
        print(
            f"[prep] {date}: 上一期期末 {asof.date()}，"
            f"期初持仓 {len(codes)} 只（{os.path.basename(path)}）",
            flush=True,
        )

    return date_codes


def prep_target_codes(
    dates: List[str],
    weight_glob: str = WEIGHT_GLOB,
    business_csv: str = BUSINESS_CSV,
    cache_file: Optional[str] = DEFAULT_CACHE_FILE,
    refresh_cache: bool = False,
    model: str = DEFAULT_MODEL,
) -> Dict[str, str]:
    """
    先汇总所有调仓日的唯一成分股，每只只判定一次，再按日期组装选股结果。
    12 个调仓日共 3600 次持仓出现，唯一代码约 461 只；
    有缓存时热启动接近 0 次调用。
    """
    date_codes = collect_date_codes(dates, weight_glob)
    unique_codes = sorted({code for codes in date_codes.values() for code in codes})
    total_appearances = sum(len(codes) for codes in date_codes.values())

    print(
        f"[prep] {len(dates)} 个调仓日，持仓出现 {total_appearances} 次，"
        f"唯一代码 {len(unique_codes)} 只",
        flush=True,
    )

    descriptions = load_business_descriptions(business_csv)
    cache = load_cache(cache_file) if cache_file else {}
    if cache_file:
        print(
            f"[prep] 缓存 {cache_file}：{len(cache)} 条记录"
            + ("（--refresh-cache，本次忽略已有缓存）" if refresh_cache else ""),
            flush=True,
        )
    else:
        print("[prep] 未启用磁盘缓存（--no-cache），仅在本次运行内去重", flush=True)

    results: Dict[str, dict] = {}
    api_calls = 0
    cache_hits = 0

    for i, code in enumerate(unique_codes, 1):
        description = descriptions.get(code)
        try:
            record, from_cache = classify_with_cache(
                code,
                description,
                model=model,
                cache=cache,
                cache_file=cache_file,
                refresh=refresh_cache,
            )
        except Exception as exc:
            # 不链式抛出原始异常，避免 traceback 中输出未脱敏的请求信息
            raise RuntimeError(
                f"判定 {code} 时失败：{redact_secrets(exc)}"
            ) from None

        results[code] = record
        if from_cache:
            cache_hits += 1
        elif description and description.strip():
            api_calls += 1

        if i % 50 == 0 or i == len(unique_codes):
            print(
                f"[prep] 进度 {i}/{len(unique_codes)}："
                f"API {api_calls} 次，缓存命中 {cache_hits} 次",
                flush=True,
            )

    target_code_dict: Dict[str, str] = {}
    for date, codes in date_codes.items():
        selected = [code for code in codes if results[code]["is_in_chain"]]
        if not selected:
            raise ValueError(f"{date} 选股结果为空，终止流水线")
        target_code_dict[date] = " ".join(selected)
        print(
            f"[prep] {date}: 期初持仓 {len(codes)} 只，命中 {len(selected)} 只",
            flush=True,
        )

    print(
        f"[prep] 完成：唯一代码 {len(unique_codes)} 只，"
        f"本次 API 调用 {api_calls} 次，缓存命中 {cache_hits} 次",
        flush=True,
    )
    return target_code_dict


def write_target_code_csv(target_code_dict: Dict[str, str], output_path: str) -> None:
    """原子写入目标 CSV，避免中途失败留下半截文件。"""
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_date", "target_code"])
        for k, v in target_code_dict.items():
            writer.writerow([k, v])
    os.replace(tmp_path, output_path)
    print(f"[prep] 已写入 {output_path}", flush=True)


# ---------------------------------------------------------------------------
# eval：读取与校验
# ---------------------------------------------------------------------------
def read_target_code_csv(file_path: str) -> pd.DataFrame:
    """读取目标 CSV，并把 target_code 列拆分为代码列表。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"找不到目标文件: {file_path}（去掉 --skip-prep 可重新选股）"
        )

    df = pd.read_csv(
        file_path,
        dtype={"target_code": str, "query_date": str},
    )
    df["target_code"] = (
        df["target_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split()
    )
    return df


def validate_target_df(df: pd.DataFrame) -> pd.Series:
    """校验目标 CSV 的日期与选股结果，返回解析后的 query_date 序列。"""
    if df.empty:
        raise ValueError("目标 CSV 为空，无法回测")

    query_dates = pd.to_datetime(df["query_date"], errors="coerce")
    if query_dates.isna().any():
        bad = df.loc[query_dates.isna(), "query_date"].tolist()
        raise ValueError(f"存在无法解析的 query_date: {bad}")
    if query_dates.duplicated().any():
        dup = df.loc[query_dates.duplicated(), "query_date"].tolist()
        raise ValueError(f"存在重复的 query_date: {dup}")
    if not query_dates.is_monotonic_increasing:
        raise ValueError("query_date 必须按升序排列")

    empty_rows = df.loc[df["target_code"].map(len) == 0, "query_date"].tolist()
    if empty_rows:
        raise ValueError(f"以下调仓日的选股为空，无法回测: {empty_rows}")

    return query_dates


def parse_date_arg(value: Optional[str], name: str) -> Optional[pd.Timestamp]:
    """解析命令行日期参数；未提供时返回 None。"""
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"无法解析 {name}: {value!r}")
    return ts


def trim_to_backtest_range(
    df: pd.DataFrame,
    query_dates: pd.Series,
    start_ts: Optional[pd.Timestamp],
    end_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
    """
    按回测区间裁剪调仓日，返回 (裁剪后的 df, query_dates, 实际起始日)。

    每个调仓日的持仓区间为 [query_date, 下一个 query_date)，最后一期到 end_ts。
    起点落在两个调仓日之间时，沿用此前最近调仓日的选股并从起点开始持有；
    起点早于第一个调仓日时，实际从第一个调仓日开始；起点之前已经结束的调仓日
    以及终点及之后才开始的调仓日都会被丢弃。
    """
    if start_ts is not None and start_ts >= end_ts:
        raise ValueError(
            f"--start-date ({start_ts.date()}) 必须早于 "
            f"--end-date ({end_ts.date()})"
        )

    period_ends = list(query_dates.iloc[1:]) + [end_ts]
    keep = [
        i
        for i, (qdate, period_end) in enumerate(zip(query_dates, period_ends))
        if qdate < end_ts and (start_ts is None or period_end > start_ts)
    ]

    if not keep:
        range_desc = (
            f"[{start_ts.date() if start_ts is not None else '起点'}, "
            f"{end_ts.date()}]"
        )
        raise ValueError(
            f"回测区间 {range_desc} 内没有任何调仓日，请检查 --start-date/--end-date"
        )

    df = df.iloc[keep].reset_index(drop=True)
    query_dates = query_dates.iloc[keep].reset_index(drop=True)

    first_query = query_dates.iloc[0]
    if start_ts is not None and start_ts < first_query:
        print(
            f"[eval] --start-date ({start_ts.date()}) 早于第一个调仓日 "
            f"{first_query.date()}，实际从 {first_query.date()} 开始回测",
            flush=True,
        )
        effective_start = first_query
    else:
        effective_start = start_ts if start_ts is not None else first_query

    return df, query_dates, effective_start


def period_start_dates(
    query_dates: pd.Series,
    first_start: Optional[pd.Timestamp] = None,
) -> List[pd.Timestamp]:
    """返回每期的起始日期；第一期可被回测起始日覆盖。"""
    starts = [ts for ts in query_dates]
    if first_start is not None:
        starts[0] = first_start
    return starts


def collect_codes(df: pd.DataFrame) -> List[str]:
    """展开 target_code 列，返回去重后的代码列表。"""
    return (
        df["target_code"]
        .explode()
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def _price_path(code: str, k_dir: str = K_DIR) -> str:
    return os.path.join(k_dir, normalize_code(code).replace(".", "_") + ".csv")


def _last_price_date(path: str) -> Optional[pd.Timestamp]:
    """读取行情 CSV 的最后一个有效交易日。"""
    raw = pd.read_csv(path, usecols=["date"], encoding="utf-8-sig")
    dates = pd.to_datetime(raw["date"], errors="coerce").dropna()
    return dates.max() if not dates.empty else None


def required_k_end_dates(
    df: pd.DataFrame,
    periods: List[dict],
) -> Dict[str, pd.Timestamp]:
    """每个代码需要覆盖的最晚日期：它出现的最后一个区间的终点。"""
    required: Dict[str, pd.Timestamp] = {}
    for i, period in enumerate(periods):
        period_end = pd.to_datetime(period["end_date"])
        for raw_code in df.iloc[i]["target_code"]:
            code = normalize_code(raw_code)
            if code not in required or period_end > required[code]:
                required[code] = period_end
    return required


def update_k_data_if_needed(
    df: pd.DataFrame,
    periods: List[dict],
    k_dir: str = K_DIR,
    adjustflag: str = K_ADJUST_FLAG,
) -> None:
    """
    回测前检查行情覆盖：若个股 K 线未覆盖其最后一个区间的终点，
    通过 baostock 增量抓取 [最后日期+1, 区间终点] 并原子写回个股 CSV。

    抓取异常会汇总后抛出，避免数据不足时静默截断回测；空结果
    （停牌/退市/非交易日）只提示，不阻断回测。
    """
    required = required_k_end_dates(df, periods)
    todo: List[Tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    for code, required_end in required.items():
        path = _price_path(code, k_dir)
        if not os.path.exists(path):
            continue  # 缺失文件由 validate_price_files 统一报错
        try:
            last = _last_price_date(path)
        except Exception:  # noqa: BLE001
            continue  # 损坏文件同样交由 validate_price_files 报错
        if last is not None and last < required_end:
            todo.append((code, path, last + pd.Timedelta(days=1), required_end))

    if todo:
        print(
            f"[kdata] {len(todo)} 只持仓股票的行情未覆盖区间终点，开始增量更新 ...",
            flush=True,
        )
        try:
            from data.fetch_daily_k import fetch_one, login, logout
        except ImportError as exc:  # baostock 未安装等
            raise RuntimeError(
                "行情数据需要增量更新，但无法导入 data.fetch_daily_k"
                "（请确认已安装 baostock，并在项目根目录运行）"
            ) from exc

        login()
        updated = 0
        appended = 0
        empty: List[str] = []
        failed: List[str] = []
        try:
            for i, (code, path, start_ts, end_ts) in enumerate(todo, 1):
                try:
                    new = fetch_one(
                        code,
                        start_ts.strftime("%Y-%m-%d"),
                        end_ts.strftime("%Y-%m-%d"),
                        adjustflag,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{code} ({exc})")
                    continue

                if new.empty:
                    empty.append(code)
                    continue

                old = pd.read_csv(path, encoding="utf-8-sig")
                old["date"] = pd.to_datetime(old["date"], errors="coerce")
                old = old.dropna(subset=["date"])
                new = new.reindex(columns=old.columns)
                merged = (
                    pd.concat([old, new], ignore_index=True)
                    .drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                added = max(len(merged) - len(old), 0)
                tmp_path = f"{path}.tmp"
                merged.to_csv(tmp_path, index=False, encoding="utf-8-sig")
                os.replace(tmp_path, path)
                updated += 1
                appended += added

                if i % 10 == 0 or i == len(todo):
                    print(
                        f"[kdata] 进度 {i}/{len(todo)}："
                        f"更新 {updated} 只，新增 {appended} 条",
                        flush=True,
                    )
                time.sleep(0.05)
        finally:
            logout()

        print(f"[kdata] 更新完成：{updated} 只，新增 {appended} 条", flush=True)
        if empty:
            preview = ", ".join(empty[:10])
            more = " ..." if len(empty) > 10 else ""
            print(
                f"[kdata] {len(empty)} 只无新增行情"
                f"（停牌/退市/非交易日）：{preview}{more}",
                flush=True,
            )
        if failed:
            preview = "\n  ".join(failed[:10])
            more = f"\n  ... 共 {len(failed)} 只" if len(failed) > 10 else ""
            raise RuntimeError(
                f"以下股票行情更新失败，终止回测：\n  {preview}{more}"
            )
    else:
        print("[kdata] 行情数据已覆盖回测终点，无需更新", flush=True)

    latest: Optional[pd.Timestamp] = None
    for code in required:
        path = _price_path(code, k_dir)
        if not os.path.exists(path):
            continue
        try:
            last = _last_price_date(path)
        except Exception:  # noqa: BLE001
            continue
        if last is not None and (latest is None or last > latest):
            latest = last
    requested_end = max(required.values()) if required else None
    if latest is not None and requested_end is not None:
        if latest < requested_end:
            print(
                f"[kdata] 注意：行情最新日期 {latest.date()} 早于请求终点 "
                f"{requested_end.date()}（数据源尚未发布或为非交易日）",
                flush=True,
            )
        else:
            print(
                f"[kdata] 行情最新日期 {latest.date()}"
                f"（请求终点 {requested_end.date()}）",
                flush=True,
            )


def validate_price_files(
    df: pd.DataFrame,
    start_dates: List[pd.Timestamp],
    k_dir: str = K_DIR,
) -> None:
    """校验行情文件存在、可读，且每期股票在区间首个交易日已有数据。"""
    first_dates: Dict[str, pd.Timestamp] = {}
    all_dates = set()
    missing, invalid = [], []

    for code in collect_codes(df):
        try:
            path = _price_path(code, k_dir)
        except ValueError as exc:
            raise ValueError(f"无法识别的股票代码 {code!r}: {exc}") from exc

        if not os.path.exists(path):
            missing.append(f"{code} -> {path}")
            continue

        try:
            raw = pd.read_csv(path, usecols=["date"])
        except Exception as exc:
            invalid.append(f"{path} ({exc})")
            continue

        parsed = pd.to_datetime(raw["date"], errors="coerce").dropna()
        if parsed.empty:
            invalid.append(path)
            continue

        first_dates[code] = parsed.min()
        all_dates.update(parsed.tolist())

    if missing:
        preview = "\n  ".join(missing[:10])
        more = f"\n  ... 共 {len(missing)} 个" if len(missing) > 10 else ""
        raise ValueError(f"缺少以下行情文件：\n  {preview}{more}")

    if invalid:
        preview = "\n  ".join(invalid[:10])
        more = f"\n  ... 共 {len(invalid)} 个" if len(invalid) > 10 else ""
        raise ValueError(f"以下行情文件无法读取或没有有效日期：\n  {preview}{more}")

    calendar = pd.DatetimeIndex(sorted(all_dates))

    problems = []
    for i in range(len(df)):
        start_ts = start_dates[i]
        pos = calendar.searchsorted(start_ts, side="left")
        if pos >= len(calendar):
            problems.append(f"{start_ts.date()}: 行情数据中没有此后交易日")
            continue

        first_trading_day = calendar[pos]
        for code in df.iloc[i]["target_code"]:
            first = first_dates.get(code)
            if first is not None and first > first_trading_day:
                problems.append(
                    f"{start_ts.date()}: {code} 首个交易日 {first.date()} "
                    f"晚于区间首个交易日 {first_trading_day.date()}"
                )

    if problems:
        preview = "\n  ".join(problems[:10])
        more = f"\n  ... 共 {len(problems)} 条" if len(problems) > 10 else ""
        raise ValueError(
            "以下股票在区间起点尚无行情（会导致该期实际起点被静默后移）：\n  "
            f"{preview}{more}"
        )


def prep_periods(
    df: pd.DataFrame,
    query_dates: pd.Series,
    start_dates: List[pd.Timestamp],
    end_date: str,
    k_dir: str = K_DIR,
) -> List[dict]:
    """按调仓日构造持仓区间：每期起点取自 start_dates，最后一期在 end_date 结束。"""
    periods = []

    for i in range(len(df)):
        start_date = start_dates[i].strftime("%Y-%m-%d")
        if i + 1 < len(df):
            end_date_i = query_dates.iloc[i + 1].strftime("%Y-%m-%d")
        else:
            end_date_i = end_date

        code_list = [normalize_code(code) for code in df.iloc[i]["target_code"]]
        periods.append({
            "name": f"period_{i+1}",
            "csv_files": [
                os.path.join(k_dir, code.replace(".", "_") + ".csv")
                for code in code_list
            ],
            "weights": None,
            "start_date": start_date,
            "end_date": end_date_i,
        })

    return periods


# ---------------------------------------------------------------------------
# 基准文件与备份
# ---------------------------------------------------------------------------
def resolve_benchmark(explicit: Optional[str] = None) -> Optional[str]:
    """显式路径优先；否则自动查找 hs300etf_510300_*_performance.csv 的最新文件。"""
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"指定的基准文件不存在: {explicit}")
        return explicit

    legacy = "hs300etf_510300_performance.csv"
    if os.path.exists(legacy):
        return legacy

    candidates = glob.glob("hs300etf_510300_*_performance.csv")
    if candidates:
        return max(candidates, key=os.path.getmtime)

    return None


def backup_if_exists(path: str) -> Optional[str]:
    """若文件存在，复制为带时间戳的 .bak 文件并返回备份路径。"""
    if not os.path.exists(path):
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{path}.bak-{stamp}"
    shutil.copy2(path, backup_path)
    return backup_path


def atomic_copy(src: str, dst: str) -> None:
    """通过临时文件 + os.replace 原子替换目标文件。"""
    tmp_path = f"{dst}.tmp"
    shutil.copyfile(src, tmp_path)
    os.replace(tmp_path, dst)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM 选股 + 多期等权买入持有回测流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-prep", action="store_true",
        help="跳过 LLM 选股，直接复用 --target-csv 指定的文件",
    )
    parser.add_argument(
        "--target-csv", default=DEFAULT_TARGET_CSV,
        help="选股结果 CSV 路径（prep 写入、eval 读取）",
    )
    parser.add_argument(
        "--cache-file", default=DEFAULT_CACHE_FILE,
        help="LLM 判定缓存（JSONL，追加写入，可断点续跑）",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="不读写磁盘缓存（本次运行内仍按唯一代码去重）",
    )
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="忽略已有缓存，重新调用 LLM（新结果追加写回缓存）",
    )
    parser.add_argument(
        "--publish", action="store_true",
        help=f"全流程成功后，原子替换 {CANONICAL_TARGET_CSV}",
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START_DATE,
        help="回测起始日期；默认固定为 2021-01-01（2021h1 使用 2020h2 权重）",
    )
    parser.add_argument(
        "--end-date", default=default_end_date(),
        help=f"回测结束日期；默认运行当天，且不晚于 {MAX_END_DATE}",
    )
    parser.add_argument(
        "--no-update-k", action="store_true",
        help="跳过回测前的 K 线增量更新（离线或数据已就绪时使用）",
    )
    parser.add_argument(
        "--benchmark", default=None,
        help="基准 CSV 路径；默认自动查找 hs300etf_510300_*_performance.csv 的最新文件",
    )
    parser.add_argument(
        "--risk-free-rate", type=float, default=0.02,
        help="年化无风险利率（小数），用于标准夏普；默认 0.02，与 eval_hs300_perform.py 一致",
    )
    parser.add_argument(
        "--weight-glob", default=WEIGHT_GLOB,
        help="沪深300权重 CSV 通配符；每个调仓日取上一期期末的持仓作为本期期初成分股",
    )
    parser.add_argument(
        "--business-csv", default=BUSINESS_CSV,
        help="主营业务 CSV",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 基准文件提前解析，避免昂贵的选股跑完后才发现路径问题
    benchmark_file = resolve_benchmark(args.benchmark)
    if benchmark_file:
        print(f"[benchmark] 使用 {benchmark_file}")
    else:
        print("[benchmark] 警告：未找到基准文件，回测将不包含基准与 IR 比较")

    # 1) 选股（或复用）
    if args.skip_prep:
        if not os.path.exists(args.target_csv):
            raise FileNotFoundError(
                f"--skip-prep 需要已存在的文件: {args.target_csv}"
            )
        print(f"[prep] 跳过选股，复用 {args.target_csv}")
    else:
        cache_file = None if args.no_cache else args.cache_file
        target_code_dict = prep_target_codes(
            QUERY_DATES,
            args.weight_glob,
            args.business_csv,
            cache_file=cache_file,
            refresh_cache=args.refresh_cache,
        )
        write_target_code_csv(target_code_dict, args.target_csv)

    # 2) 读取并按回测区间裁剪
    start_ts = parse_date_arg(args.start_date, "--start-date")
    end_ts = parse_date_arg(args.end_date, "--end-date")
    if end_ts is None:
        raise ValueError("必须提供 --end-date")

    min_start_ts = pd.Timestamp(DEFAULT_START_DATE)
    if start_ts is not None and start_ts < min_start_ts:
        print(
            f"[eval] --start-date ({start_ts.date()}) 早于 {DEFAULT_START_DATE}，"
            f"已按 {DEFAULT_START_DATE} 处理",
            flush=True,
        )
        start_ts = min_start_ts

    max_end_ts = pd.Timestamp(MAX_END_DATE)
    if end_ts > max_end_ts:
        print(
            f"[eval] --end-date ({end_ts.date()}) 晚于 {MAX_END_DATE}，"
            f"已截断到 {MAX_END_DATE}",
            flush=True,
        )
        end_ts = max_end_ts

    df = read_target_code_csv(args.target_csv)
    query_dates = validate_target_df(df)
    df, query_dates, effective_start = trim_to_backtest_range(
        df, query_dates, start_ts, end_ts
    )
    start_dates = period_start_dates(query_dates, effective_start)
    periods = prep_periods(
        df, query_dates, start_dates, end_ts.strftime("%Y-%m-%d")
    )
    print(
        f"[eval] 请求回测区间 {start_dates[0].date()} ~ {end_ts.date()}，"
        f"共 {len(periods)} 期"
        f"（首个调仓日 {query_dates.iloc[0].date()}）",
        flush=True,
    )

    last_query = query_dates.iloc[-1]
    if end_ts > last_query:
        print(
            f"[eval] 终止日期 {end_ts.date()} 晚于最后一个调仓日 "
            f"{last_query.date()}，最后一期沿用该日的持仓与选股结果",
            flush=True,
        )

    # 回测前补齐行情数据，避免终点被静默截断
    if args.no_update_k:
        print("[kdata] 已跳过 K 线增量更新（--no-update-k）", flush=True)
    else:
        update_k_data_if_needed(df, periods)
    validate_price_files(df, start_dates)

    # 3) 备份会被覆盖的日志与净值图
    for path in (DEFAULT_LOG_FILE, DEFAULT_PLOT_FILE):
        backup_path = backup_if_exists(path)
        if backup_path:
            print(f"[backup] {path} -> {backup_path}")

    # 4) 回测
    nav, _ = run_backtest(
        periods,
        align="outer",
        plot=True,
        plot_file=DEFAULT_PLOT_FILE,
        benchmark_file=benchmark_file,
        log_file=DEFAULT_LOG_FILE,
        risk_free_rate=args.risk_free_rate,
    )

    # 5) 可选：发布为 canonical 目标文件（仅在回测确实产生结果后）
    if args.publish:
        if nav is None:
            raise RuntimeError("回测未产生有效结果，跳过发布 target_code.csv")
        backup_path = backup_if_exists(CANONICAL_TARGET_CSV)
        if backup_path:
            print(f"[backup] {CANONICAL_TARGET_CSV} -> {backup_path}")
        atomic_copy(args.target_csv, CANONICAL_TARGET_CSV)
        print(f"[publish] 已更新 {CANONICAL_TARGET_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
