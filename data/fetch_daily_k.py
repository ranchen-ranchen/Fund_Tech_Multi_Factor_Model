"""
fetch_daily_k.py
基于 baostock 批量获取 A 股日 K 线数据（前复权）

依赖:
    pip install baostock pandas

输出:
    data/all_daily_k.csv          # 所有股票合并后的长表
    data/all_daily_k.parquet      # 同上（需要 pyarrow / fastparquet）
    data/individual/sh.600519.csv # 每只股票单独一份（可选）
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pandas as pd
import baostock as bs

# ============================ 配置区 ============================
# 股票代码列表：支持 '600519'、'sh.600519'、'SH600519'、'600519.SH' 等写法
STOCK_CODES: List[str] = [
    "600519",          # 贵州茅台
    "000001",          # 平安银行
    "300750",          # 宁德时代
    "601318",          # 中国平安
    "sh.688981",       # 中芯国际
]

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

FREQUENCY = "d"        # d=日线, w=周线, m=月线
ADJUST_FLAG = "1"      # 1=后复权, 2=前复权, 3=不复权  ← 前复权

OUT_DIR = "stock_daily_k"
SAVE_INDIVIDUAL = True   # 是否额外保存每只股票单独的 csv
MAX_RETRY = 3            # 单只股票失败重试次数
RETRY_SLEEP = 1.5        # 重试间隔基数（秒）
# ===============================================================

# baostock 日线可用字段
FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,turn,pctChg,tradestatus,isST"
)

FLOAT_COLS = ["open", "high", "low", "close", "preclose",
              "volume", "amount", "turn", "pctChg"]
INT_COLS = ["tradestatus", "isST"]

# 输出列顺序
OUT_COLS = ["date", "code", "open", "high", "low", "close", "preclose",
            "volume", "amount", "turn", "pctChg", "tradestatus", "isST"]


# ----------------------------------------------------------------------
# 代码规范化
# ----------------------------------------------------------------------
def normalize_code(code: str) -> str:
    """
    将各种写法的股票代码统一为 baostock 要求的格式：sh.600000 / sz.000001 / bj.430047
    """
    raw = str(code).strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("股票代码为空")

    # 形如 sh.600000 或 600000.sh
    if "." in raw:
        parts = raw.split(".")
        if len(parts) != 2:
            raise ValueError(f"无法识别的股票代码: {code}")
        a, b = parts
        if a in ("sh", "sz", "bj"):
            return f"{a}.{b}"
        if b in ("sh", "sz", "bj"):
            return f"{b}.{a}"
        raise ValueError(f"无法识别的股票代码: {code}")

    # 形如 sh600000 / sz000001
    for pre in ("sh", "sz", "bj"):
        if raw.startswith(pre) and raw[2:].isdigit():
            return f"{pre}.{raw[2:]}"

    if not raw.isdigit():
        raise ValueError(f"无法识别的股票代码: {code}")

    # 纯数字：按前缀推断交易所
    if raw.startswith(("60", "68", "51", "58", "11", "90", "50")):
        return f"sh.{raw}"
    if raw.startswith(("00", "30", "12", "15", "16", "18", "20", "39", "13")):
        return f"sz.{raw}"
    if raw.startswith(("43", "83", "87", "88", "92")):
        return f"bj.{raw}"

    # 兜底
    return f"sh.{raw}" if raw[0] in ("5", "6", "9") else f"sz.{raw}"


# ----------------------------------------------------------------------
# baostock 登入 / 登出
# ----------------------------------------------------------------------
def login() -> None:
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_code} - {lg.error_msg}")
    print(f"[INFO] baostock 登录成功 (user_id={lg.user_id})")


def logout() -> None:
    try:
        bs.logout()
        print("[INFO] baostock 已登出")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 登出异常: {e}")


# ----------------------------------------------------------------------
# 单只股票数据获取
# ----------------------------------------------------------------------
def fetch_one(bs_code: str,
              start_date: str,
              end_date: str,
              frequency: str = FREQUENCY,
              adjustflag: str = ADJUST_FLAG,
              max_retry: int = MAX_RETRY) -> pd.DataFrame:
    """
    获取单只股票的日 K 数据，返回规范化后的 DataFrame（可能为空）。
    带重试机制。
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retry + 1):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                adjustflag=adjustflag,
            )

            if rs.error_code != "0":
                raise RuntimeError(
                    f"baostock 返回错误 {rs.error_code}: {rs.error_msg}"
                )

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            df = pd.DataFrame(rows, columns=rs.fields)
            return _clean(df, bs_code)

        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retry:
                wait = RETRY_SLEEP * attempt
                print(f"    [WARN] {bs_code} 第 {attempt} 次失败({e})，{wait:.1f}s 后重试")
                time.sleep(wait)

    raise RuntimeError(f"{bs_code} 获取失败，已重试 {max_retry} 次: {last_err}")


def _clean(df: pd.DataFrame, bs_code: str) -> pd.DataFrame:
    """类型转换 + 排序 + 去重"""
    if df is None or df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    df = df.copy()

    # 数值列转换（baostock 返回空字符串时 to_numeric 会得到 NaN）
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # 日期列
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 剔除停牌日（tradestatus == 0）——如需保留停牌数据，注释掉下面两行
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == 1]

    df = (df.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))

    # 补齐缺失列，统一列顺序
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS]


# ----------------------------------------------------------------------
# 批量获取
# ----------------------------------------------------------------------
def get_daily_data(codes: List[str],
                   start_date: str = START_DATE,
                   end_date: str = END_DATE,
                   out_dir: str = OUT_DIR,
                   save_individual: bool = SAVE_INDIVIDUAL) -> pd.DataFrame:
    """
    批量获取日 K 数据，返回合并后的长表 DataFrame，并落盘。
    """
    if not codes:
        raise ValueError("股票代码列表为空")

    os.makedirs(out_dir, exist_ok=True)
    ind_dir = os.path.join(out_dir, "individual")
    if save_individual:
        os.makedirs(ind_dir, exist_ok=True)

    login()
    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    try:
        total = len(codes)
        for i, raw_code in enumerate(codes, start=1):
            try:
                bs_code = normalize_code(raw_code)
            except ValueError as e:
                print(f"[{i}/{total}] 跳过 {raw_code}: {e}")
                failed.append(str(raw_code))
                continue

            print(f"[{i}/{total}] 获取 {bs_code} ...", end=" ", flush=True)
            try:
                df = fetch_one(bs_code, start_date, end_date)
            except Exception as e:  # noqa: BLE001
                print(f"失败: {e}")
                failed.append(bs_code)
                continue

            if df.empty:
                print("无数据（可能区间内未上市或全部停牌）")
                failed.append(bs_code)
                continue

            print(f"{len(df)} 条  {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
            frames.append(df)

            if save_individual:
                fname = os.path.join(ind_dir, f"{bs_code.replace('.', '_')}.csv")
                df.to_csv(fname, index=False, encoding="utf-8-sig")

            time.sleep(0.05)  # 轻微限速，避免请求过密

    finally:
        logout()

    if not frames:
        print("[WARN] 未获取到任何数据")
        return pd.DataFrame(columns=OUT_COLS)

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["code", "date"])
                .reset_index(drop=True))

    # 落盘：合并表
    csv_path = os.path.join(out_dir, "all_daily_k.csv")
    all_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 合并数据已保存: {csv_path}  形状={all_df.shape}")

    try:
        pq_path = os.path.join(out_dir, "all_daily_k.parquet")
        all_df.to_parquet(pq_path, index=False)
        print(f"[INFO] Parquet 已保存: {pq_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Parquet 保存失败（可忽略，安装 pyarrow 后可用）: {e}")

    if failed:
        print(f"[WARN] 以下标的未成功获取: {failed}")

    return all_df


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = get_daily_data(
        codes=STOCK_CODES,
        start_date=START_DATE,
        end_date=END_DATE,
        out_dir=OUT_DIR,
        save_individual=SAVE_INDIVIDUAL,
    )

    if not data.empty:
        print("\n===== 数据概览 =====")
        print(data.head(10).to_string(index=False))
        print("\n各股票记录数：")
        print(data.groupby("code").size().to_string())
        print("\n字段类型：")
        print(data.dtypes.to_string())