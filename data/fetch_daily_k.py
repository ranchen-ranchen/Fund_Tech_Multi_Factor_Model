"""
fetch_daily_k.py
基于 baostock 批量获取 A 股日 K 线数据（前复权）

特性：
    - 断点续传：以 out_dir/individual/*.csv 作为缓存，重启自动跳过已抓取标的
    - 单只股票失败重试
    - 结果汇总为 all_daily_k.csv / all_daily_k.parquet

依赖:
    pip install baostock pandas
    # 可选：pip install pyarrow   （保存 parquet 用）

输出:
    stock_daily_k/all_daily_k.csv          # 所有股票合并后的长表
    stock_daily_k/all_daily_k.parquet      # 同上（需要 pyarrow / fastparquet）
    stock_daily_k/individual/sh_600519.csv # 每只股票单独一份（同时作为缓存）
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import pandas as pd
import baostock as bs

# ============================ 配置区 ============================
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
ADJUST_FLAG = "1"      # 1=后复权, 2=前复权, 3=不复权

OUT_DIR = "stock_daily_k"
SAVE_INDIVIDUAL = True   # 是否额外保存每只股票单独的 csv（断点续传依赖它）
RESUME = True            # ★ 断点续传开关
MAX_RETRY = 3            # 单只股票失败重试次数
RETRY_SLEEP = 1.5        # 重试间隔基数（秒）
# ===============================================================

FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,turn,pctChg,tradestatus,isST"
)

FLOAT_COLS = ["open", "high", "low", "close", "preclose",
              "volume", "amount", "turn", "pctChg"]
INT_COLS = ["tradestatus", "isST"]

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

    for pre in ("sh", "sz", "bj"):
        if raw.startswith(pre) and raw[2:].isdigit():
            return f"{pre}.{raw[2:]}"

    if not raw.isdigit():
        raise ValueError(f"无法识别的股票代码: {code}")

    if raw.startswith(("60", "68", "51", "58", "11", "90", "50")):
        return f"sh.{raw}"
    if raw.startswith(("00", "30", "12", "15", "16", "18", "20", "39", "13")):
        return f"sz.{raw}"
    if raw.startswith(("43", "83", "87", "88", "92")):
        return f"bj.{raw}"

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
# 缓存辅助
# ----------------------------------------------------------------------
def _individual_path(ind_dir: str, bs_code: str) -> str:
    """个股 csv 路径（也用作缓存文件）。"""
    return os.path.join(ind_dir, f"{bs_code.replace('.', '_')}.csv")


def _load_cached(path: str) -> Optional[pd.DataFrame]:
    """
    尝试从缓存文件加载数据。
    - 文件不存在 / 空 / 解析失败 → 返回 None
    - 成功 → 返回规范化后的 DataFrame
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 读取缓存失败 {path}: {e}")
        return None

    if df is None or df.empty:
        return None

    # 类型规范化，保持与其他来源一致
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return None

    # 补齐列顺序
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS].reset_index(drop=True)


# ----------------------------------------------------------------------
# 单只股票数据获取
# ----------------------------------------------------------------------
def fetch_one(bs_code: str,
              start_date: str,
              end_date: str,
              frequency: str = FREQUENCY,
              adjustflag: str = ADJUST_FLAG,
              max_retry: int = MAX_RETRY) -> pd.DataFrame:
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
    if df is None or df.empty:
        return pd.DataFrame(columns=OUT_COLS)

    df = df.copy()

    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] == 1]

    df = (df.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))

    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS]


# ----------------------------------------------------------------------
# 批量获取（★ 支持断点续传）
# ----------------------------------------------------------------------
def get_daily_data(codes: List[str],
                   start_date: str = START_DATE,
                   end_date: str = END_DATE,
                   out_dir: str = OUT_DIR,
                   save_individual: bool = SAVE_INDIVIDUAL,
                   resume: bool = RESUME) -> pd.DataFrame:
    """
    批量获取日 K 数据，返回合并后的长表 DataFrame，并落盘。

    断点续传逻辑：
      - 若 resume=True 且 out_dir/individual/{code}.csv 已存在且非空，
        则直接加载缓存、跳过网络请求；
      - 抓取成功的标的会立即写入 individual csv，供后续运行复用；
      - 因此中断后重新运行此函数即可"接着上一次的进度继续"。
    """
    if not codes:
        raise ValueError("股票代码列表为空")

    os.makedirs(out_dir, exist_ok=True)
    ind_dir = os.path.join(out_dir, "individual")

    # 断点续传依赖个股缓存文件 → 开启续传时强制保存 individual
    if resume and not save_individual:
        print("[INFO] 断点续传需要个股文件作为缓存，已自动开启 save_individual")
        save_individual = True
    if save_individual:
        os.makedirs(ind_dir, exist_ok=True)

    login()
    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    cache_hits: List[str] = []
    fetched: List[str] = []

    try:
        total = len(codes)
        for i, raw_code in enumerate(codes, start=1):
            try:
                bs_code = normalize_code(raw_code)
            except ValueError as e:
                print(f"[{i}/{total}] 跳过 {raw_code}: {e}")
                failed.append(str(raw_code))
                continue

            fname = _individual_path(ind_dir, bs_code)

            # ---------- ① 断点续传：优先命中缓存 ----------
            if resume:
                cached = _load_cached(fname)
                if cached is not None:
                    print(f"[{i}/{total}] [cache] {bs_code}  命中缓存，{len(cached)} 条 "
                          f"({cached['date'].iloc[0].date()} ~ {cached['date'].iloc[-1].date()})")
                    frames.append(cached)
                    cache_hits.append(bs_code)
                    continue

            # ---------- ② 走网络抓取 ----------
            print(f"[{i}/{total}] [fetch] {bs_code} ...", end=" ", flush=True)
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
            fetched.append(bs_code)

            # ---------- ③ 立即落盘，作为下次运行的缓存 ----------
            if save_individual:
                try:
                    df.to_csv(fname, index=False, encoding="utf-8-sig")
                except Exception as e:  # noqa: BLE001
                    print(f"    [WARN] 写入缓存失败 {fname}: {e}")

            time.sleep(0.05)  # 轻微限速

    finally:
        logout()

    # ---------- 汇总 ----------
    print(f"\n[INFO] 抓取 {len(fetched)} 只，缓存命中 {len(cache_hits)} 只，失败 {len(failed)} 只")

    if not frames:
        print("[WARN] 未获取到任何数据")
        return pd.DataFrame(columns=OUT_COLS)

    all_df = (pd.concat(frames, ignore_index=True)
                .sort_values(["code", "date"])
                .reset_index(drop=True))

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
        resume=RESUME,
    )

    if not data.empty:
        print("\n===== 数据概览 =====")
        print(data.head(10).to_string(index=False))
        print("\n各股票记录数：")
        print(data.groupby("code").size().to_string())
        print("\n字段类型：")
        print(data.dtypes.to_string())


