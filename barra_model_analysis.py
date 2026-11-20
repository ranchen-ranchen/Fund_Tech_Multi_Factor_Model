import glob
import os
import re
import socket
import time
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from utils.text_utils import (
    get_stock_data,
    normalize_code,
    query_pb_mrq,
    read_hs300_weight,
    read_target_code_weight,
)
from strategy.calc_volatility_exposure import compute_volatility_exposure
from strategy.calc_momentum_exposure import momentum_exposure_histogram
from strategy.calc_size_factor_exposure import calc_size_factor_exposure
from strategy.calc_excess_return import calc_excess_return_distribution
from strategy.calc_value_exposure import value_factor_exposure
from strategy.factor_return_cross_sec_regression import build_risk_model_cross_section
from utils.plot_utils import plot_clustered_bars


##############
# 测试区间：起始日期固定为 2021-01-01，终止日期为运行当天，但不晚于 2026-12-31。
#
# 归因口径（point-in-time）：
#   - 每期因子暴露只使用期初（调仓日）及之前的数据：
#       size / value 取调仓日当天（或最近可用交易日）的截面；
#       momentum / volatility 取上一期 [前一调仓日, 调仓日] 的数据。
#   - 基准成分股与权重只使用 data/hs300_weight_*.csv：
#     上一期期末的持仓文件作为本期期初的基准持仓。
#     不使用 fetch_hs300_constituents.py / hs300_constituents_2021_2026.csv。
#   - 基准收益 = 期初基准持仓的加权收益（权重合计不足 100% 的部分视为现金，
#     现金收益按 0 计算）；不再使用 510300 ETF 的实际收益序列作为基准。
#   - 收益区间为 [调仓日, 下一调仓日]；若股票/基准数据不同步，
#     取二者共同的最后交易日作为本期有效终点。
#   - 2021h1 使用 2020h2 的期初权重；因子窗口为 2020-07-01 ~ 2021-01-01，
#     size 取 2020-12-31，value 取 2020-12-31 的 PB。
##############
TEST_START = date(2021, 1, 1)
TEST_END_LIMIT = date(2026, 12, 31)
PRICE_FETCH_START = '2020-07-01'
PRICE_HISTORY_START = '2020-07-01'
PRICE_HISTORY_MARKER = 'data/.price_history_2020h2.done'
WEIGHT_GLOB = 'data/hs300_weight_*h*.csv'
VALUATION_CSV = 'data/hs300_valuation_6m.csv'
CASH_RETURN = 0.0   # 基准中现金仓位的收益假设

# 每次运行前自动增量更新 hfq / bfq 行情；离线运行时设为 False 可跳过
UPDATE_PRICE_DATA = True
PRICE_DATA_SPECS = (
    ('后复权 hfq', 'stock_daily_k_hfq', '1'),
    ('不复权 bfq', 'stock_daily_k_bfq', '3'),
)

FACTOR_NAMES = ['size', 'value', 'momentum', 'volatility']
EMPTY_PRICE_COLS = ['date', 'code', 'open', 'high', 'low', 'close',
                    'preclose', 'volume', 'amount', 'turn', 'pctChg']


def get_test_period() -> tuple[date, date]:
    """返回 (起始日期, 终止日期)：起点固定 2021-01-01，终点为运行当天且不晚于 2026-12-31。"""
    today = date.today()
    end_date = min(today, TEST_END_LIMIT)
    if end_date < TEST_START:
        raise RuntimeError(
            f"系统日期 {today} 早于固定起始日期 {TEST_START}，无法运行测试。"
        )
    return TEST_START, end_date


def build_rebalance_dates(start: date, end: date) -> list[str]:
    """生成半年调仓日列表（6/30、12/31），首日固定为 start，末日固定为 end。"""
    dates = [start]
    for year in range(start.year, end.year + 1):
        for month, day in ((6, 30), (12, 31)):
            d = date(year, month, day)
            if start < d < end:
                dates.append(d)
    if end > start:
        dates.append(end)
    return [d.isoformat() for d in dates]


def weight_asof_files(weight_glob: str = WEIGHT_GLOB) -> list[tuple[pd.Timestamp, str]]:
    """返回 [(期末日期, weight 文件路径)]，按期末日期升序。

    文件名约定为 hs300_weight_{year}h{1|2}.csv：
    h1 的期末为当年 6/30，h2 的期末为当年 12/31。
    """
    files: list[tuple[pd.Timestamp, str]] = []
    for path in glob.glob(weight_glob):
        m = re.search(r'hs300_weight_(\d{4})h([12])\.csv$', os.path.basename(path))
        if not m:
            print(f"[warn] 跳过无法解析文件名的权重文件: {path}")
            continue
        year, half = int(m.group(1)), int(m.group(2))
        asof = pd.Timestamp(f'{year}-06-30' if half == 1 else f'{year}-12-31')
        files.append((asof, path))
    if not files:
        raise FileNotFoundError(f'未找到任何沪深300权重文件: {weight_glob}')
    files.sort(key=lambda x: x[0])
    return files


def weight_file_for(asof_date: str) -> str:
    """返回不晚于 asof_date 的最近一期 weight 文件（上一期期末持仓）。

    本期期初同时是上一期期末，因此：
      2021-01-01 -> 2020h2，2021-06-30 -> 2021h1，2021-12-31 -> 2021h2，...
    """
    ts = pd.Timestamp(asof_date)
    candidates = [(a, p) for a, p in weight_asof_files() if a <= ts]
    if not candidates:
        raise FileNotFoundError(
            f'未找到 {asof_date} 及之前任何一期的沪深300权重文件')
    asof, path = candidates[-1]
    if asof != ts:
        print(f"[info] {asof_date} 使用上一期期末 {asof.date()} 的持仓：{path}")
    return path


def valuation_date_for(asof_date: str, csv_path: str = VALUATION_CSV) -> str:
    """返回不晚于 asof_date 的最近一期估值日期（point-in-time）。"""
    dates = pd.to_datetime(
        pd.read_csv(csv_path, usecols=['target_date'])['target_date'],
        errors='coerce',
    ).dropna()
    earlier = dates[dates <= pd.Timestamp(asof_date)]
    if earlier.empty:
        raise ValueError(f'估值文件 {csv_path} 中没有 {asof_date} 及之前的任何数据')
    val_date = earlier.max().date().isoformat()
    if val_date != asof_date:
        print(f"[info] {asof_date} 估值尚未披露，使用 {val_date}：{csv_path}")
    return val_date


def load_bench_weights(csv_path: str) -> pd.Series:
    """读取基准权重，统一代码格式并按代码合并重复行，权重转为小数。"""
    w = read_hs300_weight(csv_path)
    w.index = [normalize_code(str(c)) for c in w.index]
    dup = w.index[w.index.duplicated()].unique().tolist()
    if dup:
        print(f"[warn] {csv_path} 中存在重复代码 {dup}，已按代码合并权重")
    return w.groupby(level=0)['weight'].sum() / 100.0


def load_target_weights(csv_path: str) -> dict[str, pd.Series]:
    """读取目标持仓文件，返回 {query_date: Series(code -> weight)}。"""
    df = pd.read_csv(csv_path, dtype=str)
    required_cols = {'query_date', 'target_code'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"目标持仓文件 {csv_path} 缺少列: {sorted(missing)}")

    weight_dfs = read_target_code_weight(csv_path)
    result: dict[str, pd.Series] = {}
    for (_, row), wdf in zip(df.iterrows(), weight_dfs):
        qdate = str(row['query_date']).strip()
        if wdf.empty:
            print(f"[warn] {csv_path} 中 {qdate} 的目标持仓为空，跳过该期")
            continue
        s = wdf['weight'].copy()
        s.index = [normalize_code(str(c)) for c in s.index]
        s = s.groupby(level=0).sum()
        if qdate in result:
            raise ValueError(f"目标持仓文件 {csv_path} 中 query_date 重复: {qdate}")
        result[qdate] = s
    return result


def required_end_by_code(code_lists: list, end_dates: list) -> dict:
    """每只股票需要覆盖到的日期：它所在最后一期的期末。"""
    required = {}
    for codes, end in zip(code_lists, end_dates):
        for code in (normalize_code(c) for c in codes):
            if end > required.get(code, ''):
                required[code] = end
    return required


def price_file_path(out_dir: str, code: str) -> str:
    return os.path.join(out_dir, 'individual', f"{code.replace('.', '_')}.csv")


def last_cached_date(path: str):
    """个股行情缓存中的最后一个交易日；文件不存在或无效时返回 None。"""
    if not os.path.exists(path):
        return None
    try:
        dates = pd.read_csv(path, usecols=['date'], encoding='utf-8-sig')['date']
    except Exception:  # noqa: BLE001
        return None
    dates = pd.to_datetime(dates, errors='coerce').dropna()
    return dates.max().date().isoformat() if not dates.empty else None


def load_stock_data(path: str) -> pd.DataFrame:
    """读取个股行情；文件缺失或损坏时返回空表，由因子函数按缺失值处理。"""
    if not os.path.exists(path):
        return pd.DataFrame(columns=EMPTY_PRICE_COLS)
    try:
        return get_stock_data(path)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 读取行情失败 {path}: {e}")
        return pd.DataFrame(columns=EMPTY_PRICE_COLS)


def probe_vendor_last_date(end_date: str, probe_code: str = 'sh.600000') -> str:
    """用一只正常交易的股票探测数据源当前能提供的最新交易日。"""
    from data.fetch_daily_k import fetch_one, login, logout
    start = (pd.Timestamp(end_date) - pd.Timedelta(days=30)).date().isoformat()
    login()
    try:
        df = fetch_one(probe_code, start, end_date, adjustflag='1')
    finally:
        logout()
    if df.empty:
        return end_date
    last = pd.to_datetime(df['date'], errors='coerce').max()
    return end_date if pd.isna(last) else last.date().isoformat()


def rebuild_combined_csv(out_dir: str) -> None:
    """用 individual/ 下全部个股文件重建合并表。

    get_daily_data 只会把本次更新的子集写入 all_daily_k.csv，需重建以保证完整。
    """
    files = sorted(glob.glob(os.path.join(out_dir, 'individual', '*.csv')))
    if not files:
        return
    combined = (pd.concat(
        (pd.read_csv(f, encoding='utf-8-sig') for f in files),
        ignore_index=True,
    ).sort_values(['code', 'date']).reset_index(drop=True))
    path = os.path.join(out_dir, 'all_daily_k.csv')
    combined.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"[数据更新] 已重建合并行情 {path}（{len(files)} 只，{len(combined)} 行）")


def update_price_data(required: dict, start_date: str, end_date: str,
                      vendor_last: str) -> None:
    """增量更新 hfq / bfq 日线到 min(所需期末, 数据源最新交易日)。

    只请求未覆盖目标的股票；当天已尝试更新但仍不达标（退市/停牌）的不重复请求。
    """
    from data.fetch_daily_k import get_daily_data
    today = date.today().isoformat()
    for label, out_dir, adjustflag in PRICE_DATA_SPECS:
        stale = []
        for code, needed in required.items():
            target = min(needed, vendor_last)
            path = price_file_path(out_dir, code)
            last = last_cached_date(path)
            if last is not None and last >= target:
                continue
            if (last is not None
                    and date.fromtimestamp(os.path.getmtime(path)).isoformat() >= today):
                continue   # 今日已更新过但仍不达标（退市/停牌），避免重复请求
            stale.append(code)
        if not stale:
            print(f"[数据更新] {label}: 无需更新（{len(required)} 只，"
                  f"数据源最新 {vendor_last}）")
            continue
        print(f"[数据更新] {label}: 更新 {len(stale)}/{len(required)} 只"
              f"（{start_date} ~ {end_date}，数据源最新 {vendor_last}）")
        get_daily_data(codes=sorted(stale), adjustflag=adjustflag,
                       start_date=start_date, end_date=end_date,
                       out_dir=out_dir, resume=False)
        rebuild_combined_csv(out_dir)


def backfill_price_history(universe: list, start_date: str,
                           marker: str = PRICE_HISTORY_MARKER) -> None:
    """为第一期股票池补齐 start_date 到缓存首日之前的行情缺口。

    2021h1 的因子窗口需要 2020-07-01 ~ 2020-12-31 的行情，而现有缓存大多
    从 2021-01-04 开始。这里只抓每只股票缺失的前缀区间并合并，完成后写
    marker，后续运行跳过。
    """
    if os.path.exists(marker):
        print(f"[数据更新] 已存在历史回补标记 {marker}，跳过")
        return

    from data.fetch_daily_k import fetch_one, login, logout

    start_ts = pd.Timestamp(start_date)
    print(f"[数据更新] 为 2021h1 因子窗口补齐第一期股票池 {len(universe)} 只的"
          f" {start_date} 起历史缺口（仅 hfq；size 用 hfq 的 amount/turn，"
          f"与 bfq 一致）...")

    updated = 0
    empty = 0
    failed = []
    missing = []
    login()
    try:
        # 只回补 hfq：
        #   - momentum / volatility 用 hfq 收盘价；
        #   - size 用 amount / turn，这两个字段 hfq 与 bfq 完全相同。
        for label, out_dir, adjustflag in PRICE_DATA_SPECS[:1]:
            for i, code in enumerate(sorted(universe), 1):
                path = price_file_path(out_dir, code)
                if not os.path.exists(path):
                    missing.append(f"{out_dir}/{code}")
                    continue
                try:
                    old = pd.read_csv(path, encoding='utf-8-sig')
                    old['date'] = pd.to_datetime(old['date'], errors='coerce')
                    old = old.dropna(subset=['date'])
                    if old.empty:
                        continue
                    first = old['date'].min()
                    if first <= start_ts:
                        continue
                    prefix_end = (first - pd.Timedelta(days=1)).date().isoformat()
                    if prefix_end < start_date:
                        continue

                    new = fetch_one(code, start_date, prefix_end,
                                    adjustflag=adjustflag)
                    if new.empty:
                        empty += 1
                        continue
                    new = new.reindex(columns=old.columns)
                    merged = (
                        pd.concat([old, new], ignore_index=True)
                        .drop_duplicates(subset=['date'], keep='last')
                        .sort_values('date')
                        .reset_index(drop=True)
                    )
                    tmp_path = f"{path}.tmp"
                    merged.to_csv(tmp_path, index=False, encoding='utf-8-sig')
                    os.replace(tmp_path, path)
                    updated += 1
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{label} {code}: {exc}")

                if i % 20 == 0 or i == len(universe):
                    print(f"[数据更新] {label}: {i}/{len(universe)}，"
                          f"已补 {updated} 只，无数据 {empty}，失败 {len(failed)}",
                          flush=True)
                time.sleep(0.05)
    finally:
        logout()

    if missing:
        print(f"[warn] {len(missing)} 只缺少个股文件，将由行情更新流程补抓: "
              f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    if failed:
        print(f"[warn] {len(failed)} 只历史回补失败（下次运行会重试）:")
        for item in failed[:10]:
            print(f"       {item}")
        if len(failed) > 10:
            print(f"       ... 共 {len(failed)} 只")
        return

    with open(marker, 'w', encoding='utf-8') as f:
        f.write(f"price history backfilled from {start_date}\n")
    print(f"[数据更新] 历史回补完成（补 {updated} 只），写入标记 {marker}")


def check_price_data_coverage(periods: list, required: dict,
                              expected_date: str, vendor_last: str | None = None) -> None:
    """检查各期所需 hfq / bfq 行情是否覆盖到各自最后一期的期末。"""
    if vendor_last is None:
        print(f"\n[数据检查] 测试期末: {expected_date}；未探测数据源最新交易日")
    else:
        print(f"\n[数据检查] 测试期末: {expected_date}；数据源最新交易日: {vendor_last}")
        if vendor_last < expected_date:
            print(f"[数据检查] 注意：股票行情受数据源限制最多到 {vendor_last}，"
                  f"比测试期末少 {expected_date} 当日数据")
    cap_date = vendor_last if vendor_last is not None else expected_date

    for label, key in (("后复权 hfq", "df_hfq_list"), ("不复权 bfq", "df_bfq_list")):
        last_dates = {}
        for p in periods:
            for code, df in zip(p["code_list"], p[key]):
                d = pd.to_datetime(df["date"], errors="coerce").max()
                if (code not in last_dates
                        or pd.isna(last_dates[code])
                        or (pd.notna(d) and d > last_dates[code])):
                    last_dates[code] = d
        short = []
        for code, d in last_dates.items():
            target = min(required[code], cap_date)
            if pd.isna(d) or d < pd.Timestamp(target):
                short.append((code, d, target))
        if not short:
            print(f"[数据检查] {label}: {len(last_dates)} 只均已覆盖所需区间")
            continue
        counts = Counter(
            '无有效数据' if pd.isna(d) else d.date().isoformat()
            for _, d, _ in short
        )
        print(f"[数据检查] {label}: {len(short)}/{len(last_dates)} 只未覆盖所需区间"
              f"（可能为退市/长期停牌）")
        for dt, n in sorted(counts.items(), reverse=True):
            print(f"           最后日期 {dt}: {n} 只")
        for code, d, target in sorted(short, key=lambda x: str(x[1]))[:5]:
            last = '无数据' if pd.isna(d) else d.date().isoformat()
            print(f"           {code}: 最后 {last}，应覆盖至 {target}")
        if len(short) > 5:
            print(f"           ... 共 {len(short)} 只")


def make_return_calendar(stock_dfs: list, start_date: str, end_date: str) -> pd.DatetimeIndex:
    """汇总区间内所有个股的交易日，用作占位基准的日期序列。"""
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    dates = []
    for df in stock_dfs:
        if df is None or len(df) == 0:
            continue
        d = pd.to_datetime(df['date'], errors='coerce')
        dates.extend(d[(d >= start_ts) & (d <= end_ts)].dropna().tolist())
    return pd.DatetimeIndex(sorted(set(dates)))


def resolve_target_csv() -> str:
    """优先使用已发布的 target_code.csv，否则回退到 tmp_target_code.csv。"""
    for path in ('target_code.csv', 'tmp_target_code.csv'):
        if os.path.exists(path):
            return path
    raise FileNotFoundError('未找到 target_code.csv / tmp_target_code.csv')


def period_label(nominal_end: str, effective_end: str) -> str:
    """按收益区间的半年度生成标签；有效终点被截断时加 *。"""
    ts = pd.Timestamp(nominal_end)
    label = f"{ts.year}h{1 if ts.month <= 6 else 2}"
    if effective_end != nominal_end:
        label += '*'
    return label


def main() -> None:
    # 防止 baostock 网络异常时无限阻塞；异常会被下面的 try/except 捕获并回退
    socket.setdefaulttimeout(60)

    # ---------- 测试区间 ----------
    test_start, test_end = get_test_period()
    expected_data_date = test_end.isoformat()

    date_list = build_rebalance_dates(test_start, test_end)
    if len(date_list) < 3:
        raise RuntimeError(f"测试区间 {date_list} 不足两期，无法进行归因")

    # 需要上一期数据构造期初暴露，因此从第二个调仓日开始归因
    analysis_starts = date_list[:-1]
    analysis_ends = date_list[1:]
    prior_starts = [PRICE_HISTORY_START] + analysis_starts[:-1]

    print(f"测试区间: {date_list[0]} ~ {date_list[-1]}")
    print(f"归因区间: {analysis_starts[0]} ~ {analysis_ends[-1]}，共 {len(analysis_starts)} 期")
    print("[info] 每期因子暴露均为 point-in-time；2021h1 使用 2020h2 权重和 "
          "2020-07-01 ~ 2020-12-31 的因子窗口。")

    # ---------- 目标持仓与期初基准持仓 ----------
    target_csv = resolve_target_csv()
    target_weights_all = load_target_weights(target_csv)
    missing_targets = [s for s in analysis_starts if s not in target_weights_all]
    if missing_targets:
        raise ValueError(
            f"{target_csv} 缺少调仓日 {missing_targets} 的目标持仓，"
            f"请先更新目标文件。"
        )
    print(f"[info] 目标持仓文件: {target_csv}")

    bench_weights = {}
    universes = []
    for s in analysis_starts:
        path = weight_file_for(s)
        bench_weights[s] = load_bench_weights(path)
        print(f"[info] {s} 期初基准持仓（上一期期末）: {path}"
              f"（{len(bench_weights[s])} 只）")
        universe = sorted(set(target_weights_all[s].index) | set(bench_weights[s].index))
        if not universe:
            raise ValueError(f"{s} 的目标持仓与基准持仓均为空")
        universes.append(universe)

    # 目标权重应合计为 1（等权）；只提示，不修改
    for s in analysis_starts:
        total = float(target_weights_all[s].sum())
        if abs(total - 1.0) > 1e-6:
            print(f"[warn] {s} 目标权重合计为 {total:.6f}，不等于 1")

    # ---------- 行情更新 ----------
    required_dates = required_end_by_code(universes, analysis_ends)
    vendor_last = None
    if UPDATE_PRICE_DATA:
        try:
            vendor_last = probe_vendor_last_date(analysis_ends[-1])
            update_price_data(required_dates, PRICE_FETCH_START,
                              analysis_ends[-1], vendor_last)
            # 2021h1 的因子窗口需要 2020h2 行情；首次运行回补后写 marker
            backfill_price_history(universes[0], PRICE_HISTORY_START)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 行情自动更新失败：{e}\n       继续使用本地已有数据。")

    # ---------- 逐期构造因子暴露、因子收益与归因 ----------
    periods_for_coverage = []
    detail_rows = []

    for idx, (s, e, prior) in enumerate(zip(analysis_starts, analysis_ends, prior_starts)):
        universe = universes[idx]
        print(f"\n[period] {s} ~ {e}（因子窗口 {prior} ~ {s}，股票池 {len(universe)} 只）")

        hfq_paths = [price_file_path('stock_daily_k_hfq', c) for c in universe]
        bfq_paths = [price_file_path('stock_daily_k_bfq', c) for c in universe]
        hfq_list = [load_stock_data(path) for path in hfq_paths]
        bfq_list = [load_stock_data(path) for path in bfq_paths]

        # 股票行情的“市场最后交易日”：取各股最后日期的众数，避免个别文件
        # 已更新/未更新造成日期错配；有数据源探测结果时以二者较早者为准。
        stock_dates = []
        for df in hfq_list:
            if len(df):
                d = pd.to_datetime(df['date'], errors='coerce').max()
                if pd.notna(d):
                    stock_dates.append(d.normalize())
        if not stock_dates:
            print("[warn] 股票池内没有任何行情，跳过该期")
            continue
        counts = pd.Series(stock_dates).value_counts()
        stock_last = max(counts[counts == counts.max()].index)
        market_last = stock_last
        if vendor_last is not None:
            market_last = min(market_last, pd.Timestamp(vendor_last))

        end_ts = min(pd.Timestamp(e), market_last)
        if end_ts <= pd.Timestamp(s):
            print(f"[warn] 共同收益终点 {end_ts.date()} 不晚于起点，跳过该期")
            continue
        end_eff = end_ts.date().isoformat()
        if end_eff != e:
            print(f"[info] 股票行情共同终点为 {end_eff}（名义期末 {e}）")

        # ---- 期初因子暴露（全部使用调仓日及之前的数据） ----
        try:
            val_date = valuation_date_for(s)
            df_pb = query_pb_mrq(csv_path=VALUATION_CSV, stock_codes=universe,
                                 target_date=val_date, target_mode='exact')
            df_value = value_factor_exposure(
                df=df_pb, target_date=val_date, plot=False).set_index('code')

            df_vol, _ = compute_volatility_exposure(
                stock_data=hfq_list, start_date=prior, end_date=s,
                stock_names=universe, annualize=False, plot=False)
            df_vol = df_vol.set_index('stock')

            df_mom = momentum_exposure_histogram(
                price_dfs=hfq_list, start_date=prior, end_date=s, names=universe,
                skip_days=0, winsorize=True, return_fig=False, plot=False
            ).set_index('stock')

            # amount / turn 在 hfq 与 bfq 中完全一致，size 直接用 hfq，
            # 避免额外回补 2020h2 的 bfq 数据。
            df_size, _ = calc_size_factor_exposure(
                data=dict(zip(universe, hfq_list)), date=s, plot=False)
            df_size = df_size.set_index('code')
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {s} 因子暴露计算失败：{exc}，跳过该期")
            continue

        # ---- 每期个股收益与“持仓加权”基准收益 ----
        # 用一个零收益的占位基准，仅复用 calc_excess_return_distribution 的个股收益计算；
        # 基准收益随后按“期初基准权重 × 个股收益”重算（现金按 0 收益）。
        calendar = make_return_calendar(hfq_list, s, end_eff)
        if len(calendar) < 2:
            print(f"[warn] {s} ~ {end_eff} 交易日不足，跳过该期")
            continue
        placeholder_bench = pd.DataFrame({'date': calendar, 'close': 1.0})
        try:
            df_exc, _ = calc_excess_return_distribution(
                benchmark_df=placeholder_bench, stock_dfs=hfq_list,
                start_date=s, end_date=end_eff, stock_names=universe,
                return_df=True, verbose=False, plot=False)
        except ValueError as exc:
            print(f"[warn] {s} ~ {end_eff} 收益计算失败：{exc}，跳过该期")
            continue
        df_exc = df_exc.set_index('stock')

        bench_w = bench_weights[s].reindex(universe).fillna(0.0)
        stock_ret = df_exc['stock_return']
        valid_ret = pd.Series(np.isfinite(stock_ret.to_numpy()), index=universe)
        bench_weight_sum = float(bench_weights[s].sum())
        cash_weight = max(0.0, 1.0 - bench_weight_sum)
        bench_valid_weight = float(bench_w[valid_ret].sum())
        bench_ret = float(
            (bench_w[valid_ret] * stock_ret[valid_ret]).sum()
            + CASH_RETURN * cash_weight
        )
        df_exc['benchmark_return'] = bench_ret
        df_exc['excess_return'] = stock_ret - bench_ret

        target_only = set(target_weights_all[s].index) - set(bench_weights[s].index)
        target_only_weight = float(
            target_weights_all[s].reindex(sorted(target_only)).sum()
        )

        # ---- 对齐到同一股票池 ----
        X = pd.DataFrame(index=universe, dtype=float)
        X['size'] = df_size.reindex(universe)['size_exposure']
        X['value'] = df_value.reindex(universe)['value_z']
        X['momentum'] = df_mom.reindex(universe)['momentum_z']
        X['volatility'] = df_vol.reindex(universe)['vol_exposure']
        mcap = df_size.reindex(universe)['float_mv'].rename('market_cap')
        ret = df_exc.reindex(universe)['excess_return'].rename('return')

        # ---- 横截面回归：只用有完整暴露/收益/市值的股票 ----
        regression_mask = (
            X.notna().all(axis=1)
            & np.isfinite(X).all(axis=1)
            & mcap.notna()
            & np.isfinite(mcap)
            & (mcap > 0)
            & ret.notna()
            & np.isfinite(ret)
        )
        if int(regression_mask.sum()) < len(FACTOR_NAMES) + 1:
            print(f"[warn] 回归有效样本仅 {int(regression_mask.sum())} 只，跳过该期")
            continue
        fac_ret, _ = build_risk_model_cross_section(
            market_cap=mcap[regression_mask].to_frame(),
            stock_return=ret[regression_mask].to_frame(),
            factor_exposure=X.loc[regression_mask],
            weight_power=0.5)
        f = fac_ret['factor_return'].reindex(FACTOR_NAMES)
        if f.isna().any():
            raise ValueError(f"{s} 因子收益缺失: {f.to_dict()}")

        # ---- 归因：统一股票池上的主动权重 ----
        w_target = target_weights_all[s].reindex(universe).fillna(0.0)
        w_bench = bench_weights[s].reindex(universe).fillna(0.0)

        finite_ret = pd.Series(np.isfinite(ret.to_numpy()), index=universe)
        attr_mask = finite_ret & X.notna().all(axis=1) & np.isfinite(X).all(axis=1)

        E = ret[attr_mask]
        Xa = X.loc[attr_mask]
        A = (w_target - w_bench)[attr_mask]

        factor_expo = Xa.to_numpy().T @ A.to_numpy()
        factor_contrib = pd.Series(factor_expo * f.to_numpy(), index=FACTOR_NAMES)
        spec_vec = E.to_numpy() - Xa.to_numpy() @ f.to_numpy()
        spec_contrib = float(A.to_numpy() @ spec_vec)
        attr_total = float(A.to_numpy() @ E.to_numpy())

        factor_total = float(factor_contrib.sum())
        model_total = factor_total + spec_contrib
        reconcile_check = attr_total - model_total

        total_active = float((w_target[finite_ret] * ret[finite_ret]).sum())
        bench_resid = float((w_bench[finite_ret] * ret[finite_ret]).sum())
        unattributed = total_active - attr_total - bench_resid

        missing_target_w = float(w_target[~finite_ret].sum())
        missing_bench_w = float(w_bench[~finite_ret].sum())
        unmodeled_target_w = float(w_target[finite_ret & ~attr_mask].sum())
        unmodeled_bench_w = float(w_bench[finite_ret & ~attr_mask].sum())

        label = period_label(e, end_eff)
        print(
            f"[归因] {label} ({s}~{end_eff}, N={int(attr_mask.sum())}): "
            f"size={factor_contrib['size']:+.2%} "
            f"value={factor_contrib['value']:+.2%} "
            f"momentum={factor_contrib['momentum']:+.2%} "
            f"volatility={factor_contrib['volatility']:+.2%} | "
            f"specific={spec_contrib:+.2%} factor合计={factor_total:+.2%} "
            f"现金/基准残差={bench_resid:+.2%} 未归因={unattributed:+.2%} "
            f"主动收益={total_active:+.2%} 校验差={reconcile_check:.2e}"
        )
        print(
            f"       基准=持仓加权 {bench_ret:+.2%}"
            f"（权重合计 {bench_weight_sum:.2%}，有效权重 {bench_valid_weight:.2%}，"
            f"现金 {cash_weight:.2%}）；目标-only {len(target_only)} 只，"
            f"目标权重 {target_only_weight:.2%}"
        )
        if target_only:
            print("       目标-only 代码（基准权重按 0 计）: "
                  + ", ".join(c.split('.')[-1] for c in sorted(target_only)))
        if abs(reconcile_check) > 1e-10:
            raise AssertionError(f"{label} 归因对账失败: {reconcile_check}")
        if missing_target_w > 1e-6 or missing_bench_w > 1e-6:
            print(f"       缺少收益数据：目标权重 {missing_target_w:.2%}，"
                  f"基准权重 {missing_bench_w:.2%}")
        if unmodeled_target_w > 1e-4 or unmodeled_bench_w > 1e-4:
            print(f"       缺少因子暴露：目标权重 {unmodeled_target_w:.2%}，"
                  f"基准权重 {unmodeled_bench_w:.2%}")

        detail_rows.append({
            'period': label, 'start': s, 'end_nominal': e, 'end_effective': end_eff,
            'n_stocks': len(universe), 'n_attr': int(attr_mask.sum()),
            'size': float(factor_contrib['size']),
            'value': float(factor_contrib['value']),
            'momentum': float(factor_contrib['momentum']),
            'volatility': float(factor_contrib['volatility']),
            'factor_total': factor_total,
            'specific': spec_contrib,
            'bench_resid': bench_resid,
            'unattributed': unattributed,
            'total': total_active,
            'reconcile_check': reconcile_check,
            'missing_target_weight': missing_target_w,
            'missing_bench_weight': missing_bench_w,
            'unmodeled_target_weight': unmodeled_target_w,
            'unmodeled_bench_weight': unmodeled_bench_w,
            'benchmark_return': bench_ret,
            'bench_weight_sum': bench_weight_sum,
            'bench_valid_weight': bench_valid_weight,
            'cash_weight': cash_weight,
            'n_target_only': len(target_only),
            'target_only_weight': target_only_weight,
            'target_only_codes': ';'.join(sorted(target_only)),
        })
        periods_for_coverage.append({
            'name': label, 'code_list': universe,
            'df_hfq_list': hfq_list, 'df_bfq_list': bfq_list,
            'start_date': s, 'end_date': end_eff,
        })

    if not detail_rows:
        raise RuntimeError("没有任何一期完成归因，请检查数据")

    # ---------- 数据覆盖检查 ----------
    # 用各期实际使用的有效终点（而不是名义期末）检查，避免数据源晚一天时误报
    coverage_required = required_end_by_code(
        [p['code_list'] for p in periods_for_coverage],
        [p['end_date'] for p in periods_for_coverage])
    check_price_data_coverage(
        periods_for_coverage, coverage_required, expected_data_date, vendor_last)

    # ---------- 明细与画图 ----------
    detail_df = pd.DataFrame(detail_rows)
    detail_path = 'factor_contrib_detail.csv'
    detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"\n[输出] 归因明细已保存: {detail_path}")

    labels = detail_df['period'].tolist()
    plot_data = detail_df[
        FACTOR_NAMES + ['specific', 'total']
    ].to_numpy(dtype=float) * 100.0
    plot_clustered_bars(
        data=plot_data,
        output_path='factor_contrib_barplot.png',
        periods=labels,
        series=FACTOR_NAMES + ['specific', 'total'],
        y_label='Contribution (%)',
        bar_fmt='%.2f',
        title='Factor contributions in different periods (point-in-time)',
    )
    print("[输出] 因子贡献图已保存: factor_contrib_barplot.png")


if __name__ == '__main__':
    main()
