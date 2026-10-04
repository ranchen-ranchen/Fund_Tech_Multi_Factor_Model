"""
一揽子股票长期持有（等权重/自定义权重）回测程序 —— 支持多时间段

用法:
    python buy_and_hold_backtest.py a.csv b.csv c.csv      # 单时间段
    python buy_and_hold_backtest.py                         # 无参数时用模拟数据演示（多时间段）

输入: 一系列股票 K 线 CSV 文件（需包含日期列与收盘价列）
输出: portfolio_backtest.log 日志文件
"""

import os
import sys
import logging
import tempfile
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 日志配置
# -----------------------------------------------------------------------------
def setup_logger(log_file: str = "portfolio_backtest.log") -> logging.Logger:
    logger = logging.getLogger("portfolio")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()          # 防止重复添加 handler

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# -----------------------------------------------------------------------------
# 读取单只股票数据
# -----------------------------------------------------------------------------
DATE_CANDIDATES  = ["date", "datetime", "time", "trade_date", "日期", "交易日期"]
CLOSE_CANDIDATES = ["close", "adj close", "adj_close", "closeprice",
                    "收盘价", "收盘", "close_price"]


def load_stock_data(csv_path: str) -> pd.Series:
    """读取单个 csv，返回以日期为索引的收盘价 Series。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"{csv_path} 中未找到日期列，现有列: {list(df.columns)}")

    close_col = next((c for c in CLOSE_CANDIDATES if c in df.columns), None)
    if close_col is None:
        raise ValueError(f"{csv_path} 中未找到收盘价列，现有列: {list(df.columns)}")

    df = df[[date_col, close_col]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (df.dropna()
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date"))
    df = df[df["close"] > 0]

    if df.empty:
        raise ValueError(f"{csv_path} 清洗后无有效数据")

    return df["close"]


# -----------------------------------------------------------------------------
# 核心回测（多时间段）
# -----------------------------------------------------------------------------
def run_backtest(
    periods: List[Dict[str, Any]],
    initial_capital: float = 1.0,
    log_file: str = "portfolio_backtest.log",
    align: str = "inner",
):
    """
    参数
    ----
    periods : 时间段配置列表，每个元素是一个字典，支持字段：
        - csv_files  : List[str]               该时间段的股票 csv 文件列表（必填）
        - weights    : Optional[List[float]]   该时间段的权重，None 表示等权重
        - start_date : Optional[str]           该时间段起始日期 'YYYY-MM-DD'
        - end_date   : Optional[str]           该时间段终止日期 'YYYY-MM-DD'
        - name       : Optional[str]           该时间段的名称（用于日志）
    initial_capital : 初始资金
    log_file        : 日志文件路径
    align           : 'inner' 取共同交易日；'outer' 取并集并前后向填充

    每个时间段结束时，将其期末净值作为下一时间段的期初净值，
    从而得到跨越多个时间段、连续变化的资产净值曲线。
    """
    logger = setup_logger(log_file)

    logger.info("=" * 78)
    logger.info("一揽子股票 · 多时间段长期持有回测")
    logger.info("=" * 78)
    logger.info(f"时间段数量   : {len(periods)}")
    logger.info(f"初始资金     : {initial_capital}")
    logger.info(f"日期对齐方式 : {align}")

    if not periods:
        logger.error("未提供任何时间段配置")
        return None, None

    # ---- 校验时间段 ----
    for k, p in enumerate(periods, 1):
        if "csv_files" not in p or not p["csv_files"]:
            raise ValueError(f"时间段 {k} 缺少 'csv_files' 或列表为空")

    cumulative_nav = 1.0          # 上一时间段结束时的累计净值
    last_date: Optional[pd.Timestamp] = None
    total_nav_parts: List[pd.DataFrame] = []
    period_overview: List[Dict[str, Any]] = []

    # =========================================================================
    # 依次处理每个时间段
    # =========================================================================
    for k, period in enumerate(periods, 1):
        pname      = period.get("name", f"Period_{k}")
        csv_files  = period["csv_files"]
        weights    = period.get("weights")
        p_start    = period.get("start_date")
        p_end      = period.get("end_date")

        logger.info("")
        logger.info("-" * 78)
        logger.info(f"【时间段 {k}/{len(periods)}】{pname}")
        logger.info("-" * 78)
        logger.info(f"  股票数量     : {len(csv_files)}")
        logger.info(f"  指定起始日期 : {p_start if p_start else '不限制'}")
        logger.info(f"  指定终止日期 : {p_end if p_end else '不限制'}")
        logger.info(f"  期初累计净值 : {cumulative_nav:.6f}")

        n = len(csv_files)

        # ---- 权重处理 ----
        if weights is None:
            weights = [1.0 / n] * n
            logger.info("  权重         : 等权重（默认）")
        else:
            if len(weights) != n:
                raise ValueError(
                    f"时间段 {k}: 权重个数({len(weights)})与股票个数({n})不一致"
                )
            total_w = float(sum(weights))
            if total_w <= 0:
                raise ValueError(f"时间段 {k}: 权重之和必须大于 0")
            weights = [w / total_w for w in weights]
            logger.info("  权重         : 用户指定（已归一化）")

        # ---- 加载数据 ----
        names, series_list = [], []
        for i, path in enumerate(csv_files):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                s = load_stock_data(path)
            except Exception as e:
                logger.error(f"  [{i+1}] 读取 {path} 失败: {e}")
                raise
            names.append(name)
            series_list.append(s)
            logger.info(
                f"  [{i+1}] {name:<18s} 权重={weights[i]:.4f}  "
                f"{len(s):>5d} 条  {s.index.min().date()} ~ {s.index.max().date()}"
            )

        # ---- 日期对齐 ----
        prices = pd.concat(series_list, axis=1, keys=names)
        prices.columns = names

        if align == "inner":
            prices = prices.dropna(how="any")
        else:
            prices = prices.sort_index().ffill().bfill()

        if prices.empty:
            logger.error(f"  时间段 {k} 对齐后没有可用的交易日期，跳过")
            continue

        # ---- 按指定日期范围筛选 ----
        if p_start is not None:
            prices = prices[prices.index >= pd.to_datetime(p_start)]
        if p_end is not None:
            prices = prices[prices.index <= pd.to_datetime(p_end)]

        # ---- 去掉与上一时间段重叠的日期，保证净值曲线无重复时间点 ----
        if last_date is not None:
            prices = prices[prices.index > last_date]

        if prices.empty:
            logger.warning(f"  时间段 {k} 过滤后没有可用的交易日期，跳过")
            continue

        logger.info(f"  实际回测区间 : {prices.index.min().date()} ~ {prices.index.max().date()}")
        logger.info(f"  交易日数     : {len(prices)}")

        # ---- 该时间段的期初资金 = 初始资金 × 累计净值 ----
        period_capital = initial_capital * cumulative_nav

        # ---- 建仓：第一天按权重买入 ----
        p0 = prices.iloc[0]
        amounts = [period_capital * w for w in weights]
        shares  = [amounts[i] / float(p0.iloc[i]) for i in range(n)]

        logger.info("")
        logger.info(f"  建仓明细（{prices.index.min().date()} 收盘价买入）：")
        logger.info(f"    {'股票':<18s}{'首日价格':>12s}{'投入金额':>16s}{'买入份额':>18s}")
        for i, name in enumerate(names):
            logger.info(
                f"    {name:<18s}{float(p0.iloc[i]):>12.4f}"
                f"{amounts[i]:>16.4f}{shares[i]:>18.6f}"
            )

        # ---- 每日市值 ----
        values = prices.mul(shares, axis=1)          # 各股票市值
        total  = values.sum(axis=1)                  # 组合总市值

        # 以 initial_capital 为单位的“绝对累计净值贡献”
        # sum_i values_i(t) / initial_capital == cumulative_nav(t)
        stock_nav = values.div(initial_capital)
        abs_nav   = total.div(initial_capital)       # 组合累计净值（跨周期连续）

        # ---- 输出每日净值明细 ----
        logger.info("")
        logger.info(f"  每日净值明细（累计净值，期初 = {cumulative_nav:.4f}）")
        hdr = (f"  {'日期':<12s}"
               + "".join(f"{nm[:11]:>13s}" for nm in names)
               + f"{'TOTAL':>13s}")
        logger.info(hdr)
        logger.info("  " + "-" * (len(hdr) - 2))

        for date in stock_nav.index:
            line = f"  {date.strftime('%Y-%m-%d'):<12s}"
            for nm in names:
                line += f"{float(stock_nav.at[date, nm]):>13.4f}"
            line += f"{float(abs_nav.at[date]):>13.4f}"
            logger.info(line)

        # ---- 记录该时间段数据，供合并 ----
        total_nav_parts.append(pd.DataFrame({"TOTAL": abs_nav}))

        period_start_nav = cumulative_nav
        period_end_nav   = float(abs_nav.iloc[-1])
        period_ret       = (period_end_nav / period_start_nav - 1.0) if period_start_nav > 0 else 0.0

        logger.info("")
        logger.info(
            f"  期初净值 : {period_start_nav:.4f}   期末净值 : {period_end_nav:.4f}   "
            f"区间收益 : {period_ret*100:.2f}%"
        )

        period_overview.append({
            "name":         pname,
            "start":        abs_nav.index.min(),
            "end":          abs_nav.index.max(),
            "days":         (abs_nav.index[-1] - abs_nav.index[0]).days,
            "trading_days": len(abs_nav),
            "start_nav":    period_start_nav,
            "end_nav":      period_end_nav,
            "return":       period_ret,
        })

        # ---- 更新累计净值与已覆盖的最新日期 ----
        cumulative_nav = period_end_nav
        last_date      = abs_nav.index[-1]

    # =========================================================================
    # 合并所有时间段
    # =========================================================================
    if not total_nav_parts:
        logger.error("没有成功执行任何时间段")
        return None, None

    nav = pd.concat(total_nav_parts, axis=0).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]

    # ---- 时间段概览 ----
    logger.info("")
    logger.info("=" * 78)
    logger.info("多时间段概览")
    logger.info("=" * 78)
    logger.info(f"{'时间段':<20s}{'起':>12s}{'止':>12s}"
                f"{'期初净值':>12s}{'期末净值':>12s}{'区间收益':>12s}")
    logger.info("-" * 80)
    for po in period_overview:
        logger.info(
            f"{po['name'][:18]:<20s}"
            f"{po['start'].strftime('%Y-%m-%d'):>12s}"
            f"{po['end'].strftime('%Y-%m-%d'):>12s}"
            f"{po['start_nav']:>12.4f}"
            f"{po['end_nav']:>12.4f}"
            f"{po['return']*100:>11.2f}%"
        )
    logger.info("-" * 80)

    # ---- 汇总统计 ----
    logger.info("")
    logger.info("=" * 78)
    logger.info("整体回测结果汇总")
    logger.info("=" * 78)

    s = nav["TOTAL"]
    final_nav = float(s.iloc[-1])
    total_ret = final_nav - 1.0
    days      = (s.index[-1] - s.index[0]).days
    years     = days / 365.25 if days > 0 else 1.0
    ann_ret   = (final_nav ** (1 / years) - 1) if (years > 0 and final_nav > 0) else 0.0

    cummax = s.cummax()
    dd     = (s - cummax) / cummax
    max_dd = float(dd.min())

    daily_ret = s.pct_change().dropna()
    ann_vol   = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0

    header = (f"{'标的':<18s}{'期末净值':>11s}{'累计收益':>12s}"
              f"{'年化收益':>12s}{'最大回撤':>12s}{'年化波动':>12s}{'夏普':>10s}")
    logger.info(header)
    logger.info("-" * 90)
    logger.info(
        f"{'TOTAL':<18s}{final_nav:>11.4f}{total_ret*100:>11.2f}%"
        f"{ann_ret*100:>11.2f}%{max_dd*100:>11.2f}%"
        f"{ann_vol*100:>11.2f}%{sharpe:>10.4f}"
    )
    logger.info("-" * 90)

    logger.info(f"回测区间     : {s.index.min().date()} ~ {s.index.max().date()}")
    logger.info(f"总交易日数   : {len(s)}")
    logger.info(f"回测结束，日志已保存至：{os.path.abspath(log_file)}")
    logger.info("=" * 78)

    summary = [{
        "name":          "TOTAL",
        "final_nav":     final_nav,
        "total_return":  total_ret,
        "annual_return": ann_ret,
        "max_drawdown":  max_dd,
        "annual_vol":    ann_vol,
        "sharpe":        sharpe,
    }]

    return nav, summary


# -----------------------------------------------------------------------------
# 演示：生成模拟数据
# -----------------------------------------------------------------------------
def _make_demo_files() -> List[str]:
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    specs = [("STOCK_A", 0.0006, 0.015),
             ("STOCK_B", 0.0003, 0.020),
             ("STOCK_C", 0.0004, 0.018)]
    files = []
    for name, mu, sigma in specs:
        rets   = np.random.normal(mu, sigma, len(dates))
        prices = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "date":   dates,
            "open":   prices * (1 + np.random.normal(0, 0.002, len(dates))),
            "high":   prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            "low":    prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            "close":  prices,
            "volume": np.random.randint(100_000, 10_000_000, len(dates)),
        })
        path = os.path.join(tempfile.gettempdir(), f"{name}.csv")
        df.to_csv(path, index=False)
        files.append(path)
        print(f"  生成模拟数据: {path}")
    return files


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行传参：单时间段，所有 csv 等权
        csv_files = sys.argv[1:]
        periods = [
            {
                "name":       "单时间段",
                "csv_files":  csv_files,
                "weights":    None,
                "start_date": None,
                "end_date":   None,
            }
        ]
    else:
        print("未提供 csv 文件，使用模拟数据演示多时间段回测...")
        demo_files = _make_demo_files()      # [STOCK_A, STOCK_B, STOCK_C]

        periods = [
            {
                "name":       "第一阶段 A+B+C",
                "csv_files":  demo_files,           # 三只股票
                "weights":    [0.4, 0.3, 0.3],      # 自定义权重
                "start_date": "2020-01-01",
                "end_date":   "2021-12-31",
            },
            {
                "name":       "第二阶段 A+B",
                "csv_files":  demo_files[:2],       # 换成 A、B 两只
                "weights":    [0.5, 0.5],
                "start_date": "2022-01-01",
                "end_date":   "2023-06-30",
            },
            {
                "name":       "第三阶段 B+C",
                "csv_files":  demo_files[1:],       # 换成 B、C 两只
                "weights":    None,                 # 等权
                "start_date": "2023-07-01",
                "end_date":   "2024-12-31",
            },
        ]

    run_backtest(
        periods=periods,
        initial_capital=1.0,
        log_file="portfolio_backtest.log",
        align="inner",
    )
