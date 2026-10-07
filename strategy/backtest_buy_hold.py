"""
一揽子股票长期持有（等权重/自定义权重）回测程序 —— 支持多时间段 + 双边交易成本

用法:
    python backtest_buy_hold.py a.csv b.csv c.csv      # 单时间段
    python backtest_buy_hold.py                         # 无参数时用模拟数据演示（多时间段）

输入: 一系列股票 K 线 CSV 文件（需包含日期列与收盘价列）
      可选基准文件 hs300etf_510300_performance.csv
输出: portfolio_backtest.log 日志文件
      portfolio_nav.png      净值曲线图（总资产净值 + 基准）

交易成本:
    默认单边 50 bp (cost_rate = 0.005)，双边各扣一次：
      - 每个时间段期初：期初资金 × cost_rate 作为买入成本
      - 每个时间段期末：期末总市值 × cost_rate 作为卖出成本

修复说明:
    原代码在基准对齐时使用了:
        benchmark_aligned = benchmark_nav.reindex(s.index).ffill()
        if benchmark_aligned.isna().any():
            benchmark_aligned = benchmark_aligned.bfill()
    其中 bfill 会用未来的基准净值回填组合起始日之前的空白区间，
    属于前视偏差（look-ahead bias），会影响基准统计与信息比率。
    本版本移除 bfill，只保留 ffill（仅填补中间缺失），
    开头缺失的区间保留为 NaN，由后续 dropna() 自动丢弃，
    保证所有指标只在组合与基准真实重叠的交易日上计算。
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
    logger.handlers.clear()

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
# 读取基准数据（hs300etf_510300_performance.csv）
# -----------------------------------------------------------------------------
def load_benchmark_data(csv_path: str) -> pd.Series:
    """
    读取基准 CSV（如 hs300etf_510300_performance.csv），
    返回以日期为索引的累计净值 Series。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到基准文件: {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Daily Series"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"{csv_path} 中未找到 '# Daily Series' 部分")

    df = pd.read_csv(csv_path, skiprows=start_idx + 1, encoding="utf-8-sig")
    if df.shape[1] < 3:
        raise ValueError(f"{csv_path} 数据列不足")

    df = df.iloc[:, [0, 2]].copy()          # 日期列、累计净值列
    df.columns = ["date", "nav"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = (df.dropna()
            .sort_values("date")
            .drop_duplicates("date")
            .set_index("date"))
    return df["nav"]


# -----------------------------------------------------------------------------
# 单只股票区间指标计算
# -----------------------------------------------------------------------------
def compute_stock_metrics(nav: pd.Series) -> Dict[str, float]:
    """
    给定一条从 1.0 开始的净值序列，计算常用绩效指标。
    年化收益与年化波动统一使用交易日（252）作为时间基准。
    """
    nav = nav.dropna()
    if len(nav) == 0:
        return {
            "final_nav": 1.0, "total_ret": 0.0, "annual_ret": 0.0,
            "max_dd": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
            "days": 0, "trading_days": 0,
        }

    final_nav = float(nav.iloc[-1])
    total_ret = final_nav - 1.0

    trading_days = len(nav)
    years = trading_days / 252 if trading_days > 0 else 1.0

    if years > 0 and final_nav > 0:
        annual_ret = final_nav ** (1 / years) - 1
    else:
        annual_ret = 0.0

    cummax = nav.cummax()
    dd = (nav - cummax) / cummax
    max_dd = float(dd.min()) if len(dd) else 0.0

    daily_ret = nav.pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
    sharpe = annual_ret / ann_vol if ann_vol > 0 else 0.0

    return {
        "final_nav":    final_nav,
        "total_ret":    total_ret,
        "annual_ret":   annual_ret,
        "max_dd":       max_dd,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "days":         (nav.index[-1] - nav.index[0]).days,
        "trading_days": trading_days,
    }


# -----------------------------------------------------------------------------
# 信息比率 / 主动管理指标
# -----------------------------------------------------------------------------
def compute_information_ratio(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    计算组合相对基准的主动管理指标：

        - 年化超额收益（Active Return）
              = mean(组合日收益 - 基准日收益) × periods_per_year
        - 年化跟踪误差（Tracking Error）
              = std(组合日收益 - 基准日收益) × sqrt(periods_per_year)
        - 信息比率（Information Ratio, IR）
              = 年化超额收益 / 年化跟踪误差

    说明
    ----
    本函数内部通过 concat + dropna 只使用组合与基准双方都有效的交易日，
    因此即使传入的 benchmark_nav 开头存在 NaN（例如组合起始早于基准起始），
    也不会产生前视偏差。对齐区间由实际重叠数据决定。
    """
    empty = {
        "active_return":     0.0,   # 年化超额收益
        "tracking_error":    0.0,   # 年化跟踪误差
        "information_ratio": 0.0,   # 信息比率
        "excess_daily_mean": 0.0,   # 日均超额收益
        "excess_daily_std":  0.0,   # 日超额收益标准差
        "n_obs":             0,     # 有效样本数
        "overlap_start":     None,  # 实际重叠起始日
        "overlap_end":       None,  # 实际重叠结束日
    }
    if portfolio_nav is None or benchmark_nav is None:
        return empty

    p = portfolio_nav.dropna()
    b = benchmark_nav.dropna()
    if p.empty or b.empty:
        return empty

    # 日期对齐（取交集）
    df = pd.concat([p, b], axis=1, keys=["port", "bench"]).dropna()
    if len(df) < 2:
        return empty

    # 日收益率
    port_ret  = df["port"].pct_change()
    bench_ret = df["bench"].pct_change()

    # 超额收益
    excess = (port_ret - bench_ret).dropna()
    if excess.empty:
        return empty

    mean_excess = float(excess.mean())
    # 无偏标准差（样本标准差，ddof=1）
    std_excess  = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0

    active_return  = mean_excess * periods_per_year
    tracking_error = std_excess * np.sqrt(periods_per_year)
    ir = active_return / tracking_error if tracking_error > 0 else 0.0

    return {
        "active_return":     active_return,
        "tracking_error":    tracking_error,
        "information_ratio": ir,
        "excess_daily_mean": mean_excess,
        "excess_daily_std":  std_excess,
        "n_obs":             int(len(excess)),
        "overlap_start":     df.index.min(),
        "overlap_end":       df.index.max(),
    }


# -----------------------------------------------------------------------------
# 可视化：总资产净值 + 基准
# -----------------------------------------------------------------------------
def plot_nav(
    nav_total: pd.Series,
    output_path: str = "portfolio_nav.png",
    dpi: int = 120,
    period_ranges: Optional[List[Dict[str, Any]]] = None,
    benchmark_nav: Optional[pd.Series] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """绘制组合总资产净值随时间变化的曲线，可叠加基准净值（已归一化）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        msg = "未安装 matplotlib，跳过绘图（pip install matplotlib 后可启用）"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
        return None

    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))

    if period_ranges:
        for pi, pr in enumerate(period_ranges):
            if pi % 2 == 1:
                s_ = pd.to_datetime(pr["start"])
                e_ = pd.to_datetime(pr["end"])
                ax.axvspan(s_, e_, color="gray", alpha=0.06, zorder=0)

    idx  = nav_total.index
    vals = nav_total.values

    ax.plot(idx, vals, color="black", linewidth=2.0, label="TOTAL NAV (net of cost)")
    ax.fill_between(idx, 1.0, vals, where=(vals >= 1.0),
                    color="green", alpha=0.12, interpolate=True)
    ax.fill_between(idx, 1.0, vals, where=(vals < 1.0),
                    color="red", alpha=0.12, interpolate=True)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

    if benchmark_nav is not None and not benchmark_nav.empty:
        b = benchmark_nav.dropna()
        if not b.empty:
            b = b / b.iloc[0]
            ax.plot(b.index, b.values,
                    color="blue", linewidth=1.5, linestyle="--",
                    label="HS300 ETF (510300)")

    if period_ranges:
        for pi, pr in enumerate(period_ranges, 1):
            if pi > 1:
                bd = pd.to_datetime(pr["start"])
                ax.axvline(x=bd, color="gray", linestyle="--",
                           linewidth=0.7, alpha=0.5)

    ax.set_ylabel("Cumulative NAV")
    ax.set_xlabel("Date")
    ax.set_title("Total Portfolio NAV vs Benchmark (net of trading costs)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if logger:
        logger.info(f"净值曲线图已保存至：{os.path.abspath(output_path)}")
    return output_path


# -----------------------------------------------------------------------------
# 核心回测（多时间段 + 双边交易成本）
# -----------------------------------------------------------------------------
def run_backtest(
    periods: List[Dict[str, Any]],
    initial_capital: float = 1.0,
    log_file: str = "portfolio_backtest.log",
    align: str = "inner",
    plot: bool = True,
    plot_file: str = "portfolio_nav.png",
    benchmark_file: Optional[str] = None,
    log_daily_stock_nav: bool = True,
    cost_rate: float = 0.005,
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
    initial_capital     : 初始资金
    log_file            : 日志文件路径
    align               : 'inner' 取共同交易日；'outer' 取并集并前后向填充
    plot                : 是否绘制净值曲线图
    plot_file           : 净值曲线图输出路径
    benchmark_file      : 基准 CSV 文件路径（如 hs300etf_510300_performance.csv）
    log_daily_stock_nav : 是否把每只股票每日净值明细写入 log（数据量大时可关闭）
    cost_rate           : 单边交易成本费率，默认 0.005（即 50 bp / 0.5%）。
                          每个时间段期初买入时：期初资金 × cost_rate 作为买入成本；
                          每个时间段期末卖出时：期末市值 × cost_rate 作为卖出成本。
                          两边合计即为双边交易成本。设为 0 可关闭成本。
    """
    if cost_rate < 0:
        raise ValueError("cost_rate 不能为负数")
    if cost_rate >= 1:
        raise ValueError("cost_rate 必须小于 1")

    logger = setup_logger(log_file)

    logger.info("=" * 78)
    logger.info("一揽子股票 · 多时间段长期持有回测（含双边交易成本）")
    logger.info("=" * 78)
    logger.info(f"时间段数量     : {len(periods)}")
    logger.info(f"初始资金       : {initial_capital}")
    logger.info(f"日期对齐方式   : {align}")
    logger.info(f"单边交易成本   : {cost_rate*100:.4f}%  ({cost_rate*10000:.2f} bp)")
    logger.info(f"双边合计       : {cost_rate*2*100:.4f}%  (每期买入 + 期末卖出各一次)")

    if not periods:
        logger.error("未提供任何时间段配置")
        return None, None

    for k, p in enumerate(periods, 1):
        if "csv_files" not in p or not p["csv_files"]:
            raise ValueError(f"时间段 {k} 缺少 'csv_files' 或列表为空")

    cumulative_nav = 1.0
    last_date: Optional[pd.Timestamp] = None
    total_nav_parts: List[pd.DataFrame] = []
    period_overview: List[Dict[str, Any]] = []
    total_cost_accum = 0.0

    # 用于跨时间段汇总每只股票的表现
    stock_overview_rows: List[Dict[str, Any]] = []

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
            if any(w < 0 for w in weights):
                raise ValueError(f"时间段 {k}: 权重不能为负数")
            total_w = float(sum(weights))
            if total_w <= 0:
                raise ValueError(f"时间段 {k}: 权重之和必须大于 0")
            weights = [w / total_w for w in weights]
            logger.info("  权重         : 用户指定（已归一化）")

        # ---- 加载数据 ----
        names, series_list = [], []
        for i, path in enumerate(csv_files):
            name = os.path.splitext(os.path.basename(path))[0]
            if name in names:
                name = f"{name}_{i+1}"
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
            prices = prices.sort_index().ffill()
            prices = prices.dropna(how="any")

        if prices.empty:
            logger.error(f"  时间段 {k} 对齐后没有可用的交易日期，跳过")
            continue

        # ---- 按指定日期范围筛选 ----
        if p_start is not None:
            prices = prices[prices.index >= pd.to_datetime(p_start)]
        if p_end is not None:
            prices = prices[prices.index <= pd.to_datetime(p_end)]

        # ---- 去掉与上一时间段重叠的日期 ----
        if last_date is not None:
            prices = prices[prices.index > last_date]

        if prices.empty:
            logger.warning(f"  时间段 {k} 过滤后没有可用的交易日期，跳过")
            continue

        logger.info(f"  实际回测区间 : {prices.index.min().date()} ~ {prices.index.max().date()}")
        logger.info(f"  交易日数     : {len(prices)}")

        # =====================================================================
        # 期初资金 & 买入交易成本
        # =====================================================================
        # 该时间段的期初资金 = 初始资金 × 累计净值（费前）
        period_capital_gross = initial_capital * cumulative_nav

        # 买入成本：期初资金 × cost_rate
        buy_cost = period_capital_gross * cost_rate
        # 实际可投入资金（费后）
        period_capital = period_capital_gross - buy_cost

        logger.info("")
        logger.info("  资金与期初交易成本：")
        logger.info(f"    期初资金(费前) : {period_capital_gross:>14.6f}")
        logger.info(f"    买入交易成本   : {buy_cost:>14.6f}  ({cost_rate*100:.4f}%)")
        logger.info(f"    期初资金(费后) : {period_capital:>14.6f}")

        # =====================================================================
        # 建仓：第一天按权重买入（使用费后资金）
        # =====================================================================
        p0 = prices.iloc[0]
        amounts = [period_capital * w for w in weights]
        shares  = [amounts[i] / float(p0.iloc[i]) for i in range(n)]

        logger.info("")
        logger.info(f"  建仓明细（{prices.index.min().date()} 收盘价买入，使用费后资金）：")
        logger.info(f"    {'股票':<18s}{'首日价格':>12s}{'投入金额':>16s}{'买入份额':>18s}")
        for i, name in enumerate(names):
            logger.info(
                f"    {name:<18s}{float(p0.iloc[i]):>12.4f}"
                f"{amounts[i]:>16.4f}{shares[i]:>18.6f}"
            )

        # ---- 每日市值（基于费后资金建仓） ----
        values = prices.mul(shares, axis=1)          # 各股票市值
        total  = values.sum(axis=1)                  # 组合总市值（期初已扣除买入成本）
        abs_nav = total.div(initial_capital).copy()  # 组合累计净值（含买入成本）

        # =====================================================================
        # 期末卖出交易成本
        # =====================================================================
        gross_final_value = float(total.iloc[-1])
        sell_cost = gross_final_value * cost_rate
        net_final_value = gross_final_value - sell_cost

        # 将卖出成本反映到最后一天的净值上
        if len(abs_nav) > 0 and cost_rate > 0:
            abs_nav.iloc[-1] = net_final_value / initial_capital

        logger.info("")
        logger.info("  期末卖出交易成本：")
        logger.info(f"    期末市值(费前) : {gross_final_value:>14.6f}")
        logger.info(f"    卖出交易成本   : {sell_cost:>14.6f}  ({cost_rate*100:.4f}%)")
        logger.info(f"    期末市值(费后) : {net_final_value:>14.6f}")

        # 累计交易成本（以初始资金为单位）
        total_cost_accum += buy_cost + sell_cost

        # ---- 各持仓股净值（按各自首日价格归一化，起始 = 1.0） ----
        stock_nav = prices.div(p0, axis=1)
        stock_nav.columns = names

        # ---- 各持仓股市值占比（按当日组合总市值计算） ----
        weights_daily = values.div(total, axis=0)

        # ---- 输出每日净值明细（TOTAL） ----
        logger.info("")
        logger.info(f"  每日总净值明细（期初 = {cumulative_nav:.4f}，已含双边成本）")
        logger.info(f"  {'日期':<12s}{'TOTAL':>13s}")
        logger.info("  " + "-" * 27)
        for date in abs_nav.index:
            logger.info(f"  {date.strftime('%Y-%m-%d'):<12s}{float(abs_nav.at[date]):>13.4f}")

        # =====================================================================
        # 各持仓股每日净值明细写入 log
        # =====================================================================
        if log_daily_stock_nav:
            logger.info("")
            logger.info(f"  各持仓股每日净值明细（每只股票以 {prices.index.min().date()} "
                        f"收盘价为 1.0000）")
            header_line = f"  {'日期':<12s}" + "".join([f"{n_:>12s}" for n_ in names])
            logger.info(header_line)
            logger.info("  " + "-" * (12 + 12 * len(names)))
            for date in stock_nav.index:
                row = f"  {date.strftime('%Y-%m-%d'):<12s}"
                for n_ in names:
                    row += f"{float(stock_nav.at[date, n_]):>12.4f}"
                logger.info(row)

        # =====================================================================
        # 各持仓股区间表现分析
        # =====================================================================
        period_start_nav = cumulative_nav
        period_end_nav   = float(abs_nav.iloc[-1])
        period_ret = (period_end_nav / period_start_nav - 1.0) if period_start_nav > 0 else 0.0

        # ---- 组合区间年化收益（使用交易日，与波动率时间基准一致） ----
        period_trading_days = len(abs_nav)
        period_years = period_trading_days / 252 if period_trading_days > 0 else 1.0
        if period_years > 0 and period_start_nav > 0 and period_end_nav > 0:
            period_ann_ret = (period_end_nav / period_start_nav) ** (1 / period_years) - 1
        else:
            period_ann_ret = 0.0

        logger.info("")
        logger.info(f"  各持仓股区间表现分析（{prices.index.min().date()} ~ "
                    f"{prices.index.max().date()}）")
        hdr = (f"  {'股票':<18s}{'权重':>9s}{'期末净值':>11s}{'区间收益':>11s}"
               f"{'年化收益':>11s}{'最大回撤':>11s}{'年化波动':>11s}"
               f"{'夏普':>9s}{'收益贡献':>11s}")
        logger.info(hdr)
        logger.info("  " + "-" * (len(hdr) - 2))

        stock_metrics_period: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(names):
            s_nav = stock_nav[name].dropna()
            m = compute_stock_metrics(s_nav)
            stock_metrics_period[name] = m

            contrib = weights[i] * m["total_ret"]

            logger.info(
                f"  {name:<18s}"
                f"{weights[i]:>9.4f}"
                f"{m['final_nav']:>11.4f}"
                f"{m['total_ret']*100:>10.2f}%"
                f"{m['annual_ret']*100:>10.2f}%"
                f"{m['max_dd']*100:>10.2f}%"
                f"{m['ann_vol']*100:>10.2f}%"
                f"{m['sharpe']:>9.4f}"
                f"{contrib*100:>10.2f}%"
            )
        logger.info("  " + "-" * (len(hdr) - 2))

        logger.info(
            f"  {'组合(TOTAL)':<18s}"
            f"{1.0:>9.4f}"
            f"{period_end_nav:>11.4f}"
            f"{period_ret*100:>10.2f}%"
            f"{period_ann_ret*100:>10.2f}%"
            f"{'':>10s}"
            f"{'':>10s}"
            f"{'':>9s}"
            f"{period_ret*100:>10.2f}%"
        )
        logger.info("")

        for i, name in enumerate(names):
            m = stock_metrics_period[name]
            stock_overview_rows.append({
                "period":       pname,
                "period_k":     k,
                "name":         name,
                "weight":       weights[i],
                "start":        prices.index.min(),
                "end":          prices.index.max(),
                "trading_days": m["trading_days"],
                "final_nav":    m["final_nav"],
                "total_ret":    m["total_ret"],
                "annual_ret":   m["annual_ret"],
                "max_dd":       m["max_dd"],
                "ann_vol":      m["ann_vol"],
                "sharpe":       m["sharpe"],
                "contrib":      weights[i] * m["total_ret"],
            })

        logger.info("  各持仓股权重漂移（首日 → 末日）：")
        for name in names:
            w_first = float(weights_daily[name].iloc[0])
            w_last  = float(weights_daily[name].iloc[-1])
            logger.info(
                f"    {name:<18s}{w_first*100:>8.2f}%  →  {w_last*100:>8.2f}%"
            )
        logger.info("")

        total_nav_parts.append(pd.DataFrame({"TOTAL": abs_nav}))

        logger.info(
            f"  期初净值 : {period_start_nav:.4f}   期末净值(费后) : {period_end_nav:.4f}   "
            f"区间收益(费后) : {period_ret*100:.2f}%"
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
    logger.info("多时间段概览（净值已扣双边交易成本）")
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

    # =========================================================================
    # 跨时间段各持仓股汇总表
    # =========================================================================
    if stock_overview_rows:
        logger.info("")
        logger.info("=" * 78)
        logger.info("跨时间段 · 各持仓股表现汇总（个股为价格收益，未含交易成本）")
        logger.info("=" * 78)
        hdr2 = (f"{'时间段':<18s}{'股票':<16s}{'权重':>8s}"
                f"{'期末净值':>11s}{'区间收益':>11s}{'年化收益':>11s}"
                f"{'最大回撤':>11s}{'年化波动':>11s}{'夏普':>9s}{'收益贡献':>11s}")
        logger.info(hdr2)
        logger.info("-" * len(hdr2))
        for row in stock_overview_rows:
            logger.info(
                f"{row['period'][:16]:<18s}"
                f"{row['name'][:14]:<16s}"
                f"{row['weight']:>8.4f}"
                f"{row['final_nav']:>11.4f}"
                f"{row['total_ret']*100:>10.2f}%"
                f"{row['annual_ret']*100:>10.2f}%"
                f"{row['max_dd']*100:>10.2f}%"
                f"{row['ann_vol']*100:>10.2f}%"
                f"{row['sharpe']:>9.4f}"
                f"{row['contrib']*100:>10.2f}%"
            )
        logger.info("-" * len(hdr2))

    # ---- 汇总统计 ----
    logger.info("")
    logger.info("=" * 78)
    logger.info("整体回测结果汇总（净值已扣双边交易成本）")
    logger.info("=" * 78)

    s = nav["TOTAL"]
    final_nav = float(s.iloc[-1])
    total_ret = final_nav - 1.0

    trading_days = len(s)
    years = trading_days / 252 if trading_days > 0 else 1.0
    ann_ret = (final_nav ** (1 / years) - 1) if (years > 0 and final_nav > 0) else 0.0

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
        f"{'TOTAL (net)':<18s}{final_nav:>11.4f}{total_ret*100:>11.2f}%"
        f"{ann_ret*100:>11.2f}%{max_dd*100:>11.2f}%"
        f"{ann_vol*100:>11.2f}%{sharpe:>10.4f}"
    )
    logger.info("-" * 90)

    logger.info(f"回测区间       : {s.index.min().date()} ~ {s.index.max().date()}")
    logger.info(f"总交易日数     : {len(s)}")
    logger.info(f"单边成本       : {cost_rate*100:.4f}%")
    logger.info(f"累计交易成本   : {total_cost_accum:.6f}  (以初始资金为单位)")
    logger.info(f"费前累计收益   : {(final_nav + total_cost_accum) - 1.0:.4%}")
    logger.info(f"费后累计收益   : {total_ret:.4%}")

    # =========================================================================
    # 基准数据加载、对齐、统计，以及组合 vs 基准的主动管理指标
    # =========================================================================
    #
    # 【修复点】
    # 旧代码:
    #     benchmark_aligned = benchmark_nav.reindex(s.index).ffill()
    #     if benchmark_aligned.isna().any():
    #         benchmark_aligned = benchmark_aligned.bfill()
    # 其中 bfill 会用未来的基准净值回填组合起始日之前的空白区间，
    # 属于前视偏差，会污染基准统计与信息比率。
    #
    # 新代码:
    #     只做 ffill（填补中间缺失），不做 bfill。
    #     组合起始早于基准起始时，开头保留 NaN，
    #     由后续 dropna() 自动丢弃，
    #     所有指标只在组合与基准真实重叠的交易日上计算。
    # =========================================================================
    benchmark_aligned = None
    ir_metrics: Dict[str, float] = {
        "active_return":     0.0,
        "tracking_error":    0.0,
        "information_ratio": 0.0,
        "excess_daily_mean": 0.0,
        "excess_daily_std":  0.0,
        "n_obs":             0,
        "overlap_start":     None,
        "overlap_end":       None,
    }

    if benchmark_file:
        try:
            benchmark_nav = load_benchmark_data(benchmark_file)
            logger.info("")
            logger.info("=" * 78)
            logger.info(f"基准数据已加载：{benchmark_file}，共 {len(benchmark_nav)} 条")
            logger.info(f"基准原始区间 : {benchmark_nav.index.min().date()} ~ {benchmark_nav.index.max().date()}")
            logger.info("=" * 78)

            # 只做前向填充，避免 bfill 带来的前视偏差
            benchmark_aligned = benchmark_nav.reindex(s.index).ffill()

            # 诊断：有多少日期基准确实缺失（通常是组合起始早于基准起始的部分）
            n_missing = int(benchmark_aligned.isna().sum())
            if n_missing > 0:
                first_valid = benchmark_aligned.first_valid_index()
                logger.warning(
                    f"基准在组合区间内有 {n_missing} 个交易日无有效数据"
                    f"（首个有效日: {first_valid.date() if first_valid is not None else 'N/A'}），"
                    f"这些日期将被 dropna() 排除，不参与基准统计与信息比率计算。"
                )

            if benchmark_aligned.notna().any():
                b = benchmark_aligned.dropna()
                if len(b) > 0:
                    b_final = float(b.iloc[-1])
                    b_start = float(b.iloc[0])
                    b_total_ret = b_final / b_start - 1.0

                    b_trading_days = len(b)
                    b_years = b_trading_days / 252 if b_trading_days > 0 else 1.0
                    b_ann_ret = (b_final / b_start) ** (1 / b_years) - 1 \
                        if (b_years > 0 and b_start > 0 and b_final > 0) else 0.0

                    b_cummax = b.cummax()
                    b_dd = (b - b_cummax) / b_cummax
                    b_max_dd = float(b_dd.min())
                    b_daily_ret = b.pct_change().dropna()
                    b_ann_vol = float(b_daily_ret.std() * np.sqrt(252)) if len(b_daily_ret) > 1 else 0.0
                    b_sharpe = b_ann_ret / b_ann_vol if b_ann_vol > 0 else 0.0

                    logger.info("")
                    logger.info("=" * 78)
                    logger.info("基准（HS300 ETF 510300）业绩汇总（与组合真实重叠区间）")
                    logger.info("=" * 78)
                    logger.info(f"基准区间     : {b.index.min().date()} ~ {b.index.max().date()}")
                    logger.info(f"基准交易日数 : {len(b)}")
                    logger.info(header)
                    logger.info("-" * 90)
                    logger.info(
                        f"{'BENCHMARK':<18s}{b_final:>11.4f}{b_total_ret*100:>11.2f}%"
                        f"{b_ann_ret*100:>11.2f}%{b_max_dd*100:>11.2f}%"
                        f"{b_ann_vol*100:>11.2f}%{b_sharpe:>10.4f}"
                    )
                    logger.info("-" * 90)

                    # =====================================================
                    # 组合 vs 基准 · 主动管理指标（信息比率等）
                    # 组合净值已扣双边交易成本
                    # =====================================================
                    ir_metrics = compute_information_ratio(s, b, periods_per_year=252)

                    logger.info("")
                    logger.info("=" * 78)
                    logger.info("组合(费后) vs 基准 · 主动管理指标（Information Ratio）")
                    logger.info("=" * 78)
                    if ir_metrics["overlap_start"] is not None:
                        logger.info(
                            f"实际重叠区间           : "
                            f"{ir_metrics['overlap_start'].date()} ~ "
                            f"{ir_metrics['overlap_end'].date()}"
                        )
                    logger.info(f"有效样本交易日         : {ir_metrics['n_obs']}")
                    logger.info(f"日均超额收益           : {ir_metrics['excess_daily_mean']*100:>10.4f}%")
                    logger.info(f"日超额收益标准差       : {ir_metrics['excess_daily_std']*100:>10.4f}%")
                    logger.info(f"年化超额收益(Active)   : {ir_metrics['active_return']*100:>10.2f}%")
                    logger.info(f"年化跟踪误差(TE)       : {ir_metrics['tracking_error']*100:>10.2f}%")
                    logger.info(f"信息比率 (IR)          : {ir_metrics['information_ratio']:>10.4f}")

                    # 直观解读
                    ir_val = ir_metrics["information_ratio"]
                    if ir_metrics["tracking_error"] <= 0:
                        ir_hint = "无法评估（跟踪误差为 0，组合与基准几乎完全同步）"
                    elif ir_val >= 1.0:
                        ir_hint = "优秀（IR ≥ 1，主动管理创造显著超额收益）"
                    elif ir_val >= 0.5:
                        ir_hint = "良好（0.5 ≤ IR < 1，主动管理有一定成效）"
                    elif ir_val > 0:
                        ir_hint = "一般（0 < IR < 0.5，超额收益有限）"
                    else:
                        ir_hint = "偏弱（IR ≤ 0，未跑赢基准）"
                    logger.info(f"评估                   : {ir_hint}")
                    logger.info("=" * 78)
            else:
                logger.warning("基准数据与组合日期无重叠，无法计算基准指标与信息比率")
                benchmark_aligned = None
        except Exception as e:
            logger.warning(f"加载基准数据失败：{e}，将不绘制基准曲线")
            benchmark_aligned = None

    # ---- 可视化 ----
    if plot:
        try:
            plot_nav(
                nav_total=s,
                output_path=plot_file,
                period_ranges=period_overview,
                benchmark_nav=benchmark_aligned,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"绘制净值曲线图失败：{e}")

    logger.info(f"回测结束，日志已保存至：{os.path.abspath(log_file)}")
    logger.info("=" * 78)

    summary = [{
        "name":              "TOTAL",
        "final_nav":         final_nav,
        "total_return":      total_ret,
        "annual_return":     ann_ret,
        "max_drawdown":      max_dd,
        "annual_vol":        ann_vol,
        "sharpe":            sharpe,
        "active_return":     ir_metrics["active_return"],
        "tracking_error":    ir_metrics["tracking_error"],
        "information_ratio": ir_metrics["information_ratio"],
        "total_cost":        total_cost_accum,
        "cost_rate":         cost_rate,
    }]

    return nav, summary


# -----------------------------------------------------------------------------
# 演示：生成模拟数据
# -----------------------------------------------------------------------------
def _make_demo_files() -> List[str]:
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    specs = [
        ("STOCK_A", 0.0006, 0.015),
        ("STOCK_B", 0.0003, 0.020),
        ("STOCK_C", 0.0004, 0.018),
        ("STOCK_D", 0.0005, 0.016),
        ("STOCK_E", 0.0002, 0.022),
        ("STOCK_F", 0.0007, 0.017),
        ("STOCK_G", 0.0001, 0.019),
    ]
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
        demo_files = _make_demo_files()

        periods = [
            {
                "name":       "第一阶段 A+B+C+D+E",
                "csv_files":  demo_files[:5],
                "weights":    [0.3, 0.25, 0.2, 0.15, 0.1],
                "start_date": "2020-01-01",
                "end_date":   "2021-12-31",
            },
            {
                "name":       "第二阶段 A+C+F+G",
                "csv_files":  [demo_files[0], demo_files[2], demo_files[5], demo_files[6]],
                "weights":    [0.4, 0.3, 0.2, 0.1],
                "start_date": "2022-01-01",
                "end_date":   "2023-06-30",
            },
            {
                "name":       "第三阶段 B+C+F",
                "csv_files":  [demo_files[1], demo_files[2], demo_files[5]],
                "weights":    None,
                "start_date": "2023-07-01",
                "end_date":   "2024-12-31",
            },
        ]

    run_backtest(
        periods=periods,
        initial_capital=1.0,
        log_file="portfolio_backtest.log",
        align="inner",
        plot=True,
        plot_file="portfolio_nav.png",
        benchmark_file="hs300etf_510300_performance.csv",
        log_daily_stock_nav=True,
        cost_rate=0.005,   # 单边 50 bp，双边各扣一次
    )
