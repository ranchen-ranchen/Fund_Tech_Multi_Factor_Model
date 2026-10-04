"""
一揽子股票长期持有（等权重/自定义权重）回测程序 —— 支持多时间段

用法:
    python backtest_buy_hold.py a.csv b.csv c.csv      # 单时间段
    python backtest_buy_hold.py                         # 无参数时用模拟数据演示（多时间段）

输入: 一系列股票 K 线 CSV 文件（需包含日期列与收盘价列）
      可选基准文件 hs300etf_510300_performance.csv
输出: portfolio_backtest.log 日志文件
      portfolio_nav.png      净值曲线图（总资产净值 + 基准）
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
    文件格式：前面是 Performance Summary，之后是 '# Daily Series'，
    随后是列名行（,Close,Cumulative NAV,Drawdown）和每日数据。
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

    # 跳过 '# Daily Series' 行及其之前的所有行，下一行作为列名
    df = pd.read_csv(csv_path, skiprows=start_idx + 1, encoding="utf-8-sig")
    # 列名可能为 ['Unnamed: 0', 'Close', 'Cumulative NAV', 'Drawdown']
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
    """
    绘制组合总资产净值随时间变化的曲线，可叠加基准净值。

    参数
    ----
    nav_total     : 组合总净值 Series（索引为日期）
    output_path   : 输出图片路径
    dpi           : 图像分辨率
    period_ranges : 时间段配置列表，每个元素包含 'name' / 'start' / 'end'。
                    用于绘制交替背景阴影和分界线；为 None 时忽略。
    benchmark_nav : 基准净值 Series（索引为日期），为 None 时不绘制。
    logger        : 可选日志对象
    """
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

    # 中文字体兼容
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))

    # ---- 时间段交替背景阴影 ----
    if period_ranges:
        for pi, pr in enumerate(period_ranges):
            if pi % 2 == 1:
                s_ = pd.to_datetime(pr["start"])
                e_ = pd.to_datetime(pr["end"])
                ax.axvspan(s_, e_, color="gray", alpha=0.06, zorder=0)

    # ---- 总净值曲线 ----
    idx  = nav_total.index
    vals = nav_total.values

    ax.plot(idx, vals, color="black", linewidth=2.0, label="TOTAL NAV")
    ax.fill_between(idx, 1.0, vals, where=(vals >= 1.0),
                    color="green", alpha=0.12, interpolate=True)
    ax.fill_between(idx, 1.0, vals, where=(vals < 1.0),
                    color="red", alpha=0.12, interpolate=True)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

    # ---- 基准净值曲线 ----
    if benchmark_nav is not None and not benchmark_nav.empty:
        ax.plot(benchmark_nav.index, benchmark_nav.values,
                color="blue", linewidth=1.5, linestyle="--",
                label="HS300 ETF (510300)")

    # ---- 时间段分界线 ----
    if period_ranges:
        for pi, pr in enumerate(period_ranges, 1):
            if pi > 1:
                bd = pd.to_datetime(pr["start"])
                ax.axvline(x=bd, color="gray", linestyle="--",
                           linewidth=0.7, alpha=0.5)

    ax.set_ylabel("Cumulative NAV")
    ax.set_xlabel("Date")
    ax.set_title("Total Portfolio NAV vs Benchmark", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # 日期格式优化
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
# 核心回测（多时间段）
# -----------------------------------------------------------------------------
def run_backtest(
    periods: List[Dict[str, Any]],
    initial_capital: float = 1.0,
    log_file: str = "portfolio_backtest.log",
    align: str = "inner",
    plot: bool = True,
    plot_file: str = "portfolio_nav.png",
    benchmark_file: Optional[str] = None,
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
    plot            : 是否绘制净值曲线图
    plot_file       : 净值曲线图输出路径
    benchmark_file  : 基准 CSV 文件路径（如 hs300etf_510300_performance.csv）

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

    cumulative_nav = 1.0
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
            # outer：仅前向填充，避免 bfill 引入前视偏差
            prices = prices.sort_index().ffill()

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
        abs_nav = total.div(initial_capital)         # 组合累计净值（跨周期连续）

        # ---- 输出每日净值明细（仅 TOTAL） ----
        logger.info("")
        logger.info(f"  每日总净值明细（期初 = {cumulative_nav:.4f}）")
        logger.info(f"  {'日期':<12s}{'TOTAL':>13s}")
        logger.info("  " + "-" * 27)
        for date in abs_nav.index:
            logger.info(f"  {date.strftime('%Y-%m-%d'):<12s}{float(abs_nav.at[date]):>13.4f}")

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

    # =========================================================================
    # 基准数据加载、对齐与统计
    # =========================================================================
    benchmark_aligned = None
    if benchmark_file:
        try:
            benchmark_nav = load_benchmark_data(benchmark_file)
            logger.info("")
            logger.info("=" * 78)
            logger.info(f"基准数据已加载：{benchmark_file}，共 {len(benchmark_nav)} 条")
            logger.info(f"基准原始区间 : {benchmark_nav.index.min().date()} ~ {benchmark_nav.index.max().date()}")
            logger.info("=" * 78)

            # 对齐到组合的日期索引（前向填充）
            benchmark_aligned = benchmark_nav.reindex(s.index).ffill()
            # 若开头仍有缺失，用后向填充补齐（仅影响开头极少数点）
            if benchmark_aligned.isna().any():
                benchmark_aligned = benchmark_aligned.bfill()

            if benchmark_aligned.notna().any():
                b = benchmark_aligned.dropna()
                if len(b) > 0:
                    b_final = float(b.iloc[-1])
                    b_start = float(b.iloc[0])
                    b_total_ret = b_final / b_start - 1.0
                    b_days = (b.index[-1] - b.index[0]).days
                    b_years = b_days / 365.25 if b_days > 0 else 1.0
                    b_ann_ret = (b_final ** (1 / b_years) - 1) if (b_years > 0 and b_final > 0) else 0.0
                    b_cummax = b.cummax()
                    b_dd = (b - b_cummax) / b_cummax
                    b_max_dd = float(b_dd.min())
                    b_daily_ret = b.pct_change().dropna()
                    b_ann_vol = float(b_daily_ret.std() * np.sqrt(252)) if len(b_daily_ret) > 1 else 0.0
                    b_sharpe = b_ann_ret / b_ann_vol if b_ann_vol > 0 else 0.0

                    logger.info("")
                    logger.info("=" * 78)
                    logger.info("基准（HS300 ETF 510300）业绩汇总（与组合同区间）")
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
            else:
                logger.warning("基准数据与组合日期无重叠，无法计算基准指标")
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
        plot=True,                     # 是否输出净值图
        plot_file="portfolio_nav.png", # 输出图片路径
        benchmark_file="hs300etf_510300_performance.csv",  # 基准文件
    )
