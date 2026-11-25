#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSI 300 ETF (510300) Buy-and-Hold Performance Analysis
Data source: akshare (East Money)
Adjustment: back-adjusted (hfq)

测试区间固定为 2021-01-01 至运行脚本当天（不晚于 2026-12-31），
不支持自定义起止日期。

用法示例
--------
# 默认：510300，2021-01-01 至今，后复权
python eval_hs300_perform.py

# 更换标的、复权方式、无风险利率
python eval_hs300_perform.py -s 159919 --adjust qfq --rf 0.015

# 保存图表但不弹窗显示（服务器/无显示器环境使用）
python eval_hs300_perform.py --no-show
"""

import argparse
from datetime import datetime

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

DEFAULT_SYMBOL = "510300"
TEST_START = "20210101"        # 固定测试起始日期（YYYYMMDD）
TEST_END_LIMIT = "20261231"    # 结束日期上限（YYYYMMDD）
DEFAULT_RF = 0.02
ADJUST_CHOICES = {"hfq": "hfq", "qfq": "qfq", "none": ""}
ADJUST_LABELS = {"hfq": "hfq (back-adjusted)",
                 "qfq": "qfq (forward-adjusted)",
                 "none": "none (unadjusted)"}


# ==================== 1. Data Fetching ====================

def get_test_period() -> tuple[str, str]:
    """
    返回固定的测试起止日期（YYYYMMDD）。

    起始日期固定为 2021-01-01；结束日期为运行脚本的当天，
    但不晚于 2026-12-31。
    """
    today = datetime.today().strftime("%Y%m%d")
    end_date = min(today, TEST_END_LIMIT)
    if end_date < TEST_START:
        raise RuntimeError(
            f"系统日期 {today} 早于固定起始日期 {TEST_START}，无法运行测试。"
        )
    return TEST_START, end_date


def fetch_etf_data(symbol: str = DEFAULT_SYMBOL,
                   start_date: str = TEST_START,
                   end_date: str | None = None,
                   adjust: str = "hfq") -> pd.DataFrame:
    """
    从 akshare 获取 ETF 后复权日K数据。

    Parameters
    ----------
    symbol : str
        ETF 代码，沪深300ETF为 "510300"
    start_date, end_date : str
        日期范围，格式 "YYYYMMDD"，应来自 get_test_period()；
        end_date 为 None 时取今天（不晚于 2026-12-31）
    adjust : str
        复权方式："hfq"=后复权, "qfq"=前复权, ""=不复权

    Returns
    -------
    pd.DataFrame，索引为日期，含列 open/close/high/low/volume/amount 等
    """
    if end_date is None:
        end_date = get_test_period()[1]

    print(f"Fetching back-adjusted daily data for {symbol} ({start_date} ~ {end_date}) ...")

    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"akshare returned no data for {symbol}. "
            f"Please check the symbol and date range ({start_date} ~ {end_date})."
        )

    # Data cleaning
    df["日期"] = pd.to_datetime(df["日期"])
    df.sort_values("日期", inplace=True)
    df.set_index("日期", inplace=True)

    # Drop suspended days (zero volume)
    df = df[df["成交量"] > 0].copy()

    if df.empty:
        raise RuntimeError("Data is empty after removing suspended days.")

    # Rename Chinese columns to English
    df.rename(columns={
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "chg",
        "换手率": "turnover",
    }, inplace=True)

    print(f"Retrieved {len(df)} records, "
          f"range: {df.index[0].date()} ~ {df.index[-1].date()}\n")

    return df


# ==================== 2. Performance Metrics ====================

def calc_performance_metrics(close: pd.Series,
                             rf_annual: float = DEFAULT_RF,
                             trading_days: int = 252) -> dict:
    """
    基于后复权收盘价序列计算买入持有策略的绩效指标。

    Parameters
    ----------
    close : pd.Series
        后复权收盘价序列（日期索引）
    rf_annual : float
        年化无风险利率，默认 2%
    trading_days : int
        年交易日数，默认 252

    Returns
    -------
    dict

    口径
    ----
    - Annualized Return 为几何 CAGR（首尾收盘价）。
    - Sharpe Ratio 为标准的算术口径：
          (mean(日收益) × trading_days - rf_annual) / 年化波动
    - Sharpe Ratio (CAGR/vol) 为旧的 CAGR/vol 口径，仅用于对比。
    """
    if len(close) < 2:
        raise ValueError("价格序列不足两个交易日，无法计算收益指标。")

    # ---- Daily returns ----
    daily_ret = close.pct_change().dropna()

    # ---- Cumulative NAV (starts at 1.0) ----
    cum_nav = (1 + daily_ret).cumprod()
    cum_nav = pd.concat([pd.Series([1.0], index=[close.index[0]]), cum_nav])

    # ---- Total return ----
    total_return = close.iloc[-1] / close.iloc[0] - 1

    # ---- Annualized return (geometric) ----
    n_days = len(close)
    years = n_days / trading_days
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # ---- Annualized volatility ----
    annual_vol = daily_ret.std() * np.sqrt(trading_days)

    # ---- Arithmetic annualized return (standard Sharpe numerator) ----
    annual_return_arith = (
        float(daily_ret.mean() * trading_days) if len(daily_ret) > 0 else 0.0
    )

    # ---- Sharpe ratio ----
    # 标准口径：算术年化收益 - rf，再除以年化波动。
    # 旧口径（CAGR/vol）保留为 sharpe_cagr，仅用于对比。
    sharpe_cagr = (
        (annual_return - rf_annual) / annual_vol if annual_vol > 0 else 0.0
    )
    sharpe = (
        (annual_return_arith - rf_annual) / annual_vol if annual_vol > 0 else 0.0
    )

    # ---- Max drawdown ----
    running_max = cum_nav.cummax()
    drawdown = cum_nav / running_max - 1.0
    max_dd = drawdown.min()

    # ---- Drawdown start/end dates ----
    dd_end_idx = drawdown.idxmin()
    dd_start_idx = cum_nav.loc[:dd_end_idx].idxmax()

    # ---- Calmar ratio ----
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.inf

    # ---- Daily win rate ----
    win_rate = (daily_ret > 0).mean()

    return {
        "Trading Days": n_days,
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Annualized Return (Arithmetic)": annual_return_arith,
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Sharpe Ratio (CAGR/vol)": sharpe_cagr,
        "Max Drawdown": max_dd,
        "Max Drawdown Start": dd_start_idx,
        "Max Drawdown End": dd_end_idx,
        "Calmar Ratio": calmar,
        "Daily Win Rate": win_rate,
        "Initial Price": close.iloc[0],
        "Final Price": close.iloc[-1],
        "Cumulative NAV": cum_nav,
        "Drawdown Series": drawdown,
    }


def save_report_to_csv(metrics: dict,
                       df: pd.DataFrame,
                       symbol: str,
                       csv_path: str = None,
                       adjust: str = "hfq") -> str:
    """
    将绩效报告保存为 CSV 文件。

    Parameters
    ----------
    metrics : dict
        绩效指标字典
    df : pd.DataFrame
        原始行情数据
    symbol : str
        ETF 代码，用于默认文件名
    csv_path : str, optional
        输出路径，默认 "hs300etf_{symbol}_{start}_{end}_performance.csv"
    adjust : str
        复权方式，写入报告

    Returns
    -------
    str : 实际保存路径
    """
    start_date = df.index[0].date().isoformat()
    end_date = df.index[-1].date().isoformat()

    if csv_path is None:
        csv_path = (
            f"hs300etf_{symbol}_{df.index[0]:%Y%m%d}_{df.index[-1]:%Y%m%d}"
            f"_performance.csv"
        )

    # Scalar metrics table
    summary_rows = [
        ("Symbol", symbol),
        ("Start Date", start_date),
        ("End Date", end_date),
        ("Adjustment", ADJUST_LABELS.get(adjust, adjust)),
        ("Initial Price", f"{metrics['Initial Price']:.4f}"),
        ("Final Price", f"{metrics['Final Price']:.4f}"),
        ("Trading Days", metrics["Trading Days"]),
        ("Total Return", f"{metrics['Total Return']:.6f}"),
        ("Annualized Return", f"{metrics['Annualized Return']:.6f}"),
        ("Annualized Return (Arithmetic)", f"{metrics['Annualized Return (Arithmetic)']:.6f}"),
        ("Annualized Volatility", f"{metrics['Annualized Volatility']:.6f}"),
        ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.6f}"),
        ("Sharpe Ratio (CAGR/vol)", f"{metrics['Sharpe Ratio (CAGR/vol)']:.6f}"),
        ("Max Drawdown", f"{metrics['Max Drawdown']:.6f}"),
        ("Max Drawdown Start", metrics["Max Drawdown Start"].date().isoformat()),
        ("Max Drawdown End", metrics["Max Drawdown End"].date().isoformat()),
        ("Calmar Ratio", f"{metrics['Calmar Ratio']:.6f}"),
        ("Daily Win Rate", f"{metrics['Daily Win Rate']:.6f}"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Daily series table
    daily_df = pd.DataFrame({
        "Close": df["close"],
        "Cumulative NAV": metrics["Cumulative NAV"],
        "Drawdown": metrics["Drawdown Series"],
    })

    # Write both tables into one CSV with a separator line
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("# Performance Summary\n")
        summary_df.to_csv(f, index=False)
        f.write("\n# Daily Series\n")
        daily_df.to_csv(f)

    print(f"Report saved to: {csv_path}")
    return csv_path


# ==================== 3. Visualization ====================

def plot_results(metrics: dict,
                 symbol: str = DEFAULT_SYMBOL,
                 save_path: str = None,
                 show: bool = True):
    """绘制净值曲线与回撤曲线。"""
    cum = metrics["Cumulative NAV"]
    dd = metrics["Drawdown Series"]

    if save_path is None:
        save_path = (
            f"hs300etf_{symbol}_{cum.index[0]:%Y%m%d}_{cum.index[-1]:%Y%m%d}"
            f"_performance.png"
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Cumulative NAV
    axes[0].plot(cum.index, cum.values, color="#1f77b4", linewidth=1.5)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Cumulative NAV")
    axes[0].set_title(
        f"CSI 300 ETF ({symbol}) Buy-and-Hold NAV (Back-Adjusted)\n"
        f"{cum.index[0].date()} ~ {cum.index[-1].date()}",
        fontsize=13, fontweight="bold",
    )
    axes[0].grid(alpha=0.3)

    # Drawdown
    axes[1].fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    axes[1].plot(dd.index, dd.values, color="#d62728", linewidth=1)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"Drawdown (Max Drawdown {metrics['Max Drawdown']:.2%})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ==================== 4. CLI ====================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="沪深300 ETF 买入持有绩效分析（数据源：akshare/东方财富）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s", "--symbol", default=DEFAULT_SYMBOL,
        help="ETF 代码",
    )
    parser.add_argument(
        "-a", "--adjust", choices=sorted(ADJUST_CHOICES), default="hfq",
        help="复权方式：hfq=后复权, qfq=前复权, none=不复权",
    )
    parser.add_argument(
        "-r", "--rf", type=float, default=DEFAULT_RF,
        metavar="RATE",
        help="年化无风险利率（小数），用于夏普比率",
    )
    parser.add_argument(
        "--csv", default=None, metavar="PATH",
        help="CSV 输出路径（默认自动按标的+区间命名）",
    )
    parser.add_argument(
        "--png", default=None, metavar="PATH",
        help="图片输出路径（默认自动按标的+区间命名）",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="跳过绘图（只输出报告 CSV）",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="保存图片但不弹窗显示（服务器/无显示器环境使用）",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbol = args.symbol
    start, end = get_test_period()

    # Fetch data
    df = fetch_etf_data(symbol, start, end, adjust=ADJUST_CHOICES[args.adjust])

    # Compute metrics
    metrics = calc_performance_metrics(df["close"], rf_annual=args.rf)

    actual_start, actual_end = df.index[0], df.index[-1]
    if actual_start.strftime("%Y%m%d") != start or actual_end.strftime("%Y%m%d") != end:
        print(f"Note: 实际数据区间为 {actual_start.date()} ~ {actual_end.date()}，"
              f"与请求区间（{start} ~ {end}）不同（非交易日/数据边界）。\n")

    # Print report
    print("=" * 60)
    print(f"  CSI 300 ETF ({symbol}) Buy-and-Hold Performance Report")
    print(f"  Holding period: {actual_start.date()} to {actual_end.date()}")
    print(f"  Adjustment: {ADJUST_LABELS[args.adjust]}")
    print("=" * 60)
    print(f"  Initial Price (hfq)     : {metrics['Initial Price']:.4f}")
    print(f"  Final Price (hfq)       : {metrics['Final Price']:.4f}")
    print(f"  Trading Days            : {metrics['Trading Days']}")
    print("-" * 60)
    print(f"  Total Return            : {metrics['Total Return']:.2%}")
    print(f"  Annualized Return       : {metrics['Annualized Return']:.2%}  (CAGR)")
    print(f"  Annualized Return (Arith) : {metrics['Annualized Return (Arithmetic)']:.2%}")
    print(f"  Annualized Volatility   : {metrics['Annualized Volatility']:.2%}")
    print(f"  Sharpe Ratio (arith, rf={args.rf:.1%}) : {metrics['Sharpe Ratio']:.4f}")
    print(f"  Sharpe Ratio (CAGR/vol) : {metrics['Sharpe Ratio (CAGR/vol)']:.4f}  (legacy)")
    print(f"  Max Drawdown            : {metrics['Max Drawdown']:.2%}")
    print(f"  Max Drawdown Period     : {metrics['Max Drawdown Start'].date()} "
          f"-> {metrics['Max Drawdown End'].date()}")
    print(f"  Calmar Ratio            : {metrics['Calmar Ratio']:.4f}")
    print(f"  Daily Win Rate          : {metrics['Daily Win Rate']:.2%}")
    print("=" * 60)

    # Save CSV report
    save_report_to_csv(metrics, df, symbol,
                       csv_path=args.csv, adjust=args.adjust)

    # Plot
    if args.no_plot:
        print("Skipped plotting (--no-plot).")
    else:
        plot_results(metrics, symbol,
                     save_path=args.png, show=not args.no_show)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
