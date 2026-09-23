"""
CSI 300 ETF (510300) Buy-and-Hold Performance Analysis
Data source: akshare (East Money)
Adjustment: back-adjusted (hfq)
Holding period: 2021-01-01 to 2026-01-01
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 1. Data Fetching ====================

def fetch_etf_data(symbol: str = "510300",
                   start_date: str = "20210101",
                   end_date: str = "20260101",
                   adjust: str = "hfq") -> pd.DataFrame:
    """
    从 akshare 获取 ETF 后复权日K数据。

    Parameters
    ----------
    symbol : str
        ETF 代码，沪深300ETF为 "510300"
    start_date, end_date : str
        日期范围，格式 "YYYYMMDD"
    adjust : str
        复权方式："hfq"=后复权, "qfq"=前复权, ""=不复权

    Returns
    -------
    pd.DataFrame，索引为日期，含列 open/close/high/low/volume/amount 等
    """
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
            f"Please check the symbol and date range."
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
                             rf_annual: float = 0.02,
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
    """
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

    # ---- Sharpe ratio ----
    sharpe = (annual_return - rf_annual) / annual_vol if annual_vol > 0 else 0.0

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
        "Annualized Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
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
                       csv_path: str = None) -> str:
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
        输出路径，默认 "hs300etf_{symbol}_performance.csv"

    Returns
    -------
    str : 实际保存路径
    """
    if csv_path is None:
        csv_path = f"hs300etf_{symbol}_performance.csv"

    start_date = df.index[0].date().isoformat()
    end_date = df.index[-1].date().isoformat()

    # Scalar metrics table
    summary_rows = [
        ("Symbol", symbol),
        ("Start Date", start_date),
        ("End Date", end_date),
        ("Adjustment", "hfq (back-adjusted)"),
        ("Initial Price", f"{metrics['Initial Price']:.4f}"),
        ("Final Price", f"{metrics['Final Price']:.4f}"),
        ("Trading Days", metrics["Trading Days"]),
        ("Total Return", f"{metrics['Total Return']:.6f}"),
        ("Annualized Return", f"{metrics['Annualized Return']:.6f}"),
        ("Annualized Volatility", f"{metrics['Annualized Volatility']:.6f}"),
        ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.6f}"),
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

def plot_results(metrics: dict, symbol: str = "510300"):
    """绘制净值曲线与回撤曲线。"""
    cum = metrics["Cumulative NAV"]
    dd = metrics["Drawdown Series"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Cumulative NAV
    axes[0].plot(cum.index, cum.values, color="#1f77b4", linewidth=1.5)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Cumulative NAV")
    axes[0].set_title(f"CSI 300 ETF ({symbol}) Buy-and-Hold NAV (Back-Adjusted)",
                      fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Drawdown
    axes[1].fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.4)
    axes[1].plot(dd.index, dd.values, color="#d62728", linewidth=1)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"Drawdown (Max Drawdown {metrics['Max Drawdown']:.2%})")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"hs300etf_{symbol}_performance.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==================== 4. Main ====================

if __name__ == "__main__":
    SYMBOL = "510300"
    START = "20210101"
    END = "20260101"

    # Fetch data
    df = fetch_etf_data(SYMBOL, START, END, adjust="hfq")

    # Compute metrics
    metrics = calc_performance_metrics(df["close"], rf_annual=0.02)

    # Print report
    print("=" * 60)
    print(f"  CSI 300 ETF ({SYMBOL}) Buy-and-Hold Performance Report")
    print(f"  Holding period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Adjustment: hfq (back-adjusted)")
    print("=" * 60)
    print(f"  Initial Price (hfq)     : {metrics['Initial Price']:.4f}")
    print(f"  Final Price (hfq)       : {metrics['Final Price']:.4f}")
    print(f"  Trading Days            : {metrics['Trading Days']}")
    print("-" * 60)
    print(f"  Total Return            : {metrics['Total Return']:.2%}")
    print(f"  Annualized Return       : {metrics['Annualized Return']:.2%}")
    print(f"  Annualized Volatility   : {metrics['Annualized Volatility']:.2%}")
    print(f"  Sharpe Ratio (rf=2%)    : {metrics['Sharpe Ratio']:.4f}")
    print(f"  Max Drawdown            : {metrics['Max Drawdown']:.2%}")
    print(f"  Max Drawdown Period     : {metrics['Max Drawdown Start'].date()} "
          f"-> {metrics['Max Drawdown End'].date()}")
    print(f"  Calmar Ratio            : {metrics['Calmar Ratio']:.4f}")
    print(f"  Daily Win Rate          : {metrics['Daily Win Rate']:.2%}")
    print("=" * 60)

    # Save CSV report
    save_report_to_csv(metrics, df, SYMBOL)

    # Plot
    plot_results(metrics, SYMBOL)





    