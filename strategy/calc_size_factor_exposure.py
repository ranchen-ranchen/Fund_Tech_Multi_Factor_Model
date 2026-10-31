"""
计算个股在指定时间对规模因子（Size）的暴露并可视化分布
思路：流动市值 ≈ 成交额 / 换手率，规模因子 = ln(流动市值)，再做标准化
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calc_size_factor_exposure(
    data: Dict[str, pd.DataFrame],
    date,
    *,
    turn_col: str = "turn",
    amount_col: str = "amount",
    date_col: str = "date",
    min_turn: float = 0.0,
    winsorize: Optional[float] = 0.01,
    zscore_ddof: int = 0,
    plot: bool = True,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[plt.Figure]]:
    """
    参数
    ----
    data : dict[str, pd.DataFrame]
        key 为股票代码，value 为日 K 数据，需包含 date / turn / amount 列。
        若指定日期该股票无数据，则自动回退到该日期之前最近的一个交易日。
    date : 任意可被 pd.Timestamp 解析的日期，如 "2024-03-01"
    turn_col, amount_col, date_col : 列名
    min_turn : 换手率下限，<= 该值的样本视为无效（如停牌 turn=0）
    winsorize : 标准化前的去极值分位数（如 0.01 表示 1%/99% 分位缩尾）；None 表示不做
    zscore_ddof : 标准化时标准差自由度
    plot : 是否绘制直方图
    bins : 直方图分箱数
    ax : 可传入已有 Axes；否则自动新建
    title : 图标题

    返回
    ----
    result_df : 每只股票一行，含 code / turn / amount / float_mv / log_mv / size_exposure / used_date
    fig : matplotlib Figure（plot=False 时为 None）
    """
    target = pd.Timestamp(date).normalize()

    # ---------- 1. 逐只股票取指定日（或之前最近一日）数据 ----------
    rows = []
    for code, df in data.items():
        if df is None or len(df) == 0:
            continue
        d = df[[date_col, turn_col, amount_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col]).dt.normalize()
        d = d.dropna(subset=[date_col])

        sub = d[d[date_col] == target]
        if sub.empty:                                  # 回退到最近一个可用交易日
            sub = d[d[date_col] <= target].sort_values(date_col).tail(1)
        if sub.empty:                                  # 早于全部历史，跳过
            continue

        r = sub.iloc[-1]
        rows.append(
            {
                "code": code,
                "used_date": r[date_col],
                turn_col: float(r[turn_col]) if pd.notna(r[turn_col]) else np.nan,
                amount_col: float(r[amount_col]) if pd.notna(r[amount_col]) else np.nan,
            }
        )

    if not rows:
        raise ValueError(f"{target.date()} 及其之前没有任何股票的可用数据")

    df_all = pd.DataFrame(rows)

    # ---------- 2. 流动市值 = 成交额 / 换手率 ----------
    df_all = df_all.dropna(subset=[turn_col, amount_col])
    df_all = df_all[(df_all[turn_col] > min_turn) & (df_all[amount_col] > 0)].copy()

    if df_all.empty:
        raise ValueError(f"{target.date()} 无有效样本（换手率/成交额异常）")

    # 换手率若为百分数（如 1.5 表示 1.5%），此处常数不影响取对数后的标准化结果
    df_all["float_mv"] = df_all[amount_col] / df_all[turn_col]
    df_all["log_mv"] = np.log(df_all["float_mv"])

    # ---------- 3. 去极值 + 标准化 ----------
    x = df_all["log_mv"]
    if winsorize is not None and 0 < winsorize < 0.5:
        lo, hi = x.quantile(winsorize), x.quantile(1 - winsorize)
        x = x.clip(lo, hi)

    mu = x.mean()
    sd = x.std(ddof=zscore_ddof)
    sd = sd if (np.isfinite(sd) and sd > 0) else 1.0

    df_all["size_exposure"] = (x - mu) / sd
    df_all = df_all.sort_values("size_exposure").reset_index(drop=True)

    # ---------- 4. 直方图可视化 ----------
    fig = None
    if plot:
        sns.set_theme(style="whitegrid")
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        sns.histplot(
            df_all["size_exposure"],
            bins=bins,
            kde=True,
            color="steelblue",
            edgecolor="white",
            ax=ax,
        )
        ax.axvline(0, color="crimson", linestyle="--", linewidth=1.2, label="mean = 0")
        ax.set_xlabel("Size factor exposure (standardized ln float mkt cap)")
        ax.set_ylabel("Number of stocks")
        ttl = title or f"Size Factor Exposure Distribution on {target.date()}"
        ax.set_title(f"{ttl}  (N={len(df_all)})")
        ax.legend()
#        fig.tight_layout()
#        plt.show()

    return df_all, fig


# =======================================================================
# 以下为完整可运行示例
# =======================================================================
if __name__ == "__main__":
    np.random.seed(2024)

    # 构造 40 只股票、60 个交易日的模拟日 K 数据
    dates = pd.bdate_range("2024-01-02", periods=60)
    stock_data = {}
    for i in range(40):
        code = f"{600000 + i:06d}"
        float_mv = np.random.uniform(2e9, 5e11) * (
            1 + 0.03 * np.random.randn(len(dates))
        )                                                    # 流通市值（元）
        turn = np.random.lognormal(mean=0.0, sigma=0.5, size=len(dates))  # 换手率(%)
        amount = float_mv * turn / 100.0                     # 成交额（元）
        stock_data[code] = pd.DataFrame(
            {"date": dates, "turn": turn, "amount": amount}
        )

    # 人为制造停牌样本，测试异常值处理
    stock_data["600000"].loc[stock_data["600000"].index[-5:], "turn"] = 0.0

    result, fig = calc_size_factor_exposure(
        stock_data,
        date="2024-03-15",
        winsorize=0.01,
        bins=20,
    )

    print(result.head(10).to_string(index=False))
    print(f"\n样本股票数: {len(result)}")
    print(f"暴露度均值: {result['size_exposure'].mean():.6f}")
    print(f"暴露度标准差: {result['size_exposure'].std(ddof=0):.6f}")
    print(f"使用日期范围: {result['used_date'].min().date()} ~ {result['used_date'].max().date()}")
