
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time

import seaborn as sns
import matplotlib.pyplot as plt


def plot_clustered_bars(data, output_path=None, periods=None, series=None,
                        y_label="Value", bar_fmt="%.2f", title="Factor contributions"):
    """
    Visualize data as a clustered bar chart using seaborn.

    Parameters
    ----------
    data : list of list
        Nested list with `n_clusters` rows and `n_series` columns.
    output_path : str, optional
        If given, the figure is saved to this path.
    periods : list[str], optional
        每个簇的标签，长度必须等于行数。缺省时：11 行使用 2021h1~2026h1，
        其他行数使用 P1, P2, ...
    series : list[str], optional
        每个序列的标签，长度必须等于列数。缺省时：5 列使用
        size/value/momentum/volatility/total，其他列数使用 S1, S2, ...
    y_label : str
        y 轴标题。
    bar_fmt : str
        柱顶数值格式。
    title : str
        图标题。
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D data, got shape {arr.shape}")

    n_clusters, n_series = arr.shape

    if periods is None:
        if n_clusters == 11:
            periods = ['2021h1', '2021h2', '2022h1', '2022h2', '2023h1',
                       '2023h2', '2024h1', '2024h2', '2025h1', '2025h2', '2026h1']
        else:
            periods = [f"P{i + 1}" for i in range(n_clusters)]
    if series is None:
        if n_series == 5:
            series = ['size', 'value', 'momentum', 'volatility', 'total']
        else:
            series = [f"S{i + 1}" for i in range(n_series)]

    periods = list(periods)
    series = list(series)
    if len(periods) != n_clusters:
        raise ValueError(
            f"periods 长度 {len(periods)} 与数据行数 {n_clusters} 不一致"
        )
    if len(series) != n_series:
        raise ValueError(
            f"series 长度 {len(series)} 与数据列数 {n_series} 不一致"
        )

    # Build a long-format DataFrame, which is the natural input for seaborn.
    df = pd.DataFrame(
        {
            "Periods": np.repeat(periods, n_series),
            "Series": np.tile(series, n_clusters),
            "Value": arr.reshape(-1),
        }
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, n_clusters * 1.1), 6))

    sns.barplot(
        data=df,
        x="Periods",
        y="Value",
        hue="Series",
        palette="muted",
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Periods")
    ax.set_ylabel(y_label)
    ax.legend(title="Factor contrib", ncol=min(n_series, 4), loc="upper right")

    # Annotate each bar with its value.
    for container in ax.containers:
        ax.bar_label(container, fmt=bar_fmt, fontsize=7, padding=2)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, ax





def plot_heatmap(code, data, x_ticks, y_ticks):
    img_name = 'ror_sens_{0:}.png'.format(code)
    # 1. 构造数据
    df = pd.DataFrame(data, index=y_ticks, columns=x_ticks)

    # 2. 设置画布
    plt.figure(figsize=(10, 8))
    # 3. 绘制热力图
    sns.heatmap(
    df,
    annot=True,        # 显示单元格数值
    fmt=".2f",         # 数值保留2位小数
    cmap="YlOrRd",     # 配色方案
    linewidths=0.5,    # 格子边框宽度
    cbar=True          # 显示颜色条
    )
    # 4. 标题+显示
    plt.title(f"The sensitivity of RoR of {code:} to parameters", fontsize=14)
    plt.xlabel("buy parameter")
    plt.ylabel("sell parameter")
    #plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    #plt.show()


def plot_k_line(df, mid, upper, lower, code):
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    df["date"] = pd.to_datetime(df["date"])
    df["date_num"] = df["date"].apply(mdates.date2num)

    cols = ["open","high","low","close"]
    df[cols] = df[cols].astype(float)
    ohlc = df[["date_num","open","high","low","close"]].values
    fig, ax = plt.subplots(figsize=(16, 8))
    # 绘制K线
    candlestick_ohlc(ax, ohlc, width=0.6, colorup="#E74C3C", colordown="#3498DB")
    df["ma20"] = mid
    df["upper"] = upper
    df["lower"] = lower
    # 绘制布林带 + 均线
    sns.lineplot(data=df, x="date", y="ma20", color="#F39C12", linewidth=2, label="MA20", ax=ax)
    sns.lineplot(data=df, x="date", y="upper", color="#27AE60", linewidth=1.5, label="Upper", ax=ax)
    sns.lineplot(data=df, x="date", y="lower", color="#8E44AD", linewidth=1.5, label="Lower", ax=ax)
    # 美化坐标轴
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    plt.title("{0:} K-line & BOLL".format(code), fontsize=16)
    plt.ylabel("Stock Price")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    img_name = 'img_{0:}.png'.format(code)
    plt.savefig(img_name)




if __name__ == "__main__":
    pass
