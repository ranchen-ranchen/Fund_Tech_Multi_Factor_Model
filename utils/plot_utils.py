
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time

import seaborn as sns
import matplotlib.pyplot as plt


def plot_clustered_bars(data, output_path=None):
    """
    Visualize data as a clustered bar chart using seaborn.

    Parameters
    ----------
    data : list of list
        Nested list with 10 groups, each group containing 5 values.
        Shape must be (10, 5): 10 clusters, 5 bars per cluster.
    output_path : str, optional
        If given, the figure is saved to this path.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape != (10, 5):
        raise ValueError(
            f"Expected data with shape (10, 5), got {arr.shape}"
        )

    n_clusters, n_series = arr.shape

    # Build a long-format DataFrame, which is the natural input for seaborn.
    df = pd.DataFrame(
        {
            "Periods": np.repeat(
                [f"P{i + 1}" for i in range(n_clusters)], n_series
            ),
            "Series": np.tile(['size', 'value', 'momentum', 'volatility', 'total'], n_clusters),
            "Value": arr.reshape(-1),
        }
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

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

    ax.set_title("Factor contributions in 10 periods", fontsize=14)
    ax.set_xlabel("Periods")
    ax.set_ylabel("Value")
    ax.legend(title="Factor contrib", ncol=n_series, loc="upper right")

    # Annotate each bar with its value.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

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
