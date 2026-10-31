"""
动量因子暴露计算 + 横截面标准化 + seaborn 直方图可视化

依赖: numpy, pandas, seaborn, matplotlib
运行: python momentum_exposure.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


def momentum_exposure_histogram(
    price_dfs,
    start_date,
    end_date,
    names=None,
    skip_days=0,
    winsorize=True,
    winsor_limits=(0.01, 0.99),
    bins=20,
    figsize=(10, 6),
    price_col="close",
    date_col="date",
    title=None,
    return_fig=False,
):
    """
    计算一系列个股在指定时间段内的动量因子暴露，横截面标准化后画直方图。

    Parameters
    ----------
    price_dfs : list[pd.DataFrame] 或单个 pd.DataFrame
        每个 DataFrame 至少包含 date、close 两列（日 K 数据）。
    start_date, end_date : str 或 datetime-like
        计算区间起止日期（含端点）。动量 = 区间内 末价/首价 - 1。
    names : list[str], optional
        股票名称/代码列表，长度须与 price_dfs 一致；缺省用 stock_0, stock_1...
    skip_days : int
        区间末端跳过最近多少个交易日再取价格（默认 0，即用区间最后一个收盘价）。
    winsorize : bool
        是否在标准化前对原始暴露做缩尾处理。
    winsor_limits : tuple(float, float)
        缩尾分位数。
    bins : int
        直方图分箱数。
    figsize : tuple
        图形尺寸。
    price_col, date_col : str
        价格列与日期列列名。
    title : str, optional
        图标题，缺省自动生成。
    return_fig : bool
        True 时返回 (结果 DataFrame, matplotlib Figure)。

    Returns
    -------
    pd.DataFrame
        列: stock, momentum_raw, momentum_z
        （返回 (df, fig) 当 return_fig=True）
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")
    if skip_days < 0:
        raise ValueError("skip_days 必须 >= 0")

    if isinstance(price_dfs, pd.DataFrame):
        price_dfs = [price_dfs]
    if not isinstance(price_dfs, (list, tuple)):
        raise TypeError("price_dfs 必须是 DataFrame 或 DataFrame 列表")

    if names is None:
        names = [f"stock_{i}" for i in range(len(price_dfs))]
    if len(names) != len(price_dfs):
        raise ValueError("names 长度必须与 price_dfs 一致")

    # ---------- 1. 逐只股票计算原始动量暴露 ----------
    raw = []
    for df in price_dfs:
        if df is None or len(df) == 0:
            raw.append(np.nan)
            continue
        if date_col not in df.columns or price_col not in df.columns:
            raise KeyError(f"DataFrame 缺少必需列: {date_col!r} / {price_col!r}")

        d = df[[date_col, price_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        d = (
            d.dropna()
             .sort_values(date_col)
             .drop_duplicates(subset=date_col)
        )
        d = d[(d[date_col] >= start) & (d[date_col] <= end)]

        # 需要至少 2 个有效价格点（考虑 skip_days 后）
        if len(d) < 2 + skip_days:
            raw.append(np.nan)
            continue

        p_first = float(d[price_col].iloc[0])
        p_last = float(d[price_col].iloc[-1 - skip_days])
        if not np.isfinite(p_first) or not np.isfinite(p_last) or p_first <= 0:
            raw.append(np.nan)
            continue

        raw.append(p_last / p_first - 1.0)

    res = pd.DataFrame({"stock": list(names), "momentum_raw": raw})

    # ---------- 2. 横截面标准化（winsorize + z-score） ----------
    valid_mask = res["momentum_raw"].notna()
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError("所选时间段内没有可用于计算的股票，请检查数据与日期区间")

    x = res.loc[valid_mask, "momentum_raw"].copy()

    if winsorize and n_valid >= 5:  # 样本太少时缩尾无意义
        lo = x.quantile(winsor_limits[0])
        hi = x.quantile(winsor_limits[1])
        x = x.clip(lo, hi)

    mu = x.mean()
    sd = x.std(ddof=1)  # 样本标准差
    if not np.isfinite(sd) or sd == 0:
        print("[警告] 横截面标准差为 0 或不可用，标准化结果全部置为 0")
        z = pd.Series(np.zeros(len(x)), index=x.index)
    else:
        z = (x - mu) / sd

    res["momentum_z"] = np.nan
    res.loc[valid_mask, "momentum_z"] = z.values

    plot_df = res.dropna(subset=["momentum_z"]).copy()

    # ---------- 3. seaborn 直方图 ----------
    fig, ax = plt.subplots(figsize=figsize)
    use_kde = len(plot_df) >= 3
    sns.histplot(
        data=plot_df,
        x="momentum_z",
        bins=bins,
        kde=use_kde,
        color="steelblue",
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    ax.axvline(0.0, color="crimson", linestyle="--", linewidth=1.2,
               label="横截面均值 (z=0)")

    # 标注关键统计量
    stats_text = (
        f"n = {len(plot_df)}\n"
        f"mean = {plot_df['momentum_z'].mean():.3f}\n"
        f"std  = {plot_df['momentum_z'].std(ddof=1):.3f}\n"
        f"min  = {plot_df['momentum_z'].min():.3f}\n"
        f"max  = {plot_df['momentum_z'].max():.3f}"
    )
    ax.text(
        0.985, 0.97, stats_text,
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )

    ax.set_xlabel("Standardized Momentum Exposure (z-score)")
    ax.set_ylabel("Number of Stocks")
    if title is None:
        title = (f"Momentum Factor Exposure Distribution "
                 f"({start.date()} ~ {end.date()}, n={len(plot_df)})")
    ax.set_title(title)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if return_fig:
        return res, fig
    return res


# ============================== 示例运行 ==============================
if __name__ == "__main__":
    # 构造 40 只股票 2022-01-03 ~ 2023-12-29 的模拟日 K 数据
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", "2023-12-29")

    price_dfs, stock_names = [], []
    for i in range(40):
        n = len(dates)
        # 每只股票给一个不同的漂移，制造横截面差异
        drift = rng.normal(0.0002, 0.0005)
        rets = rng.normal(drift, 0.02, size=n)
        close = 10.0 * np.exp(np.cumsum(rets))
        price_dfs.append(pd.DataFrame({"date": dates, "close": close}))
        stock_names.append(f"STK{i:02d}")

    # 计算 2023 年动量暴露并可视化
    result_df, fig = momentum_exposure_histogram(
        price_dfs=price_dfs,
        start_date="2023-01-01",
        end_date="2023-12-31",
        names=stock_names,
        skip_days=0,        # 若想剔除最近 5 个交易日，可设为 5
        winsorize=True,
        bins=15,
        figsize=(10, 6),
        return_fig=True,
    )

    print("前 10 行结果（按标准化暴露排序）:")
    print(result_df.sort_values("momentum_z", ascending=False).head(10).to_string(index=False))

    # 也可保存图片
    # fig.savefig("momentum_exposure_hist.png", dpi=150, bbox_inches="tight")
    plt.show()
