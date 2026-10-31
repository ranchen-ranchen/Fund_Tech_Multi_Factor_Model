import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def calc_excess_return_distribution(
    stock_dfs,
    benchmark_df,
    start_date,
    end_date,
    stock_names=None,
    bins=30,
    kde=True,
    figsize=(10, 6),
    title=None,
    ax=None,
    return_df=False,
    verbose=True,
):
    """
    计算一组股票在 [start_date, end_date] 区间内相对基准的超额收益，
    并用 seaborn 直方图展示「超额收益 -> 股票数目」的横截面分布。

    参数
    ----------
    stock_dfs : list[pd.DataFrame]
        每只股票一个 DataFrame，必须包含 'date' 和 'close' 两列。
    benchmark_df : pd.DataFrame
        基准指数日线数据，同样包含 'date' 和 'close' 两列。
    start_date, end_date : str 或 datetime-like
        计算区间（含边界）。
    stock_names : list[str], 可选
        股票名称列表，长度须与 stock_dfs 一致；缺省用 stock_0, stock_1 ...
    bins : int
        直方图分箱数。
    kde : bool
        是否叠加核密度曲线。
    figsize : tuple
        图形尺寸（仅在不传 ax 时生效）。
    title : str, 可选
        图标题。
    ax : matplotlib.axes.Axes, 可选
        指定坐标轴，便于多图拼版。
    return_df : bool
        为 True 时额外返回每只股票的收益明细 DataFrame。
    verbose : bool
        是否打印统计摘要。

    返回
    -------
    ax : matplotlib.axes.Axes
        或 (result_df, ax)，当 return_df=True。
    """

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if start > end:
        raise ValueError("start_date 不能晚于 end_date")

    # ---------- 内部工具 ----------
    def _clean(df):
        """统一格式：date 转 datetime，close 转数值，按日期排序去重。"""
        d = df.loc[:, ["date", "close"]].copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=["date", "close"])
        d = d.sort_values("date").drop_duplicates(subset="date", keep="last")
        return d

    def _period_return(df):
        """区间收益率：区间内首个有效收盘价 -> 区间内最后一个有效收盘价。"""
        d = _clean(df)
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if len(d) < 2:
            return np.nan
        p0, p1 = d["close"].iloc[0], d["close"].iloc[-1]
        if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
            return np.nan
        return p1 / p0 - 1.0

    # ---------- 计算收益率 ----------
    bench_ret = _period_return(benchmark_df)
    if not np.isfinite(bench_ret):
        raise ValueError("基准数据在指定区间内有效样本不足（<2 个交易日）")

    n = len(stock_dfs)
    if stock_names is None:
        stock_names = [f"stock_{i}" for i in range(n)]
    if len(stock_names) != n:
        raise ValueError("stock_names 长度必须与 stock_dfs 一致")

    stock_rets = np.array([_period_return(df) for df in stock_dfs], dtype=float)
    excess = stock_rets - bench_ret

    result = pd.DataFrame(
        {
            "stock": stock_names,
            "stock_return": stock_rets,
            "benchmark_return": bench_ret,
            "excess_return": excess,
        }
    ) # .set_index("stock")

    valid = result["excess_return"].dropna()
    n_valid, n_total = len(valid), len(result)
    if n_valid == 0:
        raise ValueError("没有任何股票在指定区间内获得有效收益率")

    # ---------- 绘图 ----------
    sns.set_theme(style="whitegrid", font_scale=1.0)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        x=valid.values,
        bins=bins,
        kde=kde,
        color="steelblue",
        edgecolor="white",
        alpha=0.85,
        ax=ax,
    )

    mean_, med_ = valid.mean(), valid.median()
    ax.axvline(0, color="crimson", ls="--", lw=1.2, label="0")
    ax.axvline(mean_, color="darkorange", ls="-", lw=1.5, label=f"mean = {mean_:.2%}")
    ax.axvline(med_, color="seagreen", ls=":", lw=1.5, label=f"median = {med_:.2%}")

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Excess return")
    ax.set_ylabel("Number of stocks")
    if title is None:
        title = (
            f"Cross-sectional distribution of excess return "
            f"({start:%Y-%m-%d} ~ {end:%Y-%m-%d}, N={n_valid})"
        )
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)

    if verbose:
        print(f"有效股票数: {n_valid}/{n_total}，基准区间收益: {bench_ret:.2%}")
        print(
            f"超额收益 均值: {mean_:.2%}，中位数: {med_:.2%}，"
            f"标准差: {valid.std():.2%}，胜率(>0): {(valid > 0).mean():.2%}"
        )

    return (result, ax) if return_df else ax


# ============================================================
# 使用示例：构造模拟数据并调用
# ============================================================
if __name__ == "__main__":
    def make_random_walk(seed, start="2023-01-01", end="2023-12-31",
                         s0=10.0, mu=0.0004, sigma=0.02):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, end)
        rets = rng.normal(mu, sigma, size=len(dates))
        close = s0 * np.exp(np.cumsum(rets))
        return pd.DataFrame({"date": dates, "close": close})

    # 50 只股票 + 1 个基准
    stock_dfs = [make_random_walk(seed=i) for i in range(50)]
    stock_names = [f"60000{i:02d}.SH" for i in range(50)]
    benchmark_df = make_random_walk(seed=999, mu=0.0002, sigma=0.012)

    result_df, ax = calc_excess_return_distribution(
        stock_dfs=stock_dfs,
        benchmark_df=benchmark_df,
        start_date="2023-03-01",
        end_date="2023-09-30",
        stock_names=stock_names,
        bins=15,
        return_df=True,
    )

    plt.tight_layout()
    plt.savefig("excess_return_hist.png", dpi=150)
    plt.show()

    # 收益明细（按超额收益排序）
    print(result_df.sort_values("excess_return", ascending=False).head())

