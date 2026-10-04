"""
多因子趋势选股全流程 Demo
------------------------------------------------
1. 趋势状态 -> 可排序因子分数
2. 横截面处理（去极值 + 标准化）
3. 中性化（行业 + 市值回归取残差）
4. 因子合成（等权/自定义权重）
5. 组合优化（均值-方差 + 行业中性 + 个股上限）
"""
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ============================================================
# 0. 全局配置
# ============================================================
CFG = dict(
    n_stocks=80,          # 股票池规模
    n_days=800,           # 交易日数
    n_industries=6,       # 行业数
    lookback_short=20,    # 短期均线
    lookback_mid=60,      # 中期窗口（趋势因子主窗口）
    lookback_long=120,    # 长期均线
    rebalance_freq=20,    # 每 20 个交易日调仓
    cov_window=252,       # 协方差估计窗口
    max_weight=0.05,      # 单票权重上限
    risk_aversion=1.0,    # 风险厌恶系数 λ
    alpha_scale=0.05,     # 因子分数 -> 预期年化收益 的缩放
    trading_days=252,
    seed=7,
)


# ============================================================
# 1. 数据生成（真实场景替换为行情/财务/行业数据接口）
# ============================================================
def generate_data(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n, T = cfg["n_stocks"], cfg["n_days"]
    dates = pd.bdate_range("2019-01-01", periods=T)
    ids = [f"S{i:03d}" for i in range(n)]
    inds = [f"IND{i}" for i in range(cfg["n_industries"])]
    industry_map = {s: inds[i % len(inds)] for i, s in enumerate(ids)}

    mkt = rng.normal(0.0003, 0.009, T)                        # 市场因子
    ind_r = {k: rng.normal(0.0, 0.005, T) for k in inds}      # 行业因子
    beta = rng.uniform(0.6, 1.4, n)                           # 个股 beta
    latent = rng.normal(0.0, 1.5e-4, n)                       # 个股真实 alpha

    R = np.empty((T, n))
    for j, s in enumerate(ids):
        R[:, j] = (latent[j] + beta[j] * mkt + ind_r[industry_map[s]]
                   + rng.normal(0, 0.012, T))

    px = pd.DataFrame(20.0 * np.exp(np.cumsum(R, axis=0)),
                      index=dates, columns=ids)

    mcap = pd.DataFrame(
        np.exp(np.log(rng.uniform(1e9, 5e11, n))
               + np.cumsum(rng.normal(0, 0.004, (T, n)), axis=0)),
        index=dates, columns=ids)
    return px, mcap, industry_map


# ============================================================
# 2. 趋势因子：把「趋势状态」量化成可排序的数值
# ============================================================
def _rolling_slope(logpx: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动 OLS 斜率（对 log 价格），向量化实现。"""
    arr = logpx.values.astype(float)
    T, N = arr.shape
    out = np.full((T, N), np.nan)
    if T < window:
        return pd.DataFrame(out, index=logpx.index, columns=logpx.columns)

    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    sw = sliding_window_view(arr, window, axis=0)      # (T-w+1, N, w)
    out[window - 1:] = (sw * xc).sum(axis=-1) / sxx
    return pd.DataFrame(out, index=logpx.index, columns=logpx.columns)


def compute_factors(px: pd.DataFrame, cfg: dict) -> dict:
    """
    返回 {因子名: DataFrame(T x N)}，每个因子都已是「数值越大趋势越强」的方向。
    """
    logpx = np.log(px)
    ret = px.pct_change()
    ws, wm, wl = cfg["lookback_short"], cfg["lookback_mid"], cfg["lookback_long"]
    vol = ret.rolling(wm).std()

    f = {}
    # 1) 中期动量：过去 60 日累计涨幅
    f["mom"] = px.pct_change(wm)

    # 2) 均线斜率（波动率调整）：趋势方向 + 陡峭度
    f["slope"] = _rolling_slope(logpx, wm) / vol

    # 3) 均线多头排列度
    ma_s, ma_m, ma_l = (px.rolling(w).mean() for w in (ws, wm, wl))
    f["ma_align"] = (ma_s / ma_m - 1.0) + (ma_m / ma_l - 1.0)

    # 4) 趋势效率（Kaufman Efficiency Ratio）：涨得"干净"还是"来回震荡"
    net = (logpx - logpx.shift(wm)).abs()
    path = logpx.diff().abs().rolling(wm).sum()
    f["eff"] = net / path.replace(0.0, np.nan)

    # 5) 距区间高点的位置：越接近新高越强
    f["near_high"] = px / px.rolling(wm).max() - 1.0

    # 6) 波动调整动量（类 Sharpe）
    f["sharpe"] = ret.rolling(wm).mean() / vol * np.sqrt(cfg["trading_days"])

    for k in f:
        f[k] = f[k].replace([np.inf, -np.inf], np.nan)
    return f


# ============================================================
# 3. 横截面比较：去极值 + 标准化
# ============================================================
def winsorize_zscore(s: pd.Series, n_mad: float = 3.0) -> pd.Series:
    x = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 5:
        return pd.Series(dtype=float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad > 0 and np.isfinite(mad):
        k = 1.4826 * mad          # MAD -> σ 的一致性系数
        x = x.clip(med - n_mad * k, med + n_mad * k)
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(dtype=float)
    return (x - x.mean()) / sd


# ============================================================
# 4. 中性化：对「行业哑变量 + log 市值」回归，取残差
# ============================================================
def neutralize(factor: pd.Series, industry_map: dict,
               log_mcap: pd.Series) -> pd.Series:
    y = factor.replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < 20:
        return pd.Series(dtype=float)

    ind = pd.Series({k: industry_map.get(k, "UNK") for k in y.index})
    dummies = pd.get_dummies(ind, drop_first=True).astype(float)

    X = pd.DataFrame(index=y.index)
    X["log_mcap"] = log_mcap.reindex(y.index)
    X = pd.concat([X, dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    y = y.reindex(X.index)
    if len(y) < 20:
        return pd.Series(dtype=float)

    X.insert(0, "const", 1.0)
    Xv = X.values.astype(float)
    beta, *_ = np.linalg.lstsq(Xv, y.values.astype(float), rcond=None)
    resid = pd.Series(y.values - Xv @ beta, index=y.index)

    sd = resid.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(dtype=float)
    return (resid - resid.mean()) / sd


# ============================================================
# 5. 因子合成
# ============================================================
def combine_factors(z_dict: dict, weights: dict = None) -> pd.Series:
    """等权或按 weights 加权合成，返回综合因子分数。"""
    df = pd.DataFrame(z_dict)
    if df.empty:
        return pd.Series(dtype=float)

    w = (pd.Series(1.0, index=df.columns) if weights is None
         else pd.Series(weights).reindex(df.columns).fillna(0.0))
    if w.abs().sum() == 0:
        w = pd.Series(1.0, index=df.columns)

    valid = df.notna().sum(axis=1) >= max(1, len(df.columns) // 2)
    score = (df.fillna(0.0) * w).sum(axis=1) / w.abs().sum()
    return score[valid]


# ============================================================
# 6. 组合优化：max w'μ - λ·w'Σw
#    s.t. Σw = 1, 行业中性, 0 ≤ w ≤ max_weight
# ============================================================
def optimize_portfolio(alpha: pd.Series, cov: pd.DataFrame,
                       industry_map: dict, cfg: dict) -> pd.Series:
    tickers = list(alpha.index)
    n = len(tickers)
    if n < 5:
        return pd.Series(dtype=float)

    mu = alpha.values.astype(float) * cfg["alpha_scale"]

    sigma = cov.reindex(index=tickers, columns=tickers).fillna(0.0).values
    sigma = np.nan_to_num(sigma, nan=0.0)
    sigma = 0.5 * (sigma + sigma.T) + np.eye(n) * 1e-6   # 对称 + 岭

    inds = np.array([industry_map.get(t, "UNK") for t in tickers])
    uniq = np.unique(inds)
    lam = cfg["risk_aversion"]

    def obj(w):
        return -(w @ mu - lam * (w @ sigma @ w))

    def grad(w):
        return -(mu - 2.0 * lam * (sigma @ w))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(n)}]
    base = 1.0 / len(uniq)                                # 行业中性基准
    for u in uniq:
        m = (inds == u).astype(float)
        cons.append({"type": "eq",
                     "fun": (lambda m: lambda w: float(w @ m) - base)(m),
                     "jac": (lambda m: lambda w: m)(m)})

    res = minimize(obj, np.ones(n) / n, jac=grad,
                   bounds=[(0.0, cfg["max_weight"])] * n,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-10})

    if not res.success or not np.all(np.isfinite(res.x)):
        return pd.Series(1.0 / n, index=tickers)          # 兜底：等权

    w = pd.Series(res.x, index=tickers)
    w[w < 1e-4] = 0.0
    return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / n, index=tickers)


# ============================================================
# 7. 回测主循环
# ============================================================
def run_backtest(px, mcap, industry_map, cfg):
    factors = compute_factors(px, cfg)
    ret = px.pct_change()
    dates = px.index

    start = cfg["lookback_long"] + 5
    rebal_dates = dates[start::cfg["rebalance_freq"]]

    weights_hist = {}
    for d in rebal_dates:
        log_mcap = np.log(mcap.loc[d].replace(0, np.nan))

        z_dict = {}
        for name, f in factors.items():
            z = winsorize_zscore(f.loc[d])                    # 横截面标准化
            z = neutralize(z, industry_map, log_mcap)         # 行业+市值中性化
            if len(z) > 0:
                z_dict[name] = z

        alpha = combine_factors(z_dict)                        # 因子合成
        if len(alpha) < 20:
            continue

        cov = (ret.loc[:d].iloc[:-1]                           # 用 d 之前数据估计
               .tail(cfg["cov_window"]).cov() * cfg["trading_days"])
        w = optimize_portfolio(alpha, cov, industry_map, cfg)  # 组合优化
        if len(w) > 0:
            weights_hist[d] = w

    return weights_hist, rebal_dates


def build_nav(weights_hist, ret, dates):
    eq = 1.0
    nav = pd.Series(index=dates, dtype=float)
    w_cur = None
    for i, d in enumerate(dates):
        if d in weights_hist:
            w_cur = weights_hist[d]
        nav.iloc[i] = eq
        if i + 1 < len(dates) and w_cur is not None:
            nxt = dates[i + 1]
            r = ret.loc[nxt].reindex(w_cur.index).fillna(0.0)
            eq *= 1.0 + float((w_cur.values * r.values).sum())
    return nav


def perf_stats(nav: pd.Series, freq: int = 252) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    r = nav.pct_change().dropna()
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (freq / len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = (nav / nav.cummax() - 1.0).min()
    return {"年化收益": ann_ret, "年化波动": ann_vol,
            "夏普比率": sharpe, "最大回撤": mdd}


# ============================================================
# 8. 主程序
# ============================================================
if __name__ == "__main__":
    px, mcap, industry_map = generate_data(CFG)
    print(f"数据规模: {px.shape[1]} 只股票 × {px.shape[0]} 个交易日\n")

    weights_hist, rebal_dates = run_backtest(px, mcap, industry_map, CFG)
    print(f"完成 {len(weights_hist)} 次调仓（每 {CFG['rebalance_freq']} 个交易日一次）\n")

    ret = px.pct_change()
    nav = build_nav(weights_hist, ret, px.index)

    # 对齐到首次调仓日
    s0 = rebal_dates[0] if len(rebal_dates) else px.index[0]
    nav = nav.loc[s0:]
    nav = nav / nav.iloc[0]

    # 等权基准
    bench = (1 + ret.mean(axis=1).reindex(nav.index).fillna(0.0)).cumprod()
    bench = bench / bench.iloc[0]

    stats = pd.DataFrame({
        "策略": perf_stats(nav, CFG["trading_days"]),
        "等权基准": perf_stats(bench, CFG["trading_days"]),
    })
    print("========== 绩效概览 ==========")
    print(stats.map(lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.3f}"))

    # 最近一次持仓
    last_d = max(weights_hist)
    print(f"\n========== 最近调仓日 {last_d.date()} 前 10 大持仓 ==========")
    print((weights_hist[last_d].sort_values(ascending=False).head(10) * 100)
          .round(2).astype(str) + " %")



