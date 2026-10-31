"""
风险模型：单期横截面风格因子回归（类 Barra，无市场因子、无行业因子、无行业约束）

模型： r_i = Σ_k X_{i,k} * f_k + u_i
估计： 加权最小二乘（WLS），权重 w_i = (market_cap_i) ** weight_power
       （默认 weight_power = 0.5，即“平方根市值加权”）

    f = argmin_f Σ_i w_i * (r_i - X_i' f)^2
      = (X' W X)^{-1} X' W r,   W = diag(w)

数值上等价于对 X、r 同时乘以 sqrt(w) 后做普通最小二乘（本实现方式）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _to_series(obj: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """把单列 DataFrame 或 Series 统一成 Series。"""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] != 1:
            raise ValueError(
                f"{name} 应为单列 DataFrame 或 Series，当前为 {obj.shape[1]} 列"
            )
        return obj.iloc[:, 0]
    raise TypeError(f"{name} 应为 pandas DataFrame 或 Series，实际为 {type(obj)}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def build_risk_model_cross_section(
    market_cap: pd.DataFrame,
    stock_return: pd.DataFrame,
    factor_exposure: pd.DataFrame,
    weight_power: float = 0.5,
    min_stocks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    在单一时间截面上，用风格因子解释个股同期收益（平方根市值加权 WLS）。

    Parameters
    ----------
    market_cap : pd.DataFrame
        市值。index 为股票代码，单列（也可直接传 Series）。
    stock_return : pd.DataFrame
        个股收益。index 为股票代码，单列（也可直接传 Series）。
    factor_exposure : pd.DataFrame
        已标准化的因子暴露。index 为股票代码，列为因子名（至少 1 列）。
    weight_power : float, default 0.5
        WLS 权重形式：w_i = market_cap_i ** weight_power。
        默认 0.5 即“平方根市值加权”。
    min_stocks : int, optional
        最少有效股票数。默认 None，即要求 n > 因子数（否则无法正常估计）。

    Returns
    -------
    factor_return : pd.DataFrame
        index 为因子名，列 'factor_return'。
    specific_return : pd.DataFrame
        index 为股票代码，列 'specific_return'。
    """
    # ---------- 1. 输入整理 ----------
    mcap = _to_series(market_cap, "market_cap")
    ret = _to_series(stock_return, "stock_return")

    if not isinstance(factor_exposure, pd.DataFrame):
        raise TypeError("factor_exposure 必须是 pandas DataFrame")
    X_all = factor_exposure
    if X_all.shape[1] == 0:
        raise ValueError("factor_exposure 至少需要包含一个因子")

    # ---------- 2. 股票代码对齐 ----------
    common_idx = mcap.index.intersection(ret.index).intersection(X_all.index)
    if len(common_idx) == 0:
        raise ValueError(
            "market_cap / stock_return / factor_exposure 之间没有共同的股票代码索引"
        )
    mcap = mcap.reindex(common_idx)
    ret = ret.reindex(common_idx)
    X_all = X_all.reindex(common_idx)

    # ---------- 3. 有效性过滤 ----------
    valid = (
        mcap.notna()
        & ret.notna()
        & (mcap > 0)
        & np.isfinite(mcap.to_numpy(dtype=float))
        & np.isfinite(ret.to_numpy(dtype=float))
        & X_all.notna().all(axis=1)
    )

    mcap_v = mcap.loc[valid]
    ret_v = ret.loc[valid]
    X_v = X_all.loc[valid]

    n, k = len(mcap_v), X_v.shape[1]
    required = (k + 1) if min_stocks is None else max(min_stocks, k + 1)
    if n < required:
        raise ValueError(
            f"有效样本数 {n} 不足：至少需要 {required} 只股票（因子数={k}）"
        )

    # ---------- 4. WLS 权重 ----------
    w = mcap_v.to_numpy(dtype=float) ** weight_power
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("市值权重出现非正或非有限值，请检查 market_cap 输入")
    w = w / w.mean()              # 仅做数值归一化，不改变回归结果
    sqrt_w = np.sqrt(w)           # 乘到 X 和 r 上，等价于 WLS

    Xv = X_v.to_numpy(dtype=float)
    rv = ret_v.to_numpy(dtype=float)

    # ---------- 5. 加权最小二乘 ----------
    Xw = Xv * sqrt_w[:, None]
    rw = rv * sqrt_w
    f, _, rank, _ = np.linalg.lstsq(Xw, rw, rcond=None)

    if rank < k:
        warnings.warn(
            f"因子暴露矩阵秩亏（rank={rank} < 因子数={k}），回归系数可能不唯一。",
            RuntimeWarning,
        )

    # ---------- 6. 特质收益 ----------
    specific = rv - Xv @ f

    # ---------- 7. 结果整理 ----------
    factor_return = pd.DataFrame(
        {"factor_return": f},
        index=pd.Index(X_v.columns, name="factor"),
    )
    specific_return = pd.DataFrame(
        {"specific_return": specific},
        index=pd.Index(X_v.index, name="stock"),
    )
    return factor_return, specific_return


# ----------------------------------------------------------------------
# 示例 / 自测
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(2024)

    n_stocks, n_factors = 300, 5
    stocks = [f"S{i:04d}" for i in range(n_stocks)]
    factor_names = [f"F{j}" for j in range(n_factors)]

    # 市值
    mcap = pd.DataFrame(
        {"market_cap": rng.lognormal(mean=10.0, sigma=1.0, size=n_stocks)},
        index=stocks,
    )

    # 因子暴露（模拟后标准化）
    X = pd.DataFrame(
        rng.standard_normal((n_stocks, n_factors)),
        index=stocks,
        columns=factor_names,
    )
    X = (X - X.mean()) / X.std()

    # 模拟“真实”因子收益 + 特质收益
    true_f = rng.normal(0.0, 0.02, size=n_factors)
    true_u = rng.normal(0.0, 0.01, size=n_stocks)
    true_r = X.to_numpy() @ true_f + true_u
    ret = pd.DataFrame({"return": true_r}, index=stocks)

    # 加入少量缺失值，验证对齐/清洗逻辑
    ret.iloc[3, 0] = np.nan
    X.iloc[7, 1] = np.nan

    fac_ret, spec_ret = build_risk_model_cross_section(
        market_cap=mcap,
        stock_return=ret,
        factor_exposure=X,
        weight_power=0.5,
    )

    print("=" * 50)
    print("估计的因子收益：")
    print(fac_ret.round(6))
    print("\n真实因子收益：")
    print(pd.Series(true_f, index=factor_names).round(6))
    print("\n特质收益（前 5 行）：")
    print(spec_ret.head().round(6))
    print(f"\n特质收益样本数：{len(spec_ret)}")
