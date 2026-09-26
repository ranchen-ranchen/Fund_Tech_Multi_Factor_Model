import numpy as np
import pandas as pd
import pytest

from tech_analysis import (
    add_ma,
    add_macd,
    add_adx,
    rolling_slope,
    _cluster_levels,
    apply_adx_gate,
    add_support_resistance,
    compute_trend,
    analyze,
)


# ------------------------------------------------------------
# 测试数据构造
# ------------------------------------------------------------
def make_ohlcv(n=320, trend=0.0005, seed=42):
    """构造一份模拟日线 OHLCV 数据"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    ret = trend + rng.normal(0, 0.015, n)
    close = 10 * np.cumprod(1 + ret)
    close = np.maximum(close, 0.1)  # 避免出现非正价格

    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    volume = rng.lognormal(mean=np.log(1e6), sigma=0.3, size=n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


# ------------------------------------------------------------
# 单元测试
# ------------------------------------------------------------
def test_rolling_slope_linear():
    s = pd.Series(np.arange(20, dtype=float))
    out = rolling_slope(s, 5)
    assert out.iloc[-1] == pytest.approx(1.0)


def test_cluster_levels():
    out = _cluster_levels([10, 10.1, 12], 0.015)
    assert out == pytest.approx([10.05, 12])


def test_adx_gate_shrink():
    df = pd.DataFrame(
        {
            "ADX": [0, 10, 20, 30],
            "BIG_SCORE": [4, 4, 4, 4],
            "SMALL_SCORE": [-4, -4, -4, -4],
        }
    )
    out = apply_adx_gate(df, adx_threshold=20, mode="shrink")

    assert out["BIG_SCORE"].tolist() == [0, 2, 4, 4]
    assert out["SMALL_SCORE"].tolist() == [0, -2, -4, -4]
    assert out["BIG_SCORE_RAW"].tolist() == [4, 4, 4, 4]
    assert out["ADX_WEAK"].tolist() == [True, True, False, False]


def test_support_resistance_no_lookahead():
    df = make_ohlcv(200)
    out1 = add_support_resistance(df.copy())

    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("high")] *= 5
    out2 = add_support_resistance(df2)

    # 修改最后一天的数据，不应影响之前任何一天
    pd.testing.assert_series_equal(
        out1["SUPPORT_1"].iloc[:-1],
        out2["SUPPORT_1"].iloc[:-1],
    )
    pd.testing.assert_series_equal(
        out1["RESISTANCE_1"].iloc[:-1],
        out2["RESISTANCE_1"].iloc[:-1],
    )


def test_compute_trend_columns_and_analyze(capsys):
    df = make_ohlcv(320)
    out = compute_trend(df, adx_threshold=20, adx_mode="shrink")

    expected_cols = {
        "BIG_SCORE",
        "SMALL_SCORE",
        "ADX",
        "SUPPORT_1",
        "RESISTANCE_1",
        "VOL_RATIO",
        "TREND_HEALTH",
        "BREAK_SCORE",
        "TOP_WARN",
        "BOTTOM_WARN",
    }
    assert expected_cols <= set(out.columns)

    result = analyze(out)

    assert result["big_trend"] in {"向上", "中性", "向下"}
    assert result["small_trend"] in {"向上", "中性", "向下"}
    assert isinstance(result["action"], str)

    captured = capsys.readouterr()
    assert "综合结论" in captured.out
