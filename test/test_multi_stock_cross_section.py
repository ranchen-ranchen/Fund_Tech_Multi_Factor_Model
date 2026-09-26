"""
多股票截面回测系统 —— pytest 测试
运行: pytest test/ -v          （或 pytest test/ -v -s 看 print）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# conftest.py 已把项目根加入 sys.path，这里直接导入
from multi_stock_cross_section import (
    add_atr, PositionConfig, _trend_state, signal_multiplier,
    compute_initial_stops, effective_stop, compute_position_size,
    Position, backtest_pool, report, pool_report,
)

# ===== 合成数据工厂=====
def make_synthetic_stock(n=200, seed=0, start_price=10.0, drift=0.001,
                         vol=0.02, big_score=3, small_score=3, adx=25.0,
                         health=50.0, break_score=0, top_warn=False,
                         bottom_warn=False, support_offset=0.03):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n); open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, vol / 3, n - 1))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    low  = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n),
    }, index=dates)
    df["BIG_SCORE"] = big_score
    df["SMALL_SCORE"] = small_score
    df["ADX"] = adx
    df["TREND_HEALTH"] = health
    df["BREAK_SCORE"] = break_score
    df["TOP_WARN"] = top_warn
    df["BOTTOM_WARN"] = bottom_warn
    df["SUPPORT_1"] = close * (1 - support_offset)
    return df


def build_pool():
    return {
        "STRONG": make_synthetic_stock(seed=1, drift=+0.0025, big_score=3,
                                       small_score=3, adx=32, health=60, break_score=2),
        "MEDIUM": make_synthetic_stock(seed=2, drift=+0.0010, big_score=2,
                                       small_score=2, adx=22, health=20),
        "WEAK":   make_synthetic_stock(seed=3, drift=+0.0000, big_score=1,
                                       small_score=1, adx=18, health=0),
        "SHORT":  make_synthetic_stock(seed=4, drift=-0.0015, big_score=-3,
                                       small_score=-3, adx=28, health=-50),
        "WARNY":  make_synthetic_stock(seed=5, drift=+0.0020, big_score=3,
                                       small_score=3, adx=30, health=40, top_warn=True),
        "RANGE":  make_synthetic_stock(seed=6, drift=+0.0005, big_score=1,
                                       small_score=1, adx=12, health=-10, bottom_warn=True),
    }


# ===== 把原来的 print("[OK] ...") 全部删掉，pytest 用断言说话 =====

def test_add_atr():
    n = 60
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": 10 + rng.normal(0, 0.1, n),
        "high": 10.5 + rng.normal(0, 0.1, n),
        "low": 9.5 + rng.normal(0, 0.1, n),
        "close": 10 + rng.normal(0, 0.1, n),
    }, index=dates)
    out = add_atr(df, period=14)
    assert "ATR" in out.columns and "ATR_PCT" in out.columns
    assert out["ATR"].notna().all()
    assert (out["ATR"] > 0).all()
    assert np.allclose(out["ATR_PCT"], out["ATR"] / out["close"] * 100)


def test_trend_state():
    assert _trend_state(5) == 1
    assert _trend_state(2) == 1
    assert _trend_state(1.9) == 0
    assert _trend_state(0) == 0
    assert _trend_state(-2) == -1
    assert _trend_state(-5) == -1
    assert _trend_state(np.nan) == 0


def test_signal_multiplier():
    row = pd.Series({"BIG_SCORE": 3, "SMALL_SCORE": 3, "ADX": 35,
                     "TREND_HEALTH": 60, "BREAK_SCORE": 2,
                     "TOP_WARN": False, "BOTTOM_WARN": False})
    assert 1.30 < signal_multiplier(row) <= 1.50

    row_short = row.copy(); row_short["BIG_SCORE"] = -3; row_short["SMALL_SCORE"] = -3
    assert signal_multiplier(row_short) == 0.0

    row_top = row.copy(); row_top["TOP_WARN"] = True; row_top["BREAK_SCORE"] = 0
    assert signal_multiplier(row_top) < signal_multiplier(row)

    row_low_adx  = row.copy(); row_low_adx["ADX"] = 10;  row_low_adx["BREAK_SCORE"] = 0
    row_mid_adx  = row.copy(); row_mid_adx["ADX"] = 25;  row_mid_adx["BREAK_SCORE"] = 0
    assert signal_multiplier(row_low_adx) < signal_multiplier(row_mid_adx)


def test_initial_and_effective_stop():
    cfg = PositionConfig()
    entry, atr, support = 100.0, 2.0, 96.0
    s_init, s_hard, s_tech = compute_initial_stops(entry, atr, support, cfg)
    assert abs(s_init - 96.0) < 1e-9
    assert abs(s_hard - 92.0) < 1e-9
    assert s_tech is not None and abs(s_tech - 95.52) < 1e-9

    stop = effective_stop(entry, atr, s_init, s_hard, s_tech, None, cfg)
    assert abs(stop - 96.0) < 1e-9
    assert stop < entry

    s_init2, s_hard2, s_tech2 = compute_initial_stops(entry, atr, np.nan, cfg)
    stop2 = effective_stop(entry, atr, s_init2, s_hard2, s_tech2, None, cfg)
    assert abs(stop2 - 96.0) < 1e-9


def test_position_size():
    cfg = PositionConfig()
    equity, entry, stop, mult = 1_000_000.0, 100.0, 96.0, 1.0
    shares, r, budget = compute_position_size(equity, entry, stop, mult, cfg)
    assert r == 4.0
    assert abs(budget - 10_000.0) < 1e-9
    assert shares == 2200

    assert compute_position_size(equity, entry, stop, 0.0, cfg)[0] == 0
    assert compute_position_size(equity, 100.0, 100.0, 1.0, cfg)[0] == 0


def test_position_dataclass():
    p = Position(entry_date=pd.Timestamp("2023-01-03"),
                 entry_price=10.0, shares=1000.0, initial_shares=1000.0,
                 r_per_share=0.5, s_init=9.5, s_hard=9.2)
    assert p.high_water == 10.0
    assert p.tp_taken == [] and p.partial_exits == []
    assert p.shares == p.initial_shares


def test_backtest_smoke_and_report(capsys):
    pool = build_pool()
    cfg = PositionConfig(top_n=3, min_mult=0.15, base_risk_pct=0.008,
                         max_position_pct=0.30, cooldown_bars=3)
    trades, equity = backtest_pool(pool, cfg, initial_equity=1_000_000.0)

    assert isinstance(trades, pd.DataFrame) and isinstance(equity, pd.DataFrame)
    assert {"equity", "cash", "mv", "n_positions"} <= set(equity.columns)
    assert len(equity) > 50
    assert (equity["equity"] > 0).all()
    assert (equity["n_positions"] <= cfg.top_n).all()
    assert equity["cash"].min() >= -1e-6

    metrics = pool_report(trades, equity, 1_000_000.0)
    assert "total_return" in metrics and "max_drawdown" in metrics
    assert metrics["max_drawdown"] <= 0

    if not trades.empty:
        allowed = {"stop", "stop_gap", "all_tp", "trend_reverse_big",
                   "trend_reverse_small", "dropout", "end_of_backtest"}
        assert set(trades["reason"]).issubset(allowed)


def test_no_trades_when_all_filtered():
    pool = build_pool()
    cfg = PositionConfig(top_n=3, min_mult=2.0)   # 高于乘数上限 → 全被过滤
    trades, equity = backtest_pool(pool, cfg, initial_equity=500_000.0)
    assert trades.empty
    assert (equity["n_positions"] == 0).all()
    assert np.allclose(equity["equity"], 500_000.0)


def test_single_stock():
    pool = {"ONLY": make_synthetic_stock(seed=99, drift=0.002,
                                         big_score=3, small_score=3, break_score=2)}
    cfg = PositionConfig(top_n=1, min_mult=0.10, cooldown_bars=0)
    trades, equity = backtest_pool(pool, cfg, initial_equity=1_000_000.0)
    assert len(equity) > 0
    assert (equity["n_positions"] <= 1).all()


