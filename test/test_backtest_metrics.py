"""Metric-convention tests for the backtest and benchmark evaluators.

Pins the behavior introduced by the metric-consistency patch:

- Information ratio numerator is the *arithmetic* annualized active return.
- Geometric excess return / relative CAGR are display-only fields.
- Standard Sharpe uses arithmetic annualized excess return / volatility.
- The legacy CAGR/vol ratio is still reported as ``sharpe_cagr``.
- ``run_backtest`` accepts a risk-free rate and applies it to the standard Sharpe.
- The benchmark evaluator uses the same standard Sharpe convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy.backtest_buy_hold import (  # noqa: E402
    compute_information_ratio,
    compute_sharpe_metrics,
    run_backtest,
)


def _nav_from_returns(returns, start: str = "2021-01-04") -> pd.Series:
    dates = pd.bdate_range(start, periods=len(returns) + 1)
    values = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns))])
    return pd.Series(values, index=dates)


def test_information_ratio_is_arithmetic_and_geometric_display_is_separate():
    rp = np.array([0.010, -0.005, 0.020, 0.000, -0.010, 0.015])
    rb = np.array([0.004, 0.001, 0.006, -0.002, 0.003, -0.001])
    p = _nav_from_returns(rp)
    b = _nav_from_returns(rb)

    m = compute_information_ratio(p, b, periods_per_year=252)

    excess = rp - rb
    expected_active = excess.mean() * 252
    expected_te = excess.std(ddof=1) * np.sqrt(252)
    years = len(p) / 252
    expected_port_cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    expected_bench_cagr = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
    expected_geo = (
        (p.iloc[-1] / p.iloc[0]) / (b.iloc[-1] / b.iloc[0])
    ) ** (1 / years) - 1

    assert m["n_obs"] == len(excess)
    assert m["active_return"] == pytest.approx(expected_active)
    assert m["tracking_error"] == pytest.approx(expected_te)
    assert m["information_ratio"] == pytest.approx(expected_active / expected_te)
    assert m["portfolio_cagr"] == pytest.approx(expected_port_cagr)
    assert m["benchmark_cagr"] == pytest.approx(expected_bench_cagr)
    assert m["cagr_difference"] == pytest.approx(
        expected_port_cagr - expected_bench_cagr
    )
    assert m["geometric_excess_return"] == pytest.approx(expected_geo)
    assert m["relative_cagr"] == pytest.approx(expected_geo)
    assert m["years"] == pytest.approx(years)

    # The IR numerator is the arithmetic active return, not the geometric one.
    assert abs(m["geometric_excess_return"] - m["active_return"]) > 1e-12


def test_information_ratio_geometric_can_use_initial_values():
    rp = np.array([0.010, -0.005, 0.020, 0.000])
    rb = np.array([0.004, 0.001, 0.006, -0.002])
    p = _nav_from_returns(rp) * 0.995   # simulate the first-period entry cost
    b = _nav_from_returns(rb)

    m_default = compute_information_ratio(p, b)
    m_init = compute_information_ratio(
        p, b, portfolio_initial_value=1.0, benchmark_initial_value=1.0
    )

    years = len(p) / 252
    expected_default = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    expected_init = (p.iloc[-1] / 1.0) ** (1 / years) - 1

    assert m_default["portfolio_cagr"] == pytest.approx(expected_default)
    assert m_init["portfolio_cagr"] == pytest.approx(expected_init)
    assert m_init["portfolio_cagr"] != pytest.approx(m_default["portfolio_cagr"])


def test_sharpe_metrics_standard_and_legacy():
    r = pd.Series([0.010, -0.005, 0.020, 0.000, 0.012, -0.003])
    cagr = 0.10
    rf = 0.02

    m = compute_sharpe_metrics(r, cagr, risk_free_rate=rf)

    ann_vol = r.std() * np.sqrt(252)
    ann_arith = r.mean() * 252
    assert m["annual_vol"] == pytest.approx(ann_vol)
    assert m["annual_return_arithmetic"] == pytest.approx(ann_arith)
    assert m["sharpe"] == pytest.approx((ann_arith - rf) / ann_vol)
    assert m["sharpe_cagr"] == pytest.approx(cagr / ann_vol)
    assert m["sharpe_rf0"] == pytest.approx(ann_arith / ann_vol)
    assert m["sharpe"] != pytest.approx(m["sharpe_cagr"])


def test_run_backtest_rejects_invalid_risk_free_rate():
    with pytest.raises(ValueError, match="risk_free_rate"):
        run_backtest([], risk_free_rate=1.0)


def _write_stock(path: Path, prices, dates: pd.DatetimeIndex) -> None:
    pd.DataFrame({"date": dates, "close": prices}).to_csv(path, index=False)


def _write_benchmark(path: Path, nav_values, dates: pd.DatetimeIndex) -> None:
    df = pd.DataFrame(
        {
            "Close": nav_values,
            "Cumulative NAV": nav_values,
            "Drawdown": [0.0] * len(nav_values),
        },
        index=dates,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Performance Summary\n")
        handle.write("Metric,Value\n")
        handle.write("Symbol,TEST\n\n")
        handle.write("# Daily Series\n")
        df.to_csv(handle)


def test_run_backtest_reports_standard_and_legacy_sharpe(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=5)
    stock_a = tmp_path / "a.csv"
    stock_b = tmp_path / "b.csv"
    bench = tmp_path / "bench.csv"
    _write_stock(stock_a, [10.0, 10.1, 10.2, 10.15, 10.3], dates)
    _write_stock(stock_b, [20.0, 19.8, 20.1, 20.0, 20.2], dates)
    _write_benchmark(bench, [1.0, 1.01, 1.02, 1.015, 1.03], dates)

    periods = [{
        "name": "test",
        "csv_files": [str(stock_a), str(stock_b)],
        "weights": None,
        "start_date": dates[0].date().isoformat(),
        "end_date": dates[-1].date().isoformat(),
    }]
    log_path = tmp_path / "backtest.log"

    nav, summary = run_backtest(
        periods,
        align="inner",
        plot=False,
        benchmark_file=str(bench),
        log_file=str(log_path),
        log_daily_stock_nav=False,
        risk_free_rate=0.02,
    )

    assert nav is not None and summary is not None
    row = summary[0]
    assert row["risk_free_rate"] == pytest.approx(0.02)
    assert row["sharpe"] == pytest.approx(
        (row["annual_return_arithmetic"] - 0.02) / row["annual_vol"]
    )
    assert row["sharpe_cagr"] == pytest.approx(
        row["annual_return"] / row["annual_vol"]
    )
    assert row["sharpe_rf0"] == pytest.approx(
        row["annual_return_arithmetic"] / row["annual_vol"]
    )
    assert row["sharpe"] != pytest.approx(row["sharpe_cagr"])
    assert row["information_ratio"] == pytest.approx(
        row["active_return"] / row["tracking_error"]
    )
    for key in (
        "geometric_excess_return",
        "relative_cagr",
        "cagr_difference",
        "portfolio_cagr",
        "benchmark_cagr",
    ):
        assert key in row

    log_text = log_path.read_text(encoding="utf-8")
    assert "夏普(算术" in log_text
    assert "几何超额收益" in log_text
    assert "CAGR/波动" in log_text


def test_eval_hs300_sharpe_is_arithmetic_and_legacy_is_cagr():
    pytest.importorskip("akshare")
    from eval_hs300_perform import calc_performance_metrics

    returns = np.array([0.010, -0.005, 0.020, 0.000, 0.012, -0.003])
    dates = pd.bdate_range("2021-01-04", periods=len(returns) + 1)
    close = pd.Series(
        np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + returns)]),
        index=dates,
    )
    rf = 0.02
    m = calc_performance_metrics(close, rf_annual=rf)

    daily = close.pct_change().dropna()
    ann_vol = daily.std() * np.sqrt(252)
    ann_arith = daily.mean() * 252

    assert m["Annualized Return (Arithmetic)"] == pytest.approx(ann_arith)
    assert m["Sharpe Ratio"] == pytest.approx((ann_arith - rf) / ann_vol)
    assert m["Sharpe Ratio (CAGR/vol)"] == pytest.approx(
        (m["Annualized Return"] - rf) / ann_vol
    )
    assert m["Sharpe Ratio"] != pytest.approx(m["Sharpe Ratio (CAGR/vol)"])
