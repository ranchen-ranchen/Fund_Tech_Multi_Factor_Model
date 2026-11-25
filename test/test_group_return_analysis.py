"""Tests for the two-group (AI chain vs non-chain) period return analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from group_return_analysis import analyze_period, run_analysis, summarize  # noqa: E402
from strategy.backtest_buy_hold import run_backtest  # noqa: E402


def _series(prices, start: str = "2021-01-04") -> pd.Series:
    dates = pd.bdate_range(start, periods=len(prices))
    return pd.Series(prices, index=dates, dtype=float)


def _prices() -> dict:
    return {
        "600000": _series([100, 100, 100, 100, 110]),  # AI: +10%
        "600001": _series([100, 100, 100, 100, 120]),  # AI: +20%
        "600002": _series([100, 100, 100, 100, 100]),  # non-AI: 0%
        "600003": _series([100, 100, 100, 100, 90]),   # non-AI: -10%
        "600004": _series([100, 100, 100, 100, 105]),  # non-AI: +5%
    }


def _weights() -> pd.Series:
    return pd.Series(
        {
            "600000": 60.0,
            "600001": 40.0,
            "600002": 50.0,
            "600003": 50.0,
            "600004": 50.0,
        }
    )


def _analyze(prices=None, weights=None):
    return analyze_period(
        start_date=pd.Timestamp("2021-01-04"),
        end_date=pd.Timestamp("2021-01-08"),
        target_codes=["600000", "600001"],
        weights=_weights() if weights is None else weights,
        prices=_prices() if prices is None else prices,
        bootstrap=200,
        permutations=200,
        rng=np.random.default_rng(0),
    )


def test_group_means_medians_and_cap_spread():
    row = _analyze()
    assert row is not None
    assert row["n_ai"] == 2
    assert row["n_non"] == 3
    assert row["n_ai_ret"] == 2
    assert row["n_non_ret"] == 3
    assert row["coverage_ai"] == pytest.approx(1.0)
    assert row["coverage_non"] == pytest.approx(1.0)

    assert row["mean_ai"] == pytest.approx(0.15)
    assert row["mean_non"] == pytest.approx(-0.05 / 3.0)
    assert row["spread"] == pytest.approx(0.15 + 0.05 / 3.0)
    assert row["median_ai"] == pytest.approx(0.15)
    assert row["median_non"] == pytest.approx(0.0)
    assert row["median_spread"] == pytest.approx(0.15)

    assert row["full_ew"] == pytest.approx(0.05)
    assert row["ai_excess_full_ew"] == pytest.approx(0.10)

    assert row["cap_mean_ai"] == pytest.approx(0.14)
    assert row["cap_mean_non"] == pytest.approx(-0.05 / 3.0)
    assert row["cap_spread"] == pytest.approx(0.14 + 0.05 / 3.0)


def test_missing_non_ai_stock_is_excluded_and_reported():
    prices = _prices()
    prices["600004"] = None
    row = _analyze(prices=prices)
    assert row is not None
    assert row["n_non_ret"] == 2
    assert row["coverage_non"] == pytest.approx(2.0 / 3.0)
    assert row["weight_coverage_non"] == pytest.approx(100.0 / 150.0)
    assert row["mean_non"] == pytest.approx(-0.05)
    assert row["mean_ai"] == pytest.approx(0.15)


def test_missing_ai_stock_raises():
    prices = _prices()
    prices["600001"] = None
    with pytest.raises(ValueError, match="AI-group stock"):
        _analyze(prices=prices)


def test_summary_cumulative_long_short_and_hit_rate():
    df = pd.DataFrame(
        [
            {
                "period": "2021h1",
                "mean_ai": 0.15,
                "mean_non": -0.05,
                "full_ew": 0.05,
                "spread": 0.20,
                "etf_ret": 0.0,
            },
            {
                "period": "2021h2",
                "mean_ai": -0.05,
                "mean_non": 0.10,
                "full_ew": 0.02,
                "spread": -0.15,
                "etf_ret": 0.0,
            },
        ]
    )
    full = summarize(df, bootstrap=200, seed=0)["Full"]
    assert full["n"] == 2
    assert full["mean_spread"] == pytest.approx(0.025)
    assert full["median_spread"] == pytest.approx(0.025)
    assert full["hit_rate"] == pytest.approx(0.5)
    assert full["cum_long_short"] == pytest.approx(1.20 * 0.85 - 1.0)
    assert full["cum_ai"] == pytest.approx(1.15 * 0.95 - 1.0)
    assert full["cum_non"] == pytest.approx(0.95 * 1.10 - 1.0)
    assert full["cum_full_ew"] == pytest.approx(1.05 * 1.02 - 1.0)
    assert full["cum_etf"] == pytest.approx(0.0)
    assert full["sign_p"] == pytest.approx(1.0)


def _write_stock(path: Path, prices, dates: pd.DatetimeIndex) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "close": prices,
            "high": prices,
            "low": prices,
            "volume": 1_000_000,
            "amount": 100_000_000,
            "turn": 1.0,
        }
    ).to_csv(path, index=False)


def _write_benchmark(path: Path, dates: pd.DatetimeIndex) -> None:
    df = pd.DataFrame(
        {
            "Close": [1.0] * len(dates),
            "Cumulative NAV": [1.0] * len(dates),
            "Drawdown": [0.0] * len(dates),
        },
        index=dates,
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Performance Summary\nMetric,Value\n\n# Daily Series\n")
        df.to_csv(handle)


def test_period_boundaries_follow_backtest_chain(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=10)
    individual = tmp_path / "individual"
    individual.mkdir()
    prices = {
        "600000": list(np.linspace(100, 110, 10)),
        "600001": list(np.linspace(100, 120, 10)),
        "600002": list(np.linspace(100, 100, 10)),
        "600003": list(np.linspace(100, 90, 10)),
    }
    for code, values in prices.items():
        _write_stock(individual / f"sh_{code}.csv", values, dates)

    weight_csv = tmp_path / "hs300_weight_2020h2.csv"
    pd.DataFrame(
        {
            "股票代码": list(prices),
            "占基金资产净值比例": [25.0] * len(prices),
        }
    ).to_csv(weight_csv, index=False)
    target_csv = tmp_path / "target_code.csv"
    pd.DataFrame(
        {
            "query_date": ["2021-01-01", "2021-01-08"],
            "target_code": ["600000 600001", "600000 600001"],
        }
    ).to_csv(target_csv, index=False)

    result, _ = run_analysis(
        target_csv=str(target_csv),
        weight_glob=str(tmp_path / "hs300_weight_*h*.csv"),
        k_dir=str(individual),
        benchmark_file=None,
        start_date="2021-01-01",
        end_date="2021-01-15",
        bootstrap=50,
        permutations=50,
        seed=0,
    )
    assert len(result) == 2
    first, second = result.iloc[0], result.iloc[1]
    assert first["start"] == "2021-01-04"
    assert first["end"] == "2021-01-08"
    assert second["start"] == "2021-01-11"
    assert second["end"] == "2021-01-15"


def test_ai_group_gross_return_matches_backtest(tmp_path):
    dates = pd.bdate_range("2021-01-04", periods=5)
    individual = tmp_path / "individual"
    individual.mkdir()
    prices = {
        "600000": [100, 100, 100, 100, 110],
        "600001": [100, 100, 100, 100, 120],
        "600002": [100, 100, 100, 100, 100],
        "600003": [100, 100, 100, 100, 90],
    }
    for code, values in prices.items():
        _write_stock(individual / f"sh_{code}.csv", values, dates)

    weight_csv = tmp_path / "hs300_weight_2020h2.csv"
    pd.DataFrame(
        {
            "股票代码": list(prices),
            "占基金资产净值比例": [25.0] * len(prices),
        }
    ).to_csv(weight_csv, index=False)

    target_csv = tmp_path / "target_code.csv"
    pd.DataFrame(
        {"query_date": ["2021-01-01"], "target_code": ["600000 600001"]}
    ).to_csv(target_csv, index=False)

    bench_csv = tmp_path / "bench.csv"
    _write_benchmark(bench_csv, dates)

    result, summary = run_analysis(
        target_csv=str(target_csv),
        weight_glob=str(tmp_path / "hs300_weight_*h*.csv"),
        k_dir=str(individual),
        benchmark_file=str(bench_csv),
        start_date="2021-01-01",
        end_date="2021-01-08",
        bootstrap=100,
        permutations=100,
        seed=0,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["mean_ai"] == pytest.approx(0.15)
    assert row["mean_non"] == pytest.approx(-0.05)
    assert row["spread"] == pytest.approx(0.20)
    assert summary["Full"]["cum_ai"] == pytest.approx(0.15)

    # The no-cost equal-weight backtest period return equals the AI group mean.
    periods = [
        {
            "name": "p1",
            "csv_files": [
                str(individual / "sh_600000.csv"),
                str(individual / "sh_600001.csv"),
            ],
            "weights": None,
            "start_date": "2021-01-04",
            "end_date": "2021-01-08",
        }
    ]
    _, bt_summary = run_backtest(
        periods,
        cost_rate=0.0,
        plot=False,
        benchmark_file=str(bench_csv),
        log_file=str(tmp_path / "bt.log"),
        log_daily_stock_nav=False,
    )
    assert bt_summary[0]["total_return"] == pytest.approx(row["mean_ai"], abs=1e-12)
