#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI compute chain vs non-chain: two-group period return analysis.

Question
--------
For each semi-annual period, partition the CSI 300 universe into

    AI group     = the target codes for that period (LLM is_in_chain = true)
    non-AI group = the rest of the previous period-end weight file universe

and compare equal-weight mean returns.  The spread is a long-short factor
return (long AI, short non-AI); the AI group mean is also the *gross*
equal-weight buy-and-hold portfolio return (before trading costs).

Method
------
- Universe and target: the same inputs as target_code_pipeline.py
  (tmp_target_code.csv + data/hs300_weight_*h*.csv).
- Prices: hfq caches; period return = close_end / close_start - 1 over the
  same effective boundaries as the backtest: the first trading day after the
  previous period's end, and the last trading day on/before the nominal end.
- Per period: equal-weight means, medians, cap-weighted means, full-universe
  equal-weight mean, coverage, Welch t (normal-approx p), permutation p,
  bootstrap CI, and the AI-vs-ETF context.
- Across periods: mean/median spread, t-stat on the period spreads, sign test,
  bootstrap CI, hit rate, cumulative long-short and cumulative group growth.
- Robustness windows: full sample, 2021-2024, 2025-2026, ex-last-period.

Important caveats
-----------------
- The labels come from the target CSV (LLM verdicts on *current* business
  descriptions).  They are not point-in-time; this is a descriptive
  decomposition, not a clean ex-ante factor premium.
- The long-short spread is not an investable A-share strategy (shorting
  constraints); the long-only AI-vs-universe excess is the investable-ish
  context.
- Per-period Welch/permutation p-values treat stocks as exchangeable within a
  period; the AI group is a sector cluster, so the period-level summary is the
  primary inference.
- 2026H2 is a partial period (through 2026-11-20); asterisks mark truncated
  period ends.
- Cumulative returns are the product of the per-period returns; the
  between-period gap days (the first trading day of each new period) are not
  earned, matching the backtest's buy-at-next-close convention. The ETF
  line/column is that same period chain, not the ETF's close-to-close total.
- Returns are gross; no trading costs.  The AI group's cumulative gross return
  should be compared with the backtest log's gross cumulative return.

Outputs
-------
- group_return_analysis.csv   per-period table
- group_return_analysis.md    summary + caveats
- group_return_spread.png     two-panel chart (spread bars, cumulative growth)

Usage
-----
    python group_return_analysis.py
    python group_return_analysis.py --end-date 2026-11-21
    python group_return_analysis.py --no-plot
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from target_code_pipeline import (  # noqa: E402
    DEFAULT_START_DATE,
    MAX_END_DATE,
    default_end_date,
    period_start_dates,
    prep_periods,
    read_target_code_csv,
    resolve_benchmark,
    trim_to_backtest_range,
    validate_target_df,
    weight_asof_files,
)
from utils.text_utils import get_stock_data, normalize_code, read_hs300_weight  # noqa: E402
from strategy.backtest_buy_hold import load_benchmark_data  # noqa: E402

DEFAULT_TARGET_CSV = "tmp_target_code.csv"
DEFAULT_WEIGHT_GLOB = "data/hs300_weight_*h*.csv"
DEFAULT_K_DIR = "stock_daily_k_hfq/individual"
DEFAULT_CSV_OUT = "group_return_analysis.csv"
DEFAULT_MD_OUT = "group_return_analysis.md"
DEFAULT_PNG_OUT = "group_return_spread.png"
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_PERMUTATIONS = 5_000
DEFAULT_SEED = 42

# Chart palette (dataviz reference instance, light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _code6(code: object) -> str:
    raw = str(code).strip()
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.zfill(6)


def _weight_file_for(date_value, weight_files: List[Tuple[pd.Timestamp, str]]) -> str:
    ts = pd.Timestamp(date_value)
    candidates = [(asof, path) for asof, path in weight_files if asof <= ts]
    if not candidates:
        raise ValueError(f"No weight file at or before {ts.date()}")
    return candidates[-1][1]


def _load_prices(code: str, k_dir: str) -> Optional[pd.Series]:
    path = os.path.join(k_dir, normalize_code(code).replace(".", "_") + ".csv")
    if not os.path.exists(path):
        return None
    try:
        df = get_stock_data(path)
    except Exception:  # noqa: BLE001 - missing/corrupt files are reported via None
        return None
    if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    closes = pd.to_numeric(df["close"], errors="coerce")
    series = pd.Series(closes.to_numpy(), index=dates).dropna()
    series = series[series > 0].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series if not series.empty else None


def _asof_return(series: Optional[pd.Series], start, end) -> float:
    if series is None or series.empty:
        return float("nan")
    s = series.dropna().sort_index()
    start_sub = s[s.index <= pd.Timestamp(start)]
    end_sub = s[s.index <= pd.Timestamp(end)]
    if start_sub.empty or end_sub.empty:
        return float("nan")
    p0 = float(start_sub.iloc[-1])
    p1 = float(end_sub.iloc[-1])
    if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
        return float("nan")
    return p1 / p0 - 1.0


def _period_label(nominal_end, effective_end) -> str:
    ts = pd.Timestamp(nominal_end)
    label = f"{ts.year}h{1 if ts.month <= 6 else 2}"
    if pd.Timestamp(effective_end).date() != ts.date():
        label += "*"
    return label


def _welch_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Welch t and a normal-approximation two-sided p (descriptive)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diff = float(np.mean(a)) - float(np.mean(b))
    se2 = float(np.var(a, ddof=1)) / len(a) + float(np.var(b, ddof=1)) / len(b)
    if not math.isfinite(se2):
        return float("nan"), float("nan")
    if se2 <= 0:
        if abs(diff) < 1e-15:
            return 0.0, 1.0
        return math.copysign(float("inf"), diff), 0.0
    t = diff / math.sqrt(se2)
    return t, math.erfc(abs(t) / math.sqrt(2.0))


def _permutation_p(
    a: np.ndarray,
    b: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_perm: int,
) -> float:
    """Two-sided random-label placebo: shuffle group labels, keep group sizes."""
    if n_perm <= 0 or len(a) == 0 or len(b) == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    n_a = len(a)
    threshold = abs(observed) - 1e-15
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = float(perm[:n_a].mean() - perm[n_a:].mean())
        if abs(diff) >= threshold:
            count += 1
    return (count + 1) / (n_perm + 1)


def _bootstrap_spread(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[float, float]:
    if n_boot <= 0 or len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ai = rng.choice(a, size=len(a), replace=True)
        non = rng.choice(b, size=len(b), replace=True)
        diffs[i] = float(ai.mean() - non.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_p(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _cap_mean(
    codes: Sequence[str], returns: Dict[str, float], weights: pd.Series
) -> Tuple[float, float]:
    sub = weights.reindex(list(codes)).dropna()
    if sub.empty or float(sub.sum()) <= 0:
        return float("nan"), 0.0
    values = np.array([returns[c] for c in sub.index], dtype=float)
    return float(np.average(values, weights=sub.to_numpy(dtype=float))), float(sub.sum())


# ---------------------------------------------------------------------------
# Per-period analysis
# ---------------------------------------------------------------------------
def analyze_period(
    *,
    start_date,
    end_date,
    nominal_end=None,
    target_codes: Sequence[str],
    weights: pd.Series,
    prices: Dict[str, Optional[pd.Series]],
    etf_nav: Optional[pd.Series] = None,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    permutations: int = DEFAULT_PERMUTATIONS,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """Compute one period's group returns and statistics.

    ``start_date``/``end_date`` are the *effective* period boundaries, matching
    the backtest: the first trading day after the previous period's end and the
    last trading day on or before the nominal period end.  ``nominal_end`` is
    used only for the period label (asterisk when the end was truncated).

    Stocks without a price on the effective start/end are excluded and reported
    through the coverage columns; a target stock with no price data at all
    raises, because the backtest would fail on it.

    Returns a row dict, or raises ValueError when the groups lack enough data
    (the caller can skip the period with a warning).
    """
    if rng is None:
        rng = np.random.default_rng(DEFAULT_SEED)

    effective_start = pd.Timestamp(start_date).normalize()
    effective_end = pd.Timestamp(end_date).normalize()
    nominal_end = (
        pd.Timestamp(nominal_end).normalize()
        if nominal_end is not None
        else effective_end
    )
    if effective_end <= effective_start:
        raise ValueError("Effective period end is not after the start")

    target_set = {_code6(c) for c in target_codes}
    if not target_set:
        raise ValueError("Empty AI group")
    universe = {_code6(c) for c in weights.index}
    non_set = universe - target_set
    missing_from_universe = sorted(target_set - universe)
    if missing_from_universe:
        raise ValueError(f"AI codes not in universe: {missing_from_universe}")
    if not non_set:
        raise ValueError("Non-AI group is empty")

    for code in sorted(target_set):
        series = prices.get(code)
        if series is None or series.empty:
            raise ValueError(f"AI-group stock has no price data: {code}")

    returns: Dict[str, float] = {}
    for code in universe:
        series = prices.get(code)
        if series is None or series.empty:
            continue
        if effective_start not in series.index or effective_end not in series.index:
            continue
        p0 = float(series.loc[effective_start])
        p1 = float(series.loc[effective_end])
        if not (math.isfinite(p0) and math.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            continue
        returns[code] = p1 / p0 - 1.0

    ai_codes = sorted(c for c in target_set if c in returns)
    non_codes = sorted(c for c in non_set if c in returns)
    if len(ai_codes) < 2 or len(non_codes) < 2:
        raise ValueError(
            f"Insufficient returns: AI={len(ai_codes)}, non-AI={len(non_codes)}"
        )

    ai = np.array([returns[c] for c in ai_codes], dtype=float)
    non = np.array([returns[c] for c in non_codes], dtype=float)
    all_returns = np.array(list(returns.values()), dtype=float)

    mean_ai = float(ai.mean())
    mean_non = float(non.mean())
    spread = mean_ai - mean_non
    median_ai = float(np.median(ai))
    median_non = float(np.median(non))
    full_ew = float(all_returns.mean())

    welch_t, welch_p = _welch_t(ai, non)
    perm_p = _permutation_p(ai, non, spread, rng, permutations)
    boot_lo, boot_hi = _bootstrap_spread(ai, non, rng, bootstrap)

    cap_ai, w_ai_used = _cap_mean(ai_codes, returns, weights)
    cap_non, w_non_used = _cap_mean(non_codes, returns, weights)
    cap_all, w_all_used = _cap_mean(list(returns), returns, weights)
    w_ai_total = float(weights.reindex(sorted(target_set)).dropna().sum())
    w_non_total = float(weights.reindex(sorted(non_set)).dropna().sum())
    w_universe_total = float(weights.sum())

    etf_ret = _asof_return(etf_nav, effective_start, effective_end)

    label = _period_label(nominal_end, effective_end)
    return {
        "period": label,
        "start": effective_start.date().isoformat(),
        "end": effective_end.date().isoformat(),
        "days": (effective_end - effective_start).days,
        "n_universe": len(universe),
        "n_ai": len(target_set),
        "n_non": len(non_set),
        "n_ai_ret": len(ai_codes),
        "n_non_ret": len(non_codes),
        "coverage_ai": len(ai_codes) / len(target_set),
        "coverage_non": len(non_codes) / len(non_set),
        "weight_coverage_ai": w_ai_used / w_ai_total if w_ai_total > 0 else float("nan"),
        "weight_coverage_non": w_non_used / w_non_total if w_non_total > 0 else float("nan"),
        "mean_ai": mean_ai,
        "mean_non": mean_non,
        "spread": spread,
        "median_ai": median_ai,
        "median_non": median_non,
        "median_spread": median_ai - median_non,
        "welch_t": welch_t,
        "welch_p_normal": welch_p,
        "perm_p": perm_p,
        "boot_lo": boot_lo,
        "boot_hi": boot_hi,
        "cap_mean_ai": cap_ai,
        "cap_mean_non": cap_non,
        "cap_spread": (
            cap_ai - cap_non
            if math.isfinite(cap_ai) and math.isfinite(cap_non)
            else float("nan")
        ),
        "cap_mean_universe": cap_all,
        "weight_coverage_universe": (
            w_all_used / w_universe_total if w_universe_total > 0 else float("nan")
        ),
        "full_ew": full_ew,
        "ai_excess_full_ew": mean_ai - full_ew,
        "etf_ret": etf_ret,
        "ai_excess_etf": mean_ai - etf_ret if math.isfinite(etf_ret) else float("nan"),
        "full_ew_excess_etf": full_ew - etf_ret if math.isfinite(etf_ret) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_analysis(
    target_csv: str,
    weight_glob: str,
    k_dir: str,
    benchmark_file: Optional[str],
    start_date: str,
    end_date: str,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    df = read_target_code_csv(target_csv)
    query_dates = validate_target_df(df)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    min_start = pd.Timestamp(DEFAULT_START_DATE)
    if start_ts < min_start:
        print(f"[warn] start {start_ts.date()} < {DEFAULT_START_DATE}; clamped", file=sys.stderr)
        start_ts = min_start
    max_end = pd.Timestamp(MAX_END_DATE)
    if end_ts > max_end:
        print(f"[warn] end {end_ts.date()} > {MAX_END_DATE}; clamped", file=sys.stderr)
        end_ts = max_end

    df, query_dates, effective_start = trim_to_backtest_range(
        df, query_dates, start_ts, end_ts
    )
    starts = period_start_dates(query_dates, effective_start)
    periods = prep_periods(df, query_dates, starts, end_ts.strftime("%Y-%m-%d"))

    weight_files = weight_asof_files(weight_glob)
    etf_nav = None
    if benchmark_file:
        try:
            etf_nav = load_benchmark_data(benchmark_file)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] benchmark load failed: {exc}", file=sys.stderr)

    period_specs = []
    all_codes = set()
    for i, period in enumerate(periods):
        query = query_dates.iloc[i]
        target_codes = [_code6(c) for c in df.iloc[i]["target_code"]]
        weight_path = _weight_file_for(query, weight_files)
        weight_df = read_hs300_weight(weight_path)
        weights = weight_df["weight"].copy()
        weights.index = [_code6(c) for c in weights.index]
        weights = weights.groupby(level=0).sum()
        all_codes.update(weights.index)
        period_specs.append((period, query, target_codes, weights, weight_path))

    prices_cache: Dict[str, Optional[pd.Series]] = {}
    for code in sorted(all_codes):
        prices_cache[code] = _load_prices(code, k_dir)

    calendar = sorted(
        {d for series in prices_cache.values() if series is not None for d in series.index}
    )
    if not calendar:
        raise RuntimeError("No price data available for the universe")

    rows: List[dict] = []
    previous_end: Optional[pd.Timestamp] = None
    for i, (period, query, target_codes, weights, weight_path) in enumerate(period_specs):
        nominal_end = pd.Timestamp(period["end_date"]).normalize()
        if previous_end is None:
            start_idx = bisect_left(calendar, start_ts)
        else:
            start_idx = bisect_right(calendar, previous_end)
        if start_idx >= len(calendar):
            print(
                f"[warn] {period['name']}: no trading day at/after the period start; "
                "stopping",
                file=sys.stderr,
            )
            break
        end_idx = bisect_right(calendar, nominal_end) - 1
        if end_idx < start_idx:
            print(
                f"[warn] {period['name']}: no trading day within the period; skipped",
                file=sys.stderr,
            )
            continue
        effective_start = calendar[start_idx]
        effective_end = calendar[end_idx]
        if effective_end <= effective_start:
            print(
                f"[warn] {period['name']}: effective end is not after the start; skipped",
                file=sys.stderr,
            )
            continue
        # Match the backtest: the next period buys on the first trading day after
        # this period's last trading day; between-period gap days are not earned.
        previous_end = effective_end

        try:
            row = analyze_period(
                start_date=effective_start,
                end_date=effective_end,
                nominal_end=nominal_end,
                target_codes=target_codes,
                weights=weights,
                prices=prices_cache,
                etf_nav=etf_nav,
                bootstrap=bootstrap,
                permutations=permutations,
                rng=np.random.default_rng(seed + i),
            )
        except ValueError as exc:
            print(
                f"[warn] {period['name']} "
                f"({effective_start.date()}~{effective_end.date()}): {exc}; skipped",
                file=sys.stderr,
            )
            continue

        if row is not None:
            row["query_date"] = pd.Timestamp(query).date().isoformat()
            row["weight_file"] = os.path.basename(weight_path)
            rows.append(row)
            print(
                f"[period] {row['period']} {row['start']}~{row['end']}: "
                f"AI={row['mean_ai']*100:+.2f}% non-AI={row['mean_non']*100:+.2f}% "
                f"spread={row['spread']*100:+.2f}pp N={row['n_ai_ret']}/{row['n_non_ret']}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No periods could be analyzed")

    result = pd.DataFrame(rows)
    summary = summarize(result, bootstrap=bootstrap, seed=seed)
    if etf_nav is not None and not etf_nav.empty:
        etf_total = float(etf_nav.iloc[-1] / etf_nav.iloc[0] - 1.0)
        for stats in summary.values():
            stats["window_etf_total"] = etf_total
    return result, summary


def summarize(df: pd.DataFrame, bootstrap: int, seed: int) -> Dict[str, dict]:
    rng = np.random.default_rng(seed + 10_000)
    year = df["period"].str.extract(r"(\d{4})").astype(int).iloc[:, 0]
    last_label = str(df["period"].iloc[-1]).rstrip("*")
    masks = {
        "Full": pd.Series(True, index=df.index),
        "2021–2024": year <= 2024,
        "2025–2026": year >= 2025,
        f"Ex-{last_label}": pd.Series(
            [True] * (len(df) - 1) + [False], index=df.index
        ),
    }

    out: Dict[str, dict] = {}
    for label, mask in masks.items():
        sub = df.loc[mask]
        if sub.empty:
            continue
        spreads = sub["spread"].to_numpy(dtype=float)
        n = len(spreads)
        mean_spread = float(np.mean(spreads))
        median_spread = float(np.median(spreads))
        std_spread = float(np.std(spreads, ddof=1)) if n > 1 else 0.0
        t_stat = (
            mean_spread / (std_spread / math.sqrt(n))
            if n > 1 and std_spread > 0
            else float("nan")
        )
        p_normal = (
            math.erfc(abs(t_stat) / math.sqrt(2.0))
            if math.isfinite(t_stat)
            else float("nan")
        )
        hit_rate = float(np.mean(spreads > 0))
        sign_p = _sign_test_p(int(np.sum(spreads > 0)), n)
        ci_lo, ci_hi = _bootstrap_mean_ci(spreads, rng, bootstrap)
        cum_ls = float(np.prod(1.0 + spreads) - 1.0)
        cum_ai = float(np.prod(1.0 + sub["mean_ai"].to_numpy(dtype=float)) - 1.0)
        cum_non = float(np.prod(1.0 + sub["mean_non"].to_numpy(dtype=float)) - 1.0)
        cum_full = float(np.prod(1.0 + sub["full_ew"].to_numpy(dtype=float)) - 1.0)
        etf = sub["etf_ret"].to_numpy(dtype=float)
        etf_valid = np.isfinite(etf)
        cum_etf = (
            float(np.prod(1.0 + etf[etf_valid]) - 1.0)
            if etf_valid.any()
            else float("nan")
        )
        factor_sharpe = (
            mean_spread / std_spread * math.sqrt(2.0)
            if std_spread > 0
            else float("nan")
        )
        out[label] = {
            "n": n,
            "mean_spread": mean_spread,
            "median_spread": median_spread,
            "std_spread": std_spread,
            "t_stat": t_stat,
            "p_normal": p_normal,
            "hit_rate": hit_rate,
            "sign_p": sign_p,
            "boot_lo": ci_lo,
            "boot_hi": ci_hi,
            "cum_long_short": cum_ls,
            "cum_ai": cum_ai,
            "cum_non": cum_non,
            "cum_full_ew": cum_full,
            "cum_etf": cum_etf,
            "factor_sharpe": factor_sharpe,
            "mean_ai": float(np.mean(sub["mean_ai"])),
            "mean_non": float(np.mean(sub["mean_non"])),
        }
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _pct(value, digits: int = 2, signed: bool = True) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    sign = "+" if signed and float(value) >= 0 else ""
    return f"{sign}{float(value) * 100:.{digits}f}%"


def _num(value, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _ci(lo, hi, digits: int = 2) -> str:
    if lo is None or hi is None or not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
        return "n/a"
    return f"[{float(lo) * 100:+.{digits}f}%, {float(hi) * 100:+.{digits}f}%]"


def write_markdown(
    df: pd.DataFrame,
    summary: Dict[str, dict],
    path: str,
    *,
    target_csv: str,
    weight_glob: str,
    benchmark_file: Optional[str],
) -> None:
    window = f"{df['start'].iloc[0]} to {df['end'].iloc[-1]}"
    lines: List[str] = []
    lines.append("# AI compute chain vs non-chain CSI 300 — two-group period return analysis")
    lines.append("")
    lines.append(f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"Window: {window}  ")
    lines.append(f"Target labels: `{target_csv}`  ")
    lines.append(f"Universe: `{weight_glob}`  ")
    lines.append(f"ETF context: `{benchmark_file or 'n/a'}`")
    lines.append("")

    lines.append("## Per-period group returns")
    lines.append("")
    lines.append(
        "| Period | Start | End | N AI | N non | Mean AI | Mean non | Spread | "
        "Median AI | Median non | Cap spread | Full EW | AI − full EW | ETF | AI − ETF |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, row in df.iterrows():
        lines.append(
            f"| {row['period']} | {row['start']} | {row['end']} | "
            f"{int(row['n_ai'])} | {int(row['n_non'])} | "
            f"{_pct(row['mean_ai'])} | {_pct(row['mean_non'])} | "
            f"{_pct(row['spread'])} | {_pct(row['median_ai'])} | "
            f"{_pct(row['median_non'])} | {_pct(row['cap_spread'])} | "
            f"{_pct(row['full_ew'])} | {_pct(row['ai_excess_full_ew'])} | "
            f"{_pct(row['etf_ret'])} | {_pct(row['ai_excess_etf'])} |"
        )
    lines.append("")

    lines.append("## Inference and coverage by period")
    lines.append("")
    lines.append(
        "| Period | Days | Welch t | p (normal) | Perm p | Boot 95% CI | "
        "Coverage AI | Coverage non | Weight cov AI | Weight cov non |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['period']} | {int(row['days'])} | {_num(row['welch_t'], 3)} | "
            f"{_num(row['welch_p_normal'], 4)} | {_num(row['perm_p'], 4)} | "
            f"{_ci(row['boot_lo'], row['boot_hi'])} | "
            f"{_pct(row['coverage_ai'], 1, signed=False)} | "
            f"{_pct(row['coverage_non'], 1, signed=False)} | "
            f"{_pct(row['weight_coverage_ai'], 1, signed=False)} | "
            f"{_pct(row['weight_coverage_non'], 1, signed=False)} |"
        )
    lines.append("")

    lines.append("## Period-level summary")
    lines.append("")
    lines.append(
        "| Window | N | Mean spread | Median | Std | t | p (normal) | Boot 95% CI | "
        "Hit rate | Sign p | Cum long-short | Cum AI | Cum non | Cum full EW | Cum ETF |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, s in summary.items():
        lines.append(
            f"| {label} | {s['n']} | {_pct(s['mean_spread'])} | "
            f"{_pct(s['median_spread'])} | {_pct(s['std_spread'], signed=False)} | "
            f"{_num(s['t_stat'], 2)} | {_num(s['p_normal'], 4)} | "
            f"{_ci(s['boot_lo'], s['boot_hi'])} | "
            f"{_pct(s['hit_rate'], 1, signed=False)} | {_num(s['sign_p'], 4)} | "
            f"{_pct(s['cum_long_short'])} | {_pct(s['cum_ai'])} | "
            f"{_pct(s['cum_non'])} | {_pct(s['cum_full_ew'])} | "
            f"{_pct(s['cum_etf'])} |"
        )
    lines.append("")

    full = summary.get("Full", {})
    lines.append("## Method and caveats")
    lines.append("")
    lines.append(
        f"- **Groups.** AI = the target codes for the period; non-AI = the "
        f"remaining names in the previous period-end weight file. The full "
        f"universe is 300 names; the AI group is 30–51 names."
    )
    lines.append(
        "- **Returns.** hfq total returns, `close_end / close_start − 1`, over the "
        "same effective boundaries as the backtest: the first trading day after "
        "the previous period's end and the last trading day on/before the nominal "
        "period end. Equal-weight means are gross buy-and-hold returns; no trading "
        "costs. Missing start/end prices are excluded and the coverage columns "
        "show the loss."
    )
    lines.append(
        "- **AI gross cross-check.** The AI group's per-period means are gross "
        "buy-and-hold returns; the backtest's net period returns are lower by the "
        "50 bp buy + 50 bp sell costs (roughly 1% of the period's gross capital). "
        "The existing log's '费前累计' (112.44%) is `final_nav + cumulative costs`, "
        "a linear cost add-back, not a compounded gross chain. The group's "
        f"compounded gross cumulative is {_pct(full.get('cum_ai'))}. Compare the "
        "group means with the backtest's per-period net returns plus ~1% cost, not "
        "with 112.44%. Both chains exclude the between-period gap days."
    )
    lines.append(
        f"- **ETF context.** `Cum ETF` is the product of the per-period ETF "
        f"returns (the same period chain). The ETF's close-to-close total over the "
        f"full window is {_pct(full.get('window_etf_total'))}; the difference is "
        "the between-period gap days, which the chain does not earn."
    )
    lines.append(
        "- **Primary inference.** The period-level mean spread, t-stat, sign test "
        "and bootstrap CI treat the 12 period spreads as the units. Per-period "
        "permutation p-values are descriptive random-label placebos; they shuffle "
        "stocks within a period and ignore cross-sectional correlation."
    )
    lines.append(
        "- **Caveats.** The labels use current business descriptions and are not "
        "point-in-time; the theme was chosen with hindsight. The AI group is a "
        "sector cluster, so the spread is partly a sector bet. The long-short "
        "spread is a factor return, not an investable A-share strategy. 2026H2 "
        "is partial (through 2026-11-20); asterisks mark truncated period ends."
    )
    lines.append(
        "- **Columns.** `cap_*` are weight-file-weighted means normalized within "
        "the group; `full_ew` is the equal-weight mean of all available universe "
        "stocks; `etf_ret` is the 510300 ETF cumulative-NAV return over the same "
        "dates (context, not the primary benchmark)."
    )
    lines.append("")

    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[output] markdown: {path}")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def plot_results(df: pd.DataFrame, output_path: str) -> Optional[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": BASELINE,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )

    x = np.arange(len(df))
    spreads = df["spread"].to_numpy(dtype=float) * 100.0
    mean_spread = float(np.nanmean(spreads))
    labels = df["period"].tolist()
    window = f"{df['start'].iloc[0]} to {df['end'].iloc[-1]}"

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(SURFACE)

    # ---- Top: per-period spread bars (diverging) ----
    colors = [BLUE if v >= 0 else RED for v in spreads]
    ax_top.bar(x, spreads, width=0.55, color=colors, edgecolor="none", zorder=3)
    ax_top.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=2)
    ax_top.axhline(mean_spread, color=INK_SECONDARY, linewidth=1.0, zorder=2)
    ax_top.text(
        len(df) - 0.45,
        mean_spread,
        f" mean {mean_spread:+.1f}pp",
        color=INK_SECONDARY,
        fontsize=9,
        va="bottom" if mean_spread >= 0 else "top",
        ha="right",
    )

    imax = int(np.nanargmax(spreads))
    imin = int(np.nanargmin(spreads))
    for idx in {imax, imin}:
        value = spreads[idx]
        ax_top.annotate(
            f"{value:+.1f}pp",
            (idx, value),
            xytext=(0, 6 if value >= 0 else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            color=INK_SECONDARY,
        )

    ax_top.legend(
        handles=[
            Patch(color=BLUE, label="AI ahead"),
            Patch(color=RED, label="AI behind"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=2,
    )
    ax_top.set_title(
        "Per-period spread",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax_top.set_ylabel("AI − non-AI (pp)", fontsize=10, color=INK_SECONDARY)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0f}"))
    ax_top.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax_top.tick_params(axis="both", length=0, labelsize=9, labelbottom=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["left"].set_color(BASELINE)
    ax_top.spines["bottom"].set_color(BASELINE)

    # ---- Bottom: cumulative growth (gross, ×) ----
    cum_ai = np.r_[1.0, np.cumprod(1.0 + df["mean_ai"].to_numpy(dtype=float))]
    cum_non = np.r_[1.0, np.cumprod(1.0 + df["mean_non"].to_numpy(dtype=float))]
    cum_full = np.r_[1.0, np.cumprod(1.0 + df["full_ew"].to_numpy(dtype=float))]
    etf = df["etf_ret"].to_numpy(dtype=float)
    cum_etf = (
        np.r_[1.0, np.cumprod(1.0 + etf)] if np.isfinite(etf).all() else None
    )
    x_line = np.r_[-0.6, x]

    ax_bot.plot(
        x_line,
        cum_ai,
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="AI group",
        zorder=5,
    )
    ax_bot.plot(
        x_line,
        cum_non,
        color=ORANGE,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="Non-AI group",
        zorder=4,
    )
    ax_bot.plot(
        x_line,
        cum_full,
        color=AQUA,
        linewidth=2.0,
        marker="o",
        markersize=4,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="Full universe (EW)",
        zorder=3,
    )
    if cum_etf is not None:
        ax_bot.plot(
            x_line,
            cum_etf,
            color=MUTED,
            linewidth=2.0,
            marker="o",
            markersize=4,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label="510300 ETF (chain)",
            zorder=2,
        )
        ax_bot.annotate(
            f"ETF {cum_etf[-1]:.2f}×",
            xy=(x_line[-1], cum_etf[-1]),
            xytext=(8, -2),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=MUTED,
        )

    ax_bot.annotate(
        f"AI {cum_ai[-1]:.2f}×",
        xy=(x_line[-1], cum_ai[-1]),
        xytext=(8, 2),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=INK_SECONDARY,
    )
    ax_bot.axhline(1.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax_bot.set_title(
        "Cumulative growth (gross, ×)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax_bot.set_ylabel("Cumulative growth (×)", fontsize=10, color=INK_SECONDARY)
    ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax_bot.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(labels, rotation=0, fontsize=9, color=MUTED)
    ax_bot.set_xlabel("Semi-annual period", fontsize=10, color=INK_SECONDARY)
    ax_bot.tick_params(axis="both", length=0, labelsize=9)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["left"].set_color(BASELINE)
    ax_bot.spines["bottom"].set_color(BASELINE)
    ax_bot.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=4,
        handlelength=1.6,
        columnspacing=1.4,
    )

    ax_top.set_xlim(-0.8, len(df) + 0.7)
    fig.suptitle(
        "AI compute chain vs the rest of the CSI 300",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.942,
        f"Equal-weight, gross hfq returns; semi-annual periods, {window}",
        ha="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    fig.text(
        0.06,
        0.012,
        "AI group = target codes (LLM is_in_chain verdicts on current business descriptions, not point-in-time). "
        "Non-AI = remainder of the 300-name universe. Asterisks mark period ends truncated to the last trading day; 2026H2 is partial.\n"
        "Spread = equal-weight AI mean minus non-AI mean; the long-short is a factor return, not an investable A-share strategy. "
        "Cumulative = product of period returns; between-period gap days are not earned; the ETF line is the period chain.",
        ha="left",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.065, right=0.98, hspace=0.16)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[output] chart: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-group period return analysis: AI compute chain vs non-chain CSI 300",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-csv", default=DEFAULT_TARGET_CSV)
    parser.add_argument("--weight-glob", default=DEFAULT_WEIGHT_GLOB)
    parser.add_argument("--k-dir", default=DEFAULT_K_DIR)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=default_end_date())
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-csv", default=DEFAULT_CSV_OUT)
    parser.add_argument("--output-md", default=DEFAULT_MD_OUT)
    parser.add_argument("--output-png", default=DEFAULT_PNG_OUT)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    benchmark_file = resolve_benchmark(args.benchmark)
    if benchmark_file:
        print(f"[benchmark] {benchmark_file}")
    else:
        print("[benchmark] none; ETF context will be n/a")

    result, summary = run_analysis(
        target_csv=args.target_csv,
        weight_glob=args.weight_glob,
        k_dir=args.k_dir,
        benchmark_file=benchmark_file,
        start_date=args.start_date,
        end_date=args.end_date,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )

    result.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[output] csv: {args.output_csv}")
    write_markdown(
        result,
        summary,
        args.output_md,
        target_csv=args.target_csv,
        weight_glob=args.weight_glob,
        benchmark_file=benchmark_file,
    )
    if not args.no_plot:
        plot_results(result, args.output_png)

    print("\n=== Period-level summary ===")
    for label, stats in summary.items():
        print(
            f"{label:14s} n={stats['n']:2d}  "
            f"mean={stats['mean_spread']*100:+6.2f}pp  "
            f"t={stats['t_stat']:5.2f}  hit={stats['hit_rate']*100:5.1f}%  "
            f"cumLS={stats['cum_long_short']*100:+7.2f}%  "
            f"cumAI={stats['cum_ai']*100:+7.2f}%  "
            f"cumETF={stats['cum_etf']*100:+7.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
