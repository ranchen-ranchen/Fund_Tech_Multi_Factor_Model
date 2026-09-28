# multi_stock_cross_section.py
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict

# 把项目根目录加入 sys.path
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "utils"))

from calc_utils import add_atr, trend_state


# ============================================================
# 日志基础设施
# ============================================================
def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pool_bt")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = RotatingFileHandler(
        log_dir / "run.log", maxBytes=20 * 1024 * 1024,
        backupCount=5, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class EventRecorder:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def log(self, event: str, **kw):
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **kw,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ============================================================
# 配置
# ============================================================
@dataclass
class PositionConfig:
    base_risk_pct: float = 0.01
    max_position_pct: float = 0.20
    lot_size: int = 100
    max_total_risk_pct: float = 0.06

    drawdown_halve_threshold: float = 0.10
    drawdown_halve_mult: float = 0.5
    drawdown_stop_threshold: float = 0.20
    drawdown_stop_days: int = 5

    atr_period: int = 14

    init_atr_mult: float = 3.5
    tech_stop_buffer: float = 0.005
    trail_atr_mult: float = 2.5
    trail_activate_R: float = 3.0
    hard_max_loss_pct: float = 0.10
    min_stop_atr: float = 0.5

    atr_pct_ma_window: int = 20
    dynamic_stop_min_mult: float = 0.8
    dynamic_stop_max_mult: float = 1.5

    breakeven_activate_R: float = 2.0

    tp_levels: Tuple[Tuple[float, float], ...] = (
        (1.5, 0.20),
        (3.0, 0.30),
    )

    exit_on_big_reverse: bool = True
    exit_on_small_reverse: bool = False
    big_reverse_threshold: int = -2
    small_reverse_threshold: int = -3

    cooldown_bars: int = 10

    slippage: float = 0.0005
    commission: float = 0.0003

    top_n: int = 5
    min_mult: float = 0.50
    cash_buffer: float = 0.98
    exit_on_dropout: bool = True
    dropout_buffer: int = 4
    max_new_positions_per_day: int = 1
    rebalance_weekly: bool = True

    log_position_daily: bool = False


# ============================================================
# 信号 → 仓位乘数
# ============================================================
_STATE_MULT: Dict[Tuple[int, int], float] = {
    ( 1,  1): 1.00,
    ( 1,  0): 0.70,
    ( 1, -1): 0.30,
    ( 0,  1): 0.40,
    ( 0,  0): 0.20,
    ( 0, -1): 0.00,
    (-1,  1): 0.10,
    (-1,  0): 0.00,
    (-1, -1): 0.00,
}


def signal_multiplier(row: pd.Series) -> float:
    """返回仓位乘数；返回 0 表示不允许开仓。"""
    big = float(row.get("BIG_SCORE_SMOOTH", 0) or 0)
    small = float(row.get("SMALL_SCORE_SMOOTH", 0) or 0)

    if not ((big >= 2 and small >= 0) or (big >= 0 and small >= 2)):
        return 0.0

    mult = _STATE_MULT.get((trend_state(big), trend_state(small)), 0.0)
    if mult <= 0:
        return 0.0

    health = row.get("TREND_HEALTH", np.nan)
    if not pd.isna(health):
        if health >= 40:
            mult *= 1.10
        elif health <= -40:
            mult *= 0.60

    if int(row.get("BREAK_SCORE", 0) or 0) >= 2:
        mult *= 1.15

    if bool(row.get("TOP_WARN", False)):
        mult *= 0.40
    if bool(row.get("BOTTOM_WARN", False)):
        mult *= 1.20

    return float(np.clip(mult, 0.0, 1.5))


# ============================================================
# 分层止损
# ============================================================
def compute_initial_stops(entry: float, atr: float, support: Optional[float],
                          cfg: PositionConfig, dynamic_mult: float = 1.0):
    s_init = entry - cfg.init_atr_mult * dynamic_mult * atr
    s_hard = entry * (1.0 - cfg.hard_max_loss_pct)

    s_tech = None
    if support is not None and not pd.isna(support) and 0 < support < entry:
        s_tech = support * (1.0 - cfg.tech_stop_buffer)

    return s_init, s_hard, s_tech


def effective_stop(entry: float, atr: float, s_init: float, s_hard: float,
                   s_tech: Optional[float], s_trail: Optional[float],
                   cfg: PositionConfig) -> float:
    cands = [s_init, s_hard]
    if s_tech is not None:
        cands.append(s_tech)
    if s_trail is not None:
        cands.append(s_trail)

    stop = max(cands)
    floor = entry - cfg.min_stop_atr * atr
    stop = min(stop, floor)
    stop = max(stop, s_hard)
    stop = min(stop, entry * 0.999)
    return float(stop)


# ============================================================
# 风险预算法仓位
# ============================================================
def compute_position_size(equity: float, entry: float, stop: float,
                          multiplier: float, cfg: PositionConfig,
                          size_scale: float = 1.0
                          ) -> Tuple[int, float, float]:
    """仅做风险预算 + 单票上限；其它约束由主循环统一处理。"""
    r = entry - stop
    if r <= 0 or multiplier <= 0:
        return 0, 0.0, 0.0

    risk_budget = equity * cfg.base_risk_pct * multiplier * size_scale
    shares = risk_budget / r
    cap_shares = equity * cfg.max_position_pct / entry
    shares = min(shares, cap_shares)
    shares = int(shares // cfg.lot_size) * cfg.lot_size
    return shares, float(r), float(risk_budget)


# ============================================================
# 持仓
# ============================================================
@dataclass
class Position:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    initial_shares: float
    r_per_share: float
    s_init: float
    s_hard: float
    s_tech: Optional[float] = None
    s_trail: Optional[float] = None
    stop: float = 0.0
    high_water: float = 0.0
    tp_taken: List[int] = field(default_factory=list)
    bars_held: int = 0
    realized_pnl: float = 0.0
    entry_atr: float = 0.0
    entry_mult: float = 0.0
    partial_exits: List[dict] = field(default_factory=list)

    entry_signal: dict = field(default_factory=dict)
    entry_ctx: dict = field(default_factory=dict)
    stop_history: List[dict] = field(default_factory=list)

    total_exit_amount: float = 0.0
    total_exit_shares: float = 0.0
    dropout_days: int = 0
    breakeven_activated: bool = False

    def __post_init__(self):
        if self.high_water <= 0:
            self.high_water = self.entry_price


# ============================================================
# 回测主循环
# ============================================================
def backtest_pool(
    data_dict: Dict[str, pd.DataFrame],
    cfg: Optional[PositionConfig] = None,
    initial_equity: float = 1_000_000.0,
    log_dir: Optional[str] = None,
    market_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or PositionConfig()

    # ---------- 日志初始化 ----------
    run_dir = Path(log_dir or "logs") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(run_dir)
    recorder = EventRecorder(run_dir / "events.jsonl")

    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"回测开始 | 初始权益={initial_equity:,.0f}")
    logger.info(f"日志目录: {run_dir.resolve()}")

    # ---------- 预处理（仅补上游可能缺失的列） ----------
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in data_dict.items():
        d = df.copy()
        if "ATR" not in d.columns:
            d = add_atr(d, cfg.atr_period)
        if "BIG_SCORE_SMOOTH" not in d.columns and "BIG_SCORE" in d.columns:
            d["BIG_SCORE_SMOOTH"] = d["BIG_SCORE"].ewm(span=5, adjust=False).mean()
        if "SMALL_SCORE_SMOOTH" not in d.columns and "SMALL_SCORE" in d.columns:
            d["SMALL_SCORE_SMOOTH"] = d["SMALL_SCORE"].ewm(span=5, adjust=False).mean()
        if "ATR_PCT" in d.columns:
            d["ATR_PCT_MA20"] = d["ATR_PCT"].rolling(cfg.atr_pct_ma_window).mean()
        else:
            d["ATR_PCT_MA20"] = np.nan
        prepared[sym] = d

    # ---------- 大盘过滤：只记录"允许日期集合" ----------
    market_ok_dates: Optional[set] = None
    if market_df is not None and not market_df.empty:
        m = market_df.copy()
        m["MA60"] = m["close"].rolling(60).mean()
        ok = (m["close"] > m["MA60"]) & (m["MA60"].diff() > 0)
        market_ok_dates = set(m.index[ok.fillna(False)])

    all_dates = sorted(set().union(*[set(df.index) for df in prepared.values()]))

    # ---------- 组合状态 ----------
    cash = float(initial_equity)
    positions: Dict[str, Position] = {}
    cooldowns: Dict[str, int] = {sym: 0 for sym in prepared}
    last_close: Dict[str, float] = {sym: np.nan for sym in prepared}
    trades: List[dict] = []
    equity_curve: List[dict] = []

    last_rebalance_week = None
    peak_equity = float(initial_equity)
    drawdown_block_until_idx = -1

    # ---------- 内部工具 ----------
    def _loc(df: pd.DataFrame, date) -> Optional[int]:
        if date not in df.index:
            return None
        loc = df.index.get_loc(date)
        return loc.start if isinstance(loc, slice) else int(loc)

    def _safe_float(v, default=np.nan):
        try:
            return float(v)
        except Exception:
            return default

    def _calc_equity() -> float:
        mv = sum(
            pos.shares *
            (last_close[sym] if not pd.isna(last_close[sym]) else pos.entry_price)
            for sym, pos in positions.items()
        )
        return cash + mv

    def _current_total_risk() -> float:
        return sum(
            max(0.0, pos.entry_price - pos.stop) * pos.shares
            for pos in positions.values()
        )

    def _exec_exit(sym: str, date, price: float, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        gross = pos.shares * price
        fee = gross * cfg.commission
        cash += gross - fee
        pos.realized_pnl += (price - pos.entry_price) * pos.shares - fee
        pos.total_exit_amount += gross
        pos.total_exit_shares += pos.shares

        avg_exit = (pos.total_exit_amount / pos.total_exit_shares
                    if pos.total_exit_shares > 0 else pos.entry_price)

        row = None
        df_s = prepared[sym]
        if date in df_s.index:
            r = df_s.loc[date]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            row = r

        trades.append({
            "symbol": sym,
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "exit_date": date, "exit_price": float(avg_exit),
            "shares": pos.initial_shares,
            "pnl": float(pos.realized_pnl),
            "return_pct": float(pos.realized_pnl /
                                (pos.entry_price * pos.initial_shares)),
            "bars_held": pos.bars_held,
            "reason": reason,
            "entry_mult": pos.entry_mult,
            "r_per_share": pos.r_per_share,
            "tp_taken": len(pos.tp_taken),
            **{f"in_{k}": v for k, v in pos.entry_signal.items()},
            **{f"ctx_{k}": v for k, v in pos.entry_ctx.items()},
            "exit_open": _safe_float(row["open"]) if row is not None else None,
            "exit_high": _safe_float(row["high"]) if row is not None else None,
            "exit_low": _safe_float(row["low"]) if row is not None else None,
            "exit_close": _safe_float(row["close"]) if row is not None else None,
            "exit_fee": fee,
            "high_water": pos.high_water,
            "stop_history": json.dumps(pos.stop_history, default=str,
                                       ensure_ascii=False),
            "partial_exits": json.dumps(pos.partial_exits, default=str,
                                        ensure_ascii=False),
        })

        recorder.log(
            "CLOSE", symbol=sym, date=str(date), price=float(price),
            reason=reason, pnl=float(pos.realized_pnl),
            bars_held=pos.bars_held, entry_price=pos.entry_price,
            high_water=pos.high_water,
        )
        logger.info(
            f"[CLOSE] {sym} @{price:.3f} 原因={reason} "
            f"PnL={pos.realized_pnl:+,.0f} 持仓={pos.bars_held}根"
        )
        cooldowns[sym] = cfg.cooldown_bars

    # ============================================================
    # 主循环
    # ============================================================
    for idx, date in enumerate(all_dates):
        reject_log: List[dict] = []

        # ---------- 回撤状态 ----------
        current_equity = _calc_equity()
        if current_equity > peak_equity:
            peak_equity = current_equity
        current_dd = ((peak_equity - current_equity) / peak_equity
                      if peak_equity > 0 else 0.0)

        if current_dd >= cfg.drawdown_stop_threshold:
            new_block_until = idx + cfg.drawdown_stop_days
            if new_block_until > drawdown_block_until_idx:
                drawdown_block_until_idx = new_block_until
                logger.info(
                    f"[DRAWDOWN-STOP] {date} 当前回撤={current_dd*100:.2f}% "
                    f">= {cfg.drawdown_stop_threshold*100:.0f}%，"
                    f"停止开仓至 idx={drawdown_block_until_idx}"
                )

        is_drawdown_blocked = idx < drawdown_block_until_idx
        size_scale = (cfg.drawdown_halve_mult
                      if current_dd >= cfg.drawdown_halve_threshold else 1.0)

        # ---------- 冷却期递减 ----------
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        # ---------- 调仓日 ----------
        if cfg.rebalance_weekly:
            current_week = (date.year, date.isocalendar()[1])
            if last_rebalance_week is None or current_week != last_rebalance_week:
                is_rebalance_day = True
                last_rebalance_week = current_week
            else:
                is_rebalance_day = False
        else:
            is_rebalance_day = True

        # ---------- 市场状态 ----------
        is_market_ok = (market_ok_dates is None) or (date in market_ok_dates)
        if not is_market_ok:
            logger.info(f"市场状态不佳，暂停开仓: {date}")

        # ----------------------------------------------------
        # (0) 全池打分 + 排名
        # ----------------------------------------------------
        daily_scores: Dict[str, float] = {}
        daily_rows: Dict[str, pd.Series] = {}
        daily_prev: Dict[str, pd.Series] = {}

        for s, df_s in prepared.items():
            if cooldowns[s] > 0:
                continue
            loc_s = _loc(df_s, date)
            if loc_s is None or loc_s < 1:
                continue
            prev_s = df_s.iloc[loc_s - 1]
            m_s = signal_multiplier(prev_s)
            atr_s = prev_s.get("ATR", np.nan)
            atr_pct = prev_s.get("ATR_PCT", np.nan)

            if m_s < cfg.min_mult:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "below_min_mult", "mult": float(m_s),
                })
                continue
            if pd.isna(atr_s) or atr_s <= 0:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "invalid_atr",
                })
                continue
            if pd.isna(atr_pct) or atr_pct < 1.0 or atr_pct > 8.0:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "atr_pct_out_of_range",
                    "atr_pct": float(atr_pct),
                })
                continue
            # 趋势过滤集中在 signal_multiplier 完成，此处只需补充 ADX 门槛
            adx = prev_s.get("ADX", np.nan)
            if pd.isna(adx) or adx < 20:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "adx_too_low",
                    "adx": None if pd.isna(adx) else float(adx),
                })
                continue

            daily_scores[s] = m_s
            daily_rows[s] = df_s.iloc[loc_s]
            daily_prev[s] = prev_s

        ranked = sorted(daily_scores, key=lambda x: -daily_scores[x])
        top_set = set(ranked[:cfg.top_n])

        # ----------------------------------------------------
        # (A) 持仓管理
        # ----------------------------------------------------
        for sym in list(positions.keys()):
            df = prepared[sym]
            loc = _loc(df, date)
            if loc is None or loc < 1:
                continue
            row = df.iloc[loc]
            prev = df.iloc[loc - 1]
            pos = positions[sym]

            pos.bars_held += 1
            pos.high_water = max(pos.high_water, float(row["high"]))
            last_close[sym] = float(row["close"])

            if cfg.log_position_daily:
                recorder.log(
                    "POS_DAILY", symbol=sym, date=str(date),
                    close=float(row["close"]), stop=pos.stop,
                    high_water=pos.high_water, shares=float(pos.shares),
                )

            # ① 止损
            if row["open"] <= pos.stop:
                _exec_exit(sym, date,
                           float(row["open"]) * (1 - cfg.slippage), "stop_gap")
                continue
            if row["low"] <= pos.stop:
                _exec_exit(sym, date,
                           pos.stop * (1 - cfg.slippage), "stop")
                continue

            # ② 分批止盈
            R = pos.r_per_share
            for k, (r_mult, frac) in enumerate(cfg.tp_levels):
                if k in pos.tp_taken:
                    continue
                tp_price = pos.entry_price + r_mult * R
                if row["high"] < tp_price:
                    continue
                sell = min(pos.initial_shares * frac, pos.shares)
                sell = int(sell // cfg.lot_size) * cfg.lot_size
                if sell <= 0:
                    continue
                fill = tp_price * (1 - cfg.slippage)
                gross = sell * fill
                fee = gross * cfg.commission
                cash += gross - fee
                pos.realized_pnl += (fill - pos.entry_price) * sell - fee
                pos.shares -= sell
                pos.tp_taken.append(k)
                pos.total_exit_amount += gross
                pos.total_exit_shares += sell
                pos.partial_exits.append({
                    "date": str(date), "price": float(fill),
                    "shares": float(sell), "reason": f"tp_{r_mult}R",
                    "remaining": float(pos.shares),
                    "R_multiple": (float((fill - pos.entry_price) / R)
                                   if R else None),
                })
                recorder.log(
                    "TP", symbol=sym, date=str(date),
                    r_mult=r_mult, price=float(fill),
                    shares=sell, remaining=float(pos.shares),
                )
                logger.info(
                    f"[TP] {sym} @{fill:.3f} 卖{sell}股 ({r_mult}R) "
                    f"剩余{int(pos.shares)}"
                )

            if pos.shares < cfg.lot_size:
                _exec_exit(sym, date, float(row["close"]), "all_tp")
                continue

            # ③ 趋势反转
            big_smooth = prev.get("BIG_SCORE_SMOOTH", 0)
            small_smooth = prev.get("SMALL_SCORE_SMOOTH", 0)
            if pos.tp_taken:
                big_rv = (cfg.exit_on_big_reverse and big_smooth <= -3)
                small_rv = (cfg.exit_on_small_reverse and small_smooth <= -4)
            else:
                big_rv = (cfg.exit_on_big_reverse and
                          big_smooth <= cfg.big_reverse_threshold and
                          small_smooth <= 0)
                small_rv = (cfg.exit_on_small_reverse and
                            small_smooth <= cfg.small_reverse_threshold)
            if big_rv or small_rv:
                reason = "trend_reverse_big" if big_rv else "trend_reverse_small"
                _exec_exit(sym, date,
                           float(row["open"]) * (1 - cfg.slippage), reason)
                continue

            # ④ 排名掉队
            if cfg.exit_on_dropout:
                rank = ranked.index(sym) + 1 if sym in ranked else 9999
                if rank > cfg.top_n * cfg.dropout_buffer:
                    pos.dropout_days += 1
                else:
                    pos.dropout_days = 0
                if pos.dropout_days >= 2:
                    _exec_exit(sym, date,
                               float(row["close"]) * (1 - cfg.slippage),
                               "dropout")
                    continue

            # ⑤ 止损更新
            atr_now = prev["ATR"] if not pd.isna(prev["ATR"]) else pos.entry_atr
            old_stop = pos.stop

            if (not pos.breakeven_activated and
                    pos.high_water >= pos.entry_price + cfg.breakeven_activate_R * R):
                pos.breakeven_activated = True
                if pos.s_trail is None or pos.s_trail < pos.entry_price:
                    pos.s_trail = pos.entry_price
                    recorder.log("BREAKEVEN", symbol=sym, date=str(date),
                                 high_water=pos.high_water,
                                 entry=pos.entry_price)

            if pos.high_water >= pos.entry_price + cfg.trail_activate_R * R:
                new_trail = pos.high_water - cfg.trail_atr_mult * atr_now
                if pos.s_trail is None or new_trail > pos.s_trail:
                    pos.s_trail = float(new_trail)
                    recorder.log(
                        "TRAIL_ACTIVATE", symbol=sym, date=str(date),
                        high_water=pos.high_water, trail=pos.s_trail,
                        atr_now=float(atr_now),
                    )

            sup = row.get("SUPPORT_1", np.nan)
            if not pd.isna(sup) and 0 < sup < row["close"]:
                new_tech = float(sup) * (1 - cfg.tech_stop_buffer)
                if pos.s_tech is None or new_tech > pos.s_tech:
                    pos.s_tech = new_tech

            pos.stop = effective_stop(
                pos.entry_price, atr_now,
                pos.s_init, pos.s_hard, pos.s_tech, pos.s_trail, cfg,
            )

            if abs(pos.stop - old_stop) > 1e-9:
                pos.stop_history.append({
                    "date": str(date),
                    "old": float(old_stop), "new": float(pos.stop),
                    "high_water": float(pos.high_water),
                    "s_init": float(pos.s_init),
                    "s_hard": float(pos.s_hard),
                    "s_tech": float(pos.s_tech) if pos.s_tech is not None else None,
                    "s_trail": float(pos.s_trail) if pos.s_trail is not None else None,
                })
                recorder.log(
                    "STOP_MOVE", symbol=sym, date=str(date),
                    old_stop=float(old_stop), new_stop=float(pos.stop),
                    high_water=float(pos.high_water),
                )

        # ----------------------------------------------------
        # (B) 建仓
        # ----------------------------------------------------
        slots = cfg.top_n - len(positions)
        new_positions_today = 0
        if slots > 0 and is_market_ok and is_rebalance_day and not is_drawdown_blocked:
            total_eq = _calc_equity()

            current_risk = _current_total_risk()
            remaining_risk_budget = max(
                0.0, total_eq * cfg.max_total_risk_pct - current_risk
            )

            for rank_idx, sym in enumerate(ranked, start=1):
                if (slots <= 0 or
                        new_positions_today >= cfg.max_new_positions_per_day):
                    break
                if sym in positions:
                    continue
                # 修复：当天被卖出的股票可能仍在 ranked 中，需再查冷却期
                if cooldowns.get(sym, 0) > 0:
                    continue
                if remaining_risk_budget <= 0:
                    reject_log.append({
                        "symbol": sym, "date": str(date),
                        "reason": "total_risk_limit_reached",
                    })
                    break

                m = daily_scores[sym]
                row = daily_rows[sym]
                prev = daily_prev[sym]
                atr_prev = float(prev["ATR"])

                entry = float(row["open"]) * (1 + cfg.slippage)
                support = prev.get("SUPPORT_1", np.nan)

                atr_pct = prev.get("ATR_PCT", np.nan)
                atr_pct_ma20 = prev.get("ATR_PCT_MA20", np.nan)
                if (not pd.isna(atr_pct) and not pd.isna(atr_pct_ma20)
                        and atr_pct_ma20 > 0):
                    dynamic_mult = float(np.clip(
                        atr_pct / atr_pct_ma20,
                        cfg.dynamic_stop_min_mult,
                        cfg.dynamic_stop_max_mult,
                    ))
                else:
                    dynamic_mult = 1.0

                s_init, s_hard, s_tech = compute_initial_stops(
                    entry, atr_prev, support, cfg, dynamic_mult)
                stop = effective_stop(entry, atr_prev,
                                      s_init, s_hard, s_tech, None, cfg)

                shares, r_share, risk_budget = compute_position_size(
                    total_eq, entry, stop, m, cfg, size_scale=size_scale)

                # 剩余总风险约束
                if r_share > 0:
                    max_shares_by_risk = int(
                        remaining_risk_budget / r_share // cfg.lot_size
                    ) * cfg.lot_size
                    shares = min(shares, max_shares_by_risk)

                # 单票上限 & 现金约束
                cap_shares = int(
                    total_eq * cfg.max_position_pct / entry // cfg.lot_size
                ) * cfg.lot_size
                shares = min(shares, cap_shares)
                max_sh = int((cash * cfg.cash_buffer)
                             // (entry * cfg.lot_size)) * cfg.lot_size
                shares = min(shares, max_sh)

                cost = shares * entry
                fee = cost * cfg.commission

                if shares < cfg.lot_size:
                    reject_log.append({
                        "symbol": sym, "date": str(date),
                        "reason": "size_too_small",
                        "mult": float(m), "entry": entry, "stop": stop,
                        "shares_raw": int(shares),
                    })
                    continue
                if cost + fee > cash:
                    reject_log.append({
                        "symbol": sym, "date": str(date),
                        "reason": "insufficient_cash",
                        "need": float(cost + fee), "cash": float(cash),
                    })
                    continue

                cash_before = cash
                cash -= (cost + fee)
                remaining_risk_budget -= max(0.0, entry - stop) * shares

                signal_snapshot = {
                    "BIG_SCORE": _safe_float(prev.get("BIG_SCORE", 0), 0.0),
                    "SMALL_SCORE": _safe_float(prev.get("SMALL_SCORE", 0), 0.0),
                    "BIG_SCORE_SMOOTH": _safe_float(prev.get("BIG_SCORE_SMOOTH", 0), 0.0),
                    "SMALL_SCORE_SMOOTH": _safe_float(prev.get("SMALL_SCORE_SMOOTH", 0), 0.0),
                    "ADX": _safe_float(prev.get("ADX", np.nan)),
                    "TREND_HEALTH": _safe_float(prev.get("TREND_HEALTH", np.nan)),
                    "BREAK_SCORE": int(prev.get("BREAK_SCORE", 0) or 0),
                    "TOP_WARN": bool(prev.get("TOP_WARN", False)),
                    "BOTTOM_WARN": bool(prev.get("BOTTOM_WARN", False)),
                    "support": (float(support) if not pd.isna(support) else None),
                    "ATR_prev": atr_prev,
                    "dynamic_mult": float(dynamic_mult),
                    "multiplier": float(m),
                    "rank": rank_idx,
                }
                ctx_snapshot = {
                    "total_eq": float(total_eq),
                    "cash_before": float(cash_before),
                    "risk_budget": float(risk_budget),
                    "size_scale": float(size_scale),
                    "current_dd": float(current_dd),
                    "final_shares": int(shares),
                    "entry_price": float(entry),
                    "s_init": float(s_init),
                    "s_hard": float(s_hard),
                    "s_tech": (float(s_tech) if s_tech is not None else None),
                    "stop_effective": float(stop),
                    "r_per_share": float(r_share),
                    "fee_paid": float(fee),
                }

                pos = Position(
                    entry_date=date, entry_price=entry,
                    shares=float(shares), initial_shares=float(shares),
                    r_per_share=r_share,
                    s_init=s_init, s_hard=s_hard, s_tech=s_tech,
                    stop=stop, entry_atr=atr_prev, entry_mult=m,
                )
                pos.entry_signal = signal_snapshot
                pos.entry_ctx = ctx_snapshot
                positions[sym] = pos
                last_close[sym] = float(row["close"])
                slots -= 1
                new_positions_today += 1

                recorder.log("OPEN", symbol=sym, date=str(date),
                             **signal_snapshot, **ctx_snapshot)
                logger.info(
                    f"[OPEN] {sym} @{entry:.3f} 股数={int(shares)} "
                    f"止损={stop:.3f} 排名#{rank_idx} mult={m:.2f} "
                    f"size_scale={size_scale:.2f} 风险预算={risk_budget:,.0f}"
                )

        # ----------------------------------------------------
        # (C) 记录被拒候选
        # ----------------------------------------------------
        for r in reject_log:
            recorder.log("REJECT", **r)

        # ----------------------------------------------------
        # (E) 记录净值
        # ----------------------------------------------------
        mv = sum(
            pos.shares *
            (last_close[sym] if not pd.isna(last_close[sym]) else pos.entry_price)
            for sym, pos in positions.items()
        )
        equity_curve.append({
            "date": date,
            "equity": cash + mv,
            "cash": cash,
            "mv": mv,
            "n_positions": len(positions),
        })

        recorder.log(
            "DAILY", date=str(date),
            equity=float(cash + mv), cash=float(cash), mv=float(mv),
            n_positions=len(positions),
            current_dd=float(current_dd),
            size_scale=float(size_scale),
            drawdown_blocked=bool(is_drawdown_blocked),
            total_risk=float(_current_total_risk()),
            top_rank=ranked[:cfg.top_n],
            top_scores=[round(float(daily_scores[s]), 3)
                        for s in ranked[:cfg.top_n]],
        )

    # ---------- 收尾：强制平仓 ----------
    for sym in list(positions.keys()):
        df = prepared[sym]
        _exec_exit(sym, df.index[-1],
                   float(df.iloc[-1]["close"]) * (1 - cfg.slippage),
                   "end_of_backtest")

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")

    trades_df.to_csv(run_dir / "trades.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(run_dir / "daily.csv", encoding="utf-8-sig")

    logger.info(
        f"回测完成 | 交易 {len(trades_df)} 笔 | "
        f"最终权益 {equity_df['equity'].iloc[-1]:,.0f}"
    )
    if not trades_df.empty:
        logger.info(f"离场原因分布: "
                    f"{trades_df['reason'].value_counts().to_dict()}")

    recorder.close()
    return trades_df, equity_df


# ============================================================
# 报告
# ============================================================
def report(trades: pd.DataFrame, equity: pd.DataFrame,
           initial_equity: float) -> dict:
    if equity.empty:
        print("无净值数据")
        return {}

    eq = equity["equity"]
    total_return = eq.iloc[-1] / initial_equity - 1.0
    n = len(eq)
    annual = (1 + total_return) ** (252.0 / max(n, 1)) - 1.0

    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min())

    daily = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    if trades.empty:
        metrics = dict(total_return=total_return, annual_return=annual,
                       max_drawdown=max_dd, sharpe=sharpe, n_trades=0)
    else:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / len(trades)
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        gross_win = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
        expectancy = trades["pnl"].mean()
        avg_hold = trades["bars_held"].mean()
        avg_mult = trades["entry_mult"].mean()

        metrics = dict(
            total_return=total_return, annual_return=annual,
            max_drawdown=max_dd, sharpe=sharpe,
            n_trades=len(trades), win_rate=win_rate,
            avg_win=avg_win, avg_loss=avg_loss,
            profit_factor=profit_factor, expectancy=expectancy,
            avg_hold_bars=avg_hold, avg_entry_mult=avg_mult,
        )

    print("=" * 62)
    print("【回测绩效】")
    print(f"  总收益      : {metrics['total_return']*100:+.2f}%")
    print(f"  年化收益    : {metrics['annual_return']*100:+.2f}%")
    print(f"  最大回撤    : {metrics['max_drawdown']*100:.2f}%")
    print(f"  夏普比率    : {metrics['sharpe']:.2f}")
    if not trades.empty:
        print(f"  交易次数    : {metrics['n_trades']}")
        print(f"  胜率        : {metrics['win_rate']*100:.1f}%")
        print(f"  盈亏比      : {metrics['profit_factor']:.2f}")
        print(f"  单笔期望    : {metrics['expectancy']:,.0f}")
        print(f"  平均持仓    : {metrics['avg_hold_bars']:.1f} 根K线")
        print(f"  平均仓位乘数: {metrics['avg_entry_mult']:.2f}")
    print("=" * 62)
    return metrics


def pool_report(trades: pd.DataFrame, equity: pd.DataFrame,
                initial_equity: float) -> dict:
    metrics = report(trades, equity, initial_equity)

    if trades.empty:
        return metrics

    print("\n【离场原因分布】")
    for reason, cnt in trades["reason"].value_counts().items():
        sub = trades[trades["reason"] == reason]
        print(f"  {reason:<22} {cnt:>3} 笔   "
              f"平均盈亏 {sub['pnl'].mean():>+10,.0f}")

    print("\n【分股票贡献】")
    by_sym = (trades.groupby("symbol")
                    .agg(n=("pnl", "size"),
                         pnl=("pnl", "sum"),
                         win=("pnl", lambda s: (s > 0).mean()))
                    .sort_values("pnl", ascending=False))
    for sym, r in by_sym.iterrows():
        print(f"  {sym:<10} {int(r['n']):>3}笔  "
              f"PnL {r['pnl']:>+12,.0f}  胜率 {r['win']*100:>5.1f}%")

    return metrics


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    project_root = _P(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    try:
        from strategy.tech_analysis import get_stock_data, compute_trend
    except ImportError:
        from tech_analysis import get_stock_data, compute_trend

    data_dict = {
        "sh.600584": compute_trend(get_stock_data("test_data/data_sh.600584_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.601020": compute_trend(get_stock_data("test_data/data_sh.601020_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.603986": compute_trend(get_stock_data("test_data/data_sh.603986_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.605255": compute_trend(get_stock_data("test_data/data_sh.605255_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.688256": compute_trend(get_stock_data("test_data/data_sh.688256_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.688525": compute_trend(get_stock_data("test_data/data_sh.688525_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
    }

    market_df = None

    cfg = PositionConfig(
        top_n=5,
        min_mult=0.50,
        base_risk_pct=0.01,
        max_position_pct=0.20,
        max_total_risk_pct=0.06,
        drawdown_halve_threshold=0.10,
        drawdown_halve_mult=0.5,
        drawdown_stop_threshold=0.20,
        drawdown_stop_days=5,
        cooldown_bars=10,
        init_atr_mult=3.5,
        trail_atr_mult=2.5,
        trail_activate_R=3.0,
        hard_max_loss_pct=0.10,
        breakeven_activate_R=2.0,
        tp_levels=((1.5, 0.20), (3.0, 0.30)),
        exit_on_dropout=True,
        dropout_buffer=4,
        max_new_positions_per_day=1,
        rebalance_weekly=True,
        log_position_daily=False,
    )

    trades, equity = backtest_pool(
        data_dict, cfg,
        initial_equity=1_000_000,
        log_dir="logs",
        market_df=market_df,
    )
    pool_report(trades, equity, 1_000_000)




