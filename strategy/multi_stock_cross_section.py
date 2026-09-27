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


# ============================================================
# 0. ATR 指标（若上游未提供则自动补算）
# ============================================================
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    df["ATR_PCT"] = df["ATR"] / df["close"] * 100
    return df


# ============================================================
# 0'. 日志基础设施
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
# 1. 配置
# ============================================================
@dataclass
class PositionConfig:
    # ---------- 风险预算 ----------
    base_risk_pct: float = 0.02          # 提高风险预算
    max_position_pct: float = 0.20       # 降低单票最大仓位
    lot_size: int = 100

    # ---------- ATR ----------
    atr_period: int = 14

    # ---------- 分层止损 ----------
    init_atr_mult: float = 2.0           # 降低初始止损
    tech_stop_buffer: float = 0.005
    trail_atr_mult: float = 2.5          # 移动止损更紧
    trail_activate_R: float = 2.0        # 更早激活移动止损
    hard_max_loss_pct: float = 0.06      # 硬止损更紧
    min_stop_atr: float = 0.5

    # ---------- 分批止盈 ----------
    tp_levels: Tuple[Tuple[float, float], ...] = (
        (1.5, 0.33),
        (3.0, 0.33),
    )

    # ---------- 趋势反转清仓 ----------
    exit_on_big_reverse: bool = True
    exit_on_small_reverse: bool = False
    big_reverse_threshold: int = -2
    small_reverse_threshold: int = -3

    # ---------- 冷却期 ----------
    cooldown_bars: int = 10              # 延长冷却期

    # ---------- 回测摩擦 ----------
    slippage: float = 0.0005
    commission: float = 0.0003

    # ---------- 池化参数 ----------
    top_n: int = 5
    min_mult: float = 0.30               # 大幅提高开仓门槛
    cash_buffer: float = 0.98
    exit_on_dropout: bool = True         # 开启排名掉队清仓
    dropout_buffer: int = 2              # 排名跌出 top_n * buffer 才清仓
    max_new_positions_per_day: int = 2   # 每天最多开仓数量

    # ---------- 日志 ----------
    log_position_daily: bool = False


# ============================================================
# 2. 信号 → 仓位乘数
# ============================================================
def _trend_state(score: float) -> int:
    if pd.isna(score):
        return 0
    if score >= 2:
        return 1
    if score <= -2:
        return -1
    return 0


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
    """
    使用平滑后的趋势分数，并增加硬性过滤。
    """
    big = float(row.get("BIG_SCORE_SMOOTH", 0) or 0)
    small = float(row.get("SMALL_SCORE_SMOOTH", 0) or 0)

    # 硬性条件：大趋势必须 >= 2 且小趋势 >= 0，或大趋势 >= 0 且小趋势 >= 2
    if not ((big >= 2 and small >= 0) or (big >= 0 and small >= 2)):
        return 0.0

    mult = _STATE_MULT.get((_trend_state(big), _trend_state(small)), 0.0)
    if mult <= 0:
        return 0.0

    # ADX 已在 tech_analysis 中做过 gate，这里不再重复惩罚
    # 仅保留趋势健康度和突破的微调
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
# 3. 分层止损计算
# ============================================================
def compute_initial_stops(entry: float, atr: float, support: Optional[float],
                          cfg: PositionConfig):
    s_init = entry - cfg.init_atr_mult * atr
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
# 4. 风险预算法仓位
# ============================================================
def compute_position_size(equity: float, entry: float, stop: float,
                          multiplier: float, cfg: PositionConfig
                          ) -> Tuple[int, float, float]:
    r = entry - stop
    if r <= 0 or multiplier <= 0:
        return 0, 0.0, 0.0

    risk_budget = equity * cfg.base_risk_pct * multiplier
    shares = risk_budget / r

    cap_shares = equity * cfg.max_position_pct / entry
    shares = min(shares, cap_shares)

    shares = int(shares // cfg.lot_size) * cfg.lot_size
    return shares, float(r), float(risk_budget)


# ============================================================
# 5. 持仓状态
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

    # -------- 复盘用 --------
    entry_signal: dict = field(default_factory=dict)
    entry_ctx: dict = field(default_factory=dict)
    stop_history: List[dict] = field(default_factory=list)

    # -------- 新增：准确计算平均出场价 --------
    total_exit_amount: float = 0.0
    total_exit_shares: float = 0.0

    def __post_init__(self):
        if self.high_water <= 0:
            self.high_water = self.entry_price


# ============================================================
# 6. 池化回测主循环
# ============================================================
def backtest_pool(
    data_dict: Dict[str, pd.DataFrame],
    cfg: Optional[PositionConfig] = None,
    initial_equity: float = 1_000_000.0,
    log_dir: Optional[str] = None,
    market_df: Optional[pd.DataFrame] = None,   # 新增：大盘数据，用于市场过滤
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

    # ---------- 预处理 ----------
    prepared: Dict[str, pd.DataFrame] = {}
    for sym, df in data_dict.items():
        d = df.copy()
        if "ATR" not in d.columns:
            d = add_atr(d, cfg.atr_period)
        # 确保有平滑分数
        if "BIG_SCORE_SMOOTH" not in d.columns and "BIG_SCORE" in d.columns:
            d["BIG_SCORE_SMOOTH"] = d["BIG_SCORE"].rolling(3, min_periods=1).mean()
        if "SMALL_SCORE_SMOOTH" not in d.columns and "SMALL_SCORE" in d.columns:
            d["SMALL_SCORE_SMOOTH"] = d["SMALL_SCORE"].rolling(3, min_periods=1).mean()
        prepared[sym] = d

    # 大盘过滤准备
    market_ok = {}
    if market_df is not None and not market_df.empty:
        m = market_df.copy()
        m["MA60"] = m["close"].rolling(60).mean()
        m["market_ok"] = (m["close"] > m["MA60"]) & (m["MA60"].diff() > 0)
        for date, row in m.iterrows():
            market_ok[date] = bool(row["market_ok"])
    else:
        # 如果没有大盘数据，默认始终允许交易
        for df in prepared.values():
            for date in df.index:
                market_ok[date] = True

    all_dates = sorted(set().union(*[set(df.index) for df in prepared.values()]))

    # ---------- 组合状态 ----------
    cash = float(initial_equity)
    positions: Dict[str, Position] = {}
    cooldowns: Dict[str, int] = {sym: 0 for sym in prepared}
    last_close: Dict[str, float] = {sym: np.nan for sym in prepared}
    trades: List[dict] = []
    equity_curve: List[dict] = []

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

    def _exec_exit(sym: str, date, price: float, reason: str) -> None:
        nonlocal cash
        pos = positions.pop(sym)
        gross = pos.shares * price
        fee = gross * cfg.commission
        cash += gross - fee
        pos.realized_pnl += (price - pos.entry_price) * pos.shares - fee

        # 记录最终卖出
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
            "CLOSE",
            symbol=sym, date=str(date),
            price=float(price), reason=reason,
            shares_remaining=float(pos.shares),
            initial_shares=float(pos.initial_shares),
            pnl=float(pos.realized_pnl),
            bars_held=pos.bars_held,
            entry_price=pos.entry_price,
            entry_date=str(pos.entry_date),
            high_water=pos.high_water,
            partial_exits=pos.partial_exits,
            stop_history=pos.stop_history,
        )
        logger.info(
            f"[CLOSE] {sym} @{price:.3f} 原因={reason} "
            f"PnL={pos.realized_pnl:+,.0f} 持仓={pos.bars_held}根"
        )
        cooldowns[sym] = cfg.cooldown_bars

    # ============================================================
    # 主循环
    # ============================================================
    for date in all_dates:
        reject_log: List[dict] = []

        # (D) 冷却期递减（移到每日开始）
        for sym in cooldowns:
            if cooldowns[sym] > 0:
                cooldowns[sym] -= 1

        # 市场过滤
        is_market_ok = market_ok.get(date, True)
        if not is_market_ok:
            logger.info(f"市场状态不佳，暂停开仓: {date}")

        # ----------------------------------------------------
        # (0) 全池打分 + 排名
        # ----------------------------------------------------
        daily_scores: Dict[str, float] = {}
        daily_rows: Dict[str, pd.Series] = {}
        daily_prev: Dict[str, pd.Series] = {}

        for s, df_s in prepared.items():
            # ========== 方案 A 修复：已持仓股票也参与打分与排名 ==========
            # 只跳过冷却期股票，不再跳过已持仓股票
            if cooldowns[s] > 0:
                continue
            # ============================================================
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
                    "reason": "invalid_atr", "atr": float(atr_s)
                    if not pd.isna(atr_s) else None,
                })
                continue
            if pd.isna(atr_pct) or atr_pct < 1.0 or atr_pct > 8.0:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "atr_pct_out_of_range", "atr_pct": float(atr_pct)
                    if not pd.isna(atr_pct) else None,
                })
                continue
            # 硬性趋势条件
            big_smooth = prev_s.get("BIG_SCORE_SMOOTH", 0)
            small_smooth = prev_s.get("SMALL_SCORE_SMOOTH", 0)
            adx = prev_s.get("ADX", np.nan)
            if not ((big_smooth >= 2 and small_smooth >= 0) or
                    (big_smooth >= 0 and small_smooth >= 2)):
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "trend_filter", "big": float(big_smooth),
                    "small": float(small_smooth),
                })
                continue
            if pd.isna(adx) or adx < 20:
                reject_log.append({
                    "symbol": s, "date": str(date),
                    "reason": "adx_too_low", "adx": float(adx)
                    if not pd.isna(adx) else None,
                })
                continue

            daily_scores[s] = m_s
            daily_rows[s] = df_s.iloc[loc_s]
            daily_prev[s] = prev_s

        ranked = sorted(daily_scores, key=lambda x: -daily_scores[x])
        top_set = set(ranked[:cfg.top_n])

        # ----------------------------------------------------
        # (A) 已有持仓管理
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
                    close=float(row["close"]),
                    stop=pos.stop, high_water=pos.high_water,
                    shares=float(pos.shares),
                    unrealized_R=((float(row["close"]) - pos.entry_price)
                                  / pos.r_per_share) if pos.r_per_share else None,
                )

            # ① 止损（跳空按开盘）
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
                    "R_multiple": float((fill - pos.entry_price)
                                        / pos.r_per_share)
                    if pos.r_per_share else None,
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

            # ③ 趋势反转（使用平滑分数，且如果已止盈则更严格）
            big_smooth = prev.get("BIG_SCORE_SMOOTH", 0)
            small_smooth = prev.get("SMALL_SCORE_SMOOTH", 0)
            if pos.tp_taken:
                # 已经止盈过，要求更严格才清仓
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

            # ④ 排名掉队（带缓冲）—— 方案 A 修复后，持仓股票已进入 ranked
            if cfg.exit_on_dropout:
                rank = ranked.index(sym) + 1 if sym in ranked else 9999
                if rank > cfg.top_n * cfg.dropout_buffer:
                    _exec_exit(sym, date,
                               float(row["close"]) * (1 - cfg.slippage),
                               "dropout")
                    continue

            # ⑤ 收盘后更新移动 / 技术止损
            atr_now = prev["ATR"] if not pd.isna(prev["ATR"]) else pos.entry_atr
            old_stop = pos.stop

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
        # (B) 横截面建仓
        # ----------------------------------------------------
        slots = cfg.top_n - len(positions)
        new_positions_today = 0
        if slots > 0 and is_market_ok:
            total_eq = cash + sum(
                positions[s].shares *
                (last_close[s] if not pd.isna(last_close[s])
                 else positions[s].entry_price)
                for s in positions
            )

            for rank_idx, sym in enumerate(ranked, start=1):
                if slots <= 0 or new_positions_today >= cfg.max_new_positions_per_day:
                    break
                if sym in positions:
                    continue

                m = daily_scores[sym]
                row = daily_rows[sym]
                prev = daily_prev[sym]
                atr_prev = float(prev["ATR"])

                entry = float(row["open"]) * (1 + cfg.slippage)
                support = prev.get("SUPPORT_1", np.nan)

                s_init, s_hard, s_tech = compute_initial_stops(
                    entry, atr_prev, support, cfg)
                stop = effective_stop(entry, atr_prev,
                                      s_init, s_hard, s_tech, None, cfg)

                shares, r_share, risk_budget = compute_position_size(
                    total_eq, entry, stop, m, cfg)

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
                        "cash": float(cash), "total_eq": float(total_eq),
                    })
                    continue
                if cost + fee > cash:
                    reject_log.append({
                        "symbol": sym, "date": str(date),
                        "reason": "insufficient_cash",
                        "need": float(cost + fee), "cash": float(cash),
                    })
                    continue

                # ---- 正式建仓 ----
                cash_before = cash
                cash -= (cost + fee)

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
                    "support": (float(support)
                                if not pd.isna(support) else None),
                    "ATR_prev": atr_prev,
                    "multiplier": float(m),
                    "rank": rank_idx,
                }
                ctx_snapshot = {
                    "total_eq": float(total_eq),
                    "cash_before": float(cash_before),
                    "risk_budget": float(risk_budget),
                    "cap_shares": int(total_eq * cfg.max_position_pct
                                      / entry // cfg.lot_size) * cfg.lot_size,
                    "cash_cap_shares": int(max_sh),
                    "final_shares": int(shares),
                    "entry_price": float(entry),
                    "s_init": float(s_init),
                    "s_hard": float(s_hard),
                    "s_tech": (float(s_tech)
                               if s_tech is not None else None),
                    "stop_effective": float(stop),
                    "r_per_share": float(r_share),
                    "slippage_paid": float(float(row["open"])
                                           * cfg.slippage * shares),
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
                    f"止损={stop:.3f} 排名#{rank_idx} "
                    f"mult={m:.2f} 风险预算={risk_budget:,.0f}"
                )

        # ----------------------------------------------------
        # (C) 记录被拒的候选
        # ----------------------------------------------------
        for r in reject_log:
            recorder.log("REJECT", **r)

        # ----------------------------------------------------
        # (E) 记录组合净值
        # ----------------------------------------------------
        mv = sum(
            pos.shares *
            (last_close[sym] if not pd.isna(last_close[sym])
             else pos.entry_price)
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
            top_rank=ranked[:cfg.top_n],
            top_scores=[round(float(daily_scores[s]), 3)
                        for s in ranked[:cfg.top_n]],
        )

    # ---------- 收尾 ----------
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
        logger.info(f"离场原因分布: {trades_df['reason'].value_counts().to_dict()}")

    recorder.close()
    return trades_df, equity_df


# ============================================================
# 7. 回测报告
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
            n_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win, avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_hold_bars=avg_hold,
            avg_entry_mult=avg_mult,
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
        print("-" * 62)
        print("【离场原因分布】")
        for reason, cnt in trades["reason"].value_counts().items():
            sub = trades[trades["reason"] == reason]
            print(f"  {reason:<22} {cnt:>3} 笔   平均盈亏 {sub['pnl'].mean():>+10,.0f}")
    print("=" * 62)
    return metrics


def pool_report(trades: pd.DataFrame, equity: pd.DataFrame,
                initial_equity: float) -> dict:
    metrics = report(trades, equity, initial_equity)

    if trades.empty:
        return metrics

    print("\n【分股票贡献】")
    by_sym = (trades.groupby("symbol")
                    .agg(n=("pnl", "size"),
                         pnl=("pnl", "sum"),
                         win=("pnl", lambda s: (s > 0).mean()))
                    .sort_values("pnl", ascending=False))
    for sym, r in by_sym.iterrows():
        print(f"  {sym:<10} {int(r['n']):>3}笔  "
              f"PnL {r['pnl']:>+12,.0f}  胜率 {r['win']*100:>5.1f}%")

    print("\n【离场原因分布】")
    for reason, cnt in trades["reason"].value_counts().items():
        sub = trades[trades["reason"] == reason]
        print(f"  {reason:<22} {cnt:>3} 笔   平均盈亏 {sub['pnl'].mean():>+10,.0f}")

    return metrics


# ============================================================
# 8. 使用示例
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path as _P

    project_root = _P(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from strategy.tech_analysis import get_stock_data, compute_trend

    data_dict = {
        "sh.600584": compute_trend(get_stock_data("test_data/data_sh.600584_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.601020": compute_trend(get_stock_data("test_data/data_sh.601020_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.603986": compute_trend(get_stock_data("test_data/data_sh.603986_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.605255": compute_trend(get_stock_data("test_data/data_sh.605255_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.688256": compute_trend(get_stock_data("test_data/data_sh.688256_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
        "sh.688525": compute_trend(get_stock_data("test_data/data_sh.688525_stock_price.csv"), adx_threshold=20, adx_mode="shrink"),
    }

    # 可选：加载大盘数据，例如沪深300指数
    # market_df = get_stock_data("test_data/data_sh.000300_stock_price.csv")
    market_df = None

    cfg = PositionConfig(
        top_n=5,
        min_mult=0.30,
        base_risk_pct=0.02,
        max_position_pct=0.20,
        cooldown_bars=10,
        init_atr_mult=2.0,
        trail_atr_mult=2.5,
        trail_activate_R=2.0,
        hard_max_loss_pct=0.06,
        tp_levels=((1.5, 0.33), (3.0, 0.33)),
        exit_on_dropout=True,
        dropout_buffer=2,
        max_new_positions_per_day=2,
        log_position_daily=False,
    )

    trades, equity = backtest_pool(
        data_dict, cfg,
        initial_equity=1_000_000,
        log_dir="logs",
        market_df=market_df,
    )
    pool_report(trades, equity, 1_000_000)

