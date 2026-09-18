"""
独立仓位管理系统（ATR 分层止损 + 风险预算 + 分批止盈）

依赖上游 df 至少包含：
    open/high/low/close/volume
    BIG_SCORE, SMALL_SCORE, ADX, ADX_WEAK
    SUPPORT_1, TREND_HEALTH, BREAK_SCORE
    TOP_WARN, BOTTOM_WARN
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
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
    # Wilder 平滑，与 ADX 中用的 ATR 保持一致
    df["ATR"] = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    df["ATR_PCT"] = df["ATR"] / df["close"] * 100
    return df


# ============================================================
# 1. 配置：所有可调参数集中在这里
# ============================================================
@dataclass
class PositionConfig:
    # ---------- 风险预算 ----------
    base_risk_pct: float = 0.010          # 单笔基础风险 = 权益 × 1%
    max_position_pct: float = 0.30        # 单票最大仓位占比
    lot_size: int = 100                   # A 股一手 = 100 股

    # ---------- ATR ----------
    atr_period: int = 14

    # ---------- 分层止损 ----------
    init_atr_mult: float = 2.0            # 初始止损 = entry − 2·ATR
    tech_stop_buffer: float = 0.005       # 技术止损缓冲：支撑下方 0.5%
    trail_atr_mult: float = 3.0           # 移动止损 = high_water − 3·ATR
    trail_activate_R: float = 1.0         # 浮盈 ≥ 1R 后才启用移动止损
    hard_max_loss_pct: float = 0.08       # 硬性最大亏损 8%
    min_stop_atr: float = 0.5             # 有效止损至少离入场 0.5·ATR

    # ---------- 分批止盈：(R 倍数, 卖出比例) ----------
    tp_levels: Tuple[Tuple[float, float], ...] = (
        (1.5, 0.30),
        (2.5, 0.30),
        (4.0, 0.40),
    )

    # ---------- 趋势反转清仓 ----------
    exit_on_big_reverse: bool = True
    exit_on_small_reverse: bool = False
    big_reverse_threshold: int = -2
    small_reverse_threshold: int = -3

    # ---------- 冷却期 ----------
    cooldown_bars: int = 5

    # ---------- 回测摩擦 ----------
    slippage: float = 0.0005              # 单边 0.05%
    commission: float = 0.0003            # 单边 0.03%


# ============================================================
# 2. 信号 → 仓位乘数
# ============================================================
def _trend_state(score: float) -> int:
    """±4 的分数 → 向上/中性/向下"""
    if pd.isna(score):
        return 0
    if score >= 2:
        return 1
    if score <= -2:
        return -1
    return 0


# 九宫格基础乘数（与上游 DECISION_MAP 语义一致）
_STATE_MULT: Dict[Tuple[int, int], float] = {
    ( 1,  1): 1.00,   # 主升浪
    ( 1,  0): 0.70,   # 上升途中整理
    ( 1, -1): 0.30,   # 上升趋势中的回调
    ( 0,  1): 0.40,   # 震荡反弹
    ( 0,  0): 0.20,   # 横盘
    ( 0, -1): 0.00,   # 震荡转弱
    (-1,  1): 0.10,   # 下跌趋势中的反弹（只做试探）
    (-1,  0): 0.00,   # 弱势整理
    (-1, -1): 0.00,   # 空头排列
}


def signal_multiplier(row: pd.Series) -> float:
    """
    将技术信号压缩为 [0, 1.5] 的仓位乘数。

    调节因子：
      · 九宫格基础乘数
      · ADX 趋势强度调节（无趋势降仓位）
      · TREND_HEALTH 量价健康度调节
      · BREAK_SCORE 放量突破加分
      · TOP_WARN 顶部预警强制减仓
      · BOTTOM_WARN 底部预警可小仓试多
    """
    big = float(row.get("BIG_SCORE", 0) or 0)
    small = float(row.get("SMALL_SCORE", 0) or 0)

    mult = _STATE_MULT.get((_trend_state(big), _trend_state(small)), 0.0)
    if mult <= 0:
        return 0.0

    # ---- ADX 趋势强度调节 ----
    adx = row.get("ADX", np.nan)
    if pd.isna(adx) or adx < 15:
        mult *= 0.50
    elif adx < 20:
        mult *= 0.80
    elif adx >= 30:
        mult *= 1.10

    # ---- 量价健康度 ----
    health = row.get("TREND_HEALTH", np.nan)
    if not pd.isna(health):
        if health >= 40:
            mult *= 1.10
        elif health <= -40:
            mult *= 0.60

    # ---- 放量突破加分 ----
    if int(row.get("BREAK_SCORE", 0) or 0) >= 2:
        mult *= 1.15

    # ---- 顶部/底部预警 ----
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
    """
    返回 (s_init, s_hard, s_tech) —— 各层独立价格。

      s_init = entry − init_atr_mult · ATR
      s_hard = entry · (1 − hard_max_loss_pct)      ← 绝对下限
      s_tech = support · (1 − tech_stop_buffer)     ← 若有效支撑存在
    """
    s_init = entry - cfg.init_atr_mult * atr
    s_hard = entry * (1.0 - cfg.hard_max_loss_pct)

    s_tech = None
    if support is not None and not pd.isna(support) and 0 < support < entry:
        s_tech = support * (1.0 - cfg.tech_stop_buffer)

    return s_init, s_hard, s_tech


def effective_stop(entry: float, atr: float, s_init: float, s_hard: float,
                   s_tech: Optional[float], s_trail: Optional[float],
                   cfg: PositionConfig) -> float:
    """
    有效止损 = max(各层生效价格)   —— 取最紧
    再受两个约束：
      ① 不得比 entry − min_stop_atr·ATR 更紧（防止被噪音扫损）
      ② 不得低于 s_hard（硬性最大亏损兜底）
    """
    cands = [s_init, s_hard]
    if s_tech is not None:
        cands.append(s_tech)
    if s_trail is not None:
        cands.append(s_trail)

    stop = max(cands)

    floor = entry - cfg.min_stop_atr * atr
    stop = min(stop, floor)          # 不能太紧
    stop = max(stop, s_hard)         # 不能太松
    stop = min(stop, entry * 0.999)  # 必须低于入场价
    return float(stop)


# ============================================================
# 4. 风险预算法仓位
# ============================================================
def compute_position_size(equity: float, entry: float, stop: float,
                          multiplier: float, cfg: PositionConfig) -> Tuple[int, float, float]:
    """
    返回 (shares, R_per_share, risk_budget)
      R_per_share = entry − stop        每股风险
      risk_budget = equity × base_risk_pct × multiplier
      shares      = risk_budget / R_per_share     ← 再受 max_position_pct 约束
    """
    r = entry - stop
    if r <= 0 or multiplier <= 0:
        return 0, 0.0, 0.0

    risk_budget = equity * cfg.base_risk_pct * multiplier
    shares = risk_budget / r

    # 单票仓位上限
    cap_shares = equity * cfg.max_position_pct / entry
    shares = min(shares, cap_shares)

    # 整手
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

    def __post_init__(self):
        if self.high_water <= 0:
            self.high_water = self.entry_price


# ============================================================
# 6. 回测主循环
# ============================================================
def backtest(df: pd.DataFrame,
             cfg: Optional[PositionConfig] = None,
             initial_equity: float = 1_000_000.0):
    """
    事件驱动回测：
      · 第 i 日开盘用第 i−1 日的信号 + 第 i−1 日的 ATR 建仓
      · 第 i 日盘中按 open/high/low 顺序处理止损/止盈
      · 收盘后更新 high_water、移动止损、技术止损
    返回 (trades_df, equity_df)
    """
    if cfg is None:
        cfg = PositionConfig()

    if "ATR" not in df.columns:
        df = add_atr(df, cfg.atr_period)
    df = df.copy()

    cash = float(initial_equity)
    pos: Optional[Position] = None
    cooldown = 0
    trades: List[dict] = []
    equity_curve: List[dict] = []

    def _close_position(p: Position, date, price, reason: str, commission: bool = True):
        nonlocal cash
        gross = p.shares * price
        fee = gross * cfg.commission if commission else 0.0
        cash += gross - fee
        p.realized_pnl += (price - p.entry_price) * p.shares - fee
        p.partial_exits.append({
            "date": date, "price": float(price),
            "shares": float(p.shares), "reason": reason,
        })

    for i in range(1, len(df)):
        date = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # ----------------------------------------------------
        # (A) 已有持仓 → 逐层处理
        # ----------------------------------------------------
        if pos is not None:
            pos.bars_held += 1
            pos.high_water = max(pos.high_water, float(row["high"]))

            exited = False

            # ---- ① 止损检查（盘中触发，跳空按开盘） ----
            stop_hit_price = None
            if row["open"] <= pos.stop:
                stop_hit_price = float(row["open"]) * (1 - cfg.slippage)
                stop_reason = "stop_gap"
            elif row["low"] <= pos.stop:
                stop_hit_price = pos.stop * (1 - cfg.slippage)
                stop_reason = "stop"

            if stop_hit_price is not None:
                _close_position(pos, date, stop_hit_price, stop_reason)
                # 汇总一笔完整交易
                avg_exit = (pos.entry_price * pos.initial_shares +
                            pos.realized_pnl) / pos.initial_shares
                trades.append({
                    "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                    "exit_date": date, "exit_price": float(avg_exit),
                    "shares": pos.initial_shares,
                    "pnl": float(pos.realized_pnl),
                    "return_pct": float(pos.realized_pnl /
                                        (pos.entry_price * pos.initial_shares)),
                    "bars_held": pos.bars_held,
                    "reason": stop_reason,
                    "entry_mult": pos.entry_mult,
                    "r_per_share": pos.r_per_share,
                    "tp_taken": len(pos.tp_taken),
                })
                pos = None
                cooldown = cfg.cooldown_bars
                exited = True

            # ---- ② 分批止盈（盘中触发） ----
            if not exited and pos is not None:
                R = pos.r_per_share
                for k, (r_mult, frac) in enumerate(cfg.tp_levels):
                    if k in pos.tp_taken:
                        continue
                    tp_price = pos.entry_price + r_mult * R
                    if row["high"] >= tp_price:
                        sell_shares = min(pos.initial_shares * frac, pos.shares)
                        sell_shares = int(sell_shares // cfg.lot_size) * cfg.lot_size
                        if sell_shares <= 0:
                            continue
                        fill = tp_price * (1 - cfg.slippage)
                        gross = sell_shares * fill
                        fee = gross * cfg.commission
                        cash += gross - fee
                        pos.realized_pnl += (fill - pos.entry_price) * sell_shares - fee
                        pos.shares -= sell_shares
                        pos.tp_taken.append(k)
                        pos.partial_exits.append({
                            "date": date, "price": float(fill),
                            "shares": float(sell_shares),
                            "reason": f"tp_{r_mult}R",
                        })

                if pos is not None and pos.shares < cfg.lot_size:
                    # 全部分批止盈完成
                    avg_exit = (pos.entry_price * pos.initial_shares +
                                pos.realized_pnl) / pos.initial_shares
                    trades.append({
                        "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                        "exit_date": date, "exit_price": float(avg_exit),
                        "shares": pos.initial_shares,
                        "pnl": float(pos.realized_pnl),
                        "return_pct": float(pos.realized_pnl /
                                            (pos.entry_price * pos.initial_shares)),
                        "bars_held": pos.bars_held,
                        "reason": "all_tp",
                        "entry_mult": pos.entry_mult,
                        "r_per_share": pos.r_per_share,
                        "tp_taken": len(pos.tp_taken),
                    })
                    pos = None
                    cooldown = cfg.cooldown_bars
                    exited = True

            # ---- ③ 趋势反转清仓（用前一日信号，开盘执行） ----
            if not exited and pos is not None:
                big_rv = (cfg.exit_on_big_reverse
                          and int(prev.get("BIG_SCORE", 0) or 0) <= cfg.big_reverse_threshold)
                small_rv = (cfg.exit_on_small_reverse
                            and int(prev.get("SMALL_SCORE", 0) or 0) <= cfg.small_reverse_threshold)
                if big_rv or small_rv:
                    exit_price = float(row["open"]) * (1 - cfg.slippage)
                    reason = "trend_reverse_big" if big_rv else "trend_reverse_small"
                    _close_position(pos, date, exit_price, reason)
                    avg_exit = (pos.entry_price * pos.initial_shares +
                                pos.realized_pnl) / pos.initial_shares
                    trades.append({
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
                    })
                    pos = None
                    cooldown = cfg.cooldown_bars
                    exited = True

            # ---- ④ 收盘后更新 移动止损 / 技术止损 ----
            if not exited and pos is not None:
                atr_now = prev["ATR"] if not pd.isna(prev["ATR"]) else pos.entry_atr
                R = pos.r_per_share

                # 移动止损：浮盈 ≥ trail_activate_R 后启用，只上移
                if pos.high_water >= pos.entry_price + cfg.trail_activate_R * R:
                    new_trail = pos.high_water - cfg.trail_atr_mult * atr_now
                    if pos.s_trail is None or new_trail > pos.s_trail:
                        pos.s_trail = float(new_trail)

                # 技术止损：支撑上移则同步上移，只上移
                sup = row.get("SUPPORT_1", np.nan)
                if not pd.isna(sup) and 0 < sup < row["close"]:
                    new_tech = float(sup) * (1 - cfg.tech_stop_buffer)
                    if pos.s_tech is None or new_tech > pos.s_tech:
                        pos.s_tech = new_tech

                # 重算有效止损
                pos.stop = effective_stop(
                    pos.entry_price, atr_now,
                    pos.s_init, pos.s_hard, pos.s_tech, pos.s_trail, cfg,
                )

        # ----------------------------------------------------
        # (B) 空仓 → 尝试建仓
        # ----------------------------------------------------
        if pos is None and cooldown == 0:
            mult = signal_multiplier(prev)
            atr_prev = prev.get("ATR", np.nan)

            if mult > 0 and not pd.isna(atr_prev) and atr_prev > 0:
                entry = float(row["open"]) * (1 + cfg.slippage)
                support = prev.get("SUPPORT_1", np.nan)

                s_init, s_hard, s_tech = compute_initial_stops(
                    entry, float(atr_prev), support, cfg
                )
                stop = effective_stop(
                    entry, float(atr_prev), s_init, s_hard, s_tech, None, cfg
                )

                shares, r_share, risk_budget = compute_position_size(
                    cash, entry, stop, mult, cfg
                )
                cost = shares * entry
                fee = cost * cfg.commission

                if shares >= cfg.lot_size and cost + fee <= cash:
                    cash -= (cost + fee)
                    pos = Position(
                        entry_date=date,
                        entry_price=entry,
                        shares=float(shares),
                        initial_shares=float(shares),
                        r_per_share=r_share,
                        s_init=s_init,
                        s_hard=s_hard,
                        s_tech=s_tech,
                        stop=stop,
                        entry_atr=float(atr_prev),
                        entry_mult=mult,
                    )

        if cooldown > 0:
            cooldown -= 1

        # ----------------------------------------------------
        # (C) 记录净值
        # ----------------------------------------------------
        mv = pos.shares * float(row["close"]) if pos is not None else 0.0
        equity_curve.append({"date": date, "equity": cash + mv,
                             "cash": cash, "mv": mv,
                             "in_position": pos is not None})

    # ---- 收尾：强制平仓 ----
    if pos is not None:
        last = df.iloc[-1]
        price = float(last["close"]) * (1 - cfg.slippage)
        _close_position(pos, df.index[-1], price, "end_of_backtest")
        avg_exit = (pos.entry_price * pos.initial_shares +
                    pos.realized_pnl) / pos.initial_shares
        trades.append({
            "entry_date": pos.entry_date, "entry_price": pos.entry_price,
            "exit_date": df.index[-1], "exit_price": float(avg_exit),
            "shares": pos.initial_shares,
            "pnl": float(pos.realized_pnl),
            "return_pct": float(pos.realized_pnl /
                                (pos.entry_price * pos.initial_shares)),
            "bars_held": pos.bars_held,
            "reason": "end_of_backtest",
            "entry_mult": pos.entry_mult,
            "r_per_share": pos.r_per_share,
            "tp_taken": len(pos.tp_taken),
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("date")
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


# ============================================================
# 8. 使用示例
# ============================================================
if __name__ == "__main__":
    # ---- 上游：拿到含技术信号的 df ----
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    # from utils.text_utils import read_from_csv
    from strategy.tech_analysis import get_stock_data, compute_trend
    df = get_stock_data("/home/omen/work/quant/data/data_sh.600584_stock_price.csv")
    df = compute_trend(df, adx_threshold=20, adx_mode="shrink")
    # ---- 本模块 ----
    cfg = PositionConfig(
        base_risk_pct=0.010,       # 单笔风险 1%
        init_atr_mult=2.0,         # 初始止损 2 ATR
        trail_atr_mult=3.0,        # 移动止损 3 ATR
        hard_max_loss_pct=0.08,    # 硬性最大亏损 8%
        tp_levels=((1.5, 0.3), (2.5, 0.3), (4.0, 0.4)),
    )
    trades, equity = backtest(df, cfg, initial_equity=1_000_000)
    report(trades, equity, 1_000_000)
    pass