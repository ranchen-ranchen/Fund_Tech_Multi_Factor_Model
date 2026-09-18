

# import logging
# logger = logging.getLogger(__name__) 
# import yaml
# from pathlib import Path
# config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
# with open(config_path, 'r', encoding='utf-8') as file:
#     config = yaml.safe_load(file)
# time_window = config['TIME_WINDOW']
# price_level_threshold = config['PRICE_LEVEL_THRESHOLD']
# price_trend_threshold = config['PRICE_TREND_THRESHOLD']
# trading_amount_window = config['TRADING_AMOUNT_WINDOW']
# price_trend_window = config['PRICE_TREND_WINDOW']
# trading_range_threshold = config['TRADING_RANGE_THRESHOLD']

import pandas as pd
import numpy as np
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.text_utils import read_from_csv




def get_stock_data(filename):
    df = read_from_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]



def add_ma(df, windows=(5, 10, 20, 60, 120, 250)):
    for n in windows:
        df[f"MA{n}"] = df["close"].rolling(n).mean()
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = 2 * (df["DIF"] - df["DEA"])
    return df


def add_adx(df, period=14):
    """ADX 趋势强度指标"""
    high, low, close = df["high"], df["low"], df["close"]
    up, down = high.diff(), -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return df


def rolling_slope(series, window):
    """滚动线性回归斜率（未归一化）"""
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y):
        if np.isnan(y).any():
            return np.nan
        return ((x - x_mean) * (y - y.mean())).sum() / denom

    return series.rolling(window).apply(_slope, raw=True)


# ============================================================
# 3. 大趋势 / 小趋势 打分
# ============================================================
def compute_trend(df, adx_threshold=20, adx_mode="shrink",
                  pivot_window=5, sr_lookback=120, sr_tolerance=0.015,
                  enable_volume=True):
    df = df.copy()
    df = add_ma(df)
    df = add_macd(df)
    df = add_adx(df)

     # 支撑 / 阻力
    df = add_support_resistance(df,
                                pivot_window=pivot_window,
                                lookback=sr_lookback,
                                tolerance=sr_tolerance)



    # 均线斜率，转成"日均涨跌百分比"，便于跨股票比较
    df["SLOPE_MA120"] = rolling_slope(df["MA120"], 20) / df["MA120"] * 100
    df["SLOPE_MA20"] = rolling_slope(df["MA20"], 5) / df["MA20"] * 100

    # ---- 周线 MACD 作为大趋势的辅助确认 ----
    weekly = df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    weekly = add_macd(weekly)

    df["W_DIF"] = weekly["DIF"].reindex(df.index, method="ffill")
    df["W_DEA"] = weekly["DEA"].reindex(df.index, method="ffill")

    # ---------- 大趋势打分（4 个条件，每个 ±1）----------
    big = pd.DataFrame(index=df.index)
    big["价格在MA120上方"] = np.where(df["close"] > df["MA120"], 1, -1)
    big["MA60>MA120"] = np.where(df["MA60"] > df["MA120"], 1, -1)
    big["MA120向上"] = np.where(df["SLOPE_MA120"] > 0, 1, -1)
    big["周线MACD多头"] = np.where(df["W_DIF"] > df["W_DEA"], 1, -1)
    df["BIG_SCORE"] = big.sum(axis=1)          # 取值 -4 ~ +4
    


    # ---------- 小趋势打分（4 个条件，每个 ±1）----------
    small = pd.DataFrame(index=df.index)
    small["价格在MA20上方"] = np.where(df["close"] > df["MA20"], 1, -1)
    small["MA5>MA10>MA20"] = np.where(
        (df["MA5"] > df["MA10"]) & (df["MA10"] > df["MA20"]), 1, -1)
    small["日线MACD多头"] = np.where(df["DIF"] > df["DEA"], 1, -1)
    small["MA20向上"] = np.where(df["SLOPE_MA20"] > 0, 1, -1)
    df["SMALL_SCORE"] = small.sum(axis=1)      # 取值 -4 ~ +4


    # ====== 成交量三维分析 ======
    if enable_volume:
        df = add_volume_features(df)
        df = add_breakout_confirmation(df)
        df = add_trend_health(df)
        df = add_top_bottom_warning(df)
    


    # ---------- ADX 门槛 ----------
    df = apply_adx_gate(df, adx_threshold=adx_threshold, mode=adx_mode)

    return df


# ============================================================
# 3.4 支撑 / 阻力位（无前视）
# ============================================================
def _cluster_levels(levels, tolerance):
    """把相近的价位聚成一个簇，返回簇均值列表"""
    if not levels:
        return []
    s = sorted(levels)
    clusters, cur = [], [s[0]]
    for v in s[1:]:
        if (v - cur[-1]) / cur[-1] <= tolerance:
            cur.append(v)
        else:
            clusters.append(sum(cur) / len(cur))
            cur = [v]
    clusters.append(sum(cur) / len(cur))
    return clusters


def add_support_resistance(df, pivot_window=5, lookback=120,
                           max_levels=3, tolerance=0.015):
    """
    计算每日的支撑 / 阻力位。

    无前视设计：
      · 枢轴高点/低点：center=True 窗口识别后 shift(pivot_window) 对齐，
        保证 t 时刻只使用已经在 t 收盘时被确认的枢轴。
      · 前高/前低：shift(1) 排除当日，再 rolling(lookback)。
      · 聚类、排序全部只依赖 t 及之前的数据。

    新增列：
      SUPPORT_1..3    / RESISTANCE_1..3     由近到远（SUPPORT_1 = 最近支撑）
      DIST_SUPPORT    / DIST_RESISTANCE     相对现价的百分比距离
    """
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    # ---- 1. 历史枢轴点（已对齐，无前视） ----
    win = 2 * pivot_window + 1
    roll_hi_c = high.rolling(win, center=True).max()
    roll_lo_c = low.rolling(win, center=True).min()
    # 位置 p 的枢轴 → 位置 p+pivot_window 才能看到
    pivot_high = high.where(high == roll_hi_c).shift(pivot_window)
    pivot_low  = low.where(low == roll_lo_c).shift(pivot_window)

    # ---- 2. 滚动前高 / 前低（不含当日） ----
    roll_hi = high.shift(1).rolling(lookback, min_periods=1).max()
    roll_lo = low.shift(1).rolling(lookback, min_periods=1).min()

    n = len(df)
    sup_arr = np.full((n, max_levels), np.nan)
    res_arr = np.full((n, max_levels), np.nan)

    ph_np, pl_np = pivot_high.to_numpy(), pivot_low.to_numpy()
    rh_np, rl_np = roll_hi.to_numpy(),   roll_lo.to_numpy()
    cl_np = close.to_numpy()

    for i in range(n):
        # 只保留 lookback 以内的枢轴，控制计算复杂度
        start = max(0, i - lookback)
        levels = []
        for j in range(start, i + 1):
            if not np.isnan(ph_np[j]):
                levels.append(ph_np[j])
            if not np.isnan(pl_np[j]):
                levels.append(pl_np[j])
        if not np.isnan(rh_np[i]):
            levels.append(rh_np[i])
        if not np.isnan(rl_np[i]):
            levels.append(rl_np[i])
        if not levels:
            continue

        price = cl_np[i]
        clusters = _cluster_levels(levels, tolerance)

        # 现价下方的支撑（取最近的 max_levels 个）
        sups = sorted([c for c in clusters if c < price], reverse=True)[:max_levels]
        # 现价上方的阻力
        ress = sorted([c for c in clusters if c > price])[:max_levels]

        for k, v in enumerate(sups):
            sup_arr[i, k] = v
        for k, v in enumerate(ress):
            res_arr[i, k] = v

    for k in range(max_levels):
        df[f"SUPPORT_{k+1}"]    = sup_arr[:, k]
        df[f"RESISTANCE_{k+1}"] = res_arr[:, k]

    df["DIST_SUPPORT"]    = (close - df["SUPPORT_1"])    / close * 100
    df["DIST_RESISTANCE"] = (df["RESISTANCE_1"] - close) / close * 100
    return df



# ============================================================
# 3.5 ADX 门槛：弱趋势时收缩 / 归零分数
# ============================================================
def apply_adx_gate(df, adx_threshold=20, mode="shrink"):
    """
    用 ADX 做趋势门槛：
      - mode="shrink"  : ADX < threshold 时，分数按 ADX/threshold 比例向 0 收缩
      - mode="neutral" : ADX < threshold 时，分数直接归 0（判定为震荡）
      - mode="off"     : 不做任何处理（保留原行为）

    额外写入列：
      BIG_SCORE_RAW / SMALL_SCORE_RAW  : ADX 处理前的原始分数
      ADX_WEAK                         : ADX 是否低于阈值（含 NaN）
    """
    df = df.copy()

    # 保留原始分数
    df["BIG_SCORE_RAW"]   = df["BIG_SCORE"]
    df["SMALL_SCORE_RAW"] = df["SMALL_SCORE"]

    # ADX NaN 视为无趋势（数据不足，保守处理）
    adx_filled = df["ADX"].fillna(0.0)
    df["ADX_WEAK"] = adx_filled < adx_threshold

    if mode == "off":
        return df

    if mode == "neutral":
        weak = df["ADX_WEAK"]
        df.loc[weak, "BIG_SCORE"]   = 0
        df.loc[weak, "SMALL_SCORE"] = 0

    elif mode == "shrink":
        # ADX = 0         → 系数 0   （完全归零）
        # ADX = threshold → 系数 1   （分数不变）
        # 中间线性插值
        factor = (adx_filled / adx_threshold).clip(upper=1.0)
        df["BIG_SCORE"]   = (df["BIG_SCORE"]   * factor).round().astype(int)
        df["SMALL_SCORE"] = (df["SMALL_SCORE"] * factor).round().astype(int)
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 方便 analyze 展示当前模式
    df.attrs["adx_threshold"] = adx_threshold
    df.attrs["adx_mode"]      = mode
    return df

# ============================================================
# 3.6 成交量基础指标
# ============================================================
def add_volume_features(df):
    """成交量均线、量比、OBV 等基础衍生指标"""
    df = df.copy()
    vol = df["volume"].astype(float)

    df["VOL_MA5"]  = vol.rolling(5).mean()
    df["VOL_MA20"] = vol.rolling(20).mean()
    df["VOL_MA60"] = vol.rolling(60).mean()

    # 量比：当日量 / 20 日均量（1 为正常，≥1.5 放量，≤0.6 地量）
    df["VOL_RATIO"] = vol / df["VOL_MA20"].replace(0, np.nan)
    df["VOL_RATIO_MA5"] = df["VOL_RATIO"].rolling(5).mean()

    # OBV 能量潮（累积量能方向）
    sign = np.sign(df["close"].diff()).fillna(0)
    df["OBV"] = (sign * vol).fillna(0).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # 成交量 z-score（衡量离群程度）
    std = vol.rolling(20).std()
    df["VOL_Z"] = ((vol - df["VOL_MA20"]) / std.replace(0, np.nan)).fillna(0)

    return df


# ============================================================
# 3.7 突破 + 成交量确认
# ============================================================
def add_breakout_confirmation(df, vol_ok=1.5, vol_strong=2.0, vol_weak=0.8):
    """
    识别突破并用量能确认：

      BREAK_MA20 / BREAK_MA60 / BREAK_RES   是否发生突破（布尔）
      *_OK                                  放量确认（VOL_RATIO ≥ vol_ok）
      BREAK_SCORE                           +2 强放量 / +1 普通 / 0 无 / -1 缩量假突破
      BREAK_TYPE                            文本描述
    """
    df = df.copy()
    close = df["close"]
    vol_ratio = df["VOL_RATIO"].fillna(1.0)

    prev_close = close.shift(1)
    prev_ma20  = df["MA20"].shift(1)
    prev_ma60  = df["MA60"].shift(1)
    high_60    = df["high"].shift(1).rolling(60, min_periods=20).max()

    df["BREAK_MA20"] = (prev_close <= prev_ma20) & (close > df["MA20"])
    df["BREAK_MA60"] = (prev_close <= prev_ma60) & (close > df["MA60"])
    df["BREAK_RES"]  = close > high_60

    df["BREAK_MA20_OK"] = df["BREAK_MA20"] & (vol_ratio >= vol_ok)
    df["BREAK_MA60_OK"] = df["BREAK_MA60"] & (vol_ratio >= vol_ok)
    df["BREAK_RES_OK"]  = df["BREAK_RES"]  & (vol_ratio >= vol_ok)

    any_break = df["BREAK_MA20"] | df["BREAK_MA60"] | df["BREAK_RES"]
    any_ok    = df["BREAK_MA20_OK"] | df["BREAK_MA60_OK"] | df["BREAK_RES_OK"]
    weak      = any_break & (vol_ratio < vol_weak)
    strong    = any_ok    & (vol_ratio >= vol_strong)

    score = pd.Series(0, index=df.index, dtype=int)
    score[any_break] = 1
    score[weak]      = -1
    score[strong]    = 2
    df["BREAK_SCORE"] = score

    def _type(r):
        tags = []
        if r["BREAK_MA20"]: tags.append("MA20")
        if r["BREAK_MA60"]: tags.append("MA60")
        if r["BREAK_RES"]:  tags.append("前高")
        if not tags:
            return ""
        confirmed = r["BREAK_MA20_OK"] or r["BREAK_MA60_OK"] or r["BREAK_RES_OK"]
        return f"{'放量' if confirmed else '缩量'}突破({'/'.join(tags)})"

    df["BREAK_TYPE"] = df.apply(_type, axis=1)
    return df


# ============================================================
# 3.8 趋势健康度
# ============================================================
def add_trend_health(df, window=20):
    """
    趋势健康度（-100 ~ +100）：
      · 量能结构：上涨日均量 / 下跌日均量  —— 价涨量增、价跌量缩才健康
      · OBV 一致性：OBV 斜率与价格斜率同向加分
      · 量价相关性：收益与量变化的正相关强弱
    """
    df = df.copy()
    vol = df["volume"].astype(float)
    ret = df["close"].pct_change()

    up_mask = ret > 0
    dn_mask = ret < 0
    up_avg = vol.where(up_mask).rolling(window, min_periods=5).mean()
    dn_avg = vol.where(dn_mask).rolling(window, min_periods=5).mean()
    ratio = (up_avg / dn_avg.replace(0, np.nan)).fillna(1.0)
    # ratio = 1 → 0；ratio = 2 → +50；ratio = 0.5 → -50
    df["VOL_HEALTH"] = (np.log2(ratio) * 50).clip(-100, 100)

    price_slope = rolling_slope(df["close"], window)
    obv_slope   = rolling_slope(df["OBV"], window)
    agree = np.sign(price_slope) * np.sign(obv_slope)
    df["OBV_AGREE"] = agree.fillna(0).astype(int)   # +1 同向 / -1 背离

    vol_chg = vol.pct_change().replace([np.inf, -np.inf], np.nan)
    pv_corr = ret.rolling(window).corr(vol_chg).fillna(0)

    health = df["VOL_HEALTH"] * 0.5 + df["OBV_AGREE"] * 25 + pv_corr * 25
    df["TREND_HEALTH"] = health.clip(-100, 100)
    return df


# ============================================================
# 3.9 顶部 / 底部预警
# ============================================================
def add_top_bottom_warning(df, lookback=60,
                           high_zone=0.95, low_zone=1.05,
                           huge_vol=2.5, dry_vol=0.5):
    """
    顶部预警（≥2 项触发）：
      · 高位巨量滞涨（收阴 + 量比 ≥ huge_vol + 接近 60 日高点）
      · 长上影巨量（上影线 > 2 倍实体 + 巨量 + 高位）
      · OBV 顶背离（价格创新高，OBV 未创新高）

    底部预警（≥2 项触发）：
      · 低位地量（量比 ≤ dry_vol + 接近 60 日低点）
      · 放量反弹（阳线 + 量比 ≥ 1.5 + 前 5 日均量萎缩）
      · OBV 底背离
    """
    df = df.copy()
    high, low = df["high"], df["low"]
    close, open_ = df["close"], df["open"]
    vol_ratio = df["VOL_RATIO"].fillna(1.0)

    roll_high = high.rolling(lookback, min_periods=20).max()
    roll_low  = low.rolling(lookback, min_periods=20).min()
    near_high = close >= roll_high * high_zone
    near_low  = close <= roll_low  * low_zone

    # -------- 顶部信号 --------
    sig_top_vol = near_high & (vol_ratio >= huge_vol) & (close < open_)

    body = (close - open_).abs().replace(0, np.nan)
    upper_shadow = high - np.maximum(close, open_)
    sig_top_shadow = (upper_shadow > 2 * body) & (vol_ratio >= huge_vol) & near_high

    obv_high = df["OBV"].rolling(lookback, min_periods=20).max()
    sig_top_div = (close >= roll_high) & (df["OBV"] < obv_high)

    df["TOP_SIGNALS"] = (
        sig_top_vol.astype(int) + sig_top_shadow.astype(int) + sig_top_div.astype(int)
    )
    df["TOP_WARN"] = df["TOP_SIGNALS"] >= 2

    # -------- 底部信号 --------
    sig_bot_dry = near_low & (vol_ratio <= dry_vol)

    vol_ratio_prev5 = df["VOL_RATIO"].shift(1).rolling(5).mean()
    sig_bot_rebound = (close > open_) & (vol_ratio >= 1.5) & (vol_ratio_prev5 < 0.8) & near_low

    obv_low = df["OBV"].rolling(lookback, min_periods=20).min()
    sig_bot_div = (low <= roll_low) & (df["OBV"] > obv_low)

    df["BOTTOM_SIGNALS"] = (
        sig_bot_dry.astype(int) + sig_bot_rebound.astype(int) + sig_bot_div.astype(int)
    )
    df["BOTTOM_WARN"] = df["BOTTOM_SIGNALS"] >= 2

    # 文本原因
    df["TOP_REASON"] = ""
    df.loc[sig_top_vol,    "TOP_REASON"] += "高位巨量滞涨;"
    df.loc[sig_top_shadow, "TOP_REASON"] += "长上影巨量;"
    df.loc[sig_top_div,    "TOP_REASON"] += "OBV顶背离;"
    df["TOP_REASON"] = df["TOP_REASON"].replace("", np.nan)

    df["BOTTOM_REASON"] = ""
    df.loc[sig_bot_dry,     "BOTTOM_REASON"] += "低位地量;"
    df.loc[sig_bot_rebound, "BOTTOM_REASON"] += "放量反弹;"
    df.loc[sig_bot_div,     "BOTTOM_REASON"] += "OBV底背离;"
    df["BOTTOM_REASON"] = df["BOTTOM_REASON"].replace("", np.nan)

    return df




# ============================================================
# 4. 综合研判
# ============================================================
def _state(score):
    """把 -4~+4 的分数映射为 向上 / 中性 / 向下"""
    if score >= 2:
        return "向上"
    if score <= -2:
        return "向下"
    return "中性"


# 大趋势 × 小趋势 九宫格决策表
DECISION_MAP = {
    ("向上", "向上"): ("主升浪", "持股 / 逢回调加仓", "🟢🟢"),
    ("向上", "中性"): ("上升途中的整理", "持股观察，跌破MA20减仓", "🟢"),
    ("向上", "向下"): ("上升趋势中的回调", "不追高，等小趋势重新走强再介入", "🟡"),
    ("中性", "向上"): ("震荡市中的反弹", "短线可参与，快进快出", "🟡"),
    ("中性", "中性"): ("无趋势 / 横盘", "观望，等方向明确", "⚪"),
    ("中性", "向下"): ("震荡转弱", "减仓规避", "🟠"),
    ("向下", "向上"): ("下跌趋势中的反弹", "反弹减仓，不抄底", "🟠"),
    ("向下", "中性"): ("弱势整理", "空仓或极轻仓", "🔴"),
    ("向下", "向下"): ("空头排列", "空仓等待", "🔴🔴"),
}


def analyze(df, date=None):
    """输出某一天的完整研判结果"""
    row = df.iloc[-1] if date is None else df.loc[date]

    big_state = _state(row["BIG_SCORE"])
    small_state = _state(row["SMALL_SCORE"])
    desc, action, icon = DECISION_MAP[(big_state, small_state)]

    # ADX 趋势强度
    adx = row["ADX"]
    threshold = df.attrs.get("adx_threshold", 20)
    mode      = df.attrs.get("adx_mode", "shrink")
    if pd.isna(adx):
        adx_txt = "N/A"
    elif adx >= 25:
        adx_txt = f"{adx:.1f}（趋势强劲）"
    elif adx >= threshold:
        adx_txt = f"{adx:.1f}（趋势初现）"
    else:
        tag = "已向 0 收缩" if mode == "shrink" else "已归 0 判定震荡 "
        adx_txt = f"{adx:.1f} 无趋势/震荡，分数{tag} "

    def _fmt(score, raw):
        s = int(score)
        if row["ADX_WEAK"] and int(raw) != s:
            return f"{s:+d} (原始 {int(raw):+d})"
        return f"{s:+d}"

    big_disp   = _fmt(row["BIG_SCORE"],   row["BIG_SCORE_RAW"])
    small_disp = _fmt(row["SMALL_SCORE"], row["SMALL_SCORE_RAW"])


    print("=" * 62)
    print(f"日期: {row.name.date()}   收盘价: {row['close']:.2f}")
    print("-" * 62)
    print(f"【大趋势】{big_state}   得分 {int(row['BIG_SCORE']):+d} / 4")
    print(f"   价格 vs MA120 : {row['close']:.2f} vs {row['MA120']:.2f}")
    print(f"   MA60  vs MA120: {row['MA60']:.2f} vs {row['MA120']:.2f}")
    print(f"   MA120 斜率    : {row['SLOPE_MA120']:+.3f} % / 日")
    print(f"   周线 MACD     : DIF {row['W_DIF']:+.3f} / DEA {row['W_DEA']:+.3f}")
    print("-" * 62)
    print(f"【小趋势】{small_state}   得分 {int(row['SMALL_SCORE']):+d} / 4")
    print(f"   价格 vs MA20  : {row['close']:.2f} vs {row['MA20']:.2f}")
    print(f"   MA5/10/20     : {row['MA5']:.2f} / {row['MA10']:.2f} / {row['MA20']:.2f}")
    print(f"   日线 MACD     : DIF {row['DIF']:+.3f} / DEA {row['DEA']:+.3f}")
    print(f"   MA20 斜率     : {row['SLOPE_MA20']:+.3f} % / 日")

    # ---------- 支撑 / 阻力 ----------
    price = row["close"]
    sup_cols = [f"SUPPORT_{k}"    for k in (1, 2, 3)]
    res_cols = [f"RESISTANCE_{k}" for k in (1, 2, 3)]
    supports    = [row[c] for c in sup_cols if not pd.isna(row[c])]
    resistances = [row[c] for c in res_cols if not pd.isna(row[c])]

    dist_sup = row["DIST_SUPPORT"]
    dist_res = row["DIST_RESISTANCE"]

    if not supports and not resistances:
        sr_pos = "数据不足"
    elif not pd.isna(dist_sup) and dist_sup <= 2:
        sr_pos = "接近支撑"
    elif not pd.isna(dist_res) and dist_res <= 2:
        sr_pos = "接近阻力"
    elif not pd.isna(dist_sup) and not pd.isna(dist_res):
        sr_pos = "偏支撑侧" if dist_sup < dist_res else "偏阻力侧"
    else:
        sr_pos = "中性"

    print("-" * 62)
    print("【支撑 / 阻力】")
    if supports or resistances:
        for i, s in enumerate(supports, 1):
            print(f"   支撑 {i}: {s:7.2f}   (距现价 {(price - s) / price * 100:+6.2f}%)")
        for i, r in enumerate(resistances, 1):
            print(f"   阻力 {i}: {r:7.2f}   (距现价 {(r - price) / price * 100:+6.2f}%)")
        print(f"   价格位置: {sr_pos}")
    else:
        print("   （历史样本不足）")


    # ---------- 量价分析 ----------
    print("-" * 62)
    print("【量价分析】")
    vol_ratio = row.get("VOL_RATIO", np.nan)
    if not pd.isna(vol_ratio):
        if   vol_ratio >= 2.0: tag = "巨量"
        elif vol_ratio >= 1.5: tag = "放量"
        elif vol_ratio <= 0.6: tag = "地量"
        elif vol_ratio <= 0.8: tag = "缩量"
        else: tag = "正常"
        print(f"   量比 VOL/MA20 : {vol_ratio:.2f}（{tag}）  VOL_MA5/MA20 = {row['VOL_MA5']/row['VOL_MA20']:.2f}")

    health = row.get("TREND_HEALTH", np.nan)
    if not pd.isna(health):
        if   health >=  40: htxt = f"{health:+.0f}（健康）"
        elif health >=   0: htxt = f"{health:+.0f}（一般）"
        elif health >= -40: htxt = f"{health:+.0f}（偏弱）"
        else:               htxt = f"{health:+.0f}（不健康）"
        print(f"   趋势健康度    : {htxt}   （量能结构 {row['VOL_HEALTH']:+.0f} / OBV一致 {row['OBV_AGREE']:+d}）")

    bt = row.get("BREAK_TYPE", "")
    if bt:
        print(f"   突破信号      : {bt}  (BREAK_SCORE = {row['BREAK_SCORE']:+d})")

    if row.get("TOP_WARN", False):
        print(f"   ⚠️  顶部预警   : {row['TOP_REASON']}")
    if row.get("BOTTOM_WARN", False):
        print(f"   🔔 底部预警   : {row['BOTTOM_REASON']}")


    print("-" * 62)
    print(f"【趋势强度】ADX = {adx_txt}")
    print(f"【综合结论】{icon} {desc}")
    print(f"【操作建议】{action}")
    print("=" * 62)

    return {
        "big_trend": big_state, "big_score": int(row["BIG_SCORE"]),
        "big_score_raw": int(row["BIG_SCORE_RAW"]),
        "small_trend": small_state, "small_score": int(row["SMALL_SCORE"]),
        "small_score_raw": int(row["SMALL_SCORE_RAW"]),
        "adx": round(float(adx), 2), 
        "adx_weak": bool(row["ADX_WEAK"]),
        "signal": desc, "action": action,
        "support_1":    float(supports[0])    if supports else None,
        "support_2":    float(supports[1])    if len(supports) > 1 else None,
        "support_3":    float(supports[2])    if len(supports) > 2 else None,
        "resistance_1": float(resistances[0]) if resistances else None,
        "resistance_2": float(resistances[1]) if len(resistances) > 1 else None,
        "resistance_3": float(resistances[2]) if len(resistances) > 2 else None,
        "dist_support":    None if pd.isna(dist_sup) else round(float(dist_sup), 2),
        "dist_resistance": None if pd.isna(dist_res) else round(float(dist_res), 2),
        "sr_position":  sr_pos,
                "vol_ratio":       None if pd.isna(row.get("VOL_RATIO", np.nan)) else round(float(row["VOL_RATIO"]), 2),
        "trend_health":    None if pd.isna(row.get("TREND_HEALTH", np.nan)) else round(float(row["TREND_HEALTH"]), 1),
        "break_type":      row.get("BREAK_TYPE", "") or None,
        "break_score":     int(row.get("BREAK_SCORE", 0)),
        "top_warn":        bool(row.get("TOP_WARN", False)),
        "top_reason":      None if pd.isna(row.get("TOP_REASON")) else row["TOP_REASON"],
        "bottom_warn":     bool(row.get("BOTTOM_WARN", False)),
        "bottom_reason":   None if pd.isna(row.get("BOTTOM_REASON")) else row["BOTTOM_REASON"],
    }



# ============================================================
# 5. 运行
# ============================================================
if __name__ == "__main__":
    df = get_stock_data('/home/omen/work/quant/data/data_sh.688783_stock_price.csv')
    df = compute_trend(df, adx_threshold=20, adx_mode="shrink")

    # 最近 20 天的趋势状态
    print("\n最近 20 个交易日趋势状态：")
    print(df[["close", "BIG_SCORE", "SMALL_SCORE", "ADX",
              "VOL_RATIO", "TREND_HEALTH", "BREAK_SCORE",
              "TOP_WARN", "BOTTOM_WARN"]].tail(20).round(2))

    # 最新一天的完整研判
    result = analyze(df)
