

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
def compute_trend(df):
    df = df.copy()
    df = add_ma(df)
    df = add_macd(df)
    df = add_adx(df)

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
    if adx >= 25:
        adx_txt = f"{adx:.1f}（趋势强劲）"
    elif adx >= 20:
        adx_txt = f"{adx:.1f}（趋势初现）"
    else:
        adx_txt = f"{adx:.1f}（无趋势/震荡）"

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
    print("-" * 62)
    print(f"【趋势强度】ADX = {adx_txt}")
    print(f"【综合结论】{icon} {desc}")
    print(f"【操作建议】{action}")
    print("=" * 62)

    return {
        "big_trend": big_state, "big_score": int(row["BIG_SCORE"]),
        "small_trend": small_state, "small_score": int(row["SMALL_SCORE"]),
        "adx": round(float(adx), 2), "signal": desc, "action": action,
    }


# ============================================================
# 5. 运行
# ============================================================
if __name__ == "__main__":
    df = get_stock_data('/home/omen/work/quant/data/data_sh.688783_stock_price.csv')
    df = compute_trend(df)

    # 最近 20 天的趋势状态
    print("\n最近 20 个交易日趋势状态：")
    print(df[["close", "BIG_SCORE", "SMALL_SCORE", "ADX"]].tail(20).round(2))

    # 最新一天的完整研判
    result = analyze(df)
