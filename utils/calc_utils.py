
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time




def linear_reg_for_price_trend(high_series: pd.Series, low_series: pd.Series) -> float:
    x1 = np.arange(len(high_series))
    x2 = np.arange(len(low_series))
    k1 = np.polyfit(x1, high_series, 1)[0]
    k2 = np.polyfit(x2, low_series, 1)[0]
    return k1, k2







def calc_bollinger_bands(close_series, n=20, k=2):
    # 中轨
    mid = close_series.rolling(window=n).mean()
    # 标准差
    std = close_series.rolling(window=n).std()
    # 上下轨
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower

def calc_bbi(close_series):
    ma2  = close_series.rolling(window=2).mean()
    ma5  = close_series.rolling(window=5).mean()
    ma10 = close_series.rolling(window=10).mean()
    ma20 = close_series.rolling(window=20).mean()
    bbi  = (ma2 + ma5 + ma10 + ma20) / 4.0
    return bbi

def calc_kdj(high_series, low_series, close_series, n=9, m1=3, m2=3):
    # 1. RSV
    llv = low_series.rolling(n).min()
    hhv = high_series.rolling(n).max()
    rsv = (close_series - llv) / (hhv - llv) * 100
    rsv = rsv.fillna(50)  # 同花顺前值填50

    # 2. K = 2/3*前K + 1/3*RSV
    k = np.zeros_like(rsv)
    k[0] = 50
    for i in range(1, len(rsv)):
        k[i] = 2/3 * k[i-1] + 1/3 * rsv[i]

    # 3. D = 2/3*前D + 1/3*K
    d = np.zeros_like(k)
    d[0] = 50
    for i in range(1, len(k)):
        d[i] = 2/3 * d[i-1] + 1/3 * k[i]

    # 4. J
    j = 3*k - 2*d
    return k, d, j

def strategy_kdj_to_buy(code, start_i):
    filename = "data_{0:}_stock_price.csv".format(code)
    df = pd.read_csv(filename, dtype=str)
    close_series = df['close'].astype(float)
    turn_series = df['turn'].astype(float)
    high_series = df['high'].astype(float)
    low_series = df['low'].astype(float)
    ma60 = close_series.rolling(window=60).mean()
    k, d, j = calc_kdj(high_series, low_series, close_series)
    for i in range(start_i, len(close_series)):
        # 换手率大于0 / 当天没有停牌
        if turn_series.iloc[i] > 0.0:
            # 收盘价在MA60之上且KDJ-J值小于-3.0
            if close_series.iloc[i] > ma60[i] and j[i] < -3.0:
                # 换手率小于前五天平均值 / 成交量缩量
                if turn_series[i] < turn_series[i-5:i].mean():
                    # 当天未触及涨停
                    if code.startswith(('sz.301', 'sz.300', 'sh.688')) and round((close_series.iloc[i] / close_series.iloc[i-1]), 2) < 1.2: 
                        # 符合买入条件，返回列表序号，日期，买入价格 / 当天收盘价
                        return i, df['date'][i], close_series.iloc[i]
                    # 当天未触及涨停
                    elif round((close_series.iloc[i] / close_series.iloc[i-1]), 2) < 1.1:
                        # 符合买入条件，返回列表序号，日期，买入价格 / 当天收盘价
                        return i, df['date'][i], close_series.iloc[i]

def strategy_kdj_to_sell(code, start_i):
    filename = "data_{0:}_stock_price.csv".format(code)
    df = pd.read_csv(filename, dtype=str)
    close_series = df['close'].astype(float)
    high_series = df['high'].astype(float)
    low_series = df['low'].astype(float)
    k, d, j = calc_kdj(high_series, low_series, close_series)
    ma60 = close_series.rolling(window=60).mean()
    for i in range(start_i+1, len(df['date'])):
        # 收盘价大于MA60
        if close_series.iloc[i] > ma60[i]:
            # KDJ-J值大于100后，J值跌破100后卖出
            if j[i-2] > 100 and j[i-1] > 100 and j[i] < 100:
                # 当天未触及跌停
                if code.startswith(('sz.301', 'sz.300', 'sh.688')) and round((close_series.iloc[i] / close_series.iloc[i-1]), 2) > 0.8:
                    # 符合卖出条件，返回列表序号，日期，卖出价格 / 当天收盘价
                    return i, df['date'][i], close_series.iloc[i]
                # 当天未触及跌停
                elif round((close_series.iloc[i] / close_series.iloc[i-1]), 2) > 0.9:
                    # 符合卖出条件，返回列表序号，日期，卖出价格 / 当天收盘价
                    return i, df['date'][i], close_series.iloc[i]


def strategy_5d_breakout_to_buy(code, start_i): # 输入分析股票代码和起始列表序号 / 起始日期
    filename = "data_{0:}_stock_price.csv".format(code)
    df = pd.read_csv(filename, dtype=str)
    close_series = df['close'].astype(float)
    turn_series = df['turn'].astype(float)
    for i in range(start_i, len(df['date'])):
        # 换手率大于0 / 当天没有停牌
        if turn_series.iloc[i] > 0.0:
            # 当天收盘价大于前五天收盘价最高值
            if close_series.iloc[i] > close_series[i-5:i].max() * 1.05:
                # 突破5日新高时，当日成交量大于5日平均值
                if turn_series.iloc[i] > turn_series[i-5:i].mean():
                    # 当天未触及涨停
                    if code.startswith(('sz.301', 'sz.300', 'sh.688')) and round((close_series.iloc[i] / close_series.iloc[i-1]), 2) < 1.2: 
                        # 符合买入条件，返回列表序号，日期，买入价格 / 当天收盘价
                        return i, df['date'][i], close_series.iloc[i]
                    # 当天未触及涨停
                    elif round((close_series.iloc[i] / close_series.iloc[i-1]), 2) < 1.1:
                        # 符合买入条件，返回列表序号，日期，买入价格 / 当天收盘价
                        return i, df['date'][i], close_series.iloc[i]

def strategy_5d_breakout_to_sell(code, start_i):
    filename = "data_{0:}_stock_price.csv".format(code)
    df = pd.read_csv(filename, dtype=str)
    close_series = df['close'].astype(float)
    max_price = close_series.iloc[start_i]
    for i in range(start_i+1, len(df['date'])):
        if close_series.iloc[i] > max_price:
            max_price = close_series.iloc[i]
        else:
            # 当天收盘价从买入后最高点下跌超过阈值
            if close_series.iloc[i] < max_price * 0.9:
                # 当天未触及跌停
                if code.startswith(('sz.301', 'sz.300', 'sh.688')) and round((close_series.iloc[i] / close_series.iloc[i-1]), 2) > 0.8:
                    # 符合卖出条件，返回列表序号，日期，卖出价格 / 当天收盘价
                    return i, df['date'][i], close_series.iloc[i]
                # 当天未触及跌停
                elif round((close_series.iloc[i] / close_series.iloc[i-1]), 2) > 0.9:
                    # 符合卖出条件，返回列表序号，日期，卖出价格 / 当天收盘价
                    return i, df['date'][i], close_series.iloc[i]
    


if __name__ == "__main__":
    pass
