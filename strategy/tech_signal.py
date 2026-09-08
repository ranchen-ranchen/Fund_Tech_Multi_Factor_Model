
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import time

import logging
logger = logging.getLogger(__name__) 





import yaml
from pathlib import Path
config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
time_window = config['TIME_WINDOW']
price_level_threshold = config['PRICE_LEVEL_THRESHOLD']
price_trend_threshold = config['PRICE_TREND_THRESHOLD']
trading_amount_threshold = config['TRADING_AMOUNT_THRESHOLD']
price_trend_window = config['PRICE_TREND_WINDOW']


def linear_reg_for_price_trend(high_series: pd.Series, low_series: pd.Series) -> float:
    x1 = np.arange(len(high_series))
    x2 = np.arange(len(low_series))
    k1 = np.polyfit(x1, high_series, 1)[0]
    k2 = np.polyfit(x2, low_series, 1)[0]
    return k1, k2

def diff_price_trend(high_series: pd.Series, low_series: pd.Series) -> str:
    k1, k2 = linear_reg_for_price_trend(high_series[-price_trend_window:], low_series[-price_trend_window:])
    logger.info(f'price trend slope for high {k1: .1f} and low {k2: .1f}')
    if k1 > price_trend_threshold and k2 > price_trend_threshold:
        return 'up'
    elif abs(k1) < price_trend_threshold and k2 > price_trend_threshold:
        return 'up'
    elif abs(k1) < price_trend_threshold and k2 < -1 * price_trend_threshold:
        return 'down'
    elif k1 < -1 * price_trend_threshold and k2 < -1 * price_trend_threshold:
        return 'down'
    else:
        return 'sideways'


def diff_price_level(close_series: pd.Series, high_series: pd.Series, low_series: pd.Series) -> str:
    price_level_high = 0.5*(1 + price_level_threshold) * max(high_series) + 0.5*(1 - price_level_threshold) * min(low_series)
    price_level_low = 0.5*(1 - price_level_threshold) * max(high_series) + 0.5*(1 + price_level_threshold) * min(low_series)
    logger.info(f'price_level_high: {price_level_high: .1f}, price_level_low: {price_level_low: .1f}, price: {close_series.iloc[-1]: .1f}')
    if close_series.iloc[-1] > price_level_high:
        return 'high'
    elif close_series.iloc[-1] < price_level_low:
        return 'low'
    else:
        return 'mid'

def diff_trading_amount(amount_series: pd.Series) -> str:
    logger.info(f'trading amount average: {amount_series.mean(): .1f}, trading amount: {amount_series.iloc[-1]: .1f}')
    if amount_series.iloc[-1] > (1 + trading_amount_threshold) * amount_series.mean():
        return 'high'
    elif amount_series.iloc[-1] < (1 - trading_amount_threshold) * amount_series.mean():
        return 'low'
    else:
        return 'mid'


handlers = {
    ('up', 'high', 'high'): 'sell',
    ('up', 'high', 'mid'): 'sell',
    ('up', 'high', 'low'): 'sell+',
    ('up', 'mid', 'high'): 'buy',
    ('up', 'mid', 'mid'): 'hold',
    ('up', 'mid', 'low'): 'hold',
    ('up', 'low', 'high'): 'buy+',
    ('up', 'low', 'mid'): 'buy',
    ('up', 'low', 'low'): 'hold',
    ('sideways', 'high', 'high'): 'sell',
    ('sideways', 'high', 'mid'): 'sell',
    ('sideways', 'high', 'low'): 'sell+',
    ('sideways', 'mid', 'high'): 'hold',
    ('sideways', 'mid', 'mid'): 'hold',
    ('sideways', 'mid', 'low'): 'hold',
    ('sideways', 'low', 'high'): 'buy',
    ('sideways', 'low', 'mid'): 'hold',
    ('sideways', 'low', 'low'): 'hold',
    ('down', 'high', 'high'): 'sell+',
    ('down', 'high', 'mid'): 'sell+',
    ('down', 'high', 'low'): 'sell+',
    ('down', 'mid', 'high'): 'sell',
    ('down', 'mid', 'mid'): 'sell',
    ('down', 'mid', 'low'): 'hold',
    ('down', 'low', 'high'): 'hold',
    ('down', 'low', 'mid'): 'hold',
    ('down', 'low', 'low'): 'hold'
}


signal_handlers =  {
    'sell' : -5,
    'sell+' : -10,
    'buy' : 1,
    'buy+' : 3,
    'hold' : 0
}


def slice_from_series(date: str, date_series: pd.Series, close_series: pd.Series, high_series: pd.Series, low_series: pd.Series, amount_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if (date_series == date).any():
        end = date_series[date_series == date].index[0]
        start = max(0, end - time_window)
        close = close_series.iloc[start:end] # take the data before today / today not included
        high = high_series.iloc[start:end]
        low = low_series.iloc[start:end]
        amount = amount_series.iloc[start:end]
        return close, high, low, amount
    else:
        raise ValueError("requested date not found in the csv file")


def change_position_by_tech_analysis(date_series: pd.Series, close_series: pd.Series, high_series: pd.Series, low_series: pd.Series, amount_series: pd.Series) -> pd.Series:
    logger.info(f'==== Technical analysis starts ====')
    N = len(date_series)
    position_series = pd.Series(0, index=range(N))
    for i in range(time_window, N):
        date = date_series.iloc[i]
        logger.info(f'Date: {date} | ')
        close, high, low, amount = slice_from_series(date, date_series, close_series, high_series, low_series, amount_series)
        price_level = diff_price_level(close, high, low)
        price_trend = diff_price_trend(high, low)
        trading_amount = diff_trading_amount(amount)
        situation = (price_trend, price_level, trading_amount)
        signal = handlers.get(situation, 'hold')
        logger.info(f'Date: {date} | Technical analysis situation and signal: {situation} {signal}')
        position_series.iloc[i] = max(0, min((position_series.iloc[i-1] + signal_handlers.get(signal, 0)), 10))
        logger.info(f'Date: {date} | old position: {position_series.iloc[i-1]} | new position: {position_series.iloc[i]}')
        logger.info(f'========')


    return position_series / 10.0

if __name__ == "__main__":
    pass
