
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
trading_amount_window = config['TRADING_AMOUNT_WINDOW']
price_trend_window = config['PRICE_TREND_WINDOW']
trading_range_threshold = config['TRADING_RANGE_THRESHOLD']



def log_linear_reg(price_series: pd.Series) -> float:
    x = np.arange(len(price_series))
    k = np.polyfit(x, np.log(price_series), 1)[0]
    return k

def tech_analysis_router(close_series: pd.Series) -> str:
    if (close_series.iloc[int(-1*time_window/2):].max() - close_series.iloc[int(-1*time_window/2):].min()) / close_series.iloc[int(-1*time_window/2):].mean() < trading_range_threshold:
        return 'narrow range oscillation'
    elif close_series.iloc[-1] > close_series.iloc[:-1].max():
        return 'above the previous high'
    elif close_series.iloc[-1] < close_series.iloc[:-1].min():
        return 'below the prevvious low'
    else:
        return 'nothing special'


def diff_price_trend(high_series: pd.Series, low_series: pd.Series) -> str:
    k1 = log_linear_reg(high_series[-price_trend_window:])
    k2 = log_linear_reg(low_series[-price_trend_window:])
    logger.info(f'price trend slope for high {k1: .4f} and low {k2: .4f}')
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
    logger.info(f'trading amount average: {amount_series.iloc[-trading_amount_window:].mean(): .1f}, trading amount: {amount_series.iloc[-1]: .1f}')
    if amount_series.iloc[-1] > 1.5 * amount_series.iloc[-trading_amount_window:].mean():
        return 'high'
    elif amount_series.iloc[-1] < 0.5 * amount_series.iloc[-trading_amount_window:].mean():
        return 'low'
    else:
        return 'mid'


nothing_special_handlers = {
# (price_trend, price_level, trading_amount)
    ('up', 'high', 'high'): 'hold',
    ('up', 'high', 'mid'): 'hold',
    ('up', 'high', 'low'): 'hold',
    ('up', 'mid', 'high'): 'hold',
    ('up', 'mid', 'mid'): 'hold',
    ('up', 'mid', 'low'): 'hold',
    ('up', 'low', 'high'): 'buy+',
    ('up', 'low', 'mid'): 'buy',
    ('up', 'low', 'low'): 'hold',
    ('sideways', 'high', 'high'): 'hold',
    ('sideways', 'high', 'mid'): 'hold',
    ('sideways', 'high', 'low'): 'hold',
    ('sideways', 'mid', 'high'): 'hold',
    ('sideways', 'mid', 'mid'): 'hold',
    ('sideways', 'mid', 'low'): 'hold',
    ('sideways', 'low', 'high'): 'buy',
    ('sideways', 'low', 'mid'): 'hold',
    ('sideways', 'low', 'low'): 'hold',
    ('down', 'high', 'high'): 'sell',
    ('down', 'high', 'mid'): 'sell',
    ('down', 'high', 'low'): 'sell',
    ('down', 'mid', 'high'): 'sell',
    ('down', 'mid', 'mid'): 'sell',
    ('down', 'mid', 'low'): 'hold',
    ('down', 'low', 'high'): 'hold',
    ('down', 'low', 'mid'): 'hold',
    ('down', 'low', 'low'): 'buy'
}


above_high_handlers = {
# (price_trend, trading_amount)
('up', 'high') : 'buy',
('up', 'mid') : 'buy',
('up', 'low') : 'hold',
('sideways', 'high') : 'sell',
('sideways', 'mid') : 'sell',
('sideways', 'low') : 'hold',
('down', 'high') : 'sell+',
('down', 'mid') : 'sell',
('down', 'low') : 'hold'
}


below_low_handlers = {
#  (price_trend, trading_amount)
('up', 'high') : 'sell',
('up', 'mid') : 'sell',
('up', 'low') : 'sell',
('sideways', 'high') : 'sell+',
('sideways', 'mid') : 'sell+',
('sideways', 'low') : 'sell+',
('down', 'high') : 'sell++',
('down', 'mid') : 'sell++',
('down', 'low') : 'sell++'
}


narrow_range_handlers = {
# (price_level, trading_amount)
('high', 'high') : 'sell+',
('high', 'mid') : 'sell',
('high', 'low') : 'sell',
('mid', 'high') : 'sell',
('mid', 'mid') : 'hold',
('mid', 'low') : 'hold',
('low', 'high') : 'hold',
('low', 'mid') : 'hold',
('low', 'low') : 'hold'
}

signal_handlers =  {
    'sell' : -1,
    'sell+' : -2,
    'sell++' : -3,
    'buy' : 1,
    'buy+' : 2,
    'buy++' : 3,
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
        if tech_analysis_router(close) == 'narrow range oscillation':
            router = 'narrow range oscillation'
            situation = (price_level, trading_amount)
            signal = narrow_range_handlers.get(situation, 'hold')
        elif tech_analysis_router(close) == 'above the previous high':
            router = 'above the previous high'
            situation = (price_trend, trading_amount)
            signal = above_high_handlers.get(situation, 'hold')
        elif tech_analysis_router(close) == 'below the previous low':
            router = 'below the previous low'
            situation = (price_trend, trading_amount)
            signal = below_low_handlers.get(situation, 'hold')
        elif tech_analysis_router(close) == 'nothing special':
            router = 'nothing special'
            situation = (price_trend, price_level, trading_amount)
            signal = nothing_special_handlers.get(situation, 'hold')

        logger.info(f'Date: {date} | Technical analysis: {router} | {situation} | {signal}')
        position_series.iloc[i] = max(0, min((position_series.iloc[i-1] + signal_handlers.get(signal, 0)), 10))
        logger.info(f'Date: {date} | old position: {position_series.iloc[i-1]} / 10 | new position: {position_series.iloc[i]} / 10')
        logger.info(f'========')


    return position_series / 10.0

if __name__ == "__main__":
    pass
