import pandas as pd
import numpy as np
from quant_basic import query_stock_price, calc_ref_rate_of_return, BackTest
import time


list_code = [
'sh.688783',
'sh.688256',
'sh.688589',
'sh.605255',
'sz.300308',
'sz.300502',
'sz.301377',
'sz.300641',
'sz.300561',
'sz.300476',
'sh.688525']


# for code in list_code:
#     print(code)
#     query_stock_price(code)
#     time.sleep(1)
#     calc_ref_rate_of_return(code)
#     test = BackTest()
#     test.run_strategy_kdj(code)
#     # print(test.trade_log)

code = 'sz.300561'
filename = "data_{0:}_stock_price.csv".format(code)
df = pd.read_csv(filename, dtype=str)
close_series = df['close'].astype(float)
turn_series = df['turn'].astype(float)

for i in range(len(turn_series)):
    if turn_series[i] > 0.0:
        pass
    else:
        print(turn_series[i], type(turn_series[i]))


        


# # test the function calc_kdj
# code = 'sh.688525'
# filename = "data_{0:}_stock_price.csv".format(code)
# df = pd.read_csv(filename, dtype=str)
# close_series = df['close'].astype(float)
# turn_series = df['turn'].astype(float)
# high_series = df['high'].astype(float)
# low_series = df['low'].astype(float)
# from quant_basic import calc_kdj
# k, d, j = calc_kdj(high_series, low_series, close_series)
# for i in range(len(j)):
#     print('{0:}    {1:}'.format(df['date'][i], j[i]))




# code = list_code[2]
# calc_ref_rate_of_return(code)
# test = BackTest()
# ror = test.run_strategy(code, buy_para=1.05, sell_para=0.90)
