
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time
import logging

logging.basicConfig(
    filename='query_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def query_stock_code_bs(date: str) -> pd.DataFrame:
    lg = bs.login()
    logging.info('login respond error_code:'+lg.error_code)
    logging.info('login respond  error_msg:'+lg.error_msg)
    rs = bs.query_stock_industry(date=date)
    # rs = bs.query_stock_basic(code_name="浦发银行")
    logging.info('query_stock_industry error_code:'+rs.error_code)
    logging.info('query_stock_industry respond  error_msg:'+rs.error_msg)
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    bs.logout()
    logging.info(f'{date} stock code queried')
    logging.info('logout')
    return result


def query_stock_revenue_ak(date: str):
    df = ak.stock_yjbb_em(date=date)
    list_code = []
    list_name = []
    list_revenue = []
    list_revenue_YoY = []
    list_revenue_QoQ = []
    list_gross_margin = []
    for i in range(len(df['股票代码'])):
        if df['营业总收入-同比增长'][i] > 10.0: # 营收同比增长超过10%
            if df['销售毛利率'][i] > 0.0: # 营收毛利率为正
                if 'ST' in df['股票简称'][i]: # 排除ST股
                    pass
                else:
                    list_code.append(df['股票代码'][i])
                    list_name.append(df['股票简称'][i])
                    list_revenue.append(df['营业总收入-营业总收入'][i])
                    list_revenue_YoY.append(df['营业总收入-同比增长'][i])
                    list_revenue_QoQ.append(df['营业总收入-季度环比增长'][i])
                    list_gross_margin.append(df['销售毛利率'][i])

    data = {
        'code' : list_code,
        'name' : list_code,
        'revenue' : list_revenue,
        'revenue_YoY' : list_revenue_YoY,
        'revenue_QoQ' : list_revenue_QoQ,
        'gross_margin' : list_gross_margin
    }
    pd.DataFrame(data).to_csv("data_{0:}_revenue_YoY_QoQ.csv".format(date), index=False)  # no row index




def query_stock_price_bs(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    logging.info('login respond error_code:'+lg.error_code)
    logging.info('login respond  error_msg:'+lg.error_msg)
    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    # start：开始日期（包含），格式“YYYY-MM-DD”，为空时取2015-01-01；
    # end：结束日期（包含），格式“YYYY-MM-DD”，为空时取最近一个交易日；
    # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权
    # 股票停牌时，对于日线，开、高、低、收价都相同，且都为前一交易日的收盘价，成交量、成交额为0，换手率为空。
    rs = bs.query_history_k_data_plus(code,
    "date,code,open,high,low,close,volume,amount,turn,adjustflag",
    start_date=start_date, end_date=end_date,
    frequency="d", adjustflag="1")
    logging.info('query_history_k_data_plus respond error_code:'+rs.error_code)
    logging.info('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    data = []
    while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
        data.append(rs.get_row_data())
    result = pd.DataFrame(data, columns=rs.fields)
    #### 登出系统 ####
    bs.logout()
    logging.info('{0:} stock price from {1:} {2:} queried'.format(code, start_date, end_date))
    logging.info('logout')
    return result



if __name__ == "__main__":
    pass
