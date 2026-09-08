
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time
import logging

logging.basicConfig(
    filename='save_update_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def query_stock_price(code):
    filename = "data_{0:}_stock_price.csv".format(code)
    import os
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            if df.empty:
                print("无数据行")
                query_stock_price_bs(code)
        except pd.errors.EmptyDataError:
            print("文件彻底为空，连表头都没有")
            query_stock_price_bs(code)
    else:
        query_stock_price_bs(code)




if __name__ == "__main__":
    pass
