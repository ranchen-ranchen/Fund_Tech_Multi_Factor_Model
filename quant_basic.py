
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time

def subtract_months(dt, months):
    year = dt.year
    month = dt.month
    day = dt.day
    # Calculate target year and month
    target_month = month - months
    target_year = year
    while target_month <= 0:
        target_month += 12
        target_year -= 1
    # Get valid max day of target month
    max_day = calendar.monthrange(target_year, target_month)[1]
    valid_day = min(day, max_day)
    return datetime(target_year, target_month, valid_day).date()

def query_stock_code():
    lg = bs.login()
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
    rs = bs.query_stock_industry()
    # rs = bs.query_stock_basic(code_name="浦发银行")
    print('query_stock_industry error_code:'+rs.error_code)
    print('query_stock_industry respond  error_msg:'+rs.error_msg)
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    print(result)
    result.to_csv("data_stock_code.csv", index=False)  # no row index
    bs.logout()

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


def query_stock_price_bs(code):
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)
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
    frequency="d", adjustflag="1")
    print('query_history_k_data_plus respond error_code:'+rs.error_code)
    print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)
    data = []
    while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
        data.append(rs.get_row_data())
    result = pd.DataFrame(data, columns=rs.fields)
    pd.DataFrame(result).to_csv("data_{0:}_stock_price.csv".format(code), index=False)  # no row index
    #### 登出系统 ####
    bs.logout()
    print('{0:} stock price queried and stored'.format(code))
    time.sleep(3)

def calc_ref_rate_of_return(code):
    filename = "data_{0:}_stock_price.csv".format(code)
    df = pd.read_csv(filename, dtype=str)
    rate_of_return = round((100 * (float(df['close'].iloc[-1]) / float(df['close'].iloc[0]) - 1.0)), 2)
    print('From {0:} to {1:}, the ref rate of return for {2:} is {3:}%'.format(df['date'].iloc[0], df['date'].iloc[-1], code, rate_of_return))



def plot_heatmap(code, data, x_ticks, y_ticks):
    import seaborn as sns
    import matplotlib.pyplot as plt
    img_name = 'ror_sens_{0:}.png'.format(code)
    # 1. 构造数据
    df = pd.DataFrame(data, index=y_ticks, columns=x_ticks)

    # 2. 设置画布
    plt.figure(figsize=(10, 8))
    # 3. 绘制热力图
    sns.heatmap(
    df,
    annot=True,        # 显示单元格数值
    fmt=".2f",         # 数值保留2位小数
    cmap="YlOrRd",     # 配色方案
    linewidths=0.5,    # 格子边框宽度
    cbar=True          # 显示颜色条
    )
    # 4. 标题+显示
    plt.title(f"The sensitivity of RoR of {code:} to parameters", fontsize=14)
    plt.xlabel("buy parameter")
    plt.ylabel("sell parameter")
    #plt.tight_layout()
    plt.savefig(img_name, dpi=300)
    #plt.show()

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




def plot_k_line(df, mid, upper, lower, code):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    df["date"] = pd.to_datetime(df["date"])
    df["date_num"] = df["date"].apply(mdates.date2num)

    cols = ["open","high","low","close"]
    df[cols] = df[cols].astype(float)
    ohlc = df[["date_num","open","high","low","close"]].values
    fig, ax = plt.subplots(figsize=(16, 8))
    # 绘制K线
    candlestick_ohlc(ax, ohlc, width=0.6, colorup="#E74C3C", colordown="#3498DB")
    df["ma20"] = mid
    df["upper"] = upper
    df["lower"] = lower
    # 绘制布林带 + 均线
    sns.lineplot(data=df, x="date", y="ma20", color="#F39C12", linewidth=2, label="MA20", ax=ax)
    sns.lineplot(data=df, x="date", y="upper", color="#27AE60", linewidth=1.5, label="Upper", ax=ax)
    sns.lineplot(data=df, x="date", y="lower", color="#8E44AD", linewidth=1.5, label="Lower", ax=ax)
    # 美化坐标轴
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    plt.title("{0:} K-line & BOLL".format(code), fontsize=16)
    plt.ylabel("Stock Price")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    img_name = 'img_{0:}.png'.format(code)
    plt.savefig(img_name)

def deepseek_target_analysis(list_target, filename):
    import os
    from openai import OpenAI
    client = OpenAI(
        api_key='sk-184a605693524243b32eaa3cdd833733',
        base_url="https://api.deepseek.com")
    
    f = open(filename, 'w')
    for code in list_target:
        f.write('{0:}\n'.format(code))
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
            {"role": "system", "content": "你是一位A股市场股票投资专家"},
            {"role": "user", "content": "分析{0:}归属哪个赛道？当前赛道处于景气上行 / 短期题材 / 政策风口？".format(code)},
            {"role": "user", "content": "分析{0:}的主营是什么？是否贴合当下热点？有无独家优势？".format(code)},
            {"role": "user", "content": "分析{0:}的主要客户和盈利模式".format(code)},
            {"role": "user", "content": "分析{0:}过去三年的盈利情况".format(code)}
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )
        time.sleep(3)
        f.write('{0:}\n'.format(response.choices[0].message.content))
        f.write('\n')
    f.close()



class BackTest:
    def __init__(self, fee=0.0003):
        self.cash = 0.0
        self.hold_num = 0          # 持仓股数
        self.fee = fee             # 交易手续费
        self.trade_log = []        # 交易记录

    # 买入函数
    def buy(self, price, date):
        max_num = 100
        cost = max_num * price * (1 + self.fee)
        self.cash -= cost
        self.hold_num += max_num
        self.trade_log.append({"date":date, "operate":"buy", "price":price, "num":max_num})

    # 卖出函数
    def sell(self, price, date):
        if self.hold_num == 0:
            return
        income = self.hold_num * price * (1 - self.fee)
        self.cash += income
        self.trade_log.append({"date":date, "operate":"sell", "price":price, "num":self.hold_num})
        self.hold_num = 0


    # 执行策略回测
    def run_strategy_5d_breakout(self, code):
        start_i = 5
        while True:
            result_buy = strategy_5d_breakout_to_buy(code, start_i)
            if result_buy == None:
                # print('No buy opportunity is found for {0:}'.format(code))
                break
            else:
                price = result_buy[2]
                date = result_buy[1]
                self.buy(price, date)
                start_i = result_buy[0]
                result_sell = strategy_5d_breakout_to_sell(code, start_i)
                if result_sell == None:
                    # print('No sell opportunity is found for {0:}'.format(code))
                    break
                else:
                    price = result_sell[2]
                    date = result_sell[1]
                    self.sell(price, date)
                    start_i = result_sell[0]
        # print('Trading log of {0:}: '.format(code))
        # for i in range(len(self.trade_log)):
        #     print('{0:}  {1:>4}  {2:>.2f}  {3:>d}'.format(self.trade_log[i]['date'], 
        #             self.trade_log[i]['operate'], self.trade_log[i]['price'], self.trade_log[i]['num']))
        

    def run_strategy_kdj(self, code):
        start_i = 60
        while True:
            result_buy = strategy_kdj_to_buy(code, start_i)
            if result_buy == None:
                # print('No buy opportunity is found for {0:}'.format(code))
                break
            else:
                price = result_buy[2]
                date = result_buy[1]
                self.buy(price, date)
                start_i = result_buy[0]
                result_sell = strategy_kdj_to_sell(code, start_i)
                if result_sell == None:
                    # print('No sell opportunity is found for {0:}'.format(code))
                    break
                else:
                    price = result_sell[2]
                    date = result_sell[1]
                    self.sell(price, date)
                    start_i = result_sell[0]
        
        # 如果策略最后一笔操作是买入，按最新收盘价卖出，结算最终现金数目
        if self.trade_log[-1]['operate'] == 'buy':
            filename = "data_{0:}_stock_price.csv".format(code)
            df = pd.read_csv(filename, dtype=str)
            price = float(df['close'].iloc[-1])
            date = df['date'].iloc[-1]
            self.sell(price, date)
        # 比较最终现金数目与第一笔买入花费现金数目，从而计算收益率
        self.ror = round((self.cash / (self.trade_log[0]['price'] * self.trade_log[0]['num'])*100), 2)
        
        print(f'Rate of return: {self.ror:>.2f}%')


        

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
    list_code = [
    'sh.600941']
    pass
