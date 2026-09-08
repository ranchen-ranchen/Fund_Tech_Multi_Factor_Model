
import pandas as pd
import numpy as np
import akshare as ak
import baostock as bs
from datetime import datetime, timedelta
import calendar
import time
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__) 



def vector_backtest(close_series: pd.Series, position_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    price_change_series = close_series.pct_change()
    return_series = position_series.shift(1) * price_change_series 
    for i in range(len(close_series)):
        logging.info(f'{price_change_series.iloc[i]} | {position_series.iloc[i]} | {return_series.iloc[i]}')
    
    
    cum_price_change_series = (1 + price_change_series).cumprod()
    cum_return_series = (1 + return_series).cumprod()
    return cum_price_change_series, cum_return_series

    # # 总收益率
    # total_return = cum_return_series.iloc[-1] - 1
    # # 日收益率序列
    # daily_ret = return_series.dropna()
    # # 夏普比率（无风险利率为0,全年252个交易日）
    # sharpe_ratio = np.sqrt(252) * daily_ret.mean() / daily_ret.std()
    # # 最大回撤（向量化）
    # cumulative = (1 + daily_ret).cumprod()
    # running_max = cumulative.expanding().max()
    # drawdown = (cumulative - running_max) / running_max
    # max_drawdown = drawdown.min()
    # # 胜率
    # win_rate = (daily_ret > 0).mean()
    # # 输出结果
    # logger.info("========== 回测绩效 ==========")
    # logger.info(f"总收益率:        {total_return:.2%}")
    # logger.info(f"夏普比率:        {sharpe_ratio:.2f}")
    # logger.info(f"最大回撤:        {max_drawdown:.2%}")
    # logger.info(f"胜率:            {win_rate:.2%}")
    # logger.info("===============================")
    

# # ------------------------------
# # 7. 可视化（向量化绘图）
# # ------------------------------
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# # 价格与均线
# ax1.plot(df.index, df['Close'], label='Close', alpha=0.6)
# ax1.plot(df.index, df['SMA_short'], label=f'SMA {window_short}', linestyle='--')
# ax1.plot(df.index, df['SMA_long'], label=f'SMA {window_long}', linestyle='--')
# ax1.set_title(f'{ticker} 价格与均线')
# ax1.legend()
# ax1.grid(True)

# # 累计收益曲线
# ax2.plot(df.index, df['Cum_Return'], label='Buy & Hold', alpha=0.6)
# ax2.plot(df.index, df['Cum_Strategy'], label='Strategy', linewidth=2)
# ax2.set_title('累计收益对比')
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
# plt.show()







if __name__ == "__main__":
    pass
