import pandas as pd
import baostock as bs
import time
from datetime import datetime

# ==================== 1. 读取CSV文件，提取唯一股票代码 ====================
csv_file = 'hs300_constituents_2021_2026.csv'
# 使用 utf-8-sig 编码以处理可能的 BOM 字符
df_constituents = pd.read_csv(csv_file, encoding='utf-8-sig')
# 提取 code 列的唯一值，并转换为 Python 列表
codes = df_constituents['code'].unique().tolist()
print(f'共提取到 {len(codes)} 个唯一股票代码')

# ==================== 2. 登录 baostock ====================
lg = bs.login()
if lg.error_code != '0':
    print('登录失败:', lg.error_msg)
    exit()

# ==================== 3. 生成目标日期列表（每6个月） ====================
# 从 2021-01-01 到 2026-01-01，频率为每6个月的月初（即 1月1日 和 7月1日）
# target_dates = pd.date_range(start='2021-01-01', end='2026-01-01', freq='6MS').strftime('%Y-%m-%d').tolist()
target_dates = ['2021-01-01', '2021-06-30', '2021-12-31', 
        '2022-06-30', '2022-12-31', 
        '2023-06-30', '2023-12-31', 
        '2024-06-30', '2024-12-31',
        '2025-06-30', '2025-12-31',]
print('目标日期:', target_dates)

# ==================== 4. 循环查询每个股票的数据 ====================
all_data = []
for i, code in enumerate(codes):
    print(f'正在处理 {i+1}/{len(codes)}: {code}')

    # 查询日线数据，字段包括日期、代码、PE(TTM)、PB(MRQ)
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,peTTM,pbMRQ",
        start_date='2021-01-01',
        end_date='2026-01-01',
        frequency="d",
        adjustflag="3"          
# 3 表示不复权
# 不复权价格是市场真实的成交价，与财报中披露的当前每股收益（EPS）、每股净资产（BPS）在时间上是对应的。
# 只有这样计算出的PE/PB，才能真实反映“以当前市价买入，对应多少盈利或净资产”
    )

    # 查询失败处理
    if rs.error_code != '0':
        print(f'查询 {code} 失败: {rs.error_msg}')
        time.sleep(1)           # 失败时多等一会儿
        continue

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    if not data_list:
        print(f'{code} 无数据')
        time.sleep(0.5)
        continue

    df_stock = pd.DataFrame(data_list, columns=rs.fields)
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock = df_stock.sort_values('date')

    # 对于每个目标日期，找到该日期或之前最近一个交易日的数据
    for target in target_dates:
        target_dt = pd.to_datetime(target)
        subset = df_stock[df_stock['date'] <= target_dt]
        if not subset.empty:
            row = subset.iloc[-1]   # 最近一行
            all_data.append({
                'code': code,
                'target_date': target,
                'actual_date': row['date'].strftime('%Y-%m-%d'),
                'peTTM': row['peTTM'],
                'pbMRQ': row['pbMRQ']
            })

    # 每处理完一只股票，暂停 0.5 秒，避免请求过于频繁
    time.sleep(0.5)

# ==================== 5. 登出 baostock ====================
bs.logout()

# ==================== 6. 保存结果 ====================
result_df = pd.DataFrame(all_data)
output_file = 'hs300_valuation_6m.csv'
result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f'处理完成，结果已保存到 {output_file}')
print(result_df.head())


