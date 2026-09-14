import code
import re
import pandas as pd


def extract_score(text: str):
    """
    从字符串中提取由 '==' 分隔符标识的数字。
    参数:
        text (str): 包含数字的原始字符串，数字格式如 '==123==' 或 '==-3.14=='
    返回:
        list: 提取出的数字列表，元素类型为 int 或 float。
              如果没有匹配项，返回空列表。
    """
    # 匹配模式：== 后面跟着可选负号、数字、可选小数部分，再跟 ==
    pattern = r'==(-?\d+(?:\.\d+)?)=='
    matches = re.findall(pattern, text)
    result = []
    for match in matches:
        # 根据是否包含小数点决定转换为 int 或 float
        if '.' in match:
            result.append(float(match))
        else:
            result.append(int(match))
    return result


def add_days_to_date(date_str: str, days: int) -> str:
    from datetime import datetime, timedelta
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    new_dt = dt + timedelta(days=days)
    return new_dt.strftime('%Y-%m-%d')



def check_date_out_bound(date_bound: str, date: str) -> bool:
    from datetime import datetime
    d_bound = datetime.strptime(date_bound, "%Y-%m-%d")
    d_date = datetime.strptime(date, "%Y-%m-%d")
    return d_date < d_bound



def read_from_csv(filename:str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(filename, dtype=str)
    df_cleaned = df.dropna().reset_index(drop=True) # remove the days with no trading
    date_series = df_cleaned['date']
    open_series = df_cleaned['open'].astype(float)
    close_series = df_cleaned['close'].astype(float)
    high_series = df_cleaned['high'].astype(float)
    low_series = df_cleaned['low'].astype(float)
    amount_series = df_cleaned['amount'].astype(float)
    return date_series, open_series, close_series, high_series, low_series, amount_series






# def read_from_csv(filename:str, date:str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
#     df = pd.read_csv(filename, dtype=str)
#     date_bound = df['date'][0]
#     check_date = (df['date'] == date).any()
#     while check_date == False:
#         date = add_days_to_date(date, -1)
#         if check_date_out_bound(date_bound, date):
#             raise ValueError("date is out of the boundary!!!") 
#         else:
#             check_date = (df['date'] == date).any()

#     end = df[df['date'] == date].index[0]
#     start = max(0, end - time_window)
#     open_series = df['open'].astype(float).iloc[start:end+1]
#     close_series = df['close'].astype(float).iloc[start:end+1]
#     high_series = df['high'].astype(float).iloc[start:end+1]
#     low_series = df['low'].astype(float).iloc[start:end+1]
#     amount_series = df['amount'].astype(float).iloc[start:end+1]
#     return open_series, close_series, high_series, low_series, amount_series

#     # turn_series = df['turn'].fillna(0).astype(float)

