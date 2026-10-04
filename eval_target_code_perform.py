import pandas as pd

def read_target_code_csv(file_path: str) -> pd.DataFrame:
    """
    读取 target_code.csv 文件，并返回 pandas DataFrame。
    其中 target_code 列会从空格分隔的字符串拆分为 Python list。
    """
    df = pd.read_csv(
        file_path,
        dtype={"target_code": str, "query_date": str},
    )

    df["target_code"] = (
        df["target_code"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split()
    )

    return df

df = read_target_code_csv("target_code.csv")

#### download daily data for all target codes
# merged_codes = (
#     df['target_code']
#     .explode()              # 把每行的列表展开成多行
#     .dropna()               # 如果有空值/空列表产生的 NaN，去掉
#     .drop_duplicates()      # 去重，保留第一次出现的顺序
#     .tolist()
# )
# from data.fetch_daily_k import get_daily_data
# get_daily_data(merged_codes)
print(df.shape[0])

from backtest_buy_hold import run_backtest
from utils.text_utils import normalize_code
from pprint import pprint
def prep_periods(df: pd.DataFrame):
    periods = []
    for i in range(df.shape[0]-1):
        start_date = df.iloc[i]['query_date']
        end_date = df.iloc[i+1]['query_date']
        code_list = df.iloc[i]['target_code']
        code_list = [normalize_code(code) for code in code_list]
        file_list = [f"stock_daily_k/individual/{code.replace('.', '_')}.csv" for code in code_list]
        tmp_dict = {}
        tmp_dict['name'] = f"period_{i+1}"
        tmp_dict['csv_files'] = file_list
        tmp_dict['weights'] = None
        tmp_dict['start_date'] = start_date
        tmp_dict['end_date'] = end_date
        periods.append(tmp_dict)

    return periods

periods = prep_periods(df)
run_backtest(periods)









