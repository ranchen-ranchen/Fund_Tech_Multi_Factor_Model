import re
import pandas as pd

import csv
from typing import Optional


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



def read_from_csv(filename:str) -> pd.DataFrame:
    df = pd.read_csv(filename, dtype=str)
    df_cleaned = df.dropna().reset_index(drop=True) # remove the days with no trading
    data = pd.DataFrame()
    data['date'] = df_cleaned['date']
    data['open'] = df_cleaned['open'].astype(float)
    data['close'] = df_cleaned['close'].astype(float)
    data['high'] = df_cleaned['high'].astype(float)
    data['low'] = df_cleaned['low'].astype(float)
    data['volume'] = df_cleaned['volume'].astype(float)
    data['amount'] = df_cleaned['amount'].astype(float)
    return data

def get_stock_data(filename):
    df = read_from_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume"]]





def read_hs300_constituents(csv_path: str, 
                query_date: str,
                encoding: str = "utf-8-sig",) -> list:
    """
    从CSV文件中读取 query_date 和 code 两列，返回 DataFrame。

    参数
    ----------
    csv_path : str
        CSV 文件路径。
    encoding : str, 默认 'utf-8'
        文件编码。若为中文 Windows 导出的文件，可尝试 'gbk' 或 'utf-8-sig'。
    parse_date : bool, 默认 False
        是否将 query_date 解析为 datetime 类型。

    返回
    -------
    pd.DataFrame
        包含 'query_date' 和 'code' 两列的 DataFrame。
    """
    df = pd.read_csv(
        csv_path,
        usecols=["query_date", "code"],   # 只读取需要的列
        encoding=encoding,
        dtype={"code": str},              # 股票代码保留为字符串，避免前导 0 丢失
    )


    df["code"] = df["code"].astype(str).str.replace(r"[A-Za-z.]", "", regex=True)
    # 重置索引并去除缺失行
    df = df.dropna(subset=["query_date", "code"]).reset_index(drop=True)
    return df.loc[df['query_date']==query_date]['code'].tolist()


def read_business_description(
    stock_code: str,
    csv_path: str = "main_business.csv"
) -> Optional[str]:

    # utf-8-sig 用于去掉 CSV 文件开头的 BOM
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_code = (row.get("股票代码") or "").strip()

            # 兼容股票代码列中可能包含后缀的情况
            row_match = re.search(r'(?<!\d)(\d{6})(?!\d)', row_code)
            if row_match and row_match.group(1) == stock_code:
                fields = ["主营业务", "产品类型", "产品名称", "经营范围"]
                return "；".join(
                    f"{field}：{(row.get(field) or '').strip()}"
                    for field in fields
                )

    return None



def normalize_code(code: str) -> str:
    """
    将各种写法的股票代码统一为 baostock 要求的格式：sh.600000 / sz.000001 / bj.430047
    """
    raw = str(code).strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("股票代码为空")

    if "." in raw:
        parts = raw.split(".")
        if len(parts) != 2:
            raise ValueError(f"无法识别的股票代码: {code}")
        a, b = parts
        if a in ("sh", "sz", "bj"):
            return f"{a}.{b}"
        if b in ("sh", "sz", "bj"):
            return f"{b}.{a}"
        raise ValueError(f"无法识别的股票代码: {code}")

    for pre in ("sh", "sz", "bj"):
        if raw.startswith(pre) and raw[2:].isdigit():
            return f"{pre}.{raw[2:]}"

    if not raw.isdigit():
        raise ValueError(f"无法识别的股票代码: {code}")

    if raw.startswith(("60", "68", "51", "58", "11", "90", "50")):
        return f"sh.{raw}"
    if raw.startswith(("00", "30", "12", "15", "16", "18", "20", "39", "13")):
        return f"sz.{raw}"
    if raw.startswith(("43", "83", "87", "88", "92")):
        return f"bj.{raw}"

    return f"sh.{raw}" if raw[0] in ("5", "6", "9") else f"sz.{raw}"










