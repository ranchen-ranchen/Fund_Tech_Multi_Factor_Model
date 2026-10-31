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
    data['turn'] = df_cleaned['turn'].astype(float)
    return data

def get_stock_data(filename:str) -> pd.DataFrame:
    df = read_from_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    # df = df.set_index("date").sort_index()
    return df




def read_hs300_constituents(csv_file: str, 
                query_date: str,
                encoding: str = "utf-8-sig",) -> list:
    """
    从CSV文件中读取 query_date 和 code 两列，返回 code list
    """
    df = pd.read_csv(
        csv_file,
        usecols=["query_date", "code"],   # 只读取需要的列
        encoding=encoding,
        dtype={"code": str},              # 股票代码保留为字符串，避免前导 0 丢失
    )


    df["code"] = df["code"].astype(str).str.replace(r"[A-Za-z.]", "", regex=True)
    # 重置索引并去除缺失行
    df = df.dropna(subset=["query_date", "code"]).reset_index(drop=True)
    return df.loc[df['query_date']==query_date]['code'].tolist()

def read_hs300_weight(csv_path: str) -> pd.DataFrame:
    """
    读取 CSV 中列名含"股票代码"和"占基金资产净值比例"的列，
    返回只含 code、weight 两列的 DataFrame。

    参数
    ----
    csv_path : str  CSV 文件路径

    返回
    ----
    pd.DataFrame  列名为 ['code', 'weight']
                  code 为 6 位补零字符串，weight 为 float
    """
    # encoding='utf-8-sig' 用于去除文件可能存在的 BOM 头
    # dtype=str 避免 "000858" 这类代码被读成整数 858
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype=str)

    def find_col(keyword: str) -> str:
        matches = [c for c in df.columns if keyword in c]
        if not matches:
            raise KeyError(
                f"未找到包含「{keyword}」的列，实际列名为：{list(df.columns)}"
            )
        return matches[0]

    code_col = find_col('股票代码')
    weight_col = find_col('占基金资产净值比例')

    result = df[[code_col, weight_col]].copy()
    result.columns = ['code', 'weight']

    # 规范化：代码补足 6 位，权重转为数值
    result['code'] = result['code'].str.strip().str.zfill(6)
    result['weight'] = pd.to_numeric(result['weight'].str.strip(), errors='coerce')

    return result.set_index('code') 


import pandas as pd


def read_target_code_weight(csv_file_path):
    """
    读取 CSV 文件，取前 10 行的 'target_code' 列，
    将每一行按空格拆分为股票代码列表，
    每行生成一个 pandas DataFrame，包含股票代码和等权重。

    参数:
        csv_file_path: str, CSV 文件路径

    返回:
        list[pd.DataFrame]: 长度为 10 的 DataFrame 列表
    """
    # 读取 CSV
    df = pd.read_csv(csv_file_path)

    # 取前 10 行
    first_ten = df.head(10)

    result_dfs = []

    for _, row in first_ten.iterrows():
        # 拆分股票代码（去除空字符串，处理多余空格）
        codes = [c for c in str(row['target_code']).split() if c]

        n = len(codes)
        if n == 0:
            # 若该行为空，跳过或生成空 DataFrame
            result_dfs.append(pd.DataFrame(columns=['target_code', 'weight']))
            continue

        # 等权重 = 1 / n
        equal_weight = 1.0 / n

        sub_df = pd.DataFrame({
            'target_code': codes,
            'weight': [equal_weight] * n
        })

        result_dfs.append(sub_df.set_index('target_code'))

    return result_dfs




from datetime import date, datetime
from typing import Union, Sequence


def query_pb_mrq(
    csv_path: str,
    stock_codes: Union[str, Sequence[str]],
    target_date: Union[str, date, datetime, pd.Timestamp],
    code_col: str = "code",
    target_date_col: str = "target_date",
    actual_date_col: str = "actual_date",
    pb_col: str = "pbMRQ",
) -> pd.DataFrame:
    """
    从 CSV 中查询指定股票在给定日期下，target_date 最接近该日期的 pbMRQ。
    所有股票使用同一个 target_date。

    注意：匹配依据是 CSV 中的 target_date，actual_date 仅随匹配行返回，不参与比较。

    返回列：
        code, pbMRQ
    """

    # 1. 读取并规范化 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    required = [code_col, target_date_col, actual_date_col, pb_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    df["_code_key"] = df[code_col].map(normalize_code)
    df[target_date_col] = pd.to_datetime(df[target_date_col], errors="coerce")
    df[actual_date_col] = pd.to_datetime(df[actual_date_col], errors="coerce")
    df[pb_col] = pd.to_numeric(df[pb_col], errors="coerce")

    # 2. 统一输入；查询日期只转换一次并归一化到日期
    codes = [stock_codes] if isinstance(stock_codes, str) else list(stock_codes)

    target_ts = pd.to_datetime(target_date, errors="coerce")
    if pd.notna(target_ts):
        target_ts = target_ts.normalize()

    # 3. 逐只股票查询
    rows = []
    for qc in codes:
        key = normalize_code(qc)

        # 关键改动 1：按 target_date 过滤，而不是 actual_date
        sub = df[(df["_code_key"] == key) & df[target_date_col].notna()].copy()

        if sub.empty or pd.isna(target_ts):
            rows.append({
                "query_code": str(qc),
                "code": str(qc),
                "query_date": target_ts,
                "target_date": pd.NaT,
                "actual_date": pd.NaT,
                "pbMRQ": np.nan,
                "diff_days": np.nan,
            })
            continue

        # 关键改动 2：用 target_date 与查询日期比较
        sub["diff_days"] = (sub[target_date_col] - target_ts).abs().dt.days
        matched = sub.loc[sub["diff_days"].idxmin()]

        rows.append({
            "query_code": str(qc),
            "code": matched[code_col],
            "query_date": target_ts,
            "target_date": matched[target_date_col],
            "actual_date": matched[actual_date_col],
            "pbMRQ": matched[pb_col],
            "diff_days": matched["diff_days"],
        })

    return pd.DataFrame(
        rows,
        columns=[
            "code",
            "pbMRQ"
        ],
    )










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



def get_hs300etf_close(csv_path: str) -> pd.DataFrame:
    """
    从 hs300etf_510300_performance.csv 中读取日期和收盘价。
    参数:
        csv_path: CSV 文件路径

    返回:
        pandas.DataFrame，包含两列: Date, Close
    """
    # 定位 "# Daily Series" 所在行
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    daily_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Daily Series"):
            daily_idx = i
            break

    if daily_idx is None:
        raise ValueError("未在文件中找到 '# Daily Series' 标记")

    # 跳过元数据部分，从 Daily Series 的下一行表头开始读取
    df = pd.read_csv(
        csv_path,
        skiprows=daily_idx + 1,
        encoding="utf-8-sig"
    )

    # 第一列是日期，第二列是 Close
    result = df.iloc[:, [0, 1]].copy()
    result.columns = ["date", "close"]

    # 类型转换与清洗
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.dropna(subset=["date", "close"]).reset_index(drop=True)

    return result









