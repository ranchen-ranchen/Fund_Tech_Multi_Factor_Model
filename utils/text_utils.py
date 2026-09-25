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



import pandas as pd
def read_hs300_constituents(csv_path: str, 
                    encoding: str = "utf-8-sig",
                    parse_date: bool = False) -> pd.DataFrame:
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

    if parse_date:
        df["query_date"] = pd.to_datetime(df["query_date"], errors="coerce")

    df["code"] = df["code"].astype(str).str.replace(r"[A-Za-z.]", "", regex=True)
    # 重置索引并去除缺失行
    df = df.dropna(subset=["query_date", "code"]).reset_index(drop=True)

    return df



def read_business_description(csv_path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    从给定的 CSV 文件读取股票数据。

    参数:
        csv_path: CSV 文件路径
        encoding: 文件编码，默认为 'utf-8'，如果是中文 Windows 导出的可尝试 'gbk'

    返回:
        pandas DataFrame，包含两列：
            - code: 股票代码
            - description: 由 主营业务、产品类型、产品名称、经营范围 合并而成
    """
    # 1. 读取 CSV（读取为字符串，避免股票代码被转成数字丢前导 0）
    df = pd.read_csv(csv_path, dtype=str, encoding=encoding)

    # 2. 校验列是否存在
    code_col = "股票代码"
    desc_cols = ["主营业务", "产品类型", "产品名称", "经营范围"]
    missing = [c for c in [code_col] + desc_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 文件缺少以下列: {missing}")

    # 3. 构造结果 DataFrame
    result = pd.DataFrame()
    result["code"] = df[code_col].fillna("").str.strip()

    # 4. 合并描述列：去掉空值/NaN，用分隔符连接
    def merge_description(row):
        parts = []
        for col in desc_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            val = str(val).strip()
            if val:
                parts.append(f"{col}：{val}")
        return "；".join(parts)

    result["description"] = df.apply(merge_description, axis=1)

    return result.reset_index(drop=True)


def match_business_description(path_hs300, path_business_des, date, code_col='code', reset_index=True):
    """
    找出 df1 和 df2 中 code 列值相同的元素，
    将 df2 中 code 值属于该交集的行提取出来，返回新的 DataFrame。

    参数：
        df1, df2    : pandas DataFrame，均包含 code_col 列
        code_col    : code 列名，默认 'code'
        reset_index : 是否重置结果的行索引，默认 True

    返回：
        pandas DataFrame：df2 中 code 值也出现在 df1 中的行
    """
    hs_cons = read_hs300_constituents(path_hs300)
    df1 = hs_cons[hs_cons['query_date'] == date]
    df2 = read_business_description(path_business_des)


    # 取两个 code 列的非空值交集
    common_codes = set(df1[code_col].dropna()) & set(df2[code_col].dropna())
    # 从 df2 中筛选出 code 属于交集的行
    result = df2[df2[code_col].isin(common_codes)].copy()
    if reset_index:
        result = result.reset_index(drop=True)
    return result


def fund_screen_prosperity_iterations(path_stock = 'data/hs300_constituents_2021_2026.csv', 
    path_description = 'data/stock_main_business/main_business.csv', query_date = '2021-01-01'):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from strategy.eval_business_prosperity import evaluate_prosperity
    df_des = read_business_description(path_description)
    df_tmp = read_hs300_constituents(path_stock)
    df_code = df_tmp[df_tmp['query_date'] == query_date]
    def eval_pros(row, df_des):
        result = evaluate_prosperity(company_code=row['code'], description=df_des.loc[df_des['code']== row['code'], 'description'].item())
        if result['_status'] == 'success':
                return result
        max_retries = 10
        attempt = 0
        while result['_status'] == 'failed' and attempt < max_retries:
            result = evaluate_prosperity(company_code=row['code'], description=df_des.loc[df_des['code']== row['code'], 'description'].item())
            attempt += 1
            if result['_status'] == 'success':
                return result
    
    result = df_code.apply(eval_pros, axis=1, args=(df_des, ))
    result = result.tolist()
    df_result = pd.DataFrame(result)
    if "total_score" in df_result.columns:
        df_result = df_result.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

    from strategy.eval_business_prosperity import flatten_results

    print("\n" + "-" * 70)
    print("景气度评估结果汇总")
    print("-" * 70)
    display_cols = [
            "company_code", "total_score", "prosperity_level",
            "trend",
    ]
    available_cols = [c for c in display_cols if c in df_result.columns]
    print(df_result[available_cols].to_string(index=False))
    if not df_result.empty:
        output_df = flatten_results(df_result)
        output_df.to_csv(
                "prosperity_results.csv",
                index=False,
                encoding="utf-8-sig",
        )
        print("\n完整结果已保存至 prosperity_results.csv")

    print("\n" + "-" * 70)
    print("景气等级分布统计")
    print("-" * 70)
    if "prosperity_level" in df_result.columns:
        print(df_result["prosperity_level"].value_counts().to_string())

    if "total_score" in df_result.columns:
        print("\n平均景气度得分：", round(df_result["total_score"].mean(skipna=True), 2))





def fund_screen_prosperity(path_stock = 'data/hs300_constituents_2021_2026.csv', path_description = 'data/stock_main_business/main_business.csv', query_date = '2021-01-01'):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    result = match_business_description(path_hs300 = path_stock, path_business_des = path_description, date = query_date).to_dict(orient="records")
    from strategy.eval_business_prosperity import batch_evaluate_iterations
    df_result = batch_evaluate_iterations(result, max_retries=5)

    from strategy.eval_business_prosperity import flatten_results
    print("\n" + "-" * 70)
    print("景气度评估结果汇总")
    print("-" * 70)
    display_cols = [
            "company_code", "total_score", "prosperity_level",
            "trend",
    ]
    available_cols = [c for c in display_cols if c in df_result.columns]
    print(df_result[available_cols].to_string(index=False))
    if not df_result.empty:
        output_df = flatten_results(df_result)
        output_df.to_csv(
                "prosperity_results.csv",
                index=False,
                encoding="utf-8-sig",
        )
        print("\n完整结果已保存至 prosperity_results.csv")

    print("\n" + "-" * 70)
    print("景气等级分布统计")
    print("-" * 70)
    if "prosperity_level" in df_result.columns:
        print(df_result["prosperity_level"].value_counts().to_string())

    if "total_score" in df_result.columns:
        print("\n平均景气度得分：", round(df_result["total_score"].mean(skipna=True), 2))







def fund_screen_policy_match(path_stock = 'data/hs300_constituents_2021_2026.csv', path_description = 'data/stock_main_business/main_business.csv', query_date = '2021-01-01'):
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    result = match_business_description(path_hs300 = path_stock, path_business_des = path_description, date = query_date).to_dict(orient="records")
    from strategy.eval_policy_match import batch_evaluate
    df_result = batch_evaluate(result)
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    # 展示核心字段
    display_cols = ["company_code", "total_score", "match_level"]
    available_cols = [c for c in display_cols if c in df_result.columns]
    print(df_result[available_cols].to_string(index=False))
    # 保存完整结果到 CSV（展开 JSON 字段）
    if not df_result.empty:
        # 提取维度得分
        for dim in ["policy_direction", "tech_content",
            "industry_chain", "growth_prospect"]:
            df_result[f"{dim}_score"] = df_result.apply(
                    lambda r: r.get("dimensions", {}).get(dim, {}).get("score")
                    if isinstance(r.get("dimensions"), dict) else None,
                    axis=1,
                )
            df_result[f"{dim}_reason"] = df_result.apply(
                    lambda r: r.get("dimensions", {}).get(dim, {}).get("reason")
                    if isinstance(r.get("dimensions"), dict) else None,
                    axis=1,
                )

        # 提取摘要和关键词
        df_result["summary"] = df_result.apply(
                lambda r: r.get("summary", ""), axis=1
            )
        df_result["policy_keywords"] = df_result.apply(
                lambda r: ", ".join(r.get("policy_keywords", []))
                if isinstance(r.get("policy_keywords"), list) else "",
                axis=1,
            )

        # 删除原始嵌套列，保存扁平化结果
        output_df = df_result.drop(
                columns=["dimensions"], errors="ignore"
            )
        output_df.to_csv(
                "policy_match_results.csv",
                index=False,
                encoding="utf-8-sig",
            )
        print("\n完整结果已保存至 policy_match_results.csv")

    # ---------- 6.4 快速统计 ----------
    print("\n" + "=" * 60)
    print("契合度分布统计")
    print("=" * 60)
    if "match_level" in df_result.columns:
        print(df_result["match_level"].value_counts().to_string())



def read_screened_company_codes(
    file1: str ,
    file2: str ,
    limit: int = 200
) -> list[str]:
    import csv
    from itertools import islice
    """
    分别读取两个 CSV 文件前 limit 条数据行中的 company_code，
    返回两个文件中 company_code 重叠的部分。

    注意：
    - 使用 utf-8-sig 兼容 CSV 文件可能的 BOM 头。
    - csv.DictReader 会自动把第一行当作表头，因此 limit=200 表示读取前 200 条数据行。
    - 返回结果按 file1 中 company_code 出现的顺序排列，并去重。
    """

    def read_company_codes(path: str) -> list[str]:
        codes = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in islice(reader, limit):
                code = (row.get("company_code") or "").strip()
                if code:
                    codes.append(code)
        return codes

    codes1 = read_company_codes(file1)
    codes2_set = set(read_company_codes(file2))

    result = []
    seen = set()
    for code in codes1:
        if code in codes2_set and code not in seen:
            result.append(code)
            seen.add(code)

    return result










