import baostock as bs
import pandas as pd
import sys
from datetime import datetime, timedelta


def get_pb_data(stock_list, target_date, output_csv="pb_data.csv", lookback_days=30):
    """
    从baostock获取指定股票列表在指定日期的市净率数据

    如果指定日期是非交易日（周末、节假日），或者某只股票当天停牌/无数据，
    会自动向前查找最近的交易日 / 有数据日期。

    参数:
        stock_list:    股票代码列表，格式如 ['sh.600000', 'sz.000001']
        target_date:   目标日期，格式 'YYYY-MM-DD'
        output_csv:    输出CSV文件名
        lookback_days: 向前回溯的最大天数（默认30天）
    """
    # ---------- 登录系统 ----------
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        sys.exit(1)
    print("Baostock登录成功")
    print(f"目标日期: {target_date}")

    # 计算回溯起始日期
    start_date = (datetime.strptime(target_date, "%Y-%m-%d")
                  - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    print(f"回溯范围: {start_date} ~ {target_date}（最多向前 {lookback_days} 天）\n")

    results = []

    # ---------- 逐只股票查询 ----------
    for code in stock_list:
        try:
            # 查询 [start_date, target_date] 区间内的所有日频估值数据
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,pbMRQ",
                start_date=start_date,
                end_date=target_date,
                frequency="d",
                adjustflag="3"
            )

            if rs.error_code != '0':
                print(f"  {code} 查询失败: {rs.error_msg}")
                results.append({"code": code, "date": None, "pbMRQ": None})
                continue

            # 读取所有返回行
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())

            # 从后往前找第一条 pbMRQ 非空的数据（跳过停牌日的空值）
            valid_row = None
            for row in reversed(rows):
                # row = [date, code, pbMRQ]
                if row[2] not in (None, '', '0', '0.0'):
                    valid_row = row
                    break

            if valid_row is not None:
                actual_date = valid_row[0]
                pb_value = valid_row[2]

                if actual_date == target_date:
                    print(f"  {code} -> {actual_date}  PB = {pb_value}")
                else:
                    print(f"  {code} -> 目标日无数据，向前取到 {actual_date}  PB = {pb_value}")

                results.append({
                    "code": valid_row[1],
                    "date": actual_date,
                    "pbMRQ": pb_value
                })
            else:
                # 回溯范围内也没有任何有效数据
                print(f"  {code} 在 {start_date} ~ {target_date} 范围内均无有效数据")
                results.append({"code": code, "date": None, "pbMRQ": None})

        except Exception as e:
            print(f"  {code} 处理异常: {e}")
            results.append({"code": code, "date": None, "pbMRQ": None})

    # ---------- 登出 ----------
    bs.logout()

    # ---------- 保存为 CSV ----------
    if results:
        df = pd.DataFrame(results)
        df = df[["code", "date", "pbMRQ"]]  # 保持原有列顺序
        df.to_csv(output_csv, encoding="utf-8-sig", index=False)
        print(f"\n数据已保存至: {output_csv}")
        print(df.to_string(index=False))
    else:
        print("未获取到任何数据")


if __name__ == "__main__":
    # ========== 用户输入区域 ==========
    # 股票池代码列表（格式：sh.xxxxxx 或 sz.xxxxxx）
    stock_pool = [
        "sh.600000",   # 浦发银行
        "sh.601318",   # 中国平安
        "sz.000001",   # 平安银行
        "sz.000002",   # 万科A
        "sh.600519",   # 贵州茅台
    ]

    # 指定日期（可以是交易日，也可以是非交易日，例如周末、节假日）
    query_date = "2024-12-31"

    # 输出文件名
    output_file = f"pb_data_{query_date}.csv"

    # 向前回溯的最大天数
    LOOKBACK_DAYS = 30
    # ==================================

    get_pb_data(stock_pool, query_date, output_file, lookback_days=LOOKBACK_DAYS)
