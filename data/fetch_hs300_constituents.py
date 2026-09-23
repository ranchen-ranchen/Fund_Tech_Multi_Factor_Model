"""
获取沪深300指数历史成分股名单（每半年一次快照）
时间范围：2021-01-01 ~ 2026-01-01
数据源：Baostock（免费，无需注册）
"""

import baostock as bs
import pandas as pd
import os

# ---------------------- 配置 ----------------------
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"
OUTPUT_FILE = "hs300_constituents_2021_2026.csv"

# 生成每半年一次的查询日期（6月30日、12月31日，外加起始日）
# 实际指数调整在6月和12月的第二个星期五的下一交易日生效，
# 选择6月30日/12月31日作为快照日期可以覆盖每次调整后的成分股。
date_list = [
    "2021-01-01",
    "2021-06-30",
    "2021-12-31",
    "2022-06-30",
    "2022-12-31",
    "2023-06-30",
    "2023-12-31",
    "2024-06-30",
    "2024-12-31",
    "2025-06-30",
    "2025-12-31",
    "2026-01-01",
]
# 过滤掉超出 Baostock 数据范围的日期（2026-01-01 可能暂无数据）
date_list = [d for d in date_list if d <= END_DATE]

# ---------------------- 登录 Baostock ----------------------
lg = bs.login()
print(f"登录响应: error_code={lg.error_code}, error_msg={lg.error_msg}")

if lg.error_code != "0":
    raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

# ---------------------- 逐日期查询成分股 ----------------------
all_records = []

for query_date in date_list:
    print(f"\n正在查询 {query_date} 的沪深300成分股...")
    rs = bs.query_hs300_stocks(query_date)
    print(f"查询响应: error_code={rs.error_code}, error_msg={rs.error_msg}")

    if rs.error_code != "0":
        print(f"  ⚠️ 查询失败，跳过该日期。")
        continue

    # 读取结果集
    stock_list = []
    while (rs.error_code == "0") & rs.next():
        stock_list.append(rs.get_row_data())

    if not stock_list:
        print(f"  ⚠️ 该日期无数据（可能尚未更新），跳过。")
        continue

    df_temp = pd.DataFrame(stock_list, columns=rs.fields)
    # rs.fields 通常包含: updateDate, code, code_name
    df_temp["query_date"] = query_date
    all_records.append(df_temp)
    print(f"  ✅ 获取到 {len(df_temp)} 只成分股。")

# ---------------------- 合并并保存 ----------------------
if all_records:
    result = pd.concat(all_records, ignore_index=True)
    # 整理列顺序，让 query_date 在前，便于查看
    cols = ["query_date"] + [c for c in result.columns if c != "query_date"]
    result = result[cols]

    # 保存为 CSV（兼容中文）
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ 数据已保存至: {os.path.abspath(OUTPUT_FILE)}")
    print(f"   总记录数: {len(result)}")
    print(f"   涉及日期: {result['query_date'].unique().tolist()}")
    print(f"\n数据预览（前 10 行）：")
    print(result.head(10).to_string(index=False))
else:
    print("\n❌ 未获取到任何数据，请检查网络或日期范围。")

# ---------------------- 登出 ----------------------
bs.logout()
print("\n程序执行完毕。")

