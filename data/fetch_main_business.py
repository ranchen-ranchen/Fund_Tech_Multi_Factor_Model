"""
上市公司主营业务描述批量获取程序
数据源：AKShare (同花顺-主营介绍接口)
作者：量化研究工具
用法：python fetch_main_business.py
依赖：pip install akshare pandas
"""

import akshare as ak
import pandas as pd
import time
import os
import sys
from datetime import datetime

# ==================== 配置区 ====================
OUTPUT_DIR = "./stock_main_business"      # 输出目录
REQUEST_INTERVAL = 0.5                    # 每只股票请求间隔(秒)，防止被限流
MAX_RETRIES = 3                           # 单只股票最大重试次数
BATCH_SAVE_SIZE = 100                     # 每获取多少只股票保存一次中间结果
# ================================================


def ensure_output_dir():
    """创建输出目录"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] 输出目录: {os.path.abspath(OUTPUT_DIR)}")


def get_stock_list():
    """
    获取全部A股股票列表
    返回包含 code 和 name 的 DataFrame
    """
    print("[INFO] 正在获取A股股票列表...")
    try:
        # stock_zh_a_spot_em 返回沪深京所有A股的实时行情数据
        df = ak.stock_zh_a_spot_em()
        stock_list = df[["代码", "名称"]].copy()
        stock_list.columns = ["code", "name"]
        # 过滤掉北交所(8开头/4开头)和B股，聚焦沪深主板/创业板/科创板
        stock_list = stock_list[
            ~stock_list["code"].str.startswith(("4", "8"))
        ].reset_index(drop=True)
        print(f"[INFO] 共获取到 {len(stock_list)} 只股票")
        return stock_list
    except Exception as e:
        print(f"[ERROR] 获取股票列表失败: {e}")
        sys.exit(1)


def fetch_single_stock(code, retries=MAX_RETRIES):
    """
    获取单只股票的主营业务描述
    参数:
        code: 股票代码，如 "000001"
        retries: 最大重试次数
    返回:
        dict 或 None
    """
    for attempt in range(1, retries + 1):
        try:
            df = ak.stock_zyjs_ths(symbol=code)
            if df is None or df.empty:
                return None

            # 提取第一行数据（该接口每只股票返回一行）
            row = df.iloc[0]
            result = {
                "股票代码": str(row.get("股票代码", code)),
                "主营业务": str(row.get("主营业务", "")).strip(),
                "产品类型": str(row.get("产品类型", "")).strip(),
                "产品名称": str(row.get("产品名称", "")).strip(),
                "经营范围": str(row.get("经营范围", "")).strip(),
            }
            return result

        except Exception as e:
            if attempt < retries:
                wait = attempt * 1.0
                print(f"  [WARN] {code} 第{attempt}次失败，{wait}s后重试: {e}")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {code} 获取失败(已重试{retries}次): {e}")
                return None

    return None


def load_existing_data(output_path):
    """
    加载已有数据，用于断点续传
    返回已处理的股票代码集合
    """
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path, dtype={"股票代码": str})
            processed = set(existing["股票代码"].tolist())
            print(f"[INFO] 检测到已有数据 {len(processed)} 条，将跳过已处理的股票")
            return processed, existing
        except Exception:
            pass
    return set(), pd.DataFrame()


def main():
    """主流程"""
    ensure_output_dir()

    # 输出文件路径
    output_path = os.path.join(OUTPUT_DIR, "main_business.csv")

    # 1. 获取股票列表
    stock_list = get_stock_list()

    # 2. 断点续传：加载已有数据
    processed_codes, existing_df = load_existing_data(output_path)

    # 3. 过滤待处理的股票
    pending = stock_list[~stock_list["code"].isin(processed_codes)].reset_index(
        drop=True
    )
    total = len(pending)
    print(f"[INFO] 待处理: {total} 只，已完成: {len(processed_codes)} 只")

    if total == 0:
        print("[INFO] 所有股票已处理完毕")
        return

    # 4. 批量抓取
    results = []
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for idx, row in pending.iterrows():
        code = row["code"]
        name = row["name"]

        # 进度显示
        elapsed = time.time() - start_time
        speed = (idx + 1) / elapsed if elapsed > 0 else 0
        eta_seconds = (total - idx - 1) / speed if speed > 0 else 0
        eta_str = str(pd.Timedelta(seconds=int(eta_seconds)))

        print(
            f"[{idx + 1}/{total}] {code} {name}  "
            f"| 速度: {speed:.1f}只/秒 | 预计剩余: {eta_str}",
            end="  ",
        )

        result = fetch_single_stock(code)

        if result:
            result["股票名称"] = name
            results.append(result)
            success_count += 1
            print("✓")
        else:
            fail_count += 1
            print("✗")

        # 间隔控制
        time.sleep(REQUEST_INTERVAL)

        # 批量保存中间结果
        if (idx + 1) % BATCH_SAVE_SIZE == 0 and results:
            _save_batch(results, existing_df, output_path)
            existing_df = _merge_and_read(results, existing_df, output_path)
            results = []
            print(f"  [SAVE] 已保存中间结果，累计成功: {success_count}")

    # 5. 最终保存
    if results:
        _save_batch(results, existing_df, output_path)

    # 6. 统计
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"完成！耗时: {str(pd.Timedelta(seconds=int(total_time)))}")
    print(f"成功: {success_count} | 失败: {fail_count} | 总计: {total}")
    print(f"输出文件: {os.path.abspath(output_path)}")
    print("=" * 60)

    # 7. 结果预览
    if os.path.exists(output_path):
        final_df = pd.read_csv(output_path, dtype={"股票代码": str})
        print(f"\n最终数据集大小: {len(final_df)} 行 x {len(final_df.columns)} 列")
        print(f"字段: {list(final_df.columns)}")
        print("\n示例数据（前3行）：")
        print(final_df.head(3).to_string(max_colwidth=60))


def _save_batch(new_results, existing_df, output_path):
    """保存一批结果到CSV"""
    new_df = pd.DataFrame(new_results)
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # 去重（以股票代码为准）
    combined = combined.drop_duplicates(subset=["股票代码"], keep="last")
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")


def _merge_and_read(new_results, existing_df, output_path):
    """合并后重新读取"""
    try:
        return pd.read_csv(output_path, dtype={"股票代码": str})
    except Exception:
        return existing_df


if __name__ == "__main__":
    main()