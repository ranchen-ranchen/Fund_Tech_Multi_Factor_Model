"""
上市公司主营业务景气度预期评估
==================================================
时间视角：站在 2021 年初，展望未来五年（2021-2026）
判断依据：给定公司业务描述 + 一般行业知识 + 一般技术发展趋势
约束条件：不依赖具体财务数据，不引用 2021 年之后才发生的事件
输出要求：必须给出景气度总分、等级、趋势方向与不确定性说明
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# ============================================================
# 1. 配置区
# ============================================================

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
config: dict = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("读取配置文件失败: %s", e)

PROSPERITY_SCREEN_THRESHOLD = config.get("PROSPERITY_SCREEN_THRESHOLD", 60)
MODEL_NAME = config.get("MODEL_NAME", "deepseek-flash")
MAX_WORKERS = config.get("MAX_WORKERS", 3)

client = OpenAI(api_key=api_key, base_url=base_url)

# ---------- 时间视角设定 ----------
BASE_YEAR = 2021
FORECAST_HORIZON_YEARS = 5
AS_OF_DATE = f"{BASE_YEAR}-01-01"
OUTLOOK_WINDOW = f"{BASE_YEAR}-{BASE_YEAR + FORECAST_HORIZON_YEARS}"

DIMENSION_MAX = {
    "demand": 25,
    "supply_competition": 20,
    "tech_trajectory": 20,
    "policy_environment": 15,
    "company_position": 20,
}


# ============================================================
# 2. Prompt 模板
# ============================================================

SYSTEM_PROMPT = f"""你是一位资深的产业研究员，正在以 **{BASE_YEAR} 年初** 的视角，判断上市公司主营业务在未来 {FORECAST_HORIZON_YEARS} 年（{OUTLOOK_WINDOW}）的景气度预期。

你必须严格遵循以下规则：

【时间视角约束】
1. 你的信息视野应被视为截止于 {AS_OF_DATE}。禁止使用 {BASE_YEAR} 年之后才发生的事件、政策、技术突破、公司公告或市场行情作为判断依据。
2. 你的任务是“在当时看未来”，而不是“用后来验证过去”。如果某个判断只是因为你事后知道结果才成立，请不要写出来。
3. 展望窗口为 {OUTLOOK_WINDOW}，重点判断这五年内行业景气度的总体方向与中枢水平，而非某一年的短期波动。

【信息使用约束】
4. 只允许使用三类信息：
   (a) 题目给出的公司主营业务描述与补充信息；
   (b) 一般行业知识（产业链结构、供需逻辑、竞争格局常识）；
   (c) 一般技术发展趋势知识（技术路线演进方向、渗透率提升逻辑、替代关系）。
5. 严禁依赖或编造具体财务数据，包括但不限于：营收、利润、毛利率、订单金额、产能数字、市占率具体百分比、股价、估值。
   可以使用方向性、定性表述（如“需求有望扩张”“供给可能偏紧”），但不得给出精确数字。
6. 如果公司业务描述信息不足，应明确指出信息缺口，并相应降低判断的确定性，而不是自行补全。

【评估维度】
7. 从以下五个维度综合评估：
   - 需求景气度：未来五年下游需求扩张的确定性与空间
   - 供给与竞争格局：产能投放节奏、行业集中度、竞争激烈程度、是否可能产能过剩
   - 技术路线演进：主流技术路线是否清晰、公司业务是否卡在趋势方向上、是否存在被替代风险
   - 政策与外部环境：{BASE_YEAR} 年初已知的产业政策、监管导向、贸易与地缘环境的可能影响方向
   - 公司自身定位：基于业务描述判断其在产业链中的位置、客户结构、技术积累与壁垒
8. 最终给出 0-100 的景气度总分、景气等级、趋势方向。

【不确定性】
9. 必须输出 uncertainty_notes，列出 2-4 条最主要的不确定性来源。

【输出格式】
10. 必须严格以 JSON 格式输出，不得包含 JSON 之外的任何文字。"""


USER_PROMPT_TEMPLATE = """请以 {as_of_date} 的视角，评估以下上市公司主营业务在未来五年（{outlook_window}）的景气度预期。

【公司代码】{company_code}
【主营业务描述】
{business_description}

【补充信息】（可为空）
{extra_info}

【评估基准日期】{as_of_date}
【展望窗口】{outlook_window}（未来五年）

提醒：请仅使用业务描述、一般行业知识与一般技术发展趋势知识进行定性判断；不要依赖或编造具体财务数据；不要引用 {base_year} 年之后才发生的事件。

请严格按照以下 JSON 结构输出（不要输出任何其他内容）：

{{
  "company_code": "公司代码",
  "total_score": 0-100的整数,
  "prosperity_level": "高景气/景气上行/景气平稳/景气下行/景气低迷",
  "trend": "上行/持平/下行",
  "dimensions": {{
    "demand": {{
      "score": 0-25的整数,
      "reason": "未来五年需求扩张确定性的判断理由"
    }},
    "supply_competition": {{
      "score": 0-20的整数,
      "reason": "供给与竞争格局演变的判断理由"
    }},
    "tech_trajectory": {{
      "score": 0-20的整数,
      "reason": "技术路线演进与业务卡位的判断理由"
    }},
    "policy_environment": {{
      "score": 0-15的整数,
      "reason": "政策与外部环境影响的判断理由"
    }},
    "company_position": {{
      "score": 0-20的整数,
      "reason": "公司在产业链中定位与能力的判断理由"
    }}
  }},
  "drivers": ["主要驱动因素1", "主要驱动因素2"],
  "risks": ["主要风险1", "主要风险2"],
  "uncertainty_notes": ["不确定性来源1", "不确定性来源2"],
  "summary": "一句话总结未来五年景气度结论"
}}

评分标准（总分 100，针对 {outlook_window} 五年展望）：
- 高景气 80-100：未来五年需求确定性扩张，技术路线清晰，供给格局有序，公司卡位核心环节
- 景气上行 65-79：需求趋势向上，但节奏或幅度存在一定不确定性
- 景气平稳 45-64：供需大体平衡，缺乏明确的向上或向下催化
- 景气下行 25-44：需求趋弱或供给过剩压力上升，技术路线存在被替代风险
- 景气低迷 0-24：需求萎缩或技术路线明显落后，行业长期承压"""


# ============================================================
# 3. 工具函数
# ============================================================

def _to_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _level_from_score(score: int) -> str:
    if score >= 80:
        return "高景气"
    if score >= 65:
        return "景气上行"
    if score >= 45:
        return "景气平稳"
    if score >= 25:
        return "景气下行"
    return "景气低迷"


def _extract_json(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return text.strip()


# ============================================================
# 4. 核心评估函数
# ============================================================

def evaluate_prosperity(
    company_code: str,
    business_description: str,
    extra_info: str = "",
    as_of_date: str = AS_OF_DATE,
    outlook_window: str = OUTLOOK_WINDOW,
    max_retries: int = 3,
) -> dict:
    """
    调用 LLM 评估单家公司主营业务在未来五年的景气度预期。

    参数
    ----
    company_code : str
        公司代码（如 "600519"、"000001"）
    business_description : str
        主营业务描述
    extra_info : str
        补充信息（可选）

    返回结构化 dict，包含总分、等级、趋势、各维度理由、
    驱动因素、风险与不确定性说明。
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_code=company_code,
        business_description=business_description,
        extra_info=extra_info or "（无）",
        as_of_date=as_of_date,
        outlook_window=outlook_window,
        base_year=BASE_YEAR,
    )

    last_error = "未知错误"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
                stream=False,
            )

            raw_text = response.choices[0].message.content or ""
            result = json.loads(_extract_json(raw_text))

            if "total_score" not in result:
                raise ValueError("LLM 返回缺少 total_score 字段")

            # ---------- 归一化：各维度分数求和作为总分 ----------
            dims = result.get("dimensions") or {}
            dim_sum = 0
            for key, max_score in DIMENSION_MAX.items():
                dim = dims.get(key) or {}
                score = _to_int(dim.get("score", 0), default=0)
                score = max(0, min(score, max_score))
                dim["score"] = score
                dims[key] = dim
                dim_sum += score
            result["dimensions"] = dims

            total = _to_int(result.get("total_score", dim_sum), default=dim_sum)
            if dim_sum > 0 and abs(dim_sum - total) > 5:
                total = dim_sum
            total = max(0, min(total, 100))
            result["total_score"] = total

            # ---------- 等级与分数保持一致 ----------
            result["prosperity_level"] = _level_from_score(total)

            # ---------- 字段兜底 ----------
            result["company_code"] = company_code      # 用入参覆盖，避免模型自造
            result.setdefault("trend", "持平")
            result.setdefault("drivers", [])
            result.setdefault("risks", [])
            result.setdefault("uncertainty_notes", [])
            result.setdefault("summary", "")

            result["_status"] = "success"
            result["_as_of_date"] = as_of_date
            result["_outlook_window"] = outlook_window
            return result

        except json.JSONDecodeError as e:
            last_error = f"JSON 解析失败: {e}"
            print(f"  [{company_code}] {last_error} (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except Exception as e:  # noqa: BLE001
            last_error = f"API 调用失败: {e}"
            print(f"  [{company_code}] {last_error} (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {
        "company_code": company_code,
        "total_score": None,
        "prosperity_level": "评估失败",
        "trend": None,
        "drivers": [],
        "risks": [],
        "uncertainty_notes": [],
        "summary": "",
        "_status": "failed",
        "_error": last_error,
    }


# ============================================================
# 5. 批量评估函数（并发）
# ============================================================

def batch_evaluate(
    companies: list[dict],
    as_of_date: str = AS_OF_DATE,
    outlook_window: str = OUTLOOK_WINDOW,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """
    批量评估多家公司在未来五年的业务景气度。

    参数
    ----
    companies : list[dict]
        每项格式：{"code": "公司代码", "description": "主营业务描述", "extra": "补充信息（可选）"}
    as_of_date : str
        评估基准日期，默认 2021-01-01
    outlook_window : str
        展望窗口，默认 2021-2026
    max_workers : int
        并发线程数

    返回
    ----
    pd.DataFrame，按 total_score 降序排列
    """
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                evaluate_prosperity,
                c["code"],
                c["description"],
                c.get("extra", ""),
                as_of_date,
                outlook_window,
            ): c["code"]
            for c in companies
        }

        for future in as_completed(future_map):
            code = future_map[future]
            try:
                result = future.result()
                results.append(result)
                score = result.get("total_score", "N/A")
                level = result.get("prosperity_level", "N/A")
                trend = result.get("trend") or "-"
                print(f"  ✓ {code}: {score} 分 | {level} | 趋势 {trend}")
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {code}: 异常 - {e}")
                results.append({
                    "company_code": code,
                    "total_score": None,
                    "prosperity_level": "评估失败",
                    "_status": "failed",
                    "_error": str(e),
                })

    df = pd.DataFrame(results)
    if "total_score" in df.columns:
        df = df.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

    return df


def flatten_results(df: pd.DataFrame) -> pd.DataFrame:
    """把嵌套的 dimensions / 列表字段展开为扁平列，便于落盘。"""
    if df.empty:
        return df

    out = df.copy()

    for dim in DIMENSION_MAX:
        out[f"{dim}_score"] = out.apply(
            lambda r: (r.get("dimensions") or {}).get(dim, {}).get("score")
            if isinstance(r.get("dimensions"), dict) else None,
            axis=1,
        )
        out[f"{dim}_reason"] = out.apply(
            lambda r: (r.get("dimensions") or {}).get(dim, {}).get("reason")
            if isinstance(r.get("dimensions"), dict) else None,
            axis=1,
        )

    def _join(value) -> str:
        if isinstance(value, list):
            return "；".join(str(v) for v in value)
        return ""

    out["drivers"] = out.apply(lambda r: _join(r.get("drivers")), axis=1)
    out["risks"] = out.apply(lambda r: _join(r.get("risks")), axis=1)
    out["uncertainty_notes"] = out.apply(
        lambda r: _join(r.get("uncertainty_notes")), axis=1
    )

    out = out.drop(columns=["dimensions"], errors="ignore")
    return out


# ============================================================
# 6. 主流程示例
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(f"景气度预期评估（基准：{AS_OF_DATE}，展望：{OUTLOOK_WINDOW} 未来五年）")
    print("=" * 70)

    # ---------- 6.1 单条评估示例 ----------
    print("\n" + "-" * 70)
    print("单条评估示例")
    print("-" * 70)

    single_result = evaluate_prosperity(
        company_code="XX001",
        business_description=(
            "公司主营高速光模块的研发、生产与销售，产品覆盖 100G/200G/400G 系列，"
            "下游客户以海外云计算厂商和国内通信设备商为主。公司持续投入硅光技术研发，"
            "并具备一定的封装与测试能力。"
        ),
        extra_info="公司海外客户占比较高，产品迭代速度较快。",
    )
    print(json.dumps(single_result, ensure_ascii=False, indent=2))

    # ---------- 6.2 批量评估示例 ----------
    print("\n" + "-" * 70)
    print("批量评估示例")
    print("-" * 70)

    # 实际使用时可以从 Excel / CSV 读取：
    #   df_input = pd.read_csv("companies.csv")
    #   companies_to_evaluate = [
    #       {
    #           "code": row["公司代码"],
    #           "description": row["主营业务描述"],
    #           "extra": row.get("补充信息", ""),
    #       }
    #       for _, row in df_input.iterrows()
    #   ]
    companies_to_evaluate = [
        {
            "code": "A001",
            "description": "公司专注于数据中心服务器整机的研发制造，并为客户提供定制化方案，客户以互联网与政企客户为主。",
            "extra": "公司在液冷散热方向有前期技术储备。",
        },
        {
            "code": "B001",
            "description": "公司主营光伏硅料与硅片的生产与销售，产能规模位居行业前列。",
            "extra": "公司所在环节扩产周期较长，资本开支强度较高。",
        },
        {
            "code": "C001",
            "description": "公司从事创新药研发，聚焦肿瘤免疫治疗领域的单克隆抗体和细胞疗法，拥有多个临床阶段管线。",
            "extra": "核心管线处于临床中后期，尚无商业化产品收入。",
        },
        {
            "code": "D001",
            "description": "公司主营船舶制造，涵盖集装箱船、油轮及气体运输船，手持订单以海外船东为主。",
            "extra": "造船行业属于典型的长周期、重资产行业。",
        },
        {
            "code": "E001",
            "description": "公司主要从事房地产开发与销售，兼营物业管理服务。",
            "extra": "主要布局三四线城市。",
        },
    ]

    df_result = batch_evaluate(companies_to_evaluate)

    # ---------- 6.3 输出结果 ----------
    print("\n" + "-" * 70)
    print("景气度评估结果汇总")
    print("-" * 70)

    display_cols = [
        "company_code", "total_score", "prosperity_level",
        "trend",
    ]
    available_cols = [c for c in display_cols if c in df_result.columns]
    print(df_result[available_cols].to_string(index=False))

    # ---------- 6.4 保存完整结果 ----------
    if not df_result.empty:
        output_df = flatten_results(df_result)
        output_df.to_csv(
            "prosperity_results.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\n完整结果已保存至 prosperity_results.csv")

        strong = output_df[
            output_df["total_score"].notna()
            & (output_df["total_score"] >= PROSPERITY_SCREEN_THRESHOLD)
        ]
        if not strong.empty:
            strong[available_cols].to_csv(
                "prosperity_strong.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"其中 {len(strong)} 家公司景气度 ≥ {PROSPERITY_SCREEN_THRESHOLD} 分，"
                "已保存至 prosperity_strong.csv"
            )

    # ---------- 6.5 快速统计 ----------
    print("\n" + "-" * 70)
    print("景气等级分布统计")
    print("-" * 70)
    if "prosperity_level" in df_result.columns:
        print(df_result["prosperity_level"].value_counts().to_string())

    if "total_score" in df_result.columns:
        print("\n平均景气度得分：", round(df_result["total_score"].mean(skipna=True), 2))

        