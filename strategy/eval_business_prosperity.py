# -*- coding: utf-8 -*-
"""
eval_prosperity_forecast.py

基于公司代码 + 主营业务描述，调用 DeepSeek 评估该公司在指定时点
（默认 2021 年）的业务景气度预期。

用法：
    1) 单条：evaluate_prosperity("688XXX", "公司主要从事……", as_of="2021")
    2) 批量：batch_evaluate([{"code": "...", "description": "..."}, ...], as_of="2021")
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
load_dotenv()

# 读取环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")

# 可选：读取项目配置文件
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
config: dict = {}
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("读取配置文件失败: %s", e)

MODEL_NAME = config.get("MODEL_NAME", "deepseek-flash")  # 可用模型：deepseek-flash（轻量快速）、deepseek-v4-pro（能力更强）
MAX_WORKERS = config.get("MAX_WORKERS", 3)               # 并发线程数（控制 API 速率，避免触发限流）
DEFAULT_AS_OF = str(config.get("AS_OF", "2021"))         # 默认评估时点：2021 年

import json
import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ============================================================
# 1. 配置区
# ============================================================

# DeepSeek API 使用 OpenAI 兼容格式
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# ============================================================
# 2. 评估时点背景库（可自行增删年份）
# ============================================================

TIME_CONTEXT: dict[str, str] = {
    "2020": """评估基准时点：2020年。
- 新冠疫情冲击后经济逐季修复，全年 GDP 增速约 2.3%，是全球主要经济体中唯一正增长；
- 流动性宽松，政策强调“六稳”“六保”，特别国债与专项债支撑基建投资；
- 医药生物（疫苗、检测、防护耗材）处于高景气；在线办公、在线教育、云计算需求爆发；
- 消费、旅游、航空、影院等线下服务业受冲击严重；
- 半导体国产替代加速，美国对华为等企业制裁升级，供应链自主可控成为主线；
- “新基建”成为政策热词，5G、数据中心、特高压、充电桩受市场关注。""",

    "2021": """评估基准时点：2021年（“十四五”开局之年）。
- 疫后经济复苏，出口强劲；但下半年基数效应显现，能耗双控与限电限产对中游制造形成扰动；
- “双碳”目标（2030碳达峰 / 2060碳中和）成为主线，新能源全产业链（光伏、风电、储能、新能源汽车）进入高景气扩张期；
- 全球“缺芯”持续，半导体设计、制造、设备、材料产业链景气度处于历史高位；
- 大宗商品价格大幅上涨，上游资源（煤炭、有色、化工、钢铁）盈利大幅改善；
- 房地产执行“三道红线”，融资端收紧，下半年房企信用风险开始暴露；
- 平台经济反垄断、教培“双减”、医药集采常态化，相关行业预期明显降温；
- 消费受疫情反复影响恢复偏弱，线下服务、旅游、餐饮仍有压力；
- 财政强调“跨周期调节”，专项债发行偏慢，基建投资低于预期。""",

    "2022": """评估基准时点：2022年。
- 俄乌冲突推高能源与粮食价格，全球通胀高企；
- 美联储激进加息，全球流动性收紧，成长股估值明显承压；
- 国内疫情多点散发（上海等），供应链受阻，消费与地产深度调整；
- 地产销售大幅下滑，“保交楼”政策出台，行业信用风险持续释放；
- 新能源车渗透率继续提升，光伏出口高增，储能需求爆发；
- 半导体进入下行周期，消费电子需求疲软，砍单去库存；
- 平台经济政策转向“常态化监管”，市场预期边际改善；
- 信创、军工等自主可控方向受政策与订单支撑。""",

    "2023": """评估基准时点：2023年。
- 疫情管控放开，消费与出行修复，但力度低于市场预期；
- 地产供需两端继续走弱，政策持续放松（认房不认贷、降首付）但效果有限；
- AI 大模型浪潮（ChatGPT）带动算力、光模块、服务器、液冷等环节高景气；
- 新能源车价格战激烈，光伏产业链价格大幅下跌，产能过剩担忧升温；
- 出口结构升级，“新三样”（电动车、锂电池、光伏）出口高增；
- “中特估”、高股息资产受资金青睐；
- 消费电子、半导体仍处周期底部，存储价格开始触底。""",

    "2024": """评估基准时点：2024年。
- 宏观有效需求不足，价格水平低位运行，企业盈利承压；
- “新质生产力”成为政策主线，低空经济、商业航天、AI、人形机器人受高度关注；
- AI 算力产业链持续高景气，国产算力受政策与订单支持；
- 地产政策大幅放松（限购取消、存量房贷利率下调），销售仍在筑底；
- “以旧换新”政策拉动家电、汽车消费；
- 出海成为制造业重要增长极，海外产能布局加速；
- 光伏、锂电等产能过剩行业进入出清阶段，价格战延续。""",

    "2025": """评估基准时点：2025年。
- 宏观政策“更加积极有为”，财政赤字率提升，超长期特别国债支持“两重两新”；
- AI 应用加速落地，算力、智能终端、机器人产业链维持高景气；
- 新能源行业供给侧出清推进，部分环节价格企稳回升；
- 地产仍处调整期，但一线城市出现企稳迹象；
- 外部环境不确定性上升（关税、贸易摩擦），出口链承压与出海并行；
- 消费刺激政策持续，服务消费与悦己消费受关注；
- 创新药出海授权（BD）交易活跃，生物医药预期改善。""",

    "2026": """评估基准时点：2026年（“十五五”开局之年）。
- 现代化产业体系建设为主线，强调科技自立自强与产业链安全；
- 新兴支柱产业：集成电路、航空航天、生物医药、低空经济、新型储能、智能机器人；
- 未来产业：量子科技、生物制造、氢能与核聚变能、脑机接口、具身智能、6G 通信；
- 传统产业改造升级：高端新材料、基础零部件和元器件、大型邮轮、LNG 运输船、CR450 动车组、农机装备、燃气轮机；
- 数字经济与人工智能：大模型、算力基础设施、数据要素、工业互联网、智能制造；
- 内需方面关注“两新”（大规模设备更新、消费品以旧换新）与新型城镇化；
- 外部环境复杂，关税与科技管制仍是主要不确定性来源。""",
}

GENERIC_CONTEXT_TEMPLATE = """评估基准时点：{as_of}。
请基于该时点可获得的宏观环境、产业周期位置、政策取向与市场共识，判断公司主营业务的景气度预期。
若对该时点的具体环境信息不足，请明确说明，并依据行业一般周期规律做谨慎推断，不得编造具体事件或数据。"""


def get_time_context(as_of: str) -> str:
    """按年份取评估时点背景；未收录的年份返回通用模板。"""
    key = str(as_of).strip()[:4]
    if key in TIME_CONTEXT:
        return TIME_CONTEXT[key]
    return GENERIC_CONTEXT_TEMPLATE.format(as_of=as_of)


# ============================================================
# 3. Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一位资深的行业研究员与投资分析师，擅长基于特定时间点的宏观与产业环境，评估上市公司主营业务的景气度预期。

你的评估需要严格遵循以下规则：
1. 只基于给定的公司主营业务描述与评估时点背景进行判断，不得编造具体财务数据、订单数据或未公开信息。
2. 必须"站在评估时点向前看"：模拟在该时点上市场对该公司未来 1-2 年业务景气度的预期，不得使用该时点之后才发生的信息（避免"后见之明"）。
3. 从以下四个维度进行综合评估，每个维度 0-25 分：
   - 需求景气度：下游需求增速、订单/销量/开工率趋势、产品价格走势
   - 供给与竞争格局：产能投放节奏、竞争激烈程度、公司议价能力与市场份额
   - 政策与外部环境：产业政策支持力度、监管取向、宏观与外部冲击
   - 盈利与成长预期：毛利率/净利率趋势、利润增速、成长空间与确定性
4. 给出 0-100 的景气度总分、景气等级与趋势判断（上行/持平/下行）。
5. 必须严格以 JSON 格式输出，不得包含 JSON 之外的任何文字。"""

USER_PROMPT_TEMPLATE = """请评估以下上市公司在【{as_of}】时点的业务景气度预期。

【公司代码】{company_code}
【主营业务描述】
{description}

【评估时点背景】
{time_context}
{extra_block}
请严格按照以下 JSON 结构输出（不要输出任何其他内容）：

{{
  "company_code": "公司代码",
  "as_of": "{as_of}",
  "total_score": 0-100的整数,
  "prosperity_level": "高景气/较高景气/中性/较低景气/低景气",
  "trend": "上行/持平/下行",
  "dimensions": {{
    "demand": {{
      "score": 0-25的整数,
      "reason": "需求景气度判断的简要理由"
    }},
    "supply_competition": {{
      "score": 0-25的整数,
      "reason": "供给与竞争格局判断的简要理由"
    }},
    "policy_environment": {{
      "score": 0-25的整数,
      "reason": "政策与外部环境判断的简要理由"
    }},
    "profit_growth": {{
      "score": 0-25的整数,
      "reason": "盈利与成长预期判断的简要理由"
    }}
  }},
  "summary": "一句话总结景气度结论",
  "key_drivers": ["核心驱动因素1", "核心驱动因素2"],
  "risk_factors": ["主要风险因素1", "主要风险因素2"]
}}

评分标准：
- 高景气：80-100 分，下游需求高速增长、产能供不应求、盈利预期强劲上行
- 较高景气：65-79 分，需求稳健增长、盈利能力改善、成长确定性较好
- 中性：45-64 分，供需与盈利基本平稳，缺乏明确方向
- 较低景气：25-44 分，需求走弱或竞争加剧、盈利承压
- 低景气：0-24 分，需求明显萎缩、行业大面积亏损或政策强力压制"""


# ============================================================
# 4. 工具函数
# ============================================================

def _extract_json(raw_text: str) -> str:
    """从 LLM 返回文本中提取 JSON 主体。"""
    text = raw_text.strip()
    # 去掉 markdown 代码块标记
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 兜底：截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


# ============================================================
# 5. 核心评估函数
# ============================================================

def evaluate_prosperity(
    company_code: str,
    description: str,
    as_of: str = DEFAULT_AS_OF,
    extra_context: str = "",
    max_retries: int = 3,
) -> dict:
    """
    调用 LLM 评估单家公司在指定时点的业务景气度预期。

    参数
    ----
    company_code : str
        公司代码，如 "688XXX"
    description : str
        主营业务描述
    as_of : str
        评估时点，默认 "2021"
    extra_context : str
        可选补充信息（如行业数据、公司公告要点、已知订单情况等）
    max_retries : int
        失败重试次数

    返回
    ----
    dict，包含总分、景气等级、趋势、四个维度得分与理由、驱动因素、风险因素等
    """
    time_context = get_time_context(as_of)
    extra_block = (
        f"\n【补充信息】\n{extra_context.strip()}\n"
        if extra_context and extra_context.strip()
        else ""
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_code=company_code,
        description=description,
        as_of=as_of,
        time_context=time_context,
        extra_block=extra_block,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,   # 低温度保证输出稳定
                max_tokens=3000,
                stream=False,
            )

            raw_text = response.choices[0].message.content.strip()
            result = json.loads(_extract_json(raw_text))

            # ---------- 基本校验 ----------
            if "total_score" not in result:
                raise ValueError("LLM 返回缺少 total_score 字段")

            result.setdefault("company_code", company_code)
            result.setdefault("as_of", str(as_of))

            # ---------- 归一化：确保总分等于四个维度之和 ----------
            dims = result.get("dimensions", {}) or {}
            dim_keys = ["demand", "supply_competition",
                        "policy_environment", "profit_growth"]
            dim_sum = sum(
                int(dims.get(k, {}).get("score", 0) or 0) for k in dim_keys
            )
            if dim_sum > 0:
                if abs(dim_sum - int(result["total_score"])) > 5:
                    result["total_score"] = dim_sum
            result["total_score"] = max(0, min(100, int(result["total_score"])))

            # ---------- 补齐缺失的等级/趋势 ----------
            if not result.get("prosperity_level"):
                result["prosperity_level"] = _score_to_level(result["total_score"])
            if not result.get("trend"):
                result["trend"] = "持平"

            result["_status"] = "success"
            return result

        except json.JSONDecodeError as e:
            print(f"  [{company_code}] JSON 解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:  # noqa: BLE001
            print(f"  [{company_code}] API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {
        "company_code": company_code,
        "as_of": str(as_of),
        "total_score": None,
        "prosperity_level": "评估失败",
        "trend": "未知",
        "_status": "failed",
        "_error": "达到最大重试次数",
    }


def _score_to_level(score: int) -> str:
    """按总分映射景气等级（与 Prompt 中的标准保持一致）。"""
    if score >= 80:
        return "高景气"
    if score >= 65:
        return "较高景气"
    if score >= 45:
        return "中性"
    if score >= 25:
        return "较低景气"
    return "低景气"


# ============================================================
# 6. 批量评估函数（并发）
# ============================================================

def batch_evaluate(
    companies: list[dict],
    as_of: str = DEFAULT_AS_OF,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """
    批量评估多家公司在指定时点的业务景气度预期。

    参数
    ----
    companies : list[dict]
        每项格式：{"code": "公司代码", "description": "主营业务描述",
                   "extra": "可选补充信息"}
    as_of : str
        评估时点，默认 2021
    max_workers : int
        并发线程数

    返回
    ----
    pd.DataFrame，按 total_score 降序排列
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                evaluate_prosperity,
                c["code"],
                c["description"],
                as_of,
                c.get("extra", ""),
            ): c["code"]
            for c in companies
        }

        for future in as_completed(future_map):
            code = future_map[future]
            try:
                result = future.result()
                result["company_code"] = code
                results.append(result)
                score = result.get("total_score", "N/A")
                level = result.get("prosperity_level", "N/A")
                trend = result.get("trend", "N/A")
                print(f"  ✓ {code}: {score} 分 ({level} / {trend})")
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {code}: 异常 - {e}")
                results.append({
                    "company_code": code,
                    "as_of": str(as_of),
                    "total_score": None,
                    "prosperity_level": "评估失败",
                    "trend": "未知",
                    "_status": "failed",
                })

    df = pd.DataFrame(results)
    if "total_score" in df.columns:
        df = df.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

    return df

## 增加循环评估
def _is_failed_result(result: dict) -> bool:
    """判断 evaluate_prosperity 的返回结果是否表示评估失败。"""
    if not isinstance(result, dict):
        return True
    return (
        result.get("_status") == "failed"
        or result.get("prosperity_level") == "评估失败"
    )


def batch_evaluate_iterations(
    companies: list[dict],
    as_of: str = DEFAULT_AS_OF,
    max_workers: int = MAX_WORKERS,
    max_retries: int | None = None,
) -> pd.DataFrame:
    """
    批量评估多家公司在指定时点的业务景气度预期。

    参数
    ----
    companies : list[dict]
        每项格式：{"code": "公司代码", "description": "主营业务描述",
                   "extra": "可选补充信息"}
    as_of : str
        评估时点，默认 2021
    max_workers : int
        并发线程数
    max_retries : int | None
        失败后的最大额外重试轮数。
        None 表示一直重试直到没有失败项，慎用，可能死循环。
        0 表示不重试。
        3 表示初次评估失败后，最多再重试 3 轮。

    返回
    ----
    pd.DataFrame，按 total_score 降序排列
    """
    final_results: list[dict] = []
    pending = list(companies)
    attempt = 0

    while pending:
        attempt += 1
        failed_items: list[tuple[dict, Exception]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_prosperity,
                    company["code"],
                    company["description"],
                    as_of,
                    company.get("extra", ""),
                ): company
                for company in pending
            }

            for future in as_completed(future_map):
                company = future_map[future]
                code = company["code"]

                try:
                    result = future.result()

                    # 不仅捕获异常，也识别返回结果中标记为失败的情况
                    if _is_failed_result(result):
                        if isinstance(result, dict):
                            msg = result.get("_error", "评估结果标记为失败")
                        else:
                            msg = "评估结果不是字典"
                        raise RuntimeError(msg)

                    result["company_code"] = code
                    final_results.append(result)

                    score = result.get("total_score", "N/A")
                    level = result.get("prosperity_level", "N/A")
                    trend = result.get("trend", "N/A")
                    print(f"  ✓ {code}: {score} 分 ({level} / {trend})")

                except Exception as e:  # noqa: BLE001
                    print(f"  ✗ {code}: 第 {attempt} 轮评估失败 - {e}")
                    failed_items.append((company, e))

        # 没有失败项，结束
        if not failed_items:
            break

        # 达到最大重试轮数，把仍然失败的公司写入最终结果
        if max_retries is not None and attempt > max_retries:
            for company, error in failed_items:
                code = company["code"]
                final_results.append({
                    "company_code": code,
                    "as_of": str(as_of),
                    "total_score": None,
                    "prosperity_level": "评估失败",
                    "trend": "未知",
                    "_status": "failed",
                    "_attempts": attempt,
                    "_error": str(error),
                })
            break

        # 只重试本轮失败的公司
        pending = [company for company, _ in failed_items]
        print(f"  ↻ 第 {attempt} 轮结束，仍有 {len(pending)} 家公司失败，准备重试...")

    df = pd.DataFrame(final_results)
    if "total_score" in df.columns:
        df = df.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

    return df






def flatten_results(df: pd.DataFrame) -> pd.DataFrame:
    """把嵌套的 dimensions / 列表字段展开为扁平的 CSV 友好结构。"""
    if df.empty:
        return df

    out = df.copy()

    dim_keys = ["demand", "supply_competition",
                "policy_environment", "profit_growth"]
    for dim in dim_keys:
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

    out["summary"] = out.apply(lambda r: r.get("summary", ""), axis=1)
    out["key_drivers"] = out.apply(
        lambda r: ", ".join(r.get("key_drivers", []))
        if isinstance(r.get("key_drivers"), list) else "",
        axis=1,
    )
    out["risk_factors"] = out.apply(
        lambda r: ", ".join(r.get("risk_factors", []))
        if isinstance(r.get("risk_factors"), list) else "",
        axis=1,
    )

    out = out.drop(columns=["dimensions"], errors="ignore")
    return out


# ============================================================
# 7. 主流程示例
# ============================================================

if __name__ == "__main__":

    AS_OF = "2021"   # ← 想换时点，改这里即可（如 "2024"）

    # ---------- 7.1 单条评估示例 ----------
    print("=" * 60)
    print(f"单条评估示例（评估时点：{AS_OF}）")
    print("=" * 60)

    single_result = evaluate_prosperity(
        company_code="688XXX",
        description=(
            "公司主要从事集成电路芯片的设计、研发与销售，"
            "产品覆盖模拟芯片、射频前端芯片，应用于5G通信、"
            "汽车电子和物联网领域。公司拥有自主研发的芯片架构，"
            "并在先进封装技术上持续投入。"
        ),
        as_of=AS_OF,
    )
    print(json.dumps(single_result, ensure_ascii=False, indent=2))

    # ---------- 7.2 批量评估示例 ----------
    print("\n" + "=" * 60)
    print(f"批量评估示例（评估时点：{AS_OF}）")
    print("=" * 60)

    # 实际使用时，可以从 Excel/CSV 读取：
    #   df_input = pd.read_csv("companies.csv")
    #   companies = [
    #       {"code": row["公司代码"], "description": row["主营业务描述"]}
    #       for _, row in df_input.iterrows()
    #   ]
    companies_to_evaluate = [
        {
            "code": "A001",
            "description": "公司专注于光伏逆变器与储能系统的研发制造，产品销往全球，海外收入占比超过60%。"
        },
        {
            "code": "B001",
            "description": "公司主营业务为传统燃煤发电，同时涉足少量光伏电站运营。"
        },
        {
            "code": "C001",
            "description": "公司从事创新药研发，聚焦肿瘤免疫治疗领域的单克隆抗体和CAR-T细胞疗法，拥有多个临床阶段管线。"
        },
        {
            "code": "D001",
            "description": "公司主营商业航天运载火箭的研制与发射服务，同时布局卫星互联网星座建设。"
        },
        {
            "code": "E001",
            "description": "公司主要从事房地产开发与销售，兼营物业管理服务。"
        },
        {
            "code": "F001",
            "description": "公司为消费电子品牌提供精密结构件与组装服务，主要客户为海外头部手机厂商。"
        },
    ]

    df_result = batch_evaluate_iterations(companies_to_evaluate, as_of=AS_OF)

    # ---------- 7.3 输出结果 ----------
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    display_cols = ["company_code", "total_score", "prosperity_level", "trend"]
    available_cols = [c for c in display_cols if c in df_result.columns]
    print(df_result[available_cols].to_string(index=False))

    # 保存完整结果到 CSV（展开嵌套字段）
    if not df_result.empty:
        output_df = flatten_results(df_result)
        out_path = f"prosperity_forecast_{AS_OF}.csv"
        output_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n完整结果已保存至 {out_path}")

    # ---------- 7.4 快速统计 ----------
    print("\n" + "=" * 60)
    print("景气度分布统计")
    print("=" * 60)
    if "prosperity_level" in df_result.columns:
        print(df_result["prosperity_level"].value_counts().to_string())
    if "trend" in df_result.columns:
        print("\n趋势分布：")
        print(df_result["trend"].value_counts().to_string())





