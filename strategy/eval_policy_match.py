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
MAX_WORKERS = config.get("MAX_WORKERS", 3)  # 并发线程数（控制 API 速率，避免触发限流）

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
    base_url=base_url
)

# ============================================================
# 2. 产业政策文本（可替换为从文件/数据库读取）
# ============================================================
INDUSTRY_POLICY_TEXT = """
国家产业政策重点方向（2021年，依据“十四五”规划纲要及政府工作报告）
一、新兴支柱产业（优先支持）
    集成电路：芯片设计、制造、封装测试，半导体设备与材料。纲要明确将集成电路列为前沿领域国家重大科技项目方向之一，并纳入先进制造业集群重点培育范围。
    航空航天：商业航天、卫星应用、海洋工程装备。纲要提出建设商业航天发射场，推动航空航天产业创新发展，打造战略性全局性产业链。
    生物医药：创新药、高端医疗器械、基因治疗、细胞治疗、免疫治疗。纲要提出做大做强生物经济，加快生物医药、生物育种、生物材料、生物能源等产业发展，加强基因治疗、细胞治疗等技术的深度研发与通用化应用。
    高端装备制造：工业机器人、先进轨道交通装备、高端数控机床、工程机械、先进电力装备。纲要要求培育先进制造业集群，推动上述产业创新发展。
    新能源汽车：整车制造、动力电池及关键零部件，被列为战略性新兴产业重点领域之一。
    新能源：光伏发电、风电（含海上风电）、核电、储能。纲要提出大力提升风电、光伏发电规模，有序发展海上风电，安全稳妥推动沿海核电建设，建设多能互补的清洁能源基地。
    新材料：先进结构材料、功能性材料、半导体材料，作为战略性新兴产业的重要组成部分加快关键核心技术创新应用。
二、未来产业（前瞻布局）
纲要明确提出“前瞻谋划未来产业”，在以下前沿科技和产业变革领域组织实施未来产业孵化与加速计划：
    类脑智能：类脑计算芯片、神经形态计算、脑机融合技术
    量子信息：量子计算、量子通信、量子精密测量。纲要将量子信息列入国家重大科技项目前沿领域。
    基因技术：基因编辑、基因检测、合成生物学
    未来网络：6G通信、卫星互联网、确定性网络
    深海空天开发：深海探测装备、深地深海资源开发、空天科技
    氢能与储能：氢能制储运用全链条、新型储能技术（含压缩空气储能、飞轮储能等）
三、传统产业改造升级
纲要提出“推动制造业优化升级”，重点包括：
    原材料产业：石化、钢铁、有色、建材等行业布局优化和结构调整
    消费品工业：扩大轻工、纺织等优质产品供给，推动“增品种、提品质、创品牌”
    绿色制造：加快化工、造纸等重点行业企业改造升级，完善绿色制造体系
    产业基础再造：补齐基础零部件及元器件、基础软件、基础材料、基础工艺和产业技术基础等瓶颈短板。2021年政府工作报告亦提出“实施好产业基础再造工程”。
    重大技术装备：高铁、电力装备、新能源、船舶等领域的全产业链竞争力巩固提升
    智能制造：深入实施智能制造工程，建设智能制造示范工厂，完善智能制造标准体系
四、数字经济与人工智能
纲要将“加快数字化发展 建设数字中国”单独成篇，2021年政府工作报告亦要求“加大5G网络和千兆光网建设力度，丰富应用场景”。重点方向包括：
    数字产业化：培育壮大人工智能、大数据、区块链、云计算、网络安全等新兴数字产业，提升通信设备、核心电子元器件、关键软件等产业水平。纲要明确将发展云计算、大数据、物联网、工业互联网、区块链、人工智能、虚拟现实和增强现实等七大数字经济重点产业。
    人工智能：聚焦人工智能关键算法、基础理论、基础算法、装备材料等研发突破与迭代应用，加快布局神经芯片等前沿技术。
    高端芯片与操作系统：聚焦高端芯片、操作系统、传感器等关键领域，加强通用处理器、云计算系统和软件核心技术一体化研发。
    算力基础设施与数据要素：构建基于5G的应用场景和产业生态；加快建立数据资源产权、交易流通、跨境传输和安全保护等基础制度和标准规范，培育规范的数据交易平台和市场主体。
    工业互联网与数字化转型：发展工业互联网，搭建更多共性技术研发平台；推进制造业数字化转型，培育推广一批数字化解决方案。
    数字化应用场景：在智能交通、智慧物流、智慧能源、智慧医疗等重点领域开展试点示范，深入推进服务业数字化转型。
"""

# INDUSTRY_POLICY_TEXT = """
# 国家产业政策重点方向（2026年，依据"十五五"规划纲要及政府工作报告）：
#
# 一、新兴支柱产业（优先支持）：
# - 集成电路：芯片设计、制造、封装测试、半导体设备与材料
# - 航空航天：商业航天、国产大飞机、低空装备
# - 生物医药：创新药、高端医疗器械、基因治疗
# - 低空经济：eVTOL、无人机物流、低空基础设施
# - 新型储能：锂电储能、钠电储能、氢能储运、压缩空气储能
# - 智能机器人：人形机器人、工业机器人、核心零部件
#
# 二、未来产业（前瞻布局）：
# - 量子科技、生物制造、氢能与核聚变能、脑机接口、具身智能、6G通信
#
# 三、传统产业改造升级：
# - 高端新材料、基础零部件和元器件、大型邮轮、LNG运输船、CR450动车组、农机装备、燃气轮机
#
# 四、数字经济与人工智能：
# - 人工智能大模型、算力基础设施、数据要素、工业互联网、智能制造
# """

# ============================================================
# 3. Prompt 模板
# ============================================================

SYSTEM_PROMPT = """你是一位资深的产业政策研究分析师，专门评估上市公司主营业务与国家产业政策的契合程度。

你的评估需要严格遵循以下规则：
1. 只基于给定的公司主营业务描述和产业政策文本进行判断，不得编造信息。
2. 从以下四个维度进行综合评估：
   - 政策方向匹配：主营业务是否直接属于政策明确支持或鼓励的产业方向
   - 技术含量匹配：公司业务是否涉及政策强调的关键技术或核心环节
   - 产业链地位：公司在政策重点产业链中的位置（核心/配套/边缘）
   - 发展前景：结合政策支持力度判断业务未来的政策红利空间
3. 最终给出 0-100 的契合度总分，以及对应的契合等级。
4. 必须严格以 JSON 格式输出，不得包含 JSON 之外的任何文字。"""

USER_PROMPT_TEMPLATE = """请评估以下上市公司主营业务与国家产业政策的契合程度。

【公司代码】{company_code}
【主营业务描述】
{description}

【国家产业政策文本】
{policy_text}

请严格按照以下 JSON 结构输出（不要输出任何其他内容）：

{{
  "company_code": "公司代码",
  "total_score": 0-100的整数,
  "match_level": "高度契合/中度契合/轻度契合/不契合",
  "dimensions": {{
    "policy_direction": {{
      "score": 0-25的整数,
      "reason": "政策方向匹配的简要理由"
    }},
    "tech_content": {{
      "score": 0-25的整数,
      "reason": "技术含量匹配的简要理由"
    }},
    "industry_chain": {{
      "score": 0-25的整数,
      "reason": "产业链地位评估的简要理由"
    }},
    "growth_prospect": {{
      "score": 0-25的整数,
      "reason": "发展前景评估的简要理由"
    }}
  }},
  "summary": "一句话总结评估结论",
  "policy_keywords": ["匹配到的政策关键词1", "关键词2"]
}}

评分标准：
- 高度契合：80-100分，主营业务直接属于政策明确支持的新兴支柱产业或未来产业
- 中度契合：60-79分，主营业务属于政策鼓励方向，但非最核心领域
- 轻度契合：40-59分，业务与政策有部分关联，但关联度有限
- 不契合：0-39分，主营业务与政策方向无明显关联"""

# ============================================================
# 4. 核心评估函数
# ============================================================

def evaluate_company(
    company_code: str,
    description: str,
    policy_text: str = INDUSTRY_POLICY_TEXT,
    max_retries: int = 3,
) -> dict:
    """
    调用 LLM 评估单家公司的政策契合度。

    返回结构化 dict，包含评分、等级、各维度理由等。
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_code=company_code,
        description=description,
        policy_text=policy_text,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,          # 低温度保证输出稳定
                max_tokens=3000,
                stream=False,
            )

            raw_text = response.choices[0].message.content.strip()

            # 清理可能的 markdown 代码块标记
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            result = json.loads(raw_text)

            # 基本校验
            if "total_score" not in result:
                raise ValueError("LLM 返回缺少 total_score 字段")

            # 确保公司代码字段存在
            result.setdefault("company_code", company_code)

            # 归一化：确保总分等于四个维度之和（防止 LLM 算错）
            dims = result.get("dimensions", {})
            dim_sum = sum(
                dims.get(k, {}).get("score", 0)
                for k in ["policy_direction", "tech_content",
                          "industry_chain", "growth_prospect"]
            )
            if dim_sum > 0 and abs(dim_sum - result["total_score"]) > 5:
                # 差异超过5分则以维度之和为准
                result["total_score"] = dim_sum

            result["_status"] = "success"
            return result

        except json.JSONDecodeError as e:
            print(f"  [{company_code}] JSON 解析失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            print(f"  [{company_code}] API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return {
        "company_code": company_code,
        "total_score": None,
        "match_level": "评估失败",
        "_status": "failed",
        "_error": "达到最大重试次数",
    }

# ============================================================
# 5. 批量评估函数（并发）
# ============================================================

def batch_evaluate(
    companies: list[dict],
    policy_text: str = INDUSTRY_POLICY_TEXT,
    max_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """
    批量评估多家公司。

    参数
    ----
    companies : list[dict]
        每项格式：{"code": "公司代码", "description": "主营业务描述"}
    policy_text : str
        产业政策文本，可针对不同行业替换
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
                evaluate_company,
                c["code"],
                c["description"],
                policy_text,
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
                level = result.get("match_level", "N/A")
                print(f"  ✓ {code}: {score} 分 ({level})")
            except Exception as e:
                print(f"  ✗ {code}: 异常 - {e}")
                results.append({
                    "company_code": code,
                    "total_score": None,
                    "match_level": "评估失败",
                    "_status": "failed",
                })

    df = pd.DataFrame(results)
    if "total_score" in df.columns:
        df = df.sort_values(
            "total_score", ascending=False, na_position="last"
        ).reset_index(drop=True)

    return df

# ============================================================
# 6. 主流程示例
# ============================================================

if __name__ == "__main__":

    # ---------- 6.1 单条评估示例 ----------
    print("=" * 60)
    print("单条评估示例")
    print("=" * 60)

    single_result = evaluate_company(
        company_code="688XXX",
        description=(
            "公司主要从事集成电路芯片的设计、研发与销售，"
            "产品覆盖模拟芯片、射频前端芯片，应用于5G通信、"
            "汽车电子和物联网领域。公司拥有自主研发的芯片架构，"
            "并在先进封装技术上持续投入。"
        ),
    )
    print(json.dumps(single_result, ensure_ascii=False, indent=2))

    # ---------- 6.2 批量评估示例 ----------
    print("\n" + "=" * 60)
    print("批量评估示例")
    print("=" * 60)

    # 构建待评估公司列表
    # 实际使用时，可以从 Excel/CSV 读取：
    #   df_input = pd.read_csv("companies.csv")
    #   companies = [
    #       {"code": row["公司代码"], "description": row["主营业务描述"]}
    #       for _, row in df_input.iterrows()
    #   ]
    companies_to_evaluate = [
        {
            "code": "A001",
            "description": "公司专注于人形机器人的整机设计与核心零部件（谐波减速器、伺服电机）的研发制造，产品面向工业制造和服务场景。"
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
    ]

    df_result = batch_evaluate(companies_to_evaluate)

    # ---------- 6.3 输出结果 ----------
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