
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.security_utils import redact_secrets  # noqa: E402


# ----------------------------------------------------------------------
# 客户端（进程内复用）
# ----------------------------------------------------------------------
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到环境变量 DEEPSEEK_API_KEY，请先配置你的 DeepSeek API Key")
    _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client


# ----------------------------------------------------------------------
# 判定用的 System Prompt
# ----------------------------------------------------------------------
_SYSTEM_PROMPT = """你是一名资深产业研究员，负责判断一家公司的业务是否属于「AI算力产业链」。

【AI算力产业链】包括但不限于：
- 上游：AI芯片/GPU/ASIC/FPGA、HBM与存储、光模块/光芯片、高速PCB、铜连接、液冷、服务器电源、交换机、先进封装、算力芯片相关的半导体设备与材料、EDA 等
- 中游：AI服务器、智算中心/IDC 建设与运营、云算力服务、网络设备、算力租赁
- 下游：大模型训练与推理服务、以算力为核心成本或卖点的 AI 应用

【判定规则】
1. 只要公司主营业务（或重要收入来源）落在上述产业链的核心环节，即判为 true。
2. 纯传统业务（房地产、白酒、银行保险、传统燃油车整车、传统火电、煤炭开采、纯消费品、医药、传媒等）判为 false。
3. 仅为上述产业链提供通用软件/通用服务，或只有概念性布局而无实质业务收入的，倾向于 false。
4. 不确定时，以「是否直接参与产业链核心环节的产品或服务提供」为准。

请严格以 JSON 格式输出，不要输出任何其他内容，格式为：
{"is_in_chain": true 或 false, "reason": "一句话理由"}"""


DEFAULT_MODEL = "deepseek-chat"

# 提示词一旦修改，哈希值自动变化，从而让缓存自动失效；
# 模型变更则由缓存键中的 model 字段区分。
PROMPT_VERSION = hashlib.sha1(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def classify_ai_compute(
    business_description: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> dict:
    """
    输入：一家公司的业务描述（中文/英文均可）
    输出：{"is_in_chain": bool, "reason": str}
    """
    if not business_description or not business_description.strip():
        return {"is_in_chain": False, "reason": "业务描述为空"}

    client = _get_client()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"公司业务描述：\n{business_description.strip()}"},
    ]

    last_err = None
    for _ in range(max(1, max_retries)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                timeout=timeout,
            )
            content = (resp.choices[0].message.content or "").strip()
            return _parse_result(content)
        except Exception as e:          # 网络抖动 / 限流 / JSON 解析失败，重试
            last_err = e

    raise RuntimeError(
        f"调用 DeepSeek 判定失败："
        f"{redact_secrets(last_err, getattr(client, 'api_key', None))}"
    )


def is_ai_compute(
    business_description: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> bool:
    """
    兼容旧接口：仅返回布尔值。
    需要判定理由时请使用 classify_ai_compute()。
    """
    return classify_ai_compute(
        business_description,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )["is_in_chain"]


def _parse_result(content: str) -> dict:
    """从模型返回中稳健地解析出 is_in_chain 与 reason。"""
    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.S)   # 兜底：抠出第一个 JSON 对象
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    if isinstance(data, dict):
        value = data.get("is_in_chain", data.get("result", data.get("answer")))
        if isinstance(value, bool):
            return {
                "is_in_chain": value,
                "reason": str(data.get("reason", "") or "").strip(),
            }
        if isinstance(value, (int, float)):
            return {
                "is_in_chain": bool(value),
                "reason": str(data.get("reason", "") or "").strip(),
            }
        if isinstance(value, str):
            return {
                "is_in_chain": value.strip().lower() in {"true", "yes", "y", "1", "是", "属于"},
                "reason": str(data.get("reason", "") or "").strip(),
            }

    # 最后兜底：直接在文本里找 true / false
    low = content.lower()
    return {
        "is_in_chain": "true" in low,
        "reason": content.strip()[:200],
    }


def _parse_bool(content: str) -> bool:
    """兼容旧接口：仅返回布尔值。"""
    return _parse_result(content)["is_in_chain"]


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "公司主营 400G/800G 高速光模块的研发与销售，客户为海外云计算厂商。",
        "公司主要从事磷酸铁锂正极材料及储能电池系统的生产。",
        "公司主营白酒酿造与销售，拥有多个知名白酒品牌。",
        "公司业务包括房地产开发、物业管理及商业运营。",
        "公司提供液冷服务器整机与智算中心建设运营服务。",
    ]

    for s in samples:
        print(f"[{'是' if is_ai_compute(s) else '否'}] {s}")
    