
import os
import re
import json
from openai import OpenAI


# ----------------------------------------------------------------------
# 客户端
# ----------------------------------------------------------------------
def _get_client() -> OpenAI:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未找到环境变量 DEEPSEEK_API_KEY，请先配置你的 DeepSeek API Key")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


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


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def is_ai_compute(
    business_description: str,
    model: str = "deepseek-chat",
    timeout: float = 60.0,
    max_retries: int = 2,
) -> bool:
    """
    输入：一家公司的业务描述（中文/英文均可）
    输出：bool —— 该公司是否属于 AI 算力产业链
    """
    if not business_description or not business_description.strip():
        return False

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
            return _parse_bool(content)
        except Exception as e:          # 网络抖动 / 限流 / JSON 解析失败，重试
            last_err = e

    raise RuntimeError(f"调用 DeepSeek 判定失败：{last_err}")


def _parse_bool(content: str) -> bool:
    """从模型返回中稳健地解析出布尔值"""
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
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "y", "1", "是", "属于"}

    # 最后兜底：直接在文本里找 true / false
    low = content.lower()
    if "true" in low:
        return True
    return False


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
    