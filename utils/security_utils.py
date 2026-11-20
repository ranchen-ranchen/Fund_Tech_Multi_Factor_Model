"""敏感信息脱敏工具。

用于日志、异常和终端输出，避免 DeepSeek API key、Bearer token
或带凭据的 URL 被原样打印。
"""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

# 常见密钥/凭据形态。替换时保留少量上下文（如 "Bearer "），便于排查问题。
_PATTERNS = (
    # DeepSeek / OpenAI 风格：sk-xxxxxxxx
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    # Authorization: Bearer xxxxx
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1***"),
    # Authorization: xxxxx / authorization=xxxxx
    (re.compile(r"(?i)(authorization\s*[:=]\s*)\S+"), r"\1***"),
    # api_key=xxxxx / x-api-key: xxxxx / access_token=xxxxx
    (
        re.compile(
            r"(?i)(\b(?:x-api-key|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?)"
            r"[^\s'\"&,;]+"
        ),
        r"\1***",
    ),
    # URL 查询参数中的 token/api_key
    (re.compile(r"(?i)([?&](?:api[_-]?key|access_token|token)=)[^&\s]+"), r"\1***"),
    # URL userinfo：https://user:pass@host/...
    (re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@"), r"\1***:***@"),
)


def redact_secrets(value: object, secret: Optional[str] = None) -> str:
    """返回经过脱敏的文本。

    Args:
        value: 任意可转为字符串的对象（通常是异常或日志文本）。
        secret: 已知的具体密钥值；长度 >= 8 时会做精确替换，
            用于覆盖不符合常见前缀格式的自定义 key。
    """
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 - 脱敏函数本身不能抛异常
        return "***"

    if secret and len(secret) >= 8:
        text = text.replace(secret, "***")

    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def safe_endpoint(url: object) -> str:
    """只返回 scheme://host[:port]，去掉 userinfo、路径、查询串和片段。

    用于打印 base_url 等接口地址；无法解析时返回 "***"。
    """
    try:
        parts = urlsplit(str(url).strip())
        if not parts.scheme or not parts.hostname:
            return "***"

        host = parts.hostname
        if ":" in host:  # IPv6 地址需要补回方括号
            host = f"[{host}]"
        port = parts.port
        if port:
            host = f"{host}:{port}"

        return urlunsplit((parts.scheme, host, "", "", ""))
    except (TypeError, ValueError):
        return "***"
