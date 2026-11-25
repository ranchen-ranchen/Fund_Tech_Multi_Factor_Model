"""Small formatting helpers shared by the dashboard views."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pct(value: Any, digits: int = 2, signed: bool = False) -> str:
    if not _finite(value):
        return "—"
    pattern = f"{{:+.{digits}f}}%" if signed else f"{{:.{digits}f}}%"
    return pattern.format(float(value) * 100.0)


def pp(value: Any, digits: int = 2, signed: bool = True) -> str:
    if not _finite(value):
        return "—"
    pattern = f"{{:+.{digits}f}} pp" if signed else f"{{:.{digits}f}} pp"
    return pattern.format(float(value) * 100.0)


def num(value: Any, digits: int = 4, signed: bool = False) -> str:
    if not _finite(value):
        return "—"
    pattern = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return pattern.format(float(value))


def integer(value: Any) -> str:
    if not _finite(value):
        return "—"
    return f"{int(round(float(value))):,}"


def date_only(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def truncate(text: Any, max_chars: int = 80) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def human_size(size_bytes: Any) -> str:
    if not _finite(size_bytes):
        return "—"
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [float(v) for v in values if _finite(v)]
    return sum(clean) / len(clean) if clean else None
