#!/usr/bin/env python3
"""把 Markdown 提示词文件发送给 DeepSeek（可开启联网搜索），并打印回复到屏幕。

用法：
    python knowledge_base/deepseek_prompt.py                 # 默认年份 2015，联网 + Q1–Q4
    python knowledge_base/deepseek_prompt.py --year 2024     # 调查 2024 年
    python knowledge_base/deepseek_prompt.py --no-quarters   # 联网但只发一次请求
    python knowledge_base/deepseek_prompt.py --no-search     # 关闭联网（默认单次）
    python knowledge_base/deepseek_prompt.py --verify-search # 只做联网能力自检

提示词模板支持 {year} 与 {next_year} 两个占位符，脚本会按 --year 自动替换。
例如 --year 2024 会把 {year} 换成 2024、{next_year} 换成 2025。

联网搜索说明（DeepSeek 官方文档）：
    - 使用 deepseek-flash 模型，在 tools 中声明 deepseek_web_search；
    - thinking 关闭时是普通联网搜索，开启时是「深度思考搜索」；
    - 是否搜索由模型判断，tool_choice 不能强制该工具；
    - 使用过联网工具的回复不能续写，因此联网场景默认按 Q1–Q4 分四次
      独立请求，每季度单独保存，不依赖续写。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIError, OpenAI

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.security_utils import redact_secrets, safe_endpoint  # noqa: E402

DEFAULT_PROMPT = BASE_DIR / "news_collect_prompt.md"
DEFAULT_YEAR = 2015

SEARCH_TOOL = {"type": "deepseek_web_search"}
SEARCH_KINDS = {"search", "deep_search", "image_search", "deep_image_search"}
PROBE_QUESTION = "请联网搜索：DeepSeek 全家桶应用 API 是哪一天发布的？请给出来源链接。"

QUARTERS = (
    ("Q1", "1–3 月"),
    ("Q2", "4–6 月"),
    ("Q3", "7–9 月"),
    ("Q4", "10–12 月"),
)
DEFAULT_QUARTER_TEMPLATE = (
    "本次只生成 {year} 年 {quarter}（{months}）部分："
    "直接给出该季度的事件清单并展开卡片。"
    "不要生成其他季度，完成本季度后立即停止。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 DeepSeek 执行 Markdown 提示词")
    parser.add_argument(
        "prompt_file",
        nargs="?",
        type=Path,
        default=DEFAULT_PROMPT,
        help=f"提示词 Markdown 文件（默认 {DEFAULT_PROMPT.name}）",
    )
    parser.add_argument(
        "-m", "--model", default="deepseek-flash", help="模型名（默认 deepseek-flash）"
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"调查年份（默认 {DEFAULT_YEAR}；会替换提示词中的 {{year}}/{{next_year}}）",
    )
    parser.add_argument(
        "-u",
        "--instruction",
        default="请按上述要求开始执行。",
        help="随提示词一起发送的用户消息（同样支持 {year}/{next_year}）",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="采样温度（默认 1.0；联网时服务端会忽略）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=65536,
        help="每次请求的最大输出 token 数（默认 65536；deepseek-flash 上限 393216）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="单次模式的保存文件；分季度模式下作为合并文件，季度文件写在其同目录",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="等待完整回复后一次性打印"
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否开启 deepseek_web_search 联网搜索（默认开启，用 --no-search 关闭）",
    )
    parser.add_argument(
        "--thinking",
        choices=["disabled", "enabled"],
        default="disabled",
        help="思考模式：disabled=关闭思考/普通联网搜索（默认），enabled=深度思考搜索",
    )
    parser.add_argument(
        "--quarters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按 Q1–Q4 分四次请求（联网时默认开启；用 --no-quarters 关闭）",
    )
    parser.add_argument(
        "--quarter-template",
        default=DEFAULT_QUARTER_TEMPLATE,
        help="分季度附加指令模板，可用 {year}、{next_year}、{quarter}、{months} 占位符",
    )
    parser.add_argument(
        "--verify-search",
        action="store_true",
        help="只做联网能力自检（发起一个小请求，验证服务端搜索是否真的执行），不运行提示词",
    )
    return parser.parse_args()


def field(obj: Any, name: str, default: Any = None) -> Any:
    """兼容 pydantic 对象与普通 dict 的字段读取。"""
    if obj is None:
        return default
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return default if value is None else value


def render_year_placeholders(text: str, year: int) -> str:
    """替换 {year}/{next_year}，其它花括号保持原样。"""
    return text.replace("{year}", str(year)).replace("{next_year}", str(year + 1))


def collect_evidence(
    server_tool_calls: Any, annotations: Any, usage: Any
) -> dict[str, Any]:
    """从一次响应中提取联网搜索证据。"""
    calls: list[Any] = []
    for step in server_tool_calls or []:
        calls.extend(field(step, "calls", []) or [])

    search_calls = [
        c
        for c in calls
        if field(c, "type") == "deepseek_web_search"
        or field(c, "kind") in SEARCH_KINDS
    ]

    queries: list[str] = []
    sources: dict[str, str] = {}
    errors: list[Any] = []
    for call in search_calls:
        action = field(call, "action", {}) or {}
        for query in field(action, "queries", []) or []:
            if query not in queries:
                queries.append(query)
        for source in field(call, "sources", []) or []:
            url = field(source, "url")
            if url:
                sources.setdefault(url, field(source, "title") or "")
        error = field(call, "error")
        if error:
            errors.append(error)

    citations: dict[str, str] = {}
    for annotation in annotations or []:
        if field(annotation, "type") != "url_citation":
            continue
        citation = field(annotation, "url_citation", {}) or {}
        url = field(citation, "url")
        if url:
            citations.setdefault(url, field(citation, "title") or "")

    tool_use = field(usage, "server_tool_use", {}) or {}
    tool_search = field(tool_use, "tool_search", None)
    if tool_search is None:
        tool_search = len(search_calls)

    return {
        "tool_search": int(tool_search),
        "tool_fetch": int(field(tool_use, "tool_fetch", 0) or 0),
        "tool_image_search": int(field(tool_use, "tool_image_search", 0) or 0),
        "search_calls": search_calls,
        "queries": queries,
        "sources": sources,
        "citations": citations,
        "errors": errors,
    }


def new_evidence() -> dict[str, Any]:
    return {
        "tool_search": 0,
        "tool_fetch": 0,
        "tool_image_search": 0,
        "search_calls": [],
        "queries": [],
        "sources": {},
        "citations": {},
        "errors": [],
    }


def merge_evidence(total: dict[str, Any], item: dict[str, Any]) -> None:
    """把单次请求的证据累加到总计里。"""
    for key in ("tool_search", "tool_fetch", "tool_image_search"):
        total[key] += int(item.get(key, 0) or 0)
    total["search_calls"].extend(item.get("search_calls", []))
    for query in item.get("queries", []):
        if query not in total["queries"]:
            total["queries"].append(query)
    for url, title in item.get("sources", {}).items():
        total["sources"].setdefault(url, title)
    for url, title in item.get("citations", {}).items():
        total["citations"].setdefault(url, title)
    total["errors"].extend(item.get("errors", []))


def print_verification(
    evidence: dict[str, Any], enabled: bool, prefix: str = "[联网验证] "
) -> None:
    """把联网证据打印到 stderr，stdout 只留模型正文。"""
    if not enabled:
        print(f"{prefix}未开启（--no-search）", file=sys.stderr)
        return

    print(f"{prefix}工具声明：deepseek_web_search", file=sys.stderr)
    print(f"{prefix}服务端搜索次数：{evidence['tool_search']}", file=sys.stderr)
    print(f"{prefix}返回来源数：{len(evidence['sources'])}", file=sys.stderr)
    if evidence["queries"]:
        print(f"{prefix}查询词：{' | '.join(evidence['queries'][:6])}", file=sys.stderr)
    for url, title in list(evidence["sources"].items())[:3]:
        print(f"  - {title or '(无标题)'} {url}", file=sys.stderr)
    print(
        f"{prefix}回答引用：{len(evidence['citations'])} 个来源 URL",
        file=sys.stderr,
    )
    if evidence["errors"]:
        print(f"{prefix}搜索错误：{evidence['errors']}", file=sys.stderr)

    if evidence["tool_search"] > 0 and evidence["sources"]:
        print(f"{prefix}结论：✅ 联网搜索已实际执行并返回结果", file=sys.stderr)
    elif evidence["tool_search"] > 0:
        print(f"{prefix}结论：⚠️ 搜索已执行，但没有返回来源", file=sys.stderr)
    else:
        print(
            f"{prefix}结论：⚠️ 工具已声明，但本轮模型未触发搜索"
            "（可用 --verify-search 主动自检）",
            file=sys.stderr,
        )


def print_search_event(event: Any) -> None:
    """流式输出时打印搜索过程（stderr）。"""
    event_type = field(event, "type")
    call = field(event, "call", {}) or {}
    action = field(call, "action", {}) or {}

    if event_type == "call_started":
        if field(action, "type") == "search":
            queries = field(action, "queries", []) or []
            print(
                f"[联网] 搜索 {len(queries)} 个查询：{' | '.join(queries[:4])}",
                file=sys.stderr,
                flush=True,
            )
        elif field(action, "type") == "open_page":
            print(f"[联网] 打开网页：{field(action, 'url', '')}", file=sys.stderr, flush=True)
        elif field(action, "type") == "find_in_page":
            print(f"[联网] 页内查找：{field(action, 'pattern', '')}", file=sys.stderr, flush=True)
    elif event_type == "call_completed":
        status = field(call, "status")
        sources = field(call, "sources", []) or []
        if status == "completed":
            print(f"[联网] 完成：{len(sources)} 条来源", file=sys.stderr, flush=True)
        else:
            print(
                f"[联网] 调用失败：{redact_secrets(field(call, 'error', ''))}",
                file=sys.stderr,
                flush=True,
            )


def make_client() -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("错误：未找到 DEEPSEEK_API_KEY，请在项目 .env 中配置。")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=1800.0, max_retries=2)


def build_messages(prompt: str, instruction: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": instruction},
    ]


def build_quarter_instruction(
    base: str, quarter: str, months: str, template: str, year: int
) -> str:
    requirement = render_year_placeholders(template, year)
    requirement = requirement.replace("{quarter}", quarter).replace("{months}", months)
    base = render_year_placeholders(base, year).strip()
    return f"{base}\n\n【分批要求】{requirement}" if base else requirement


def resolve_quarter_outputs(
    args: argparse.Namespace, prompt_path: Path
) -> tuple[Path, Path, str, str]:
    """返回 (输出目录, 合并文件, 文件主名, 扩展名)。"""
    if args.output is not None:
        combined_path = args.output
        out_dir = combined_path.parent
        stem = combined_path.stem
        suffix = combined_path.suffix or ".md"
    else:
        out_dir = BASE_DIR / "output" / f"{prompt_path.stem}_{args.year}"
        stem = f"{prompt_path.stem}_{args.year}"
        suffix = ".md"
        combined_path = out_dir / f"{stem}_all{suffix}"
    return out_dir, combined_path, stem, suffix


def run_completion(
    client: OpenAI,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    *,
    label: str = "",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """执行一次请求，流式打印并增量保存；返回正文、结束原因和联网证据。"""
    parts: list[str] = []
    server_tool_calls: Any = []
    annotations: Any = []
    usage: Any = None
    finish_reason: str | None = None
    error: str | None = None
    tag = label or "请求"

    out_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_path.open("w", encoding="utf-8")

    def emit(piece: str) -> None:
        if not piece:
            return
        parts.append(piece)
        print(piece, end="", flush=True)
        if out_file is not None:
            out_file.write(piece)
            out_file.flush()

    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
    }
    if args.search:
        kwargs["tools"] = [SEARCH_TOOL]
    if args.model == "deepseek-flash":
        # 显式传递 thinking，避免关闭联网时模型默认开启思考、吃掉输出预算。
        kwargs["extra_body"] = {"thinking": {"type": args.thinking}}
    if not args.search:
        # 联网时 temperature 会被服务端忽略，因此仅在关闭联网时传递。
        kwargs["temperature"] = args.temperature

    try:
        if args.no_stream:
            response = client.chat.completions.create(**kwargs, stream=False)
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            text = field(choice.message, "content", "") or ""
            emit(text)
            print(flush=True)
            server_tool_calls = field(choice.message, "server_tool_calls", []) or []
            annotations = field(choice.message, "annotations", []) or []
            usage = response.usage
        else:
            stream = client.chat.completions.create(**kwargs, stream=True)
            for chunk in stream:
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason

                event = getattr(choice, "server_tool_event", None)
                if event is not None:
                    print_search_event(event)

                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                emit(field(delta, "content", "") or "")
                step_calls = field(delta, "server_tool_calls")
                if step_calls:
                    server_tool_calls = step_calls
                step_annotations = field(delta, "annotations")
                if step_annotations:
                    annotations = step_annotations
            print(flush=True)
    except APIError as exc:
        error = redact_secrets(exc, getattr(client, "api_key", None))
        print(f"\n[错误][{tag}] API 调用失败：{error}", file=sys.stderr)
    finally:
        if out_file is not None:
            if parts and not parts[-1].endswith("\n"):
                out_file.write("\n")
            out_file.flush()
            out_file.close()

    text = "".join(parts)
    evidence = collect_evidence(server_tool_calls, annotations, usage)
    prefix = f"[联网验证][{label}] " if label else "[联网验证] "
    if error:
        print(f"{prefix}跳过：本次请求失败", file=sys.stderr)
    else:
        print_verification(evidence, enabled=args.search, prefix=prefix)

    if finish_reason == "length":
        print(
            f"[警告][{tag}] 回答达到 max_tokens={args.max_tokens} 上限，"
            f"已被截断；已保留 {len(text)} 字符。"
            "请提高 --max-tokens，或缩小该季度范围后重跑。",
            file=sys.stderr,
        )
    elif finish_reason == "content_filter":
        print(f"[警告][{tag}] 回答命中内容审核，可能不完整。", file=sys.stderr)

    if output_path is not None:
        state = "部分内容已保存" if error else "已保存"
        print(f"[保存][{tag}] {state}：{output_path}", file=sys.stderr)

    return {
        "text": text,
        "finish_reason": finish_reason,
        "evidence": evidence,
        "usage": usage,
        "error": error,
    }


def run_quarterly(
    client: OpenAI, args: argparse.Namespace, prompt: str, prompt_path: Path
) -> int:
    """联网长回答模式：Q1–Q4 四次独立请求，每季度单独保存并合并。"""
    out_dir, combined_path, stem, suffix = resolve_quarter_outputs(args, prompt_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[分季度] 调查年份：{args.year}", file=sys.stderr)
    print(f"[分季度] 输出目录：{out_dir}", file=sys.stderr)
    print(f"[分季度] 合并文件：{combined_path}", file=sys.stderr)
    print(
        f"[分季度] 每季度最多输出 {args.max_tokens} token，共 4 次独立请求",
        file=sys.stderr,
    )

    total = new_evidence()
    failed: list[str] = []
    truncated: list[str] = []
    blocked: list[str] = []

    combined_file = combined_path.open("w", encoding="utf-8")
    try:
        mode = f"联网（{args.thinking} 思考）" if args.search else "未联网"
        combined_file.write(f"# {stem}（分季度生成）\n\n")
        combined_file.write(
            f"> 调查年份：{args.year}；模型：{args.model}；{mode}；"
            "每季度独立请求，不使用续写。\n\n"
        )
        combined_file.flush()

        for quarter, months in QUARTERS:
            instruction = build_quarter_instruction(
                args.instruction, quarter, months, args.quarter_template, args.year
            )
            messages = build_messages(prompt, instruction)
            quarter_path = out_dir / f"{stem}_{quarter}{suffix}"

            print("\n" + "=" * 60, file=sys.stderr)
            print(f"[分季度] 开始 {quarter}（{months}）", file=sys.stderr)
            print("=" * 60, file=sys.stderr)

            result = run_completion(
                client,
                args,
                messages,
                label=quarter,
                output_path=quarter_path,
            )
            merge_evidence(total, result["evidence"])

            if result["error"]:
                failed.append(quarter)
            if result["finish_reason"] == "length":
                truncated.append(quarter)
            if result["finish_reason"] == "content_filter":
                blocked.append(quarter)

            combined_file.write(f"## {quarter}（{months}）\n\n")
            combined_file.write(result["text"].strip() or "（本季度无内容）")
            combined_file.write("\n")
            if result["finish_reason"] == "length":
                combined_file.write(
                    "\n> [注意] 本季度回答达到 max_tokens 上限，可能被截断。\n"
                )
            if result["finish_reason"] == "content_filter":
                combined_file.write("\n> [注意] 本季度回答命中内容审核。\n")
            if result["error"]:
                combined_file.write(f"\n> [生成失败] {result['error']}\n")
            combined_file.write("\n")
            combined_file.flush()

            print(
                f"[分季度] {quarter} 结束：{len(result['text'])} 字符，"
                f"finish_reason={result['finish_reason']}",
                file=sys.stderr,
            )
    finally:
        combined_file.close()

    print("\n" + "=" * 60, file=sys.stderr)
    print_verification(total, enabled=args.search, prefix="[联网验证][汇总] ")
    print(f"[分季度] 合并文件已保存：{combined_path}", file=sys.stderr)
    if failed:
        print(f"[分季度] 失败季度：{', '.join(failed)}", file=sys.stderr)
    if truncated:
        print(
            f"[分季度] 截断季度：{', '.join(truncated)}；"
            "请提高 --max-tokens 或单独重跑这些季度。",
            file=sys.stderr,
        )
    if blocked:
        print(f"[分季度] 内容审核拦截季度：{', '.join(blocked)}", file=sys.stderr)
    if not failed and not truncated and not blocked:
        print("[分季度] 四个季度全部完成。", file=sys.stderr)

    return 1 if (failed or truncated or blocked) else 0


def run_search_probe(client: OpenAI, args: argparse.Namespace) -> int:
    """主动自检：用一个依赖新信息的提问，验证服务端搜索真的执行。"""
    print("[联网自检] 正在发起探测请求……", file=sys.stderr)
    print(f"[联网自检] 提问：{PROBE_QUESTION}", file=sys.stderr)
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": PROBE_QUESTION}],
            tools=[SEARCH_TOOL],
            extra_body={"thinking": {"type": args.thinking}},
            max_tokens=400,
            stream=False,
        )
    except APIError as exc:
        print(
            f"[联网自检] API 调用失败："
            f"{redact_secrets(exc, getattr(client, 'api_key', None))}",
            file=sys.stderr,
        )
        return 1

    message = response.choices[0].message
    text = field(message, "content", "") or ""
    evidence = collect_evidence(
        field(message, "server_tool_calls"), field(message, "annotations"), response.usage
    )

    print("-" * 60, file=sys.stderr)
    print(text.strip(), file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    print_verification(evidence, enabled=True, prefix="[联网自检] ")

    passed = evidence["tool_search"] > 0 and len(evidence["sources"]) > 0
    print(
        f"[联网自检] 结果：{'PASS ✅ 联网能力已开启' if passed else 'FAIL ❌ 未观察到服务端搜索'}",
        file=sys.stderr,
    )
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    if not 1900 <= args.year <= 2100:
        print("错误：--year 需要在 1900–2100 之间。", file=sys.stderr)
        return 1
    client = make_client()

    if args.search and args.model != "deepseek-flash":
        print(
            f"警告：联网工具当前要求 deepseek-flash 模型，当前为 {args.model}，"
            "可能返回 400。",
            file=sys.stderr,
        )

    if args.verify_search:
        return run_search_probe(client, args)

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    prompt_path: Path = args.prompt_file
    if not prompt_path.is_absolute():
        prompt_path = (Path.cwd() / prompt_path).resolve()
    if not prompt_path.is_file():
        print(f"错误：提示词文件不存在：{prompt_path}", file=sys.stderr)
        return 1
    if args.output is not None and args.output.is_dir():
        print("错误：--output 需要文件路径，不是目录。", file=sys.stderr)
        return 1

    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt = render_year_placeholders(prompt_template, args.year)
    split_quarters = args.quarters if args.quarters is not None else args.search

    print(
        f"提示词：{prompt_path}（{len(prompt)} 字符，调查年份 {args.year}）",
        file=sys.stderr,
    )
    print(f"模型：{args.model}  接口：{safe_endpoint(base_url)}", file=sys.stderr)
    thinking_label = (
        "深度思考搜索" if args.thinking == "enabled" else "普通联网搜索"
    )
    print(
        f"联网搜索：{'开启（' + thinking_label + '）' if args.search else '关闭'}",
        file=sys.stderr,
    )
    print(
        f"分季度生成：{'开启（Q1–Q4，共 4 次请求）' if split_quarters else '关闭（单次请求）'}",
        file=sys.stderr,
    )
    print(f"每次最大输出：{args.max_tokens} token", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    if split_quarters:
        return run_quarterly(client, args, prompt, prompt_path)

    instruction = render_year_placeholders(args.instruction, args.year)
    messages = build_messages(prompt, instruction)
    result = run_completion(client, args, messages, output_path=args.output)
    if result["error"] or result["finish_reason"] in {"length", "content_filter"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
