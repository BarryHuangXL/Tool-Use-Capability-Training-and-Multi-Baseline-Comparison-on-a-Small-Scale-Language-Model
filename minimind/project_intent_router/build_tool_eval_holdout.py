"""
Build a reproducible holdout jsonl from SFT data: prefix messages up to (excluding)
the first assistant turn that contains tool_calls; store gold tool_calls for metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

# 首轮 assistant tool_calls 中出现的 function.name 必须全部落在该集合内才会进入候选池
DEFAULT_ALLOWED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # "random_number",
        "get_current_time",
        "calculate_math",
        "text_length",
        "get_current_weather",
        "get_exchange_rate",
        "unit_converter",
        "translate_text",
        # "get_news",
        # "web_search",
    }
)


def _parse_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _normalize_message(m: dict[str, Any]) -> dict[str, Any]:
    m = dict(m)
    if m.get("tool_calls") is not None and isinstance(m["tool_calls"], str):
        try:
            m["tool_calls"] = json.loads(m["tool_calls"])
        except json.JSONDecodeError:
            pass
    if m.get("tools") is not None and isinstance(m["tools"], str):
        try:
            m["tools"] = json.loads(m["tools"])
        except json.JSONDecodeError:
            pass
    return m


def first_tool_call_split(
    conversations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return (prefix_messages, gold_tool_calls OpenAI list) or None."""
    for i, raw in enumerate(conversations):
        msg = _normalize_message(dict(raw))
        calls = _parse_tool_calls(msg.get("tool_calls"))
        if msg.get("role") == "assistant" and calls:
            prefix = [_normalize_message(dict(conversations[j])) for j in range(i)]
            return prefix, calls
    return None


def _gold_tool_function_names(gold_tool_calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for c in gold_tool_calls:
        if not isinstance(c, dict):
            continue
        fn = c.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _gold_matches_allowed_tools(
    gold_tool_calls: list[dict[str, Any]], allowed: frozenset[str]
) -> bool:
    """首轮 gold 里出现的每个工具名都必须在 allowed 内；无有效 name 则丢弃。"""
    names = _gold_tool_function_names(gold_tool_calls)
    if not names:
        return False
    return all(n in allowed for n in names)


def main() -> None:
    p = argparse.ArgumentParser(description="Build tool-call eval holdout jsonl")
    p.add_argument("--input", type=str, default="dataset/sft_t2t.jsonl")
    p.add_argument("--output", type=str, default="dataset/tool_eval_holdout.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout_n", type=int, default=1000, help="Max samples to write")
    p.add_argument(
        "--pool_max",
        type=int,
        default=0,
        help="If >0, shuffle at most this many tool-lines before taking holdout_n (faster for huge files)",
    )
    p.add_argument(
        "--allowed_tools",
        type=str,
        default="",
        help="Comma-separated tool names to keep (default: built-in 10 tasks). Ignored if --allow_any_tool.",
    )
    p.add_argument(
        "--allow_any_tool",
        action="store_true",
        help="Do not filter by tool name (legacy behavior: any tool_calls row).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    if args.allow_any_tool:
        allowed: frozenset[str] | None = None
    elif args.allowed_tools.strip():
        allowed = frozenset(t.strip() for t in args.allowed_tools.split(",") if t.strip())
    else:
        allowed = DEFAULT_ALLOWED_TOOL_NAMES

    candidates: list[dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations")
            if not isinstance(conv, list):
                continue
            split = first_tool_call_split(conv)
            if not split:
                continue
            prefix, gold = split
            if not prefix:
                continue
            if allowed is not None and not _gold_matches_allowed_tools(gold, allowed):
                continue
            candidates.append(
                {
                    "messages": prefix,
                    "gold_tool_calls": gold,
                    "source_line": line_no,
                }
            )
            if args.pool_max > 0 and len(candidates) >= args.pool_max:
                break

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.holdout_n]

    out_path = os.path.abspath(args.output)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "input": os.path.abspath(args.input),
        "output": out_path,
        "seed": args.seed,
        "holdout_n": args.holdout_n,
        "pool_max": args.pool_max,
        "candidates_seen": len(candidates),
        "written": len(selected),
        "tool_name_filter": None if allowed is None else sorted(allowed),
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
