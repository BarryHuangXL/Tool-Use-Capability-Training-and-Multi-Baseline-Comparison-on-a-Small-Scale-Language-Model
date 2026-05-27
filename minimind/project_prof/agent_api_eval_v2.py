"""
Improved tool-use eval using an OpenAI-compatible chat API.
Key improvements over v1:
  - Injects a stronger system prompt that forces tool usage.
  - Adds few-shot examples in the prompt to guide correct tool selection.
  - Explicitly forbids fallback to web_search when a direct tool is available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openai import OpenAI

from project_prof.metrics_core_v2 import (
    compute_all_metrics,
    classify_bad_case_v2,
    extract_tools_from_messages,
    parse_tool_calls,
    pred_tool_name,
)

# ------------------------------------------------------------------
# System prompt & few-shot examples
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant that MUST use the provided tools to answer user questions. "
    "Follow these rules strictly:\n"
    "1. ALWAYS call one or more tools from the provided list. Do NOT answer directly unless no tool is relevant.\n"
    "2. Choose the EXACT tool name that matches the user's intent. Do NOT use 'web_search' when a direct tool exists.\n"
    "3. Fill arguments accurately based on the user's request.\n"
    "4. If multiple independent tasks are asked, call the corresponding tools in the correct order.\n"
    "5. Output tool calls in the standard format.\n"
)

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "Example 1: User asks for weather. You must call get_current_weather, not web_search.\n"
            "Example 2: User asks for exchange rate. You must call get_exchange_rate.\n"
            "Example 3: User asks for math. You must call calculate_math.\n"
            "Example 4: User asks to translate. You must call translate_text.\n"
            "Example 5: User asks for time. You must call get_current_time.\n"
            "Example 6: User asks for text length. You must call text_length.\n"
            "Example 7: User asks for unit conversion. You must call unit_converter.\n"
        ),
    },
]


def _messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tools field from messages (API uses separate tools=)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        m.pop("tools", None)
        m.pop("tool_calls", None)
        if "content" not in m:
            m["content"] = ""
        out.append({"role": m["role"], "content": str(m.get("content", ""))})
    return out


def _openai_tools_from_minimind(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    normalized: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, dict) and t.get("type") == "function" and "function" in t:
            normalized.append(t)
        elif isinstance(t, dict) and "function" in t:
            normalized.append({"type": "function", **t} if t.get("type") != "function" else t)
        else:
            normalized.append({"type": "function", "function": t})
    return normalized


def _first_tool_from_message(msg: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return (first_function_name, pred_calls_in_minimind_shape)."""
    calls_out: list[dict[str, Any]] = []
    if msg is None:
        return "", calls_out
    tcs = getattr(msg, "tool_calls", None) or []
    for tc in tcs:
        fn = getattr(tc.function, "name", "") or ""
        raw_args = getattr(tc.function, "arguments", "") or "{}"
        try:
            args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args_obj = {}
        calls_out.append({"name": fn, "arguments": args_obj})
    first = calls_out[0]["name"] if calls_out else ""
    return first, calls_out


def build_enhanced_messages(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (api_messages, api_tools) with enhanced system prompt."""
    api_tools = _openai_tools_from_minimind(tools)
    api_messages = _messages_for_api(messages)

    # Replace or prepend the system message with our stronger prompt
    if api_messages and api_messages[0].get("role") == "system":
        original = api_messages[0].get("content", "")
        api_messages[0]["content"] = SYSTEM_PROMPT + (
            f"\n\n[Original system instruction]: {original}" if original else ""
        )
    else:
        api_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Inject few-shot examples right after system
    for ex in reversed(FEW_SHOT_EXAMPLES):
        api_messages.insert(1, dict(ex))

    return api_messages, api_tools


def main() -> None:
    p = argparse.ArgumentParser(description="Tool-call holdout eval via LLM API (v2 improved)")
    p.add_argument("--holdout", type=str, default="dataset/tool_eval_holdout.jsonl")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--api_base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
    )
    p.add_argument("--api_model", type=str, default="qwen-turbo", help="Model id on your provider")
    p.add_argument("--metrics_json", type=str, default="project_prof/metrics_api_agent_v2.json")
    p.add_argument("--bad_cases_jsonl", type=str, default="project_prof/bad_cases_api_v2.jsonl")
    p.add_argument("--two_round_agent", action="store_true")
    p.add_argument("--run_name", type=str, default="api_agent_v2")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Set --api_key or env DASHSCOPE_API_KEY / OPENAI_API_KEY.")

    if not os.path.isfile(args.holdout):
        raise SystemExit(f"Holdout not found: {args.holdout}")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base_url.rstrip("/"))

    rows: list[dict[str, Any]] = []
    with open(args.holdout, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    n = 0
    latencies: list[float] = []
    details: list[dict[str, Any]] = []
    bad_f = None
    if args.bad_cases_jsonl:
        bp = os.path.abspath(args.bad_cases_jsonl)
        os.makedirs(os.path.dirname(bp) or ".", exist_ok=True)
        bad_f = open(bp, "w", encoding="utf-8")

    try:
        for i, row in enumerate(rows):
            messages = row.get("messages")
            gold = row.get("gold_tool_calls")
            if not isinstance(messages, list) or not isinstance(gold, list):
                continue
            tools_list = extract_tools_from_messages(messages)
            api_messages, api_tools = build_enhanced_messages(messages, tools_list)

            t0 = time.perf_counter()
            create_kw: dict[str, Any] = {
                "model": args.api_model,
                "messages": api_messages,
                "temperature": 0.1,
                "max_tokens": 1024,
            }
            if api_tools:
                create_kw["tools"] = api_tools
                create_kw["tool_choice"] = "auto"
            resp = client.chat.completions.create(**create_kw)
            msg = resp.choices[0].message
            first_name, pred_calls = _first_tool_from_message(msg)
            content = (msg.content or "").strip()
            text_for_parse = content
            if pred_calls:
                for c in pred_calls:
                    text_for_parse += (
                        f"\n<tool_call>{json.dumps({'name': c['name'], 'arguments': c['arguments']}, ensure_ascii=False)}</tool_call>"
                    )
            parsed_from_tags = parse_tool_calls(text_for_parse)
            pred_calls = parsed_from_tags if parsed_from_tags else pred_calls

            if args.two_round_agent and pred_calls and getattr(msg, "tool_calls", None):
                tc0 = msg.tool_calls[0]
                tid = getattr(tc0, "id", None) or "call_0"
                mock = json.dumps({"ok": True, "echo": "mock_tool_env"}, ensure_ascii=False)
                follow = list(api_messages)
                asst = {"role": "assistant", "content": msg.content or ""}
                asst["tool_calls"] = [
                    {
                        "id": getattr(t, "id", f"call_{j}"),
                        "type": "function",
                        "function": {"name": t.function.name, "arguments": t.function.arguments},
                    }
                    for j, t in enumerate(msg.tool_calls)
                ]
                follow.append(asst)
                follow.append({"role": "tool", "tool_call_id": tid, "content": mock})
                kw2: dict[str, Any] = {
                    "model": args.api_model,
                    "messages": follow,
                    "temperature": 0.2,
                    "max_tokens": 512,
                }
                if api_tools:
                    kw2["tools"] = api_tools
                    kw2["tool_choice"] = "auto"
                resp2 = client.chat.completions.create(**kw2)
                msg2 = resp2.choices[0].message
                c2 = (msg2.content or "").strip()
                text_for_parse = text_for_parse + "\n" + c2

            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            metrics = compute_all_metrics(gold, pred_calls)
            kind_v2 = classify_bad_case_v2(gold, pred_calls, metrics["structure_valid"], metrics["pred_first_name"])
            metrics["bad_kind_v2"] = kind_v2
            metrics["idx"] = i
            metrics["latency_s"] = round(elapsed, 4)
            details.append(metrics)
            n += 1

            if bad_f is not None and kind_v2 != "ok":
                bad_f.write(
                    json.dumps(
                        {
                            "idx": i,
                            "run": args.run_name,
                            "api_model": args.api_model,
                            "bad_kind_v2": kind_v2,
                            "gold_tool_calls": gold,
                            "pred_tool_calls": pred_calls,
                            "model_output": text_for_parse,
                            "messages": messages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if bad_f is not None:
            bad_f.close()

    summary = {
        "backend": "openai_compatible_api",
        "api_model": args.api_model,
        "api_base_url": args.api_base_url,
        "holdout": os.path.abspath(args.holdout),
        "evaluated": n,
        "structure_valid_rate": sum(1 for d in details if d["structure_valid"]) / n if n else 0.0,
        "first_tool_name_accuracy": sum(d["first_tool_name_accuracy"] for d in details) / n if n else 0.0,
        "tool_name_f1": sum(d["tool_name_f1"] for d in details) / n if n else 0.0,
        "args_f1": sum(d["args_f1"] for d in details) / n if n else 0.0,
        "overall_f1": sum(d["overall_f1"] for d in details) / n if n else 0.0,
        "exact_match_rate": sum(d["exact_match"] for d in details) / n if n else 0.0,
        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "two_round_agent": bool(args.two_round_agent),
    }
    cap = 0  # 0 means no truncation; save all details
    out_payload = {"summary": summary, "details": details[:cap] if cap > 0 else details, "details_truncated": (cap > 0) and (len(details) > cap), "details_total": len(details)}
    mp = os.path.abspath(args.metrics_json)
    os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
    with open(mp, "w", encoding="utf-8") as mf:
        json.dump(out_payload, mf, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {mp}")


if __name__ == "__main__":
    main()
