"""
Scan SFT jsonl for tool-calling statistics: coverage, tool name distribution, turn counts.
Writes optional JSON report (no training code changes).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any


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


def _tool_names_from_calls(calls: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for c in calls:
        fn = c.get("function") if isinstance(c, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _conversation_has_tool_calls(conversations: list[dict[str, Any]]) -> bool:
    for m in conversations:
        calls = _parse_tool_calls(m.get("tool_calls"))
        if calls:
            return True
    return False


def _collect_all_tool_names(conversations: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for m in conversations:
        out.extend(_tool_names_from_calls(_parse_tool_calls(m.get("tool_calls"))))
    return out


def audit_line(obj: dict[str, Any]) -> dict[str, Any] | None:
    conv = obj.get("conversations")
    if not isinstance(conv, list):
        return None
    if not _conversation_has_tool_calls(conv):
        return None
    names = _collect_all_tool_names(conv)
    return {
        "turns": len(conv),
        "tool_call_messages": sum(1 for m in conv if _parse_tool_calls(m.get("tool_calls"))),
        "tool_names": names,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit tool-call coverage in SFT jsonl")
    p.add_argument("--input", type=str, default="dataset/sft_t2t.jsonl", help="Path to jsonl")
    p.add_argument("--report_json", type=str, default="", help="If set, write full report to this path")
    args = p.parse_args()

    path = args.input
    if not os.path.isfile(path):
        raise SystemExit(f"Input not found: {path}")

    total = 0
    with_tools = 0
    name_counter: Counter[str] = Counter()
    turns_with_tools: list[int] = []
    tool_msgs_per_conv: list[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = audit_line(obj)
            if not info:
                continue
            with_tools += 1
            for n in info["tool_names"]:
                name_counter[n] += 1
            turns_with_tools.append(info["turns"])
            tool_msgs_per_conv.append(info["tool_call_messages"])

    ratio = (with_tools / total) if total else 0.0
    avg_turns = sum(turns_with_tools) / len(turns_with_tools) if turns_with_tools else 0.0
    avg_tool_msgs = sum(tool_msgs_per_conv) / len(tool_msgs_per_conv) if tool_msgs_per_conv else 0.0

    report = {
        "input": os.path.abspath(path),
        "total_lines": total,
        "lines_with_tool_calls": with_tools,
        "ratio": ratio,
        "unique_tool_names": len(name_counter),
        "tool_name_counts": dict(name_counter.most_common()),
        "avg_turns_per_tool_conversation": avg_turns,
        "avg_assistant_tool_call_messages_per_tool_conversation": avg_tool_msgs,
    }

    print(json.dumps({k: v for k, v in report.items() if k != "tool_name_counts"}, indent=2, ensure_ascii=False))
    print("\ntool_name_counts (top 30):")
    for name, c in name_counter.most_common(30):
        print(f"  {name}: {c}")

    if args.report_json:
        rp = os.path.abspath(args.report_json)
        parent = os.path.dirname(rp)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.report_json}")


if __name__ == "__main__":
    main()
