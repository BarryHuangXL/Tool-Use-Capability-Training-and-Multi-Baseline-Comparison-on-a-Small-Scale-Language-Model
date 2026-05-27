"""Shared tool-call eval helpers (no torch) for MiniMind / HF / API scripts."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    calls: list[dict[str, Any]] = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass
    return calls


def gold_first_tool_name(gold_calls: list[dict[str, Any]]) -> str:
    if not gold_calls:
        return ""
    c0 = gold_calls[0]
    fn = c0.get("function") if isinstance(c0, dict) else None
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return ""


def pred_tool_name(call: dict[str, Any]) -> str:
    if not isinstance(call, dict):
        return ""
    if call.get("name"):
        return str(call["name"])
    fn = call.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return ""


def arguments_json_valid(call: dict[str, Any]) -> bool:
    if not isinstance(call, dict):
        return False
    raw = call.get("arguments")
    if raw is None and isinstance(call.get("function"), dict):
        raw = call["function"].get("arguments")
    if raw is None:
        return False
    if isinstance(raw, dict):
        return True
    if isinstance(raw, str):
        try:
            json.loads(raw)
            return True
        except json.JSONDecodeError:
            return False
    return False


def extract_tools_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    tools = None
    for m in messages:
        if m.get("role") == "system" and m.get("tools"):
            t = m["tools"]
            if isinstance(t, str):
                try:
                    tools = json.loads(t)
                except json.JSONDecodeError:
                    tools = None
            elif isinstance(t, list):
                tools = t
            break
    return tools


def call_structurally_valid(calls: list[dict[str, Any]]) -> bool:
    if not calls:
        return False
    c0 = calls[0]
    name = pred_tool_name(c0)
    if not name:
        return False
    return arguments_json_valid(c0)


def classify_bad_case(
    gold_name: str, pred_calls: list[dict[str, Any]], valid_s: bool, p_name: str
) -> str:
    if not pred_calls:
        return "no_tool_call_in_output"
    if not valid_s:
        return "invalid_structure_or_args"
    if p_name != gold_name:
        return "wrong_first_tool_name"
    return "ok"
