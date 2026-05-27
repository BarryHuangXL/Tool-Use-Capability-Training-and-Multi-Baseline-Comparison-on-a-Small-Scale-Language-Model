"""
Improved tool-call eval helpers with better metrics beyond first_tool_name_accuracy.

New metrics:
- tool_name_set_f1: F1 over tool-name sets (order-agnostic, handles multi-tool)
- exact_match_rate: strict sequence match (name + args + order)
- arguments_f1: F1 over argument key-value pairs
- overall_f1: harmonic mean of tool-name F1 and args F1

New bad-case categories:
- ok
- no_tool_call_in_output
- invalid_structure_or_args
- wrong_first_tool_name -> refined to:
    - wrong_first_but_others_ok
    - missing_required_tools
    - extra_wrong_tools
    - all_tools_wrong
    - wrong_args_only (names match but args differ)
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------- original helpers (kept for compatibility) ----------

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


# ---------- new v2 helpers ----------

def _normalize_call(call: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize any tool-call shape into {"name": str, "arguments": dict}."""
    if not isinstance(call, dict):
        return None
    name = pred_tool_name(call)
    if not name:
        return None
    raw_args = call.get("arguments")
    if raw_args is None and isinstance(call.get("function"), dict):
        raw_args = call["function"].get("arguments")
    args: dict[str, Any] = {}
    if isinstance(raw_args, dict):
        args = raw_args
    elif isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    return {"name": name, "arguments": args}


def _normalize_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in calls:
        nc = _normalize_call(c)
        if nc:
            out.append(nc)
    return out


def tool_name_set_metrics(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]) -> dict[str, float]:
    """Return precision, recall, f1 over tool-name sets (order-agnostic)."""
    g = _normalize_calls(gold_calls)
    p = _normalize_calls(pred_calls)
    gold_set = {c["name"] for c in g}
    pred_set = {c["name"] for c in p}
    if not gold_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match_rate(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]) -> bool:
    """Strict exact match: same count, same order, same name, same args."""
    g = _normalize_calls(gold_calls)
    p = _normalize_calls(pred_calls)
    if len(g) != len(p):
        return False
    for cg, cp in zip(g, p):
        if cg["name"] != cp["name"]:
            return False
        if cg["arguments"] != cp["arguments"]:
            return False
    return True


def _args_to_items(args: dict[str, Any]) -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for k, v in sorted(args.items()):
        items.add((k, json.dumps(v, ensure_ascii=False, sort_keys=True)))
    return items


def arguments_f1(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]) -> dict[str, float]:
    """Argument-level F1: match args by aligned tool names (best-effort alignment)."""
    g = _normalize_calls(gold_calls)
    p = _normalize_calls(pred_calls)
    if not g and not p:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not g or not p:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Greedy alignment by name matching (simplistic but effective)
    total_tp = 0
    total_pred_items = 0
    total_gold_items = 0

    # Count per aligned pair
    matched_pred = set()
    for cg in g:
        best_match = None
        best_score = -1
        for i, cp in enumerate(p):
            if i in matched_pred:
                continue
            if cg["name"] == cp["name"]:
                g_items = _args_to_items(cg["arguments"])
                p_items = _args_to_items(cp["arguments"])
                score = len(g_items & p_items)
                if score > best_score:
                    best_score = score
                    best_match = i
        if best_match is not None:
            matched_pred.add(best_match)
            cg_items = _args_to_items(cg["arguments"])
            cp_items = _args_to_items(p[best_match]["arguments"])
            total_tp += len(cg_items & cp_items)
            total_gold_items += len(cg_items)
            total_pred_items += len(cp_items)
        else:
            total_gold_items += len(_args_to_items(cg["arguments"]))

    for i, cp in enumerate(p):
        if i not in matched_pred:
            total_pred_items += len(_args_to_items(cp["arguments"]))

    precision = total_tp / total_pred_items if total_pred_items else 0.0
    recall = total_tp / total_gold_items if total_gold_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def overall_f1(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]]) -> float:
    """Harmonic mean of tool-name-set F1 and argument F1."""
    name_f1 = tool_name_set_metrics(gold_calls, pred_calls)["f1"]
    arg_f1 = arguments_f1(gold_calls, pred_calls)["f1"]
    if name_f1 + arg_f1 == 0:
        return 0.0
    return 2 * name_f1 * arg_f1 / (name_f1 + arg_f1)


def compute_all_metrics(gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]], text: str = "") -> dict[str, Any]:
    """Compute the full metric suite for a single example."""
    valid_s = call_structurally_valid(pred_calls)
    gold_name = gold_first_tool_name(gold_calls)
    p_name = pred_tool_name(pred_calls[0]) if pred_calls else ""

    name_metrics = tool_name_set_metrics(gold_calls, pred_calls)
    arg_metrics = arguments_f1(gold_calls, pred_calls)
    ov_f1 = overall_f1(gold_calls, pred_calls)
    exact = exact_match_rate(gold_calls, pred_calls)

    return {
        "structure_valid": valid_s,
        "gold_first_name": gold_name,
        "pred_first_name": p_name,
        "first_tool_name_accuracy": 1.0 if (p_name and gold_name and p_name == gold_name) else 0.0,
        "tool_name_precision": name_metrics["precision"],
        "tool_name_recall": name_metrics["recall"],
        "tool_name_f1": name_metrics["f1"],
        "args_precision": arg_metrics["precision"],
        "args_recall": arg_metrics["recall"],
        "args_f1": arg_metrics["f1"],
        "overall_f1": ov_f1,
        "exact_match": 1.0 if exact else 0.0,
    }


def classify_bad_case_v2(
    gold_calls: list[dict[str, Any]], pred_calls: list[dict[str, Any]], valid_s: bool, p_name: str
) -> str:
    """Finer-grained bad-case classification."""
    if not pred_calls:
        return "no_tool_call_in_output"
    if not valid_s:
        return "invalid_structure_or_args"

    gold_name = gold_first_tool_name(gold_calls)
    g_norm = _normalize_calls(gold_calls)
    p_norm = _normalize_calls(pred_calls)
    gold_names = {c["name"] for c in g_norm}
    pred_names = {c["name"] for c in p_norm}

    # Check if first name is wrong
    if p_name != gold_name:
        # Check if any correct tool appears later
        if gold_name in pred_names:
            return "wrong_first_but_others_ok"
        # Check if any gold tool is present at all
        if gold_names & pred_names:
            return "wrong_first_and_missing"
        return "all_tools_wrong"

    # First name is correct; check if all required tools are present
    if not (gold_names <= pred_names):
        return "missing_required_tools"

    # First name correct, all required present; check args
    if not exact_match_rate(gold_calls, pred_calls):
        return "wrong_args_only"

    return "ok"


def bad_case_statistics(bad_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate statistics from a list of bad-case records."""
    from collections import Counter
    kinds = Counter()
    name_confusion: Counter = Counter()
    missing_tools: Counter = Counter()
    extra_tools: Counter = Counter()

    for bc in bad_cases:
        kind = bc.get("bad_kind", "unknown")
        kinds[kind] += 1

        gold = bc.get("gold_tool_calls", [])
        pred = bc.get("pred_tool_calls", [])
        g_names = {pred_tool_name(c) for c in gold}
        p_names = {pred_tool_name(c) for c in pred}

        if kind in ("wrong_first_tool_name", "wrong_first_but_others_ok", "wrong_first_and_missing", "all_tools_wrong"):
            gold_first = gold_first_tool_name(gold)
            pred_first = pred_tool_name(pred[0]) if pred else "(none)"
            if gold_first or pred_first:
                name_confusion[(pred_first, gold_first)] += 1

        if g_names - p_names:
            for t in g_names - p_names:
                missing_tools[t] += 1
        if p_names - g_names:
            for t in p_names - g_names:
                extra_tools[t] += 1

    return {
        "kind_distribution": dict(kinds),
        "top_name_confusions": name_confusion.most_common(10),
        "missing_tools": dict(missing_tools),
        "extra_tools": dict(extra_tools),
    }
