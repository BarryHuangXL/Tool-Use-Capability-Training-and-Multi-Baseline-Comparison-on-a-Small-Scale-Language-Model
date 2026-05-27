"""
Recalculate metrics with improved v2 indicators and perform deep bad-case analysis.

Usage:
    python recalculate_and_analyze.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from project_prof.metrics_core_v2 import (
    compute_all_metrics,
    classify_bad_case_v2,
    parse_tool_calls,
    bad_case_statistics,
    pred_tool_name,
    gold_first_tool_name,
)


def load_holdout(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bad_cases(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def load_metrics_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Scheme definitions
# ------------------------------------------------------------------

SCHEMES = [
    {
        "name": "MiniMind SFT (Dense)",
        "metrics_json": "project_intent_router/full_sft_768_metrics.json",
        "bad_cases_jsonl": None,  # not available
        "tag": "self",
    },
    {
        "name": "MiniMind SFT + MoE",
        "metrics_json": "project_intent_router/full_sft_768_moe_metrics.json",
        "bad_cases_jsonl": None,
        "tag": "self",
    },
    {
        "name": "MiniMind Agent RL (Dense)",
        "metrics_json": "project_intent_router/agent_768_metrics.json",
        "bad_cases_jsonl": "project_intent_router/agent_768_bad_cases.jsonl",
        "tag": "self",
    },
    {
        "name": "MiniMind Agent + MoE",
        "metrics_json": "project_intent_router/agent_768_moe_metrics.json",
        "bad_cases_jsonl": None,
        "tag": "self",
    },
    {
        "name": "HF Qwen-0.5B SFT",
        "metrics_json": "project_prof/metrics_hf_qwen_sft.json",
        "bad_cases_jsonl": "project_prof/bad_cases_hf_qwen_sft.jsonl",
        "tag": "baseline",
    },
    {
        "name": "API qwen-turbo",
        "metrics_json": "project_prof/metrics_api_qwen.json",
        "bad_cases_jsonl": "project_prof/bad_cases_api.jsonl",
        "tag": "baseline",
    },
]

HOLDOUT_PATH = "dataset/tool_eval_holdout.jsonl"


def analyze_scheme(holdout: list[dict], scheme: dict) -> dict[str, Any]:
    """Recalculate metrics for one scheme using all available data.

    This function iterates over the full evaluated sample count (typically 300)
    rather than only the saved details slice. For samples beyond the details
    list, it falls back to bad_cases when available; samples not present in
    bad_cases are assumed correct (ok) because eval scripts record every
    non-ok sample into bad_cases.
    """
    name = scheme["name"]
    metrics_path = scheme["metrics_json"]
    bad_path = scheme["bad_cases_jsonl"]

    metrics_data = load_metrics_json(metrics_path)
    bad_cases = load_bad_cases(bad_path) if bad_path else []
    summary = metrics_data.get("summary", {}) if metrics_data else {}
    details = metrics_data.get("details", []) if metrics_data else []
    total_evaluated = summary.get("evaluated", 0)

    # Build lookup for bad_cases by idx
    bad_by_idx: dict[int, dict] = {bc["idx"]: bc for bc in bad_cases if "idx" in bc}

    # Build lookup for details by idx
    detail_by_idx: dict[int, dict] = {d["idx"]: d for d in details if "idx" in d}

    per_sample_metrics: list[dict] = []
    bad_kinds_v2: list[str] = []

    # Determine the range of samples to evaluate.
    # Prefer total_evaluated from summary; fall back to max observed idx + 1.
    max_idx = max(
        ([d["idx"] for d in details] + [bc["idx"] for bc in bad_cases] + [0]),
        default=0,
    )
    n_samples = total_evaluated if total_evaluated else (max_idx + 1)
    # Clamp to holdout size
    n_samples = min(n_samples, len(holdout))

    for idx in range(n_samples):
        gold = holdout[idx].get("gold_tool_calls", []) if idx < len(holdout) else []

        if idx in detail_by_idx:
            # Use saved detail (contains raw model output)
            d = detail_by_idx[idx]
            if idx in bad_by_idx:
                bc = bad_by_idx[idx]
                pred = bc.get("pred_tool_calls", bc.get("pred_tool_calls_parsed", []))
            else:
                raw = d.get("raw_head", "")
                pred = parse_tool_calls(raw)
        elif idx in bad_by_idx:
            # Not in details but present in bad_cases (e.g., idx > 99)
            bc = bad_by_idx[idx]
            pred = bc.get("pred_tool_calls", bc.get("pred_tool_calls_parsed", []))
        elif bad_cases:
            # Not in details and not in bad_cases, but bad_cases file exists.
            # Eval scripts write every non-ok sample to bad_cases, so this
            # sample is assumed ok.
            pred = gold
        else:
            # No bad_cases file and not in details => no data for this sample.
            # Skip it rather than assuming ok.
            continue

        m = compute_all_metrics(gold, pred)
        kind_v2 = classify_bad_case_v2(gold, pred, m["structure_valid"], m["pred_first_name"])
        m["bad_kind_v2"] = kind_v2
        m["idx"] = idx
        per_sample_metrics.append(m)
        bad_kinds_v2.append(kind_v2)

    n = len(per_sample_metrics)
    if n == 0:
        return {"name": name, "error": "no data"}

    def mean(key: str) -> float:
        vals = [s[key] for s in per_sample_metrics if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    kind_dist = Counter(bad_kinds_v2)

    result = {
        "name": name,
        "evaluated": n,
        "structure_valid_rate": mean("structure_valid"),
        "first_tool_name_accuracy": mean("first_tool_name_accuracy"),
        "tool_name_precision": mean("tool_name_precision"),
        "tool_name_recall": mean("tool_name_recall"),
        "tool_name_f1": mean("tool_name_f1"),
        "args_precision": mean("args_precision"),
        "args_recall": mean("args_recall"),
        "args_f1": mean("args_f1"),
        "overall_f1": mean("overall_f1"),
        "exact_match_rate": mean("exact_match"),
        "latency_mean_s": summary.get("latency_mean_s", 0.0),
        "bad_kind_distribution_v2": {k: v for k, v in kind_dist.items() if k != "ok"},
        "bad_cases_total": sum(1 for k in bad_kinds_v2 if k != "ok"),
    }

    if bad_cases:
        result["bad_case_stats"] = bad_case_statistics(bad_cases)

    return result


def plot_comparison(results: list[dict], out_dir: str):
    """Generate comparison charts."""
    os.makedirs(out_dir, exist_ok=True)
    names = [r["name"] for r in results]
    x = np.arange(len(names))

    # 1. Overall F1 comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    f1s = [r["overall_f1"] * 100 for r in results]
    colors = ["#4A90D9" if r["name"].startswith("MiniMind") else "#E74C3C" for r in results]
    bars = ax.bar(x, f1s, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Overall F1 (%)", fontsize=12)
    ax.set_title("Overall F1 Comparison (Tool Name + Arguments)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_overall_f1.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Exact match comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    em = [r["exact_match_rate"] * 100 for r in results]
    bars = ax.bar(x, em, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Exact Match Rate (%)", fontsize=12)
    ax.set_title("Exact Match Rate Comparison (Order + Name + Args)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_exact_match.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Tool-name F1 vs Args F1
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    tn_f1 = [r["tool_name_f1"] * 100 for r in results]
    arg_f1 = [r["args_f1"] * 100 for r in results]
    bars1 = ax.bar(x - width / 2, tn_f1, width, label="Tool-Name F1", color="#4A90D9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, arg_f1, width, label="Args F1", color="#E74C3C", edgecolor="white")
    ax.set_ylabel("F1 (%)", fontsize=12)
    ax.set_title("Tool-Name F1 vs. Argument F1 Breakdown", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chart_f1_breakdown.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Bad-case kind stacked bar (for schemes with data)
    all_kinds = set()
    for r in results:
        all_kinds.update(r.get("bad_kind_distribution_v2", {}).keys())
    all_kinds.discard("ok")
    all_kinds = sorted(all_kinds)

    if all_kinds:
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(names))
        kind_colors = plt.cm.Set3(np.linspace(0, 1, len(all_kinds)))
        for i, kind in enumerate(all_kinds):
            vals = [r.get("bad_kind_distribution_v2", {}).get(kind, 0) for r in results]
            ax.bar(x, vals, bottom=bottom, label=kind, color=kind_colors[i], edgecolor="white", width=0.6)
            bottom += np.array(vals)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Bad Case Type Distribution (Stacked Bar)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "chart_badcase_stacked.png"), dpi=200, bbox_inches="tight")
        plt.close()


def generate_markdown_report(results: list[dict], out_path: str):
    lines = []
    lines.append("# 工具调用评测报告 v2\n")
    lines.append("## 1. 新指标汇总表\n")
    lines.append("| 方案 | 结构合法率 | 首工具名准确率 | **Overall F1** | **精确匹配率** | 工具名 F1 | 参数 F1 | 平均延迟 |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        lines.append(
            f"| {r['name']} | "
            f"{r['structure_valid_rate']*100:.2f}% | "
            f"{r['first_tool_name_accuracy']*100:.2f}% | "
            f"**{r['overall_f1']*100:.2f}%** | "
            f"{r['exact_match_rate']*100:.2f}% | "
            f"{r['tool_name_f1']*100:.2f}% | "
            f"{r['args_f1']*100:.2f}% | "
            f"{r['latency_mean_s']:.2f}s |"
        )
    lines.append("")

    lines.append("## 2. 指标说明\n")
    lines.append("- **Overall F1**: 综合考虑工具名识别与参数匹配的综合 F1，替代原有的 `first_tool_name_accuracy`。")
    lines.append("- **精确匹配率 (Exact Match)**: 要求工具名、参数、顺序与 gold 完全一致，是最严格的指标。")
    lines.append("- **工具名 F1**: 基于集合计算 precision/recall/f1，对多工具场景和顺序不敏感。")
    lines.append("- **参数 F1**: 基于参数键值对的匹配，评估参数填写的准确性。")
    lines.append("")

    lines.append("## 3. Bad Case 统计分析\n")
    for r in results:
        name = r["name"]
        kind_dist = r.get("bad_kind_distribution_v2", {})
        total_bad = r.get("bad_cases_total", 0)
        lines.append(f"### {name}\n")
        lines.append(f"- 总错误数: {total_bad} / {r['evaluated']}")
        lines.append(f"- 错误类型分布:")
        for kind, cnt in sorted(kind_dist.items(), key=lambda x: -x[1]):
            if kind == "ok":
                continue
            pct = cnt / r["evaluated"] * 100
            lines.append(f"  - `{kind}`: {cnt} 条 ({pct:.1f}%)")

        stats = r.get("bad_case_stats")
        if stats:
            lines.append(f"- 具体错误模式:")
            confusions = stats.get("top_name_confusions", [])
            if confusions:
                lines.append(f"  - 最常见的工具名混淆 (预测 -> 期望):")
                for (pred_n, gold_n), cnt in confusions[:5]:
                    lines.append(f"    - `{pred_n}` -> `{gold_n}`: {cnt} 次")
            missing = stats.get("missing_tools", {})
            if missing:
                lines.append(f"  - 最常被遗漏的工具:")
                for tool, cnt in Counter(missing).most_common(5):
                    lines.append(f"    - `{tool}`: {cnt} 次")
            extra = stats.get("extra_tools", {})
            if extra:
                lines.append(f"  - 最常被错误调用的多余工具:")
                for tool, cnt in Counter(extra).most_common(5):
                    lines.append(f"    - `{tool}`: {cnt} 次")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {out_path}")


def main():
    holdout = load_holdout(HOLDOUT_PATH)
    print(f"Loaded holdout: {len(holdout)} rows")

    results = []
    for scheme in SCHEMES:
        print(f"\nAnalyzing: {scheme['name']} ...")
        res = analyze_scheme(holdout, scheme)
        results.append(res)
        print(f"  overall_f1={res.get('overall_f1', 0)*100:.2f}%, exact_match={res.get('exact_match_rate', 0)*100:.2f}%")

    # Save new comparison table
    out_table = {
        "version": "v2",
        "description": "Improved metrics: overall_f1, exact_match_rate, tool_name_f1, args_f1",
        "rows": results,
    }
    table_path = "project_prof/comparison_table_v2.json"
    with open(table_path, "w", encoding="utf-8") as f:
        json.dump(out_table, f, indent=2, ensure_ascii=False)
    print(f"\nNew comparison table: {table_path}")

    # Generate charts and report
    out_dir = "ppt_assets"
    os.makedirs(out_dir, exist_ok=True)
    plot_comparison(results, out_dir)
    generate_markdown_report(results, os.path.join(out_dir, "bad_case_report_v2.md"))

    print("\nDone.")


if __name__ == "__main__":
    main()
