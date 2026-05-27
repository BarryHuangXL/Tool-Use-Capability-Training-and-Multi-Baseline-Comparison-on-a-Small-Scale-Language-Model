"""Merge several metrics_*.json files (summary section) into one table JSON for reports."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate tool-eval metrics JSON files for comparison")
    p.add_argument(
        "metrics_files",
        nargs="+",
        help="Paths to JSON files produced by eval_intent_router_metrics / eval_hf_baseline / agent_api_eval",
    )
    p.add_argument("--out", type=str, default="project_prof/comparison_table.json")
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.metrics_files:
        if not os.path.isfile(path):
            rows.append({"file": path, "error": "not_found"})
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summ = data.get("summary") or {}
        rows.append({"file": os.path.abspath(path), **summ})

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=False)

    print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
