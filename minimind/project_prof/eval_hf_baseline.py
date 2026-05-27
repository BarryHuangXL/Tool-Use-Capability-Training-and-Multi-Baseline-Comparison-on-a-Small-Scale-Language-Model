"""
Evaluate the same tool-call holdout with a small HuggingFace instruct model (Qwen / Llama / etc.)
for comparison with MiniMind. Writes metrics JSON + optional bad_cases jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from project_prof.metrics_core import (
    call_structurally_valid,
    classify_bad_case,
    extract_tools_from_messages,
    gold_first_tool_name,
    parse_tool_calls,
    pred_tool_name,
)


def main() -> None:
    p = argparse.ArgumentParser(description="HF small-model tool-call eval (comparison baseline)")
    p.add_argument("--model_name_or_path", type=str, required=True, help="e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--holdout", type=str, default="dataset/tool_eval_holdout.jsonl")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--metrics_json", type=str, default="project_prof/metrics_hf_baseline.json")
    p.add_argument("--bad_cases_jsonl", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--run_name", type=str, default="hf_baseline", help="Tag written into bad-case records")
    args = p.parse_args()

    if not os.path.isfile(args.holdout):
        raise SystemExit(f"Holdout not found: {args.holdout}")

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dt if args.device != "cpu" else torch.float32,
        device_map=None,
    )
    model = model.to(args.device).eval()

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
    n_valid_structure = 0
    n_name_match = 0
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
            tools = extract_tools_from_messages(messages)
            try:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
            except Exception as e:
                try:
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    print(f"[skip idx={i}] chat_template failed: {e}")
                    continue

            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(args.device)
            gold_name = gold_first_tool_name(gold)

            gen_kw: dict[str, Any] = {
                "inputs": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            }
            if args.temperature > 0:
                gen_kw["do_sample"] = True
                gen_kw["top_p"] = args.top_p
                gen_kw["temperature"] = args.temperature
            else:
                gen_kw["do_sample"] = False

            t0 = time.perf_counter()
            with torch.no_grad():
                generated_ids = model.generate(**gen_kw)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            gen_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred_calls = parse_tool_calls(text)

            n += 1
            valid_s = call_structurally_valid(pred_calls)
            if valid_s:
                n_valid_structure += 1
            p_name = pred_tool_name(pred_calls[0]) if pred_calls else ""
            if p_name and gold_name and p_name == gold_name:
                n_name_match += 1
            kind = classify_bad_case(gold_name, pred_calls, valid_s, p_name)
            details.append(
                {
                    "idx": i,
                    "gold_first_name": gold_name,
                    "pred_first_name": p_name,
                    "structure_valid": valid_s,
                    "bad_kind": kind,
                    "latency_s": round(elapsed, 4),
                    "raw_head": text[:500],
                }
            )
            if bad_f is not None and kind != "ok":
                bad_f.write(
                    json.dumps(
                        {
                            "idx": i,
                            "run": args.run_name,
                            "model": args.model_name_or_path,
                            "bad_kind": kind,
                            "gold_tool_calls": gold,
                            "pred_tool_calls_parsed": pred_calls,
                            "model_output": text,
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
        "backend": "hf_causal_lm",
        "model_name_or_path": args.model_name_or_path,
        "holdout": os.path.abspath(args.holdout),
        "evaluated": n,
        "structure_valid_rate": (n_valid_structure / n) if n else 0.0,
        "first_tool_name_accuracy": (n_name_match / n) if n else 0.0,
        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_p50_s": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,
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
