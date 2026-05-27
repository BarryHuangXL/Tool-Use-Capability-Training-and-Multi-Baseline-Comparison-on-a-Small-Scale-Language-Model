"""

Batch-evaluate tool-calling on holdout jsonl: JSON validity, first-tool name accuracy, latency.

Optional: write bad-case rows to jsonl for professor review.

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



from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

from project_prof.metrics_core import (

    call_structurally_valid,

    classify_bad_case,

    extract_tools_from_messages,

    gold_first_tool_name,

    parse_tool_calls,

    pred_tool_name,

)

from trainer.trainer_utils import get_model_params, setup_seed





def init_model(args: argparse.Namespace):

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    if "model" in args.load_from:

        model = MiniMindForCausalLM(

            MiniMindConfig(

                hidden_size=args.hidden_size,

                num_hidden_layers=args.num_hidden_layers,

                use_moe=bool(args.use_moe),

                inference_rope_scaling=args.inference_rope_scaling,

            )

        )

        moe_suffix = "_moe" if args.use_moe else ""

        ckp = f"./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth"

        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

    else:

        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    get_model_params(model, model.config)

    return model.eval().to(args.device), tokenizer





def main() -> None:

    p = argparse.ArgumentParser(description="Intent-router style tool-call metrics on holdout jsonl")

    p.add_argument("--holdout", type=str, default="dataset/tool_eval_holdout.jsonl")

    p.add_argument("--max_samples", type=int, default=0, help="0 = all lines in holdout")

    p.add_argument("--metrics_json", type=str, default="project_intent_router/metrics_sft.json")

    p.add_argument(

        "--bad_cases_jsonl",

        type=str,

        default="",

        help="If set, append one json per bad line (structure/name mismatch + short context)",

    )

    p.add_argument("--load_from", type=str, default="model")

    p.add_argument("--save_dir", type=str, default="out")

    p.add_argument("--weight", type=str, default="full_sft")

    p.add_argument("--hidden_size", type=int, default=768)

    p.add_argument("--num_hidden_layers", type=int, default=8)

    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])

    p.add_argument("--inference_rope_scaling", action="store_true")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--max_new_tokens", type=int, default=256)

    p.add_argument("--temperature", type=float, default=0.1)

    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()



    if not os.path.isfile(args.holdout):

        raise SystemExit(f"Holdout not found: {args.holdout}")



    setup_seed(args.seed)

    model, tokenizer = init_model(args)



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

                    open_thinking=False,

                )

            except TypeError:

                input_text = tokenizer.apply_chat_template(

                    messages,

                    tokenize=False,

                    add_generation_prompt=True,

                    tools=tools,

                )



            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(args.device)

            gold_name = gold_first_tool_name(gold)



            gen_kw: dict[str, Any] = {

                "inputs": inputs["input_ids"],

                "attention_mask": inputs["attention_mask"],

                "max_new_tokens": args.max_new_tokens,

                "pad_token_id": tokenizer.pad_token_id,

                "eos_token_id": tokenizer.eos_token_id,

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

                            "run": "minimind_eval",

                            "bad_kind": kind,

                            "gold_tool_calls": gold,

                            "pred_tool_calls_parsed": pred_calls,

                            "model_output": text,

                            "messages": messages,

                            "gold_first_name": gold_name,

                            "pred_first_name": p_name,

                        },

                        ensure_ascii=False,

                    )

                    + "\n"

                )

    finally:

        if bad_f is not None:

            bad_f.close()



    summary = {

        "holdout": os.path.abspath(args.holdout),

        "evaluated": n,

        "structure_valid_rate": (n_valid_structure / n) if n else 0.0,

        "first_tool_name_accuracy": (n_name_match / n) if n else 0.0,

        "latency_mean_s": (sum(latencies) / len(latencies)) if latencies else 0.0,

        "latency_p50_s": sorted(latencies)[len(latencies) // 2] if latencies else 0.0,

        "device": args.device,

        "max_new_tokens": args.max_new_tokens,

        "temperature": args.temperature,

        "bad_cases_file": os.path.abspath(args.bad_cases_jsonl) if args.bad_cases_jsonl else None,

    }



    cap = 0  # 0 means no truncation; save all details

    out_payload = {

        "summary": summary,

        "details": details[:cap] if cap > 0 else details,

        "details_truncated": (cap > 0) and (len(details) > cap),

        "details_total": len(details),

    }



    mp = os.path.abspath(args.metrics_json)

    parent = os.path.dirname(mp)

    if parent:

        os.makedirs(parent, exist_ok=True)

    with open(mp, "w", encoding="utf-8") as mf:

        json.dump(out_payload, mf, indent=2, ensure_ascii=False)



    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nWrote {mp}")

    if args.bad_cases_jsonl:

        print(f"Bad cases: {os.path.abspath(args.bad_cases_jsonl)}")





if __name__ == "__main__":

    main()


