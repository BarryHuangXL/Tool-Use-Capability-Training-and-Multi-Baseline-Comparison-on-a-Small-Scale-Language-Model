"""
Minimal agent loop demo: local MiniMind -> parse <tool_call> -> mock tool -> next turn.
Reuses mock executors from scripts/eval_toolcall.py.

Run from repo root:
  python project_intent_router/router_agent_demo.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import eval_toolcall as et  # noqa: E402
from trainer.trainer_utils import setup_seed  # noqa: E402


DEMO_PROMPTS = [
    "帮我算一下 256 乘以 37 等于多少",
    "北京今天天气怎么样？",
]


def run_demo_session(prompt: str, tools: list, args: argparse.Namespace, model, tokenizer) -> None:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    setup_seed(42)
    max_tool_rounds = 5

    for _ in range(max_tool_rounds):
        content = et.generate(model, tokenizer, messages, tools, args)
        tool_calls = et.parse_tool_calls(content)
        if not tool_calls:
            print("[done: no <tool_call> in assistant output]")
            return

        messages.append({"role": "assistant", "content": content})
        for tc in tool_calls:
            name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            arg_str = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else str(arguments)
            print(f"📞 [Tool Calling]: {name} | args={arg_str}")
            result = et.execute_tool(tc)
            print(f"✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}")
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    print(f"[stop: exceeded {max_tool_rounds} tool rounds]")


def main() -> None:
    p = argparse.ArgumentParser(description="Intent router demo (mock tools)")
    p.add_argument("--load_from", type=str, default="model")
    p.add_argument("--save_dir", type=str, default="out")
    p.add_argument("--weight", type=str, default="full_sft")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--show_speed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tools = et.TOOLS
    model, tokenizer = et.init_model(args)

    for i, prompt in enumerate(DEMO_PROMPTS):
        print(f"\n{'=' * 20} Demo {i + 1} {'=' * 20}\n💬: {prompt}\n")
        run_demo_session(prompt, tools, args, model, tokenizer)


if __name__ == "__main__":
    main()
