#!/bin/bash
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

echo "[1/4] Evaluating SFT Dense..."
python -u project_intent_router/eval_intent_router_metrics.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cpu --max_samples 300 --weight full_sft --use_moe 0 \
  --metrics_json project_intent_router/full_sft_768_metrics.json \
  --bad_cases_jsonl project_intent_router/full_sft_768_bad_cases.jsonl

echo "[2/4] Evaluating SFT MoE..."
python -u project_intent_router/eval_intent_router_metrics.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cpu --max_samples 300 --weight full_sft --use_moe 1 \
  --metrics_json project_intent_router/full_sft_768_moe_metrics.json \
  --bad_cases_jsonl project_intent_router/full_sft_768_moe_bad_cases.jsonl

echo "[3/4] Evaluating Agent RL Dense..."
python -u project_intent_router/eval_intent_router_metrics.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cpu --max_samples 300 --weight agent --use_moe 0 \
  --metrics_json project_intent_router/agent_768_metrics.json \
  --bad_cases_jsonl project_intent_router/agent_768_bad_cases.jsonl

echo "[4/4] Evaluating Agent MoE..."
python -u project_intent_router/eval_intent_router_metrics.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cpu --max_samples 300 --weight agent --use_moe 1 \
  --metrics_json project_intent_router/agent_768_moe_metrics.json \
  --bad_cases_jsonl project_intent_router/agent_768_moe_bad_cases.jsonl

echo "All MiniMind evaluations complete."
