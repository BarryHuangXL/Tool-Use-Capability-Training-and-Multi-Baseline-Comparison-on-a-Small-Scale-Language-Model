# agent_768的模型评估
python project_intent_router/eval_intent_router_metrics.py --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 300 --weight agent --use_moe 0 --metrics_json "project_intent_router/agent_768_metrics.json" --bad_cases_jsonl "project_intent_router/agent_768_bad_cases.jsonl"

# agent_768_moe 的模型评估
python project_intent_router/eval_intent_router_metrics.py --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 300 --weight agent --use_moe 1 --metrics_json "project_intent_router/agent_768_moe_metrics.json"

# full_sft_768
python project_intent_router/eval_intent_router_metrics.py --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 300 --weight full_sft --use_moe 0 --metrics_json "project_intent_router/full_sft_768_metrics.json"


# full_sft_768_moe
python project_intent_router/eval_intent_router_metrics.py --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 300 --weight full_sft --use_moe 1 --metrics_json "project_intent_router/full_sft_768_moe_metrics.json"

# --- 教授建议：bad case 记录（与 metrics 同一次跑）---
python project_intent_router/eval_intent_router_metrics.py --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 500 --weight full_sft --metrics_json project_intent_router/metrics_with_bad.json --bad_cases_jsonl project_prof/bad_cases_minimind.jsonl

# --- HF 小模型基线评测（需已下载权重；Llama 需 HF_TOKEN）---
python project_prof/eval_hf_baseline.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --holdout dataset/tool_eval_holdout.jsonl --device cuda --max_samples 200 --metrics_json project_prof/metrics_qwen05.json --bad_cases_jsonl project_prof/bad_cases_qwen05.jsonl

# --- 云端大模型 API 评测（千问兼容 OpenAI：设 DASHSCOPE_API_KEY）---
python project_prof/agent_api_eval.py --holdout dataset/tool_eval_holdout.jsonl --max_samples 100 --api_model qwen-turbo --metrics_json project_prof/metrics_api_qwen.json --bad_cases_jsonl project_prof/bad_cases_api.jsonl

# --- HF 小模型 SFT 基线（子集快训后再用 eval_hf_baseline 对上面对比）---
python project_prof/sft_hf_baseline.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --train_jsonl dataset/sft_t2t_mini.jsonl --output_dir out_hf_sft_qwen05 --max_samples 5000 --max_steps 150 --bf16

# --- 汇总对比表 ---
python project_prof/aggregate_comparison.py project_intent_router/metrics_with_bad.json project_prof/metrics_qwen05.json project_prof/metrics_api_qwen.json --out project_prof/comparison_table.json
