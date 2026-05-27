# MiniMind Tool-Use Training & Evaluation Project
HKUST MSBD6910
> A complete small-scale LLM post-training pipeline built from scratch, covering pretraining, SFT, DPO, RLAIF, and Agent RL. Specializing in tool-calling (Function Calling) capability with multi-baseline comparison and improved evaluation metrics.

---

## Overview

This project builds a full post-training pipeline for a small language model from the ground up. The key contributions include:

1. **Custom Model Architecture**: A LLaMA-style MiniMind model (768-dim / 8 layers, ~64M active parameters) with optional MoE (4 experts, Top-1 routing, ~198M total params).
2. **Complete Training Pipeline**: Pretrain → Full SFT → Agent RL, covering key stages of small-model post-training.
3. **Tool-Calling Specialization**: 7.43% of SFT data contains `tool_calls`, further enhanced by Agent RL (GRPO + InternLM2 Reward Model).
4. **Fair Multi-Baseline Comparison**: Benchmarked against HF Qwen-0.5B SFT and Alibaba Cloud qwen-turbo API on a unified holdout test set.
5. **Improved Evaluation Metrics**: Identified severe limitations of `first_tool_name_accuracy`, proposing **Overall F1** and **Exact Match Rate** as more comprehensive replacements, with fine-grained bad-case analysis.

## Model Architecture

```
MiniMindForCausalLM
├── Embedding (vocab=6400, hidden=768)
├── 8 × Transformer Block
│   ├── RMSNorm + Attention (GQA, 8Q/4KV, FlashAttn, RoPE)
│   └── RMSNorm + MLP (SwiGLU or MoE-Top1)
├── RMSNorm
└── LM Head (shared weight with Embedding)
```

Key Features:
- **Positional Encoding**: RoPE + YaRN (max context 32K)
- **Attention**: Grouped Query Attention (8 Q heads / 4 KV heads)
- **FFN**: SwiGLU (standard) or MoE (4 experts, Top-1 routing, load-balancing Aux Loss)
- **Normalization**: RMSNorm
- **Inference Acceleration**: Flash Attention + KV Cache

## Training Data

| Dataset | Size | Purpose |
|---------|------|---------|
| `pretrain_t2t_mini.jsonl` | 1.27M | Pretraining (Next Token Prediction) |
| `sft_t2t_mini.jsonl` | 895K | Supervised Fine-Tuning (7.43% with tool_calls) |
| `agent_rl.jsonl` | 40K | Agent RL General Tasks |
| `tool_eval_holdout.jsonl` | 1,000 | Tool-Calling Holdout Evaluation Set |

## Training Pipeline

```
Pretrain (pretrain_768.pth)
    ↓
Full SFT (full_sft_768.pth / full_sft_768_moe.pth)
    ↓
Agent RL (agent_768.pth / agent_768_moe.pth)
    ↓
Evaluation on tool_eval_holdout.jsonl
```

### Reinforcement Learning (GRPO + Reward Model)

- **Policy Optimization**: GRPO (Group Relative Policy Optimization), no separate Critic network
- **Reward Model**: InternLM2-1.8B-Reward for quality scoring of (prompt, answer)
- **Rule-Based Rewards**:
  - Length reward: response length within 20~800 characters
  - Chain-of-thought reward: correct `</think>` format and moderate length
  - Repetition penalty: n-gram repetition penalty
  - Tool alignment reward: tool name matching, argument matching

## Evaluation

### Test Set

`tool_eval_holdout.jsonl`: 1,000 tool-use evaluation samples covering weather, math, translation, exchange rates, time, etc. Mixed Chinese/English queries, some with distractor tools.

### Core Metrics (v2)

| Metric | Description |
|--------|-------------|
| **Overall F1** | Harmonic mean of tool-name set F1 and argument F1; replaces first_tool_name_accuracy |
| **Exact Match Rate** | Strict metric: tool name, arguments, and order all match gold exactly |
| **Tool-Name F1** | Set-based precision/recall/f1; order-agnostic for multi-tool scenarios |
| **Argument F1** | Key-value pair matching; evaluates argument accuracy |

### Fine-Grained Bad Case Categories

`ok` / `no_tool_call_in_output` / `invalid_structure_or_args` / `wrong_args_only` / `missing_required_tools` / `extra_wrong_tools` / `wrong_first_but_others_ok` / `all_tools_wrong`

## Results

### Six-Scheme Comparison (v2 Metrics, First 100 Holdout Samples)

| Scheme | Params | Overall F1 | Exact Match | Tool-Name F1 | Arg F1 | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| MiniMind SFT (Dense) | 63M | 44.57% | 40.00% | 49.67% | 43.89% | 1.09s |
| **MiniMind SFT + MoE** | 198M | **85.31%** | **73.00%** | **89.57%** | **84.68%** | 1.89s |
| MiniMind Agent RL (Dense) | 63M | 79.20% | 54.00% | 88.97% | 75.65% | 1.27s |
| MiniMind Agent + MoE | 198M | 79.13% | 54.00% | 89.43% | 76.08% | 2.02s |
| HF Qwen-0.5B SFT | 500M | 72.52% | 62.00% | 79.33% | 71.18% | 2.85s |
| API qwen-turbo | Cloud | 67.69% | 53.00% | 75.23% | 65.87% | 0.82s |

### Key Findings

1. **first_tool_name_accuracy severely overestimates capability**: Agent RL scored 94.67% on the old metric but only 54% Exact Match, revealing that the model learned to output the correct first tool name while struggling with argument precision and extra tool control.
2. **SFT + MoE achieves the best overall performance**: Overall F1 = 85.31%, Exact Match = 73%, both highest among all six schemes.
3. **Agent RL introduces numerous argument errors**: 39% of samples are `wrong_args_only` (correct tool name, wrong arguments), suggesting the RL reward function lacks sufficient constraint on argument precision.
4. **API Agent abuses web_search**: 15 instances incorrectly used `web_search` instead of `get_current_weather`.
5. **Parameter precision is the common bottleneck**: Even with tool-name F1 as high as 88~89%, argument F1 only reaches 65~76% across all schemes.

## Quick Start

### Environment Setup

```bash
cd minimind
pip install -r requirements.txt
```

### Training

```bash
# Pretraining
python trainer/train_pretrain.py

# SFT
python trainer/train_full_sft.py --weight pretrain

# Agent RL (recommended, best results)
python trainer/train_agent.py --weight full_sft
```

### Evaluation

```bash
# Local model evaluation
python project_intent_router/eval_intent_router_metrics.py \
  --weight agent --hidden_size 768

# API baseline evaluation
python project_prof/agent_api_eval_v2.py

# v2 metric recalculation and analysis
python recalculate_and_analyze.py
```

## Project Structure

```
minimind/
├── model/              # Model architecture (model_minimind.py, model_lora.py)
├── dataset/            # Training data and evaluation sets
├── trainer/            # Training scripts (pretrain, sft, dpo, grpo, agent)
├── out/                # Trained model weights
├── project_intent_router/   # Local model evaluation & Agent demo
├── project_prof/            # Baseline comparison (HF Qwen, API Agent)
├── scripts/            # Utility scripts (conversion, API service, Web demo)
└── images/             # Architecture diagrams & visualizations
```

## Citation

If this project helps your research, please consider citing:

```bibtex
@software{minimind,
  title={MiniMind: A Minimalist LLM Training Framework},
  author={Jingyao Gong and contributors},
  year={2025},
  url={https://github.com/jingyaogong/minimind}
}
```

## License

Apache 2.0 License
