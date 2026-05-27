# 从监督微调到评测对比：命令行流程

本文约定：**除特别说明外，均在仓库根目录**（`minimind/`）执行；需要跑 `train_*.py` 时，**先 `cd trainer`**，因为脚本里使用 `../out`、`../dataset` 等相对路径。

多行命令换行：在 **CMD** 里用行末 `^`；在 **Bash**（Linux / AutoDL）里用行末 `\`。**AutoDL / Linux 切勿复制 `^` 续行**，否则每行会被当成独立命令。也可写成一行。

---

## 0. 环境与数据

- 安装依赖：`pip install -r requirements.txt`（含 `torch`、`transformers`、`trl`、`openai` 等）。
- 准备 **`dataset/`** 下 SFT 用 jsonl（如 `sft_t2t.jsonl` / `sft_t2t_mini.jsonl`），格式与仓库 `SFTDataset` 一致（含 `conversations`，可选 `tools` / `tool_calls`）。
- **`model/`** 目录：与训练一致的 **Tokenizer**（`AutoTokenizer.from_pretrained('../model')`）。
- 评测用 **holdout**（若还没有）：在仓库根目录执行：

```bash
python project_intent_router/build_tool_eval_holdout.py \
  --input dataset/sft_t2t.jsonl \
  --output dataset/tool_eval_holdout.jsonl \
  --seed 42 --holdout_n 1000
```

默认只保留白名单内工具；若要全量 tool 样本，加 `--allow_any_tool`。

---

## 1. 全参数监督微调（MiniMind SFT）

在 **`trainer`** 目录下执行（权重写入 `../out/full_sft_{hidden_size}.pth`）：

```bash
cd trainer
python train_full_sft.py --use_wandb \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --save_weight full_sft \
  --hidden_size 768 --num_hidden_layers 8 --use_moe 0
```

按需修改：`--data_path`、`--epochs`、`--batch_size`、`--from_weight`（如已从 `none` 训过则改）、`--use_moe 1`（MoE）、`--max_seq_len` 等。训完后回到仓库根目录继续下面步骤。

---

## 2.（可选）Agent 强化学习

若要做 `agent` 权重，仍在 **`trainer`** 下：

```bash
cd trainer
python train_agent.py --use_wandb --rollout_engine torch \
  --from_weight full_sft --save_weight agent \
  --hidden_size 768 --use_moe 0
```

单卡显存紧张时可加：`--num_generations 2 --max_gen_len 384 --batch_size 2`。若使用 SGLang，需先另起进程启动服务，并设 `--rollout_engine sglang`。

---

## 3. MiniMind 工具调用评测 + Bad Case 记录

在**仓库根目录**执行；`--weight` 与训练保存前缀一致（如 `full_sft` 或 `agent`），`--use_moe` 与训练一致。

```bash
cd /path/to/minimind

python project_intent_router/eval_intent_router_metrics.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cuda \
  --max_samples 500 \
  --load_from model \
  --save_dir out \
  --weight full_sft \
  --use_moe 0 \
  --hidden_size 768 --num_hidden_layers 8 \
  --metrics_json project_intent_router/metrics_full_sft.json \
  --bad_cases_jsonl project_intent_router/bad_cases_full_sft.jsonl
```

产出：

- `metrics_*.json`：`summary`（准确率、延迟等）+ 部分 `details`。
- `bad_cases_*.jsonl`：仅 **bad_kind ≠ ok** 的样本，便于写报告 / 给教授看。

---

## 4.（对比）千问 / Llama 小模型：SFT 基线

首次会从 **Hugging Face Hub** 拉模型（需网络；Llama 需 **`huggingface-cli login`** 与官网许可）。在**仓库根目录**：

```bash
python project_prof/sft_hf_baseline.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
  --train_jsonl dataset/sft_t2t_mini.jsonl \
  --output_dir out_hf_sft_qwen05 \
  --max_samples 8000 \
  --max_steps 200 \
  --bf16 \
  --hf_endpoint https://hf-mirror.com \
  --hf_hub_timeout 300
```

**默认**只把 **含 `tools` 或 `tool_calls`** 的对话送进训练（与教授「工具/意图」基线一致）；控制台会打印 `kept_for_training` 等统计。若要整表不筛选，加 **`--no_tool_filter`**。与 `train_full_sft.py` 一致可加 **`--use_wandb`**（SwanLab）；官方 W&B 可用 **`--report_to wandb`**（勿与 swanlab 重复配置同一实验）。

**国内 / AutoDL 访问不了 huggingface.co 时**：在命令里加 `--hf_endpoint https://hf-mirror.com`，或先 `export HF_ENDPOINT=https://hf-mirror.com`；也可把已下载的模型放在磁盘上，让 **`--model_name_or_path` 指向本地目录**（含 `config.json` 与权重）。需要更长下载时间可加 `--hf_hub_timeout 300`。

Llama 示例：`--model_name_or_path meta-llama/Llama-3.2-1B-Instruct`（需 token）。训完得到 `out_hf_sft_qwen05/` 等目录。

---

## 5.（对比）HF 模型在同一 holdout 上评测

对 **刚训完的目录** 或 **官方基座** 跑同一套指标（可与 MiniMind 对比）：

```bash
python project_prof/eval_hf_baseline.py \
  --model_name_or_path out_hf_sft_qwen05 \
  --holdout dataset/tool_eval_holdout.jsonl \
  --device cuda \
  --max_samples 500 \
  --metrics_json project_prof/metrics_hf_qwen_sft.json \
  --bad_cases_jsonl project_prof/bad_cases_hf_qwen_sft.jsonl \
  --run_name hf_qwen_sft
```

---

## 6.（对比）大模型 API Agent 评测

使用 **OpenAI 兼容接口**（示例为阿里云百炼兼容地址）。先设置密钥：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="你的key"

# Linux
export DASHSCOPE_API_KEY=你的key
```

在**仓库根目录**：

```bash
python project_prof/agent_api_eval.py \
  --holdout dataset/tool_eval_holdout.jsonl \
  --max_samples 100 \
  --api_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api_model qwen-turbo \
  --metrics_json project_prof/metrics_api_qwen.json \
  --bad_cases_jsonl project_prof/bad_cases_api.jsonl
```

OpenAI 官方：把 `--api_base_url` 改为 `https://api.openai.com/v1`，并设 `OPENAI_API_KEY`。可选 **`--two_round_agent`**：首轮 tool 后注入 mock tool 结果再要第二轮回复。

---

## 7. 汇总对比表

将多份 `metrics_*.json` 的 `summary` 合并成一个 JSON，便于做表：

```bash
python project_prof/aggregate_comparison.py \
  project_intent_router/metrics_full_sft.json \
  project_prof/metrics_hf_qwen_sft.json \
  project_prof/metrics_api_qwen.json \
  --out project_prof/comparison_table.json
```

---

## 8. 流程小结（建议顺序）

| 顺序 | 步骤 | 目录 |
|------|------|------|
| 1 | 准备数据 + 生成 `tool_eval_holdout.jsonl` | 根目录 |
| 2 | `train_full_sft.py`（及可选 `train_agent.py`） | `trainer/` |
| 3 | `eval_intent_router_metrics.py` + `--bad_cases_jsonl` | 根目录 |
| 4 | `sft_hf_baseline.py` → `eval_hf_baseline.py` | 根目录 |
| 5 | `agent_api_eval.py` | 根目录 |
| 6 | `aggregate_comparison.py` | 根目录 |

以上为从 **监督微调** 到 **评测、bad case、基线对比、API 对比** 的完整命令行参考；参数请按你本机数据路径、显卡和课程要求再改一版即可。
