# Training Coding Agents with RL

A step-by-step guide for reproducing **Nemotron-style competitive-coding RL** on top of `Qwen3-1.7B` using this repository. This document is meant to be approachable for someone encountering the codebase for the first time — every path, every command, and every knob you are likely to touch is spelled out below.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Evaluation Data](#5-evaluation-data)
6. [Configuring the Training Script](#6-configuring-the-training-script)
7. [Running Training](#7-running-training)
8. [Default Hyperparameters](#8-default-hyperparameters)
9. [Out-of-Memory Survival Guide](#9-out-of-memory-survival-guide)

---

## 1. Overview

This pipeline performs **GRPO-based reinforcement learning** on a coding task, following the recipe introduced in the Nemotron paper:

- **Base model**: `Qwen/Qwen3-1.7B`
- **Training data**: `nvidia/Nemotron-RL-coding-competitive_coding`
- **Validation benchmark**: LiveBench (coding subset)
- **Reward**: a custom judge defined at [evaluation/eval/verl_custom_reward.py](evaluation/eval/verl_custom_reward.py)
- **Trainer**: [verl](https://github.com/volcengine/verl), invoked via [run_qwen3_1.7b_coding.sh](run_qwen3_1.7b_coding.sh)

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Docker (recommended) | `docker.io/snow12345/verl_geeho:latest` already bundles `verl`, `vllm`, and `megatron`. |
| W&B account | Optional but the script logs there by default. |

---

## 3. Environment Setup

### Clone the repository

Grab the code and switch to the `rl-merging` branch — that is where the coding-RL pipeline lives:

```bash
git clone https://github.com/geehokim/verl_nemotron_merge.git
cd verl_nemotron_merge
git checkout rl-merging
```

### Option A — Docker (recommended)

```bash
docker pull snow12345/verl_geeho:latest
docker run --gpus all -it --rm \
  -v /home2/geeho/tmp/verl_nemotron_merge:/workspace \
  -v /131_data/geeho:/131_data/geeho \
  snow12345/verl_geeho:latest bash
```

### ⚠️ Hard-coded paths you MUST edit inside `run_qwen3_1.7b_coding.sh`

[run_qwen3_1.7b_coding.sh](run_qwen3_1.7b_coding.sh) currently ships with absolute paths baked in for the original author's machine. Before you launch anything, open the script and **manually replace** these two lines with values that make sense on **your** system — they are literal strings in the file, not environment-variable overrides:

```bash
# Near line 40 — must point to YOUR checkout of this repository
export PYTHONPATH="/home2/geeho/tmp/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

# Near line 43 — must point to a writable directory on YOUR machine
OUTPUT_DIR="/131_data/geeho/nemotron_cascade_output/Qwen3-1.7B-coding"
```

- **`PYTHONPATH`** — the absolute path to the root of this repo. If you do not update it, `verl` will import the wrong modules (or fail to find them entirely).
- **`OUTPUT_DIR`** — where checkpoints, logs, and resume state are written. Make sure the parent exists and has tens of gigabytes free.

If you skip this step, the script will either crash immediately on import or silently write checkpoints into a path you do not own.

---

## 4. Dataset Preparation

The training data comes from Hugging Face. Pick **one** of the two methods below.

### Method 1 — `git lfs`

```bash
sudo apt install git-lfs
git lfs install

git clone https://huggingface.co/datasets/nvidia/Nemotron-RL-coding-competitive_coding
cd Nemotron-RL-coding-competitive_coding
git lfs pull
```

### Method 2 — `huggingface-cli` (recommended, faster)

```bash
pip install huggingface_hub
huggingface-cli download nvidia/Nemotron-RL-coding-competitive_coding \
  --repo-type dataset \
  --local-dir ./Nemotron-RL-coding-competitive_coding
```

### Point the script at your data

The script reads its data root from the `DATASET_ROOT` variable (defaults to `/131_data/geeho/data/Nemotron-RL-coding-competitive_coding`). Set it explicitly if your layout differs:

The directory must contain a `data/` subfolder populated with `*.parquet` shards. On the first run, a preflight step converts these shards into a verl-compatible cache at `${DATASET_ROOT}/verl_coding_cache/` — this is idempotent and only re-runs when sources change.

---

## 5. Evaluation Data

Validation uses the **LiveBench coding subset**.

1. Download `evaluation_data.zip` from the shared Drive:
   https://drive.google.com/drive/folders/1J6Ril6D4D03CWXI4bn8KnlMAq3pIQKDV?hl=ko
2. Unzip it. You will see an `evaluation_data/data/` folder.
3. Copy that `data/` folder into the repo at [evaluation/data](evaluation/data):

```bash
unzip evaluation_data.zip
cp -r evaluation_data/data verl_nemotron_merge/evaluation/data
```

The script's preflight pass converts `evaluation/data/livebench/LiveBench.json` into `livebench_verl_ready.parquet` automatically.

---

## 6. Configuring the Training Script

Open [run_qwen3_1.7b_coding.sh](run_qwen3_1.7b_coding.sh) and adjust these two lines to match your machine:

```bash
export PYTHONPATH="/home2/geeho/tmp/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"
OUTPUT_DIR="/131_data/geeho/nemotron_cascade_output/Qwen3-1.7B-coding"
```

- `PYTHONPATH` — the absolute path to this repository.
- `OUTPUT_DIR` — where checkpoints and logs are written. Make sure the parent directory exists and has enough free space.

Everything else has sensible defaults (see [§8](#8-default-hyperparameters)).

---

## 7. Running Training

### Smoke test (2–4 GPUs, minutes)

Use this to verify the full pipeline end-to-end before committing to a real run:

```bash
SMOKE_MODE=true bash run_qwen3_1.7b_coding.sh
```

Smoke mode shrinks the batch sizes, caps response length to 2048, uses only 4 GPUs, and runs on a tiny subset of training samples.

### Full training run (8 GPUs)

```bash
bash run_qwen3_1.7b_coding.sh
```

The script runs for **200 optimizer steps** (following the Nemotron paper), evaluates LiveBench every 10 steps, and saves checkpoints every 20 steps.

---

## 8. Default Hyperparameters

All values below reflect the current defaults in [run_qwen3_1.7b_coding.sh](run_qwen3_1.7b_coding.sh). Smoke-mode overrides are listed in parentheses.

### Training loop

| Knob | Default | Smoke | Meaning |
|---|---|---|---|
| `TOTAL_TRAINING_STEPS` | `200` | `200` | Optimizer steps (Nemotron recipe). |
| `TOTAL_EPOCHS` | `1` | `1` | Cap on dataset passes. |
| `TRAIN_BATCH_SIZE` | `128` | `4` | Prompts per optimizer step. |
| `VAL_BATCH_SIZE` | `64` | `4` | Prompts per validation step. |
| `SAVE_FREQ` | `20` | `10` | Checkpoint cadence (steps). |
| `TEST_FREQ` | `10` | `5` | LiveBench eval cadence (steps). |
| `TRAIN_MAX_SAMPLES` | `-1` (all) | `64` | Cap on training rows. |
| `VAL_MAX_SAMPLES` | `32` | `32` | Cap on validation rows. |

### Optimizer & loss

| Knob | Default | Meaning |
|---|---|---|
| `actor.optim.lr` | `2e-6` | Learning rate. Nemotron-style low LR — raising it is the fastest way to destabilize GRPO. |
| `actor.optim.betas` | `[0.9, 0.95]` | Adam betas. |
| `actor.use_kl_loss` | `True` |  |
| `actor.kl_loss_coef` | `0.002` | — |
| `actor.entropy_coeff` | `0.0` | — |
| `algorithm.adv_estimator` | `grpo` | GRPO advantage. |
| `LOSS_AGG_MODE` | `seq-mean-token-sum-norm` | Sequence-level aggregation. |

### Sequence lengths

| Knob | Default | Smoke | Meaning |
|---|---|---|---|
| `data.max_prompt_length` | `2048` | `2048` | Truncate prompts longer than this. |
| `MAX_RESPONSE_LENGTH` | `32768` | `2048` | Max generated tokens per rollout. **This dominates GPU memory.** |
| `ROLLOUT_MAX_NUM_BATCHED_TOKENS` | `34816` | `8192` | vLLM scheduler budget (must comfortably exceed `prompt + response`). |

### Rollout / vLLM

| Knob | Default | Smoke | Meaning |
|---|---|---|---|
| `ROLLOUT_N` | `8` | `1` | Rollouts per prompt (GRPO group size). |
| `VAL_ROLLOUT_N` | `8` | `8` | Rollouts per validation prompt. |
| `rollout.temperature` | `1.0` | — | Train-time sampling temperature. |
| `rollout.top_p` | `1.0` | — | Train-time nucleus (disabled). |
| `rollout.top_k` | `-1` | — | Train-time top-k (disabled). |
| `val_kwargs.temperature` | `0.6` | — | Qwen3-Thinking recommended eval temp. |
| `val_kwargs.top_p` | `0.95` | — | Qwen3-Thinking recommended. |
| `val_kwargs.top_k` | `20` | — | Qwen3-Thinking recommended. |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | `0.90` | `0.75` | Share of GPU memory vLLM may claim. |
| `AGENT_NUM_WORKERS` | `8` | `1` | Parallel agent workers; must divide the gen batch. |
| `rollout.enable_chunked_prefill` | `True` | — | Long-prompt friendly. |
| `rollout.enable_prefix_caching` | `True` | — | Speeds up GRPO group sampling. |
| `rollout.free_cache_engine` | `True` | — | Frees the KV cache between phases. |

### PPO / parallelism

| Knob | Default | Smoke | Meaning |
|---|---|---|---|
| `PPO_MINI_BATCH_SIZE` | `128` | `4` | PPO inner mini-batch. |
| `PPO_MICRO_BATCH_SIZE_PER_GPU` | `64` | `4` | Micro-batch per GPU for the actor update. |
| `ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | `16` | `4` | Log-prob recompute micro-batch. |
| `ULYSSES_SEQUENCE_PARALLEL_SIZE` | `4` | `4` | Ulysses SP degree — splits long sequences across GPUs. |
| `MACHINE_GPU_COUNT` | `8` | `4` | GPUs per node. |
| `WORLD_SIZE` | `1` | `1` | Number of nodes. |

### Memory offload / precision

| Knob | Default | Meaning |
|---|---|---|
| `actor.fsdp_config.param_offload` | `True` | Offload actor params to CPU. |
| `actor.fsdp_config.optimizer_offload` | `True` | Offload optimizer state to CPU. |
| `ref.fsdp_config.param_offload` | `True` | Offload reference params. |
| `actor.use_dynamic_bsz` | `True` | Pack variable-length sequences. |
| `ref.log_prob_use_dynamic_bsz` | `True` | Same for reference log-prob pass. |
| `model.enable_gradient_checkpointing` | `True` | Trades compute for memory. |
| `DTYPE` | `float16` | Actor + rollout dtype. |

---

## 9. Out-of-Memory Survival Guide

If you hit a `CUDA out of memory` error, work through these levers **in order**. Each one is labeled with the approximate cost/benefit.

### 9.1 Shrink the generation footprint (biggest wins)

1. **`ROLLOUT_GPU_MEMORY_UTILIZATION`** — lowering from `0.90 → 0.80 → 0.70` leaves more headroom for the FSDP actor shards during the update phase. Fixes most "OOM between rollout and update" crashes.

### 9.2 Shrink the update footprint

2. **`PPO_MICRO_BATCH_SIZE_PER_GPU`** — halve it (`64 → 32 → 16 → 8`). Slower, but linear memory savings on the actor update.
3. **`ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU`** — same idea for the old-logprob / reference-logprob passes (`16 → 8 → 4`).
4. **`ULYSSES_SEQUENCE_PARALLEL_SIZE`** — increase to split long sequences across more GPUs (e.g. `4 → 8` if you have 8 GPUs). Must divide `MACHINE_GPU_COUNT`.

### 9.3 Batch-level reductions

6. **`TRAIN_BATCH_SIZE`** and **`PPO_MINI_BATCH_SIZE`** — reduce together (they are equal by default). Halving both is safe and linear.
7. **`AGENT_NUM_WORKERS`** — must divide the per-step gen batch. If you shrink `TRAIN_BATCH_SIZE * ROLLOUT_N` you may need to shrink workers proportionally.
