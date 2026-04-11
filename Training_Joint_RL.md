# Joint RL Training (IF + Coding + Math)

A step-by-step guide for running **joint GRPO RL** across Instruction Following, Competitive Coding, and Math on top of `Qwen3-1.7B` in a **single training run**, using [run_qwen3_1.7b_joint.sh](run_qwen3_1.7b_joint.sh).

Most of the setup (Docker image, code checkout, OOM survival guide, etc.) is **identical to the coding-only guide** in [Training_Coding_Agents_RL.md](Training_Coding_Agents_RL.md) — this document only covers what is different: the **joint-training overview**, the **data configuration** for the three tasks, and the **unified hyperparameters**.

---

## 1. Overview — How Joint Training Works

The pipeline trains a **single policy** on a **mixed batch of IF + Coding + Math prompts** at every optimizer step, and computes a **GRPO loss per task without any extra routing logic**. The key ideas:

1. **Uniform mixed batch construction.** Three training parquets (IF, Coding, Math) are passed as a list to `data.train_files`. A custom dataset wrapper, `MultiTaskRLHFDataset`, loads each parquet into its own sub-dataset so the incompatible nested schemas (e.g. IF stores `reward_model.ground_truth` as a struct while Math stores it as a flat string) never have to be merged at the Arrow level. A custom sampler, `UniformMultiTaskSampler`, then yields indices in chunks of `train_batch_size` where **each chunk has a per-task quota that sums to the batch size** (with `B=128` and 3 tasks the quota is `(43, 43, 42)`). The result is that every optimizer step sees prompts from all three tasks in a known ratio.
2. **Rollout.** vLLM generates `ROLLOUT_N=8` responses per prompt in one call, regardless of which task each prompt came from. All prompts in the batch share the same sampling config (`temperature=1.0, top_p=1.0, top_k=-1`).
3. **Per-sample reward dispatch.** A single custom reward function, [evaluation/eval/verl_custom_reward.py](evaluation/eval/verl_custom_reward.py), inspects each sample's `data_source` field and routes to the right verifier:
    - `if` → [verl/utils/reward_score/if_reward.py](verl/utils/reward_score/if_reward.py)
    - `nemotron_cascade_rl_coding` → coding verifier
    - `nemotron_cascade_rl_math` → [verl/utils/reward_score/nemotron_cascade_rl_math.py](verl/utils/reward_score/nemotron_cascade_rl_math.py)
   No routing work happens at the training-loop level — it is all per-sample inside `compute_score`.
4. **GRPO advantage per-prompt.** GRPO normalizes each prompt's advantage against its own `ROLLOUT_N` samples **only** (the group is identified by `uid`, not by task). Mixing tasks in the same batch therefore does **not** corrupt the normalization — each prompt's `n` rollouts are compared exclusively among themselves.
5. **Joint gradient update.** The actor loss is summed across all prompts in the batch (IF + Coding + Math together) and backpropagated in one optimizer step. Because the reward scales and advantage baselines are already normalized per-prompt by GRPO, the three tasks contribute comparably without needing hand-tuned per-task loss weights.

**Validation** runs **per task, sequentially**. `_validate()` in [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) iterates over each validation parquet independently (rollout → reward → metric → dump), so the IF/AIME24/LiveBench evaluators never share state and per-task metrics land in wandb under separate `val-core/{data_source}/...` keys.

---

## 2. Data Configuration

You need **two** downloads from the shared Drive: one for evaluation data, one for training data.

> **Drive root**: https://drive.google.com/drive/folders/1J6Ril6D4D03CWXI4bn8KnlMAq3pIQKDV

### 2.1 Evaluation data (IFEval + AIME24 + LiveBench)

1. Download **`evaluation_data.zip`** from the Drive.
2. Unzip it — you will see an `evaluation_data/data/` folder.
3. Copy that `data/` folder into the repo at [evaluation/data](evaluation/data):

```bash
unzip evaluation_data.zip
cp -r evaluation_data/data/* verl_nemotron_merge/evaluation/data/
```

The joint script's preflight pass converts the raw JSON/JSONL files into verl-ready parquets automatically:

| Benchmark | Raw input | Generated parquet |
|---|---|---|
| IFEval | `evaluation/data/ifeval/input_data.jsonl` | `evaluation/data/ifeval/input_data_verl_ready.parquet` |
| AIME24 | `evaluation/data/aime24/aime24_verl_ready.parquet` | (shipped ready) |
| LiveBench (coding) | `evaluation/data/livebench/LiveBench.json` | `evaluation/data/livebench/livebench_verl_ready.parquet` |

### 2.2 Training data (IF + Coding + Math)

All three training datasets live in the **`training_data/`** folder on the same Drive.

1. Download the entire `training_data/` folder from the Drive.
2. Place each dataset under your preferred data root. The defaults in [run_qwen3_1.7b_joint.sh](run_qwen3_1.7b_joint.sh) expect:

    | Task | Expected path |
    |---|---|
    | **IF**     | `/131_data/geeho/data/Nemotron-Cascade-RL-Instruction-Following/ifrl_if_verl_ready.parquet` |
    | **Coding** | `/131_data/geeho/data/Nemotron-RL-coding-competitive_coding/` (same folder as the coding-only guide; preflight will populate `verl_coding_cache/` from `data/`) |
    | **Math**   | `/131_data/geeho/data/Nemotron-Cascade-RL-Math/math_verl_ready.parquet` |

    The coding dataset is identical to the one in [Training_Coding_Agents_RL.md §4](Training_Coding_Agents_RL.md#4-dataset-preparation) — if you already downloaded it for coding-only training you can reuse it in place; just drop the IF and Math parquets next to it under the same parent.

### 2.3 Paths you MUST edit in `run_qwen3_1.7b_joint.sh`

Open [run_qwen3_1.7b_joint.sh](run_qwen3_1.7b_joint.sh) and replace the following literals if your layout differs from the defaults:

```bash
# Output directory — checkpoints, logs, resume state, validation dumps
OUTPUT_DIR="/131_data/geeho/rl_merging_output/Qwen3-1.7B-joint-if-coding-math"

# Training data roots (one per task)
IF_TRAIN_PARQUET="/131_data/geeho/data/Nemotron-Cascade-RL-Instruction-Following/ifrl_if_verl_ready.parquet"
CODING_DATASET_ROOT="/131_data/geeho/data/Nemotron-RL-coding-competitive_coding"
MATH_TRAIN_PARQUET="/131_data/geeho/data/Nemotron-Cascade-RL-Math/math_verl_ready.parquet"
```

Everything else in the script (evaluation parquet paths, Python env activation, preflight converters) is relative to the repository and should not need changes.

---

## 3. Unified Hyperparameters

The joint run uses **one shared set of hyperparameters** across all three tasks — no per-task overrides — to keep the training loop simple and the GRPO advantage comparison fair. The relevant knobs are defined near the top of [run_qwen3_1.7b_joint.sh](run_qwen3_1.7b_joint.sh):

| Knob | Value | Notes |
|---|---|---|
| `rollout.temperature` | `1.0` | Unified across IF/Coding/Math. |
| `rollout.top_p` / `top_k` | `1.0` / `-1` | Disabled; pure temperature sampling. |
| `MAX_RESPONSE_LENGTH` | `32768` | Same budget for every task. Dominates GPU memory. |
| `ROLLOUT_N` | `8` | GRPO group size (rollouts per prompt). |
| `VAL_ROLLOUT_N` | `8` | Rollouts per validation prompt. |
| `TRAIN_BATCH_SIZE` | `128` | Prompts per optimizer step. Sampler splits this across tasks as `(43, 43, 42)`. |
| `VAL_BATCH_SIZE` | `128` | Per-task validation batch. |
| `PPO_MINI_BATCH_SIZE` | `128` | Same as train batch (no PPO inner loop fragmentation). |
| `actor.use_kl_loss` | `True` | KL regularization against the reference policy. |
| `actor.kl_loss_coef` | `0.002` | Unified KL coefficient. |
| `actor.entropy_coeff` | `0.0` | No entropy bonus. |
| `algorithm.adv_estimator` | `grpo` | Per-`uid` group normalization. |
| `reward_manager.name` | `naive` | **Required** for mixed IF/Coding/Math batches — `prime`'s `ProcessPoolExecutor` breaks the coding verifier's `signal.alarm()`. |

Validation sampling keeps Qwen3-Thinking's recommended defaults (`val_kwargs.temperature=0.6, top_p=0.95, top_k=20`).

---

## 4. Running Training

Identical to the coding-only flow, just with a different entrypoint script.

### Smoke test

```bash
SMOKE_MODE=true bash run_qwen3_1.7b_joint.sh
```

Shrinks batch size, response length, and per-task sample counts so you can validate the full IF + Coding + Math pipeline end-to-end in minutes.

### Full run (8 GPUs)

```bash
bash run_qwen3_1.7b_joint.sh
```

The run evaluates IFEval + AIME24 + LiveBench every `TEST_FREQ` steps (sequentially, per task) and saves checkpoints every `SAVE_FREQ` steps into `OUTPUT_DIR`.
