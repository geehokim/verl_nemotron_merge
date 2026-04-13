#!/bin/bash
# =============================================================================
# Joint RL training (Qwen3-1.7B) across IF + Coding + Math in a SINGLE run.
#
# Design:
#   - Three training parquets are passed as a list to data.train_files.
#     RLHFDataset concatenates them, so each mini-batch is naturally MIXED.
#   - Rewards are dispatched per-sample by `data_source` inside
#     evaluation/eval/verl_custom_reward.py::compute_score, so no per-sample
#     routing work is needed at the training loop level:
#         "if"                        -> IF reward    (if_reward.py)
#         "nemotron_cascade_rl_coding" -> coding reward
#         "nemotron_cascade_rl_math"   -> math reward  (nemotron_cascade_rl_math.py)
#   - GRPO's advantage is computed per-uid (per-prompt, within the n rollouts
#     of that prompt), so mixing tasks in one batch does not corrupt the
#     normalization — each prompt is normalized only against its own n samples.
#   - Reward manager = `naive`. Rationale: `prime` spawns a ProcessPoolExecutor
#     and the coding verifier uses `signal.alarm()` which does not work inside
#     worker subprocesses; that would silently break coding scoring in mixed
#     batches. `naive` runs scoring sequentially in-process, which is slower
#     but correct for all three task types.
#
# Validation: IFEval + AIME24 + LiveBench (all 3 parquets passed as a list).
# Metrics are automatically broken down by data_source inside ray_trainer._validate,
# so you get per-task val metrics in wandb.
# =============================================================================

set -x
set -e
export PYTHONWARNINGS="ignore::UserWarning:megatron"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Ensure runtime dependencies for IFEval + coding rewards are installed.
python - <<'PY'
import importlib
import subprocess
import sys

required = {
    "langdetect": "langdetect",
    "immutabledict": "immutabledict",
    "nltk": "nltk",
    "pyarrow": "pyarrow",
    "numpy": "numpy",
    "ftlangdetect": "fasttext-langdetect",
}
missing = []
for module_name, pip_name in required.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(pip_name)

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(set(missing))])

import nltk
for path, pkg in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]:
    try:
        nltk.data.find(path)
    except Exception:
        nltk.download(pkg, quiet=True)
PY

TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
if [[ "${TRAINER_LOGGER}" == *wandb* ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY:-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}"
else
    export WANDB_MODE=disabled
fi
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_V1=1

# =============================================================================
# Project / output
# =============================================================================

PROJECT_NAME="rl_merging"
OUTPUT_DIR="/131_data/geeho/rl_merging_output/Qwen3-1.7B-joint-if-coding-math"
BASE_MODEL="Qwen/Qwen3-1.7B"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8
SAVE_FREQ=20
DTYPE=float16
LOSS_AGG_MODE=seq-mean-token-sum-norm
VAL_ROLLOUT_N=8
TOTAL_EPOCHS=1
VAL_ONLY=false

SMOKE_MODE="${SMOKE_MODE:-false}"

# ----- Unified hyperparameters (per your spec) -----
TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=128
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=32768
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE_PER_GPU=64
ROLLOUT_N=8
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_GPU_MEMORY_UTILIZATION=0.85
ROLLOUT_MAX_NUM_BATCHED_TOKENS=34816
ULYSSES_SEQUENCE_PARALLEL_SIZE=4
TEST_FREQ=25
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=32
# agent_loop chunks the gen batch across num_workers; must divide evenly.
AGENT_NUM_WORKERS=8

if [ "${SMOKE_MODE}" = "true" ]; then
    MACHINE_GPU_COUNT=2
    TRAIN_BATCH_SIZE=4
    VAL_BATCH_SIZE=4
    MAX_RESPONSE_LENGTH=2048
    PPO_MINI_BATCH_SIZE=4
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_N=1
    ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.75
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048
    ULYSSES_SEQUENCE_PARALLEL_SIZE=2
    TEST_FREQ=5
    TRAIN_MAX_SAMPLES=64
    VAL_MAX_SAMPLES=1
    SAVE_FREQ=10
    AGENT_NUM_WORKERS=1
fi

EXPERIMENT_NAME="qwen3-1.7b-joint-if-coding-math"
if [ "${SMOKE_MODE}" = "true" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-smoke"
fi

echo "SMOKE_MODE=${SMOKE_MODE}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}, ROLLOUT_N=${ROLLOUT_N}"

# =============================================================================
# Training data (mixed: IF + Coding + Math)
# =============================================================================

# --- IF ---
IF_TRAIN_PARQUET="/131_data/geeho/data/Nemotron-Cascade-RL-Instruction-Following/ifrl_if_verl_ready.parquet"

# --- Coding: preflight-converted shards, read from manifest ---
CODING_DATASET_ROOT="/131_data/geeho/data/Nemotron-RL-coding-competitive_coding"
CODING_RAW_DIR="${CODING_DATASET_ROOT}/data"
CODING_TRAIN_CONVERTED_DIR="${CODING_DATASET_ROOT}/verl_coding_cache"
CODING_TRAIN_MANIFEST="${CODING_DATASET_ROOT}/train_files_manifest.json"
CODING_TRAIN_CONVERTER="${REPO_ROOT}/evaluation/data/coding/ensure_competitive_coding_verl_parquet.py"

# --- Math ---
MATH_TRAIN_PARQUET="/131_data/geeho/data/Nemotron-Cascade-RL-Math/math_verl_ready.parquet"

if [ ! -f "${IF_TRAIN_PARQUET}" ]; then
    echo "Missing IF training parquet: ${IF_TRAIN_PARQUET}"; exit 1
fi
if [ ! -f "${MATH_TRAIN_PARQUET}" ]; then
    echo "Missing Math training parquet: ${MATH_TRAIN_PARQUET}"; exit 1
fi
if [ ! -d "${CODING_RAW_DIR}" ]; then
    echo "Missing Coding raw parquet dir: ${CODING_RAW_DIR}"; exit 1
fi
if [ ! -f "${CODING_TRAIN_CONVERTER}" ]; then
    echo "Missing coding converter: ${CODING_TRAIN_CONVERTER}"; exit 1
fi

# Preflight: convert coding shards (idempotent; skips when already converted).
python "${CODING_TRAIN_CONVERTER}" \
    --input_dir "${CODING_RAW_DIR}" \
    --output_dir "${CODING_TRAIN_CONVERTED_DIR}" \
    --manifest_path "${CODING_TRAIN_MANIFEST}" \
    --data_source nemotron_cascade_rl_coding \
    --split train

# Build the combined train_files list (Hydra list syntax) by reading the
# coding manifest and appending IF and Math parquets.
TRAIN_FILES_HYDRA=$(python - <<PY
import json
from pathlib import Path

coding_manifest = Path("${CODING_TRAIN_MANIFEST}")
if not coding_manifest.exists():
    raise FileNotFoundError(f"coding manifest not found: {coding_manifest}")
coding = json.loads(coding_manifest.read_text(encoding="utf-8")).get("train_files", [])
if not coding:
    raise RuntimeError(f"empty train_files in coding manifest: {coding_manifest}")

if_parquet = "${IF_TRAIN_PARQUET}"
math_parquet = "${MATH_TRAIN_PARQUET}"

all_files = [if_parquet] + list(coding) + [math_parquet]
# RLHFDataset accepts a python list passed via Hydra list syntax: [a,b,c]
print("[" + ",".join(f"'{p}'" for p in all_files) + "]")
PY
)
echo "TRAIN_FILES_HYDRA=${TRAIN_FILES_HYDRA}"

# =============================================================================
# Validation data: IFEval + AIME24 + LiveBench
# =============================================================================

IFEVAL_JSON="${REPO_ROOT}/evaluation/data/ifeval/input_data.jsonl"
IFEVAL_CONVERTER="${REPO_ROOT}/evaluation/data/ifeval/convert_ifeval_to_verl_parquet.py"
IFEVAL_VAL_PARQUET="${REPO_ROOT}/evaluation/data/ifeval/input_data_verl_ready.parquet"

LIVEBENCH_JSON="${REPO_ROOT}/evaluation/data/livebench/LiveBench.json"
LIVEBENCH_CONVERTER="${REPO_ROOT}/evaluation/data/livebench/convert_livebench_to_verl_parquet.py"
LIVEBENCH_VAL_PARQUET="${REPO_ROOT}/evaluation/data/livebench/livebench_verl_ready.parquet"

# NOTE: `_verl_ready.parquet` has data_source="local/aime24" which does not
# route anywhere in verl/utils/reward_score/__init__.py (NotImplementedError).
# `aime24_verl_ready.parquet` is an in-place copy with data_source rewritten to
# "aime24" so it lands in the `aime` reward path (line 77 of the router).
AIME24_VAL_PARQUET="${REPO_ROOT}/evaluation/data/aime24/aime24_verl_ready.parquet"

CUSTOM_REWARD_FN="${REPO_ROOT}/evaluation/eval/verl_custom_reward.py"
if [ ! -f "${CUSTOM_REWARD_FN}" ]; then
    echo "Missing custom reward function file: ${CUSTOM_REWARD_FN}"; exit 1
fi

# Preflight: IFEval val parquet (build if missing).
if [ ! -f "${IFEVAL_VAL_PARQUET}" ]; then
    if [ ! -f "${IFEVAL_JSON}" ]; then
        echo "Missing IFEVAL input JSONL: ${IFEVAL_JSON}"; exit 1
    fi
    python "${IFEVAL_CONVERTER}" \
        --input_jsonl "${IFEVAL_JSON}" \
        --output_parquet "${IFEVAL_VAL_PARQUET}" \
        --data_source ifeval \
        --split test \
        --prompt_language en
fi

# Preflight: LiveBench val parquet (build if missing).
if [ ! -f "${LIVEBENCH_VAL_PARQUET}" ]; then
    if [ ! -f "${LIVEBENCH_JSON}" ]; then
        echo "Missing LiveBench input JSON: ${LIVEBENCH_JSON}"; exit 1
    fi
    python "${LIVEBENCH_CONVERTER}" \
        --input_json "${LIVEBENCH_JSON}" \
        --output_parquet "${LIVEBENCH_VAL_PARQUET}" \
        --data_source livebench/coding \
        --split test \
        --prompt_language en
fi

if [ ! -f "${AIME24_VAL_PARQUET}" ]; then
    echo "Missing AIME24 val parquet: ${AIME24_VAL_PARQUET}"; exit 1
fi

VAL_FILES_HYDRA="['${IFEVAL_VAL_PARQUET}','${AIME24_VAL_PARQUET}','${LIVEBENCH_VAL_PARQUET}']"
echo "VAL_FILES_HYDRA=${VAL_FILES_HYDRA}"

# =============================================================================
# Common training arguments
# =============================================================================

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILES_HYDRA}"
    data.val_files="${VAL_FILES_HYDRA}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.validation_shuffle=False
    data.shuffle=True
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"

    # --- Multi-task: custom dataset wrapper + balanced-batch sampler ---
    # MultiTaskRLHFDataset loads each parquet into its own RLHFDataset, so the
    # incompatible nested schemas across IF/Coding/Math (e.g. reward_model
    # being a struct in IF but a flat string in Math) never have to be merged.
    # Without this the default RLHFDataset would crash inside
    # datasets.concatenate_datasets at startup. The same wrapper is applied to
    # val_files automatically — each val parquet stays in its own sub-dataset
    # so val schemas don't need to align either; per-data_source val metrics
    # are computed inside ray_trainer.process_validation_metrics regardless of
    # how val batches are sliced.
    +data.custom_cls.path=pkg://verl.utils.dataset.multitask_rl_dataset
    +data.custom_cls.name=MultiTaskRLHFDataset

    # UniformMultiTaskSampler yields indices in chunks of train_batch_size where
    # each chunk has a per-task quota that sums to train_batch_size. With B=128
    # and 3 tasks the quota is (43,43,42). epoch_policy=largest means one full
    # pass over the largest task per epoch (smaller tasks reshuffle and cycle).
    # create_rl_sampler enforces dataloader_num_workers=0 for custom samplers.
    +data.sampler.class_path=pkg://verl.experimental.dataset.uniform_multitask_sampler
    +data.sampler.class_name=UniformMultiTaskSampler
    +data.multitask_sampler.task_field=data_source
    +data.multitask_sampler.epoch_policy=largest
    data.dataloader_num_workers=0

    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    # KL loss per your spec.
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.002
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    # Unified sampling: temperature=1.0 across IF/Coding/Math.
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    algorithm.use_kl_in_reward=False

    # naive reward manager is the only safe choice for MIXED if/coding/math
    # batches: prime's ProcessPoolExecutor breaks signal.alarm() in the coding
    # verifier. See header comment for details.
    reward_manager.name=naive
    reward_manager.source=register
    custom_reward_function.path="${CUSTOM_REWARD_FN}"
    custom_reward_function.name=compute_score

    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${MACHINE_GPU_COUNT}"
    trainer.nnodes="${WORLD_SIZE}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train=False
    trainer.val_only="${VAL_ONLY}"
    trainer.resume_mode=auto
    trainer.save_freq="${SAVE_FREQ}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.validation_data_dir="${OUTPUT_DIR}/validation_outputs"
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model']

    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}"

    # Validation sampling: Qwen3 Thinking recommended defaults.
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True

    +data.apply_chat_template_kwargs.enable_thinking=True

    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.actor.fsdp_config.dtype="${DTYPE}"
    actor_rollout_ref.rollout.dtype="${DTYPE}"
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
)

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.total_epochs="${TOTAL_EPOCHS}"

echo "=============================================="
echo "Joint if+coding+math RL training complete."
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="
