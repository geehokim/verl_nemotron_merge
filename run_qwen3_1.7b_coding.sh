#!/bin/bash
# Coding RL training (Qwen3-1.7B) with:
# - preflight automatic train parquet compatibility conversion
# - LiveBench-only validation
# - coding custom reward routed through evaluation/eval/verl_custom_reward.py

set -x
set -e
export PYTHONWARNINGS="ignore::UserWarning:megatron"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Ensure runtime dependencies used by coding converter/reward are available.
python - <<'PY'
import importlib
import subprocess
import sys

required_pip = {
    "pyarrow": "pyarrow",
    "numpy": "numpy",
    "ftlangdetect": "fasttext-langdetect",
}
missing = []
for module_name, pip_name in required_pip.items():
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(pip_name)

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(set(missing))])
PY

export WANDB_API_KEY="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"
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONPATH="/home2/geeho/tmp/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_NAME="nemotron-cascade-coding"
OUTPUT_DIR="/131_data/geeho/nemotron_cascade_output/Qwen3-1.7B-coding"
BASE_MODEL="Qwen/Qwen3-1.7B"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8
SAVE_FREQ=50
DTYPE=float16
LOSS_AGG_MODE=seq-mean-token-sum-norm
VAL_ROLLOUT_N=1
TOTAL_EPOCHS=1
# Nemotron paper: 200 optimizer steps for Code RL.
TOTAL_TRAINING_STEPS=200
VAL_ONLY=false

# Smoke mode for low-resource bring-up (e.g., 2xA5000).
# Usage: SMOKE_MODE=true bash run_qwen3_1.7b_coding.sh
SMOKE_MODE="${SMOKE_MODE:-false}"

TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=64
# Kept at 32k here for Qwen3-1.7B
# on this hardware; revisit if scaling up the model or GPU count.
MAX_RESPONSE_LENGTH=32768
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE_PER_GPU=64
ROLLOUT_N=8
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_GPU_MEMORY_UTILIZATION=0.90
ROLLOUT_MAX_NUM_BATCHED_TOKENS=34816
ULYSSES_SEQUENCE_PARALLEL_SIZE=4
TEST_FREQ=10
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=32
# agent_loop chunks the gen batch across num_workers; must divide evenly.
AGENT_NUM_WORKERS=8

if [ "${SMOKE_MODE}" = "true" ]; then
    MACHINE_GPU_COUNT=4
    TRAIN_BATCH_SIZE=4
    VAL_BATCH_SIZE=4
    MAX_RESPONSE_LENGTH=2048
    PPO_MINI_BATCH_SIZE=4
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_N=1
    ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.75
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    ULYSSES_SEQUENCE_PARALLEL_SIZE=4
    TEST_FREQ=5
    TRAIN_MAX_SAMPLES=64
    VAL_MAX_SAMPLES=32
    SAVE_FREQ=10
    AGENT_NUM_WORKERS=1
fi

EXPERIMENT_NAME="qwen3-1.7b-ifrl-coding-livebench"
if [ "${SMOKE_MODE}" = "true" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-smoke2gpu"
fi

echo "SMOKE_MODE=${SMOKE_MODE}"
echo "MACHINE_GPU_COUNT=${MACHINE_GPU_COUNT}, TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
echo "ROLLOUT_N=${ROLLOUT_N}, MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}, ULYSSES_SP=${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
echo "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}, VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}"

DATASET_ROOT="${DATASET_ROOT:-/131_data/geeho/data/Nemotron-RL-coding-competitive_coding}"
TRAIN_PARQUET_DIR="${TRAIN_PARQUET_DIR:-${DATASET_ROOT}/data}"
TRAIN_CONVERTER="${REPO_ROOT}/evaluation/data/coding/ensure_competitive_coding_verl_parquet.py"
TRAIN_CONVERTED_DIR="${DATASET_ROOT}/verl_coding_cache"
TRAIN_MANIFEST="${DATASET_ROOT}/train_files_manifest.json"

LIVEBENCH_JSON="${REPO_ROOT}/evaluation/data/livebench/LiveBench.json"
LIVEBENCH_CONVERTER="${REPO_ROOT}/evaluation/data/livebench/convert_livebench_to_verl_parquet.py"
LIVEBENCH_VAL_PARQUET="${REPO_ROOT}/evaluation/data/livebench/livebench_verl_ready.parquet"

CUSTOM_REWARD_FN="${REPO_ROOT}/evaluation/eval/verl_custom_reward.py"

if [ ! -f "${CUSTOM_REWARD_FN}" ]; then
    echo "Missing custom reward function file: ${CUSTOM_REWARD_FN}"
    exit 1
fi

if [ ! -f "${TRAIN_CONVERTER}" ]; then
    echo "Missing train converter: ${TRAIN_CONVERTER}"
    exit 1
fi

if [ ! -f "${LIVEBENCH_CONVERTER}" ]; then
    echo "Missing LiveBench converter: ${LIVEBENCH_CONVERTER}"
    exit 1
fi

if [ ! -d "${DATASET_ROOT}/data" ]; then
    echo "Missing train parquet directory: ${DATASET_ROOT}/data"
    exit 1
fi

if ! compgen -G "${DATASET_ROOT}/data/*.parquet" > /dev/null; then
    echo "No parquet files found in train parquet directory: ${DATASET_ROOT}/data"
    exit 1
fi

# Preflight: convert training data shards only when source is not VERL-compatible.
python "${TRAIN_CONVERTER}" \
    --input_dir "${DATASET_ROOT}/data" \
    --output_dir "${TRAIN_CONVERTED_DIR}" \
    --manifest_path "${TRAIN_MANIFEST}" \
    --data_source nemotron_cascade_rl_coding \
    --split train

# Preflight: build LiveBench validation parquet.
python "${LIVEBENCH_CONVERTER}" \
    --input_json "${LIVEBENCH_JSON}" \
    --output_parquet "${LIVEBENCH_VAL_PARQUET}" \
    --data_source livebench/coding \
    --split test \
    --prompt_language en

TRAIN_FILES_HYDRA=$(python - <<PY
import json
from pathlib import Path
manifest = Path("${TRAIN_MANIFEST}")
if not manifest.exists():
    raise FileNotFoundError(f"train manifest not found: {manifest}")
data = json.loads(manifest.read_text(encoding="utf-8"))
train_files = data.get("train_files", [])
if not train_files:
    raise RuntimeError(
        f"empty train_files in manifest: {manifest}\\nmode={data.get('mode')}"
    )
print("[" + ",".join(f"'{p}'" for p in train_files) + "]")
PY
)

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILES_HYDRA}"
    data.val_files="${LIVEBENCH_VAL_PARQUET}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.max_prompt_length=2048
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.validation_shuffle=False
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"

    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    # Train rollout: unconstrained sampling (Nemotron-style), only temperature set.
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    algorithm.use_kl_in_reward=False

    reward_manager.name=prime
    reward_manager.source=register
    custom_reward_function.path="${CUSTOM_REWARD_FN}"
    custom_reward_function.name=compute_score

    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.n_gpus_per_node="${MACHINE_GPU_COUNT}"
    trainer.nnodes="${WORLD_SIZE}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train=False
    trainer.val_only="${VAL_ONLY}"
    trainer.resume_mode=auto
    trainer.save_freq="${SAVE_FREQ}"
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model']

    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
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
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${OUTPUT_DIR}"
