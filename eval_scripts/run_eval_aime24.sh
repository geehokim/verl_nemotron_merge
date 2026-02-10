#!/usr/bin/env bash
# Run VERL validation-only evaluation on AIME25 parquet data.
#
# Design notes:
# - Uses `verl.trainer.main_ppo` with `trainer.val_only=True` to reuse the same
#   generation + reward pipeline used during RL training.
# - Sets `data.train_files` equal to the validation parquet because VERL still
#   constructs a train dataset before entering val-only mode.
# - Keeps logger to console by default so this script works without WANDB login.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_eval_env.sh"

# -----------------------------
# User-overridable configuration
# -----------------------------

# Model under evaluation (initial model default requested by user).
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"

# Parse a model-family directory name from MODEL_PATH in a depth-agnostic way.
# Priority:
# 1) Prefer a "Qwen3-..." token anywhere in the path so checkpoint depth does
#    not matter (e.g., .../Qwen3-1.7B-math/stage1/global_step_80/actor).
# 2) Fall back to the last path segment for non-Qwen3 model names.
#
# This parser is intentionally shell-only and side-effect-free so it can be
# reused in variable default expressions.
parse_model_dir_from_path() {
    local model_path="$1"
    local trimmed_path="${model_path%/}"

    # If a Qwen3 token exists at any depth, that token becomes the model dir.
    if [[ "${trimmed_path}" =~ (Qwen3[^/[:space:]]*) ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi

    # Fallback keeps compatibility for arbitrary local/HF model identifiers.
    # Basename handling is stable for both "/a/b/c" and "org/model-name".
    printf '%s\n' "${trimmed_path##*/}"
}

# Benchmark-specific dataset.
AIME_VAL_PARQUET="${AIME_VAL_PARQUET:-${NEMOTRON_REPO_ROOT}/nemotron_evaluation/data/aime24/test_verl_ready_with_instruction.parquet}"

# Output and run identity.
# We derive the output folder from MODEL_PATH so model checkpoints of different
# directory depths still map to a consistent model-level evaluation directory.
# Example:
#   MODEL_PATH=/.../Qwen3-1.7B-math/stage1/global_step_80/actor
#   -> OUTPUT_DIR=/.../nemotron_cascade_output/Qwen3-1.7B-math/evaluation_output/aime25
MODEL_DIR_NAME="${MODEL_DIR_NAME:-$(parse_model_dir_from_path "${MODEL_PATH}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ddn/vuvlm/geeho/nemotron_cascade_output}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_DIR_NAME}/evaluation_output/aime24}"
PROJECT_NAME="${PROJECT_NAME:-nemotron-cascade-parallel}"
# Keep timestamp overridable so wrapper scripts can enforce one shared suffix
# across multiple benchmark runs for the same model.
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
# Include parsed model directory in the default experiment name so logs are
# self-descriptive even when multiple checkpoints are evaluated sequentially.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-aime24-valonly-${MODEL_DIR_NAME}-${TIMESTAMP}}"

# Cluster and precision parameters.
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-float16}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-sum-norm}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${AIME_VAL_PARQUET}" ]]; then
    echo "AIME parquet not found: ${AIME_VAL_PARQUET}" >&2
    exit 1
fi

# We keep reward routing on VERL defaults for AIME. `data_source=aime24` is
# handled by `verl.utils.reward_score.aime.compute_score`.
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${AIME_VAL_PARQUET}" \
    data.val_files="${AIME_VAL_PARQUET}" \
    data.train_batch_size=1 \
    data.val_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.dtype="${DTYPE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype="${DTYPE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    reward_manager.name=naive \
    reward_manager.source=register \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_only=True \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +reward_model.reward_kwargs.overlong_filtering=False
