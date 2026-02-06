#!/bin/bash
# =============================================================================
# Nemotron-Cascade 8B SWE RL Training Script (Single Stage)
# =============================================================================
#
# This script implements the Nemotron-Cascade paper's SWE RL training recipe
# as described in Section 4.7.
#
# Key Features:
#   - GRPO algorithm with token-level loss, no KL regularization
#   - Execution-free reward function (lexical similarity based)
#   - Agentless-style code repair task
#
# Hyperparameters (from paper):
#   - batch_size=128
#   - learning_rate=2.5e-6
#   - rollouts=16
#   - temperature=1.0
#   - max_response_length=16384 (16K)
#   - max_prompt_length=16384 (16K)
#
# Reward Function Cases:
#   1. lexical_similarity == 1 → reward = 1.0
#   2. patch identical to original → reward = 0.0
#   3. patch cannot be parsed → reward = -1.0
#   4. otherwise → lexical_similarity score
#
# Reference:
#   Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for
#   General-Purpose Reasoning Models (Section 4.7 - SWE RL)
# =============================================================================

set -x  # Enable command tracing for debugging
set -e  # Exit on error

# =============================================================================
# Environment Setup
# =============================================================================

source /home/nsml/verl/bin/activate
python -m wandb login 733bd860c7324e3d31ada0288891ee6f19d42c38

# Environment variables for cache management (consistent with math config)
export TRANSFORMERS_VERBOSITY=error

export XDG_CACHE_HOME=/mnt/tmp/nsml/cache
export PIP_CACHE_DIR=/mnt/tmp/nsml/pip-cache

export HF_HOME=/mnt/tmp/nsml/huggingface
export TRANSFORMERS_CACHE=/mnt/tmp/nsml/huggingface/hub
export HF_DATASETS_CACHE=/mnt/tmp/nsml/huggingface/datasets

export TORCH_HOME=/mnt/tmp/nsml/torch
export TRITON_CACHE_DIR=/mnt/tmp/nsml/triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/tmp/nsml/torchinductor

export CUDA_CACHE_PATH=/mnt/tmp/nsml/cuda-cache
export TMPDIR=/mnt/tmp/nsml/tmp

export WANDB_API_KEY="733bd860c7324e3d31ada0288891ee6f19d42c38"
export PYTHONPATH="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME="nemotron-cascade-swe"
OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/nemotron_cascade_swe_output/DeepSeek-R1-Distill-Qwen-1.5B-swe"
# BASE_MODEL="/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF"
BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8

# VLLM_USE_V1=1: Use vLLM v1 API for better performance and stability (recommended)
# VLLM_USE_V1=0: Use legacy vLLM API (may cause hang issues)
VLLM_USE_V1=1

# Checkpoint settings for OOM recovery
SAVE_FREQ=10

# Precision and loss aggregation settings (aligned with math config for stability)
DTYPE=float16
LOSS_AGG_MODE=seq-mean-token-sum-norm

# =============================================================================
# Dataset Configuration
# =============================================================================
#
# NOTE: Update these paths to your SWE dataset in verl-ready parquet format.
# The parquet file should contain:
#   - prompt: The problem description and buggy code
#   - data_source: "nemotron_cascade_rl_swe" (required for reward function routing)
#   - ground_truth: Golden patch in unified diff format (git diff output)
#   - extra_info (REQUIRED): {
#       "code_context": dict[str, str],  # Original file contents (path -> content)
#       "problem_id": str                # Problem identifier (optional)
#     }
#
# Example format:
#   {
#     "prompt": "<problem description>\n<buggy code>",
#     "data_source": "nemotron_cascade_rl_swe",
#     "ground_truth": "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-buggy\n+fixed",
#     "extra_info": {
#       "code_context": {"file.py": "buggy\n"},
#       "problem_id": "issue_123"
#     }
#   }
# =============================================================================

# TODO: Update these paths to your SWE dataset
train_files_list=(
    "/path/to/your/swe_train_verl_ready.parquet"
)

val_files_list=(
    "/path/to/your/swe_val_verl_ready.parquet"
)

# =============================================================================
# Training Arguments (Nemotron-Cascade SWE RL - Section 4.7)
# =============================================================================

COMMON_ARGS=(
    # Algorithm: GRPO (Group Relative Policy Optimization)
    algorithm.adv_estimator=grpo

    # Data configuration
    data.train_files=$train_files_list
    data.val_files=$val_files_list
    data.train_batch_size=128
    data.max_prompt_length=16384         # 16K input context
    data.max_response_length=16384       # 16K output length
    data.filter_overlong_prompts=True

    # Actor configuration (from paper: lr=2.5e-6)
    actor_rollout_ref.actor.optim.lr=1e-6
    # actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    # Increased batch sizes for better stability (aligned with math config)
    actor_rollout_ref.actor.ppo_mini_batch_size=128
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16

    # No KL regularization (from paper: "no KL regularization")
    # But we keep explicit settings for clarity
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.002
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # Rollout configuration (from paper: rollouts=16, temp=1.0)
    # Optimized for speed and stability (aligned with math config)
    actor_rollout_ref.rollout.n=16        # 16 rollouts per prompt (SWE requirement)
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9  # Increased for better throughput
    actor_rollout_ref.rollout.max_num_batched_tokens=34816
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1  # Reduced for better stability

    # Reference model configuration
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    # Algorithm configuration (no KL in reward)
    algorithm.use_kl_in_reward=False

    # Reward manager for SWE reward function
    # Uses nemotron_cascade_rl_swe.compute_score via data_source field
    reward_manager.name=naive
    reward_manager.source=register

    # Trainer configuration
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.n_gpus_per_node=$MACHINE_GPU_COUNT
    trainer.nnodes=$WORLD_SIZE
    trainer.test_freq=10
    trainer.val_before_train=False

    # Memory optimization and advanced settings (aligned with stable math config)
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34816
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34816
    # actor_rollout_ref.rollout.mode=async  # Disabled for stability
    actor_rollout_ref.actor.use_dynamic_bsz=True  # Enable for better throughput
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True  # Enable for ref model
    actor_rollout_ref.rollout.enable_prefix_caching=False  # Disable for stability
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.model.use_remove_padding=True

    # Checkpoint & Resume Settings (for OOM recovery)
    # resume_mode=auto: Automatically resume from latest checkpoint if available
    # This saves model, optimizer, lr_scheduler, dataloader state, and iteration number
    trainer.resume_mode=auto
    trainer.save_freq=$SAVE_FREQ
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model']

    # Ulysses sequence parallel for long context handling (aligned with math config)
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4

    # Validation config (SWE-specific: keep temperature=1.0, rollouts=16)
    data.val_batch_size=128
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.n=16
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.model.trust_remote_code=True

    # Precision settings (aligned with math config for stability)
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE
    actor_rollout_ref.rollout.dtype=$DTYPE
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE
)

# =============================================================================
# Single Stage Training (16K context)
# =============================================================================
#
# From paper Section 4.7:
# "We apply a straightforward extension of SWE RL on top of the RLHF model,
# using GRPO and execution-free reward functions."
#
# Note: Multi-stage training (16K → 24K) can be enabled later by adding
# additional stages similar to the math RL script.
# =============================================================================

echo "=============================================="
echo "Starting SWE RL Training"
echo "=============================================="
echo "  Algorithm: GRPO (no KL regularization)"
echo "  batch_size: 128"
echo "  learning_rate: 2.5e-6"
echo "  rollouts: 16"
echo "  temperature: 1.0"
echo "  max_prompt_length: 16384 (16K)"
echo "  max_response_length: 16384 (16K)"
echo "  resume_mode: auto (will resume from checkpoint if exists)"
echo "=============================================="
echo ""
echo "Reward Function (SEARCH/REPLACE format, execution-free):"
echo "  Model outputs: <think>...</think><solution>SEARCH/REPLACE blocks</solution>"
echo "  Parsing: Extract SEARCH/REPLACE blocks from model output"
echo "  Comparison: Apply edits -> Generate predicted diff -> Compare with golden diff"
echo "  Returns: -1.0 for format errors, 0.0-1.0 for valid edits (similarity score)"
echo "=============================================="

HYDRA_FULL_ERROR=1  python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    trainer.experiment_name="deepseek-qwen-distill-1.5b-swe" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$OUTPUT_DIR

echo "=============================================="
echo "DeepSeek-R1-Distill-Qwen-1.5B SWE RL Training Complete!"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Final model: $OUTPUT_DIR/global_step_*/actor"
echo ""
echo "To evaluate the model, use your SWE evaluation pipeline"
echo "(e.g., SWE-bench evaluation or Agentless framework)"
