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

export HF_HOME=/mnt/tmp
export HF_HUB_CACHE=/mnt/tmp/hf/hub
export HF_DATASETS_CACHE=/mnt/tmp/hf/datasets
export XDG_CACHE_HOME=/mnt/tmp/.cache
export TRANSFORMERS_VERBOSITY=error
export VLLM_USE_V1=1

export WANDB_API_KEY="733bd860c7324e3d31ada0288891ee6f19d42c38"
export PYTHONPATH="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

# =============================================================================
# Configuration
# =============================================================================

PROJECT_NAME="nemotron-cascade-swe"
OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/nemotron_cascade_swe_output"
BASE_MODEL="/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8

# Checkpoint settings for OOM recovery
SAVE_FREQ=1

# =============================================================================
# Dataset Configuration
# =============================================================================
#
# NOTE: Update these paths to your SWE dataset in verl-ready parquet format.
# The parquet file should contain:
#   - prompt: The problem description and buggy code
#   - data_source: "nemotron_cascade_rl_swe" (required for reward function routing)
#   - ground_truth: Ground truth patch in unified diff format
#   - extra_info (optional): {"original_code": "...", "problem_id": "..."}
#
# Example format:
#   {
#     "prompt": "<problem description>\n<buggy code>",
#     "data_source": "nemotron_cascade_rl_swe",
#     "ground_truth": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-buggy\n+fixed",
#     "extra_info": {"original_code": "...", "problem_id": "issue_123"}
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

    # Curriculum Sampler for dynamic filtering (requires num_workers=0)
    data.dataloader_num_workers=0
    +data.sampler.class_path=pkg://verl.experimental.dataset.nemotron_cascade_sampler
    +data.sampler.class_name=NemotronCascadeCurriculumSampler
    +data.sampler.hard_resample_prob=0.10
    +data.sampler.easy_resample_prob=0.01

    # Actor configuration (from paper: lr=2.5e-6)
    actor_rollout_ref.actor.optim.lr=2.5e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2

    # No KL regularization (from paper: "no KL regularization")
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    # Rollout configuration (from paper: rollouts=16, temp=1.0)
    actor_rollout_ref.rollout.n=16        # 16 rollouts per prompt
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_num_batched_tokens=34816
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True

    # Reference model configuration
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
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

    # Memory optimization
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.free_cache_engine=True

    # Checkpoint & Resume Settings (for OOM recovery)
    trainer.resume_mode=auto
    trainer.save_freq=$SAVE_FREQ
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
echo "Reward Function (execution-free):"
echo "  Case 1: lexical_similarity == 1 → reward = 1.0"
echo "  Case 2: patch == original → reward = 0.0"
echo "  Case 3: parse error → reward = -1.0"
echo "  Case 4: otherwise → lexical_similarity"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    trainer.experiment_name="8b-swe-rl" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$OUTPUT_DIR

echo "=============================================="
echo "Nemotron-Cascade 8B SWE RL Training Complete!"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Final model: $OUTPUT_DIR/global_step_*/actor"
echo ""
echo "To evaluate the model, use your SWE evaluation pipeline"
echo "(e.g., SWE-bench evaluation or Agentless framework)"
