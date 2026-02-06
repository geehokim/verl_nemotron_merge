#!/bin/bash
# =============================================================================
# Nemotron-Cascade 8B Math RL Training Script (3-Stage Curriculum)
# =============================================================================
#
# This script implements the Nemotron-Cascade paper's Math RL training recipe
# by running 3 sequential training stages with different configurations:
#
# Stage 1: 2 epochs, max_response_length=24000, temperature=1.0, overlong_filtering=True
# Stage 2: 2 epochs, max_response_length=32000, temperature=1.0, overlong_filtering=False
# Stage 3: 2 epochs, max_response_length=40000, temperature=0.8, overlong_filtering=False
#
# Each stage resumes from the previous stage's checkpoint.
# ~114 iterations ≈ 1 epoch (batch_size=128, dataset_size≈14.5K)
#
# Reference:
#   Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models
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

PROJECT_NAME="nemotron-cascade-math"
OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/nemotron_cascade_output"
BASE_MODEL="/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF"

WORLD_SIZE=1
MACHINE_GPU_COUNT=4

train_files_list=(
    "/mnt/ddn/vuvlm/geeho/datasets/Nemotron-Cascade-RL-Math/math_verl_ready.parquet"
)

val_files_list=(
    "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/aime25/test_verl_ready.parquet"
)

# =============================================================================
# Common Training Arguments (shared across all stages)
# =============================================================================

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files=$train_files_list
    data.val_files=$val_files_list
    data.train_batch_size=128
    data.max_prompt_length=2048
    data.filter_overlong_prompts=True
    # Curriculum Sampler for dynamic filtering (requires num_workers=0)
    data.dataloader_num_workers=0
    +data.sampler.class_path=pkg://verl.experimental.dataset.nemotron_cascade_sampler
    +data.sampler.class_name=NemotronCascadeCurriculumSampler
    +data.sampler.hard_resample_prob=0.10
    +data.sampler.easy_resample_prob=0.01
    # Actor config
    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # Rollout config (optimized for speed)
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.top_p=0.95
    # Ref config
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    # Algorithm config
    algorithm.use_kl_in_reward=False
    # Reward manager (naive with overlong filtering support)
    reward_manager.name=naive
    reward_manager.source=register
    # Trainer config
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.n_gpus_per_node=$MACHINE_GPU_COUNT
    trainer.nnodes=$WORLD_SIZE
    trainer.save_freq=100
    trainer.test_freq=50
    trainer.val_before_train=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
)

# =============================================================================
# STAGE 1: 2 epochs, max_len=24000, temp=1.0, overlong_filtering=True
# =============================================================================

echo "=============================================="
echo "Starting STAGE 1: 2 epochs (~228 iterations)"
echo "  max_response_length=24000"
echo "  temperature=1.0"
echo "  overlong_filtering=True"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    data.max_response_length=24000 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +reward_model.reward_kwargs.overlong_filtering=True \
    trainer.experiment_name="8b-stage1" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$OUTPUT_DIR/stage1

echo "STAGE 1 Complete!"

# =============================================================================
# STAGE 2: 2 epochs, max_len=32000, temp=1.0, overlong_filtering=False
# =============================================================================

echo "=============================================="
echo "Starting STAGE 2: 2 epochs (~228 iterations)"
echo "  max_response_length=32000"
echo "  temperature=1.0"
echo "  overlong_filtering=False"
echo "=============================================="

# Get the latest checkpoint from Stage 1
STAGE1_CKPT=$(ls -td $OUTPUT_DIR/stage1/global_step_* 2>/dev/null | head -1)
if [ -z "$STAGE1_CKPT" ]; then
    echo "ERROR: No Stage 1 checkpoint found!"
    exit 1
fi
echo "Resuming from Stage 1 checkpoint: $STAGE1_CKPT"

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$STAGE1_CKPT/actor \
    data.max_response_length=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.experiment_name="8b-stage2" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$OUTPUT_DIR/stage2

echo "STAGE 2 Complete!"

# =============================================================================
# STAGE 3: 2 epochs, max_len=40000, temp=0.8, overlong_filtering=False
# =============================================================================

echo "=============================================="
echo "Starting STAGE 3: 2 epochs (~228 iterations)"
echo "  max_response_length=40000"
echo "  temperature=0.8"
echo "  overlong_filtering=False"
echo "=============================================="

# Get the latest checkpoint from Stage 2
STAGE2_CKPT=$(ls -td $OUTPUT_DIR/stage2/global_step_* 2>/dev/null | head -1)
if [ -z "$STAGE2_CKPT" ]; then
    echo "ERROR: No Stage 2 checkpoint found!"
    exit 1
fi
echo "Resuming from Stage 2 checkpoint: $STAGE2_CKPT"

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$STAGE2_CKPT/actor \
    data.max_response_length=40000 \
    actor_rollout_ref.rollout.temperature=0.8 \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.experiment_name="8b-stage3" \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$OUTPUT_DIR/stage3

echo "=============================================="
echo "Nemotron-Cascade 8B Math RL Training Complete!"
echo "=============================================="
echo ""
echo "Stage 1 output: $OUTPUT_DIR/stage1"
echo "Stage 2 output: $OUTPUT_DIR/stage2"
echo "Stage 3 output: $OUTPUT_DIR/stage3"
echo ""
echo "Final model: $OUTPUT_DIR/stage3/global_step_*/actor"

