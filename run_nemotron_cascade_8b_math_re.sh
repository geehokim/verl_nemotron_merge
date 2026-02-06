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
# === Checkpoint & Resume Feature ===
# - trainer.resume_mode=auto: Automatically resume from latest checkpoint if exists
# - trainer.save_freq=1: Save checkpoint every iteration (model, optimizer, dataloader state)
# - If OOM occurs, just re-run the script and it will resume from the last checkpoint
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

PROJECT_NAME="nemotron-cascade-math"
OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Nemotron-Cascade-8B-RLHF-math_new"
BASE_MODEL="/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF"
# BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

#BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"


WORLD_SIZE=1
MACHINE_GPU_COUNT=8
# VLLM_USE_V1=1: Use vLLM v1 API for better performance and stability (recommended)
# VLLM_USE_V1=0: Use legacy vLLM API (may cause hang issues)
VLLM_USE_V1=1

# Checkpoint settings for OOM recovery
SAVE_FREQ=10                     # Save checkpoint every iteration for OOM recovery

DTYPE=float16
# LOSS_AGG_MODE=seq-mean-token-sum-norm
LOSS_AGG_MODE=token-mean

train_files_list=(
    # "/mnt/ddn/vuvlm/geeho/Precision-RL-verl/sanity_test/math_1460_nemotron.parquet"
    "/mnt/ddn/vuvlm/geeho/datasets/Nemotron-Cascade-RL-Math/math_verl_ready_MERGED_SYSTEM.parquet"
)

val_files_list=(
    "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/aime25/test_verl_ready_with_instruction.parquet"
)

# =============================================================================
# Common Training Arguments (shared across all stages)
# =============================================================================

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files=$train_files_list
    data.val_files=$val_files_list
    data.train_batch_size=2
    data.max_prompt_length=2048
    data.filter_overlong_prompts=True
    # Add instruction to user prompts: "please reason step by step. answer with \\boxed{}"
    # This instructs the model to provide step-by-step reasoning and boxed answers
    # data.user_prompt_suffix="\n\n Please reason step by step, and you should write with the correct answer within \\boxed{}"
    # Curriculum Sampler for dynamic filtering (requires num_workers=0)
    # +data.sampler.class_path=pkg://verl.experimental.dataset.nemotron_cascade_sampler
    # +data.sampler.class_name=NemotronCascadeCurriculumSampler
    # +data.sampler.hard_resample_prob=0.10
    # +data.sampler.easy_resample_prob=0.01
    # Actor config
    actor_rollout_ref.actor.optim.lr=2e-6
    # actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    # KL loss and entropy regularization for better training stability
    # kl_loss_coef: Controls KL divergence penalty between actor and reference model
    # entropy_coeff: Entropy bonus to encourage exploration
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.002
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # Rollout config (optimized for speed)
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9
    actor_rollout_ref.rollout.max_num_batched_tokens=34816  # 더 큰 배치 처리
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.n=8
    # Ref config
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    # Algorithm config
    algorithm.use_kl_in_reward=False
    # Reward manager (naive with overlong filtering support)
    reward_manager.name=naive
    reward_manager.source=register
    # Trainer config
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.n_gpus_per_node=$MACHINE_GPU_COUNT
    trainer.nnodes=$WORLD_SIZE
    trainer.test_freq=10
    trainer.val_before_train=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True 
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True 
    actor_rollout_ref.ref.fsdp_config.param_offload=True 
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=26624
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=26624
    # actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True 
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.model.use_remove_padding=True
    # === Checkpoint & Resume Settings ===
    # resume_mode=auto: Automatically resume from latest checkpoint if available
    # This saves model, optimizer, lr_scheduler, dataloader state, and iteration number
    trainer.resume_mode=auto
    trainer.save_freq=$SAVE_FREQ
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model','model'] 
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8
    # val config
    data.val_batch_size=128
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.n=8
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    # precision
    actor_rollout_ref.actor.fsdp_config.dtype=$DTYPE
    actor_rollout_ref.rollout.dtype=$DTYPE 
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE 
)

# =============================================================================
# STAGE 1: 2 epochs, max_len=24000, temp=1.0, overlong_filtering=True
# =============================================================================

echo "=============================================="
echo "Starting STAGE 1: 1 epochs (~228 iterations)"
echo "  max_response_length=4096"
echo "  temperature=1.0"
echo "  overlong_filtering=True"
echo "  resume_mode=auto (will resume from checkpoint if exists)"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    data.max_response_length=24576 \
    actor_rollout_ref.rollout.temperature=1.0 \
    +reward_model.reward_kwargs.overlong_filtering=True \
    trainer.experiment_name="8b-stage1" \
    trainer.total_epochs=1 \
    trainer.default_local_dir=$OUTPUT_DIR/stage1

# echo "STAGE 1 Complete!"

# =============================================================================
# STAGE 2: 2 epochs, max_len=32000, temp=1.0, overlong_filtering=False
# =============================================================================

echo "=============================================="
echo "Starting STAGE 2: 2 epochs (~228 iterations)"
echo "  max_response_length=24576"
echo "  temperature=0.8"
echo "  overlong_filtering=False"
echo "  kl_loss_coef=0.001"
echo "  resume_mode=auto (will resume from checkpoint if exists)"
echo "=============================================="


# # Get the latest checkpoint from Stage 1
# STAGE1_CKPT=$(ls -td $OUTPUT_DIR/stage1/global_step_* 2>/dev/null | head -1)
# if [ -z "$STAGE1_CKPT" ]; then
#     echo "ERROR: No Stage 1 checkpoint found!"
#     exit 1
# fi
# echo "Resuming from Stage 1 checkpoint: $STAGE1_CKPT"

# actor_rollout_ref.model.path=$STAGE1_CKPT/actor/huggingface \
# data.max_response_length=24576/ 2 = 12288 \
# Get the latest checkpoint from Stage 1
# STAGE1_CKPT=$(ls -td $OUTPUT_DIR/stage1/global_step_* 2>/dev/null | head -1)
# if [ -z "$STAGE1_CKPT" ]; then
#     echo "ERROR: No Stage 1 checkpoint found!"
#     exit 1
# fi
# echo "Resuming from Stage 1 checkpoint: $STAGE1_CKPT"

# actor_rollout_ref.model.path=$BASE_MODEL \
# actor_rollout_ref.model.path=$STAGE2_CKPT/actor/huggingface \

STAGE2_CKPT=$(ls -td $OUTPUT_DIR/stage2/global_step_* 2>/dev/null | head -1)
if [ -z "$STAGE2_CKPT" ]; then
    echo "ERROR: No Stage 2 checkpoint found!"
    exit 1
fi
echo "Resuming from Stage 2 checkpoint: $STAGE2_CKPT"

HYDRA_FULL_ERROR=1  python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    data.max_response_length=32768 \
    actor_rollout_ref.rollout.temperature=1 \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.experiment_name="deepseek-qwen-distill-1.5b-math-stage2" \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$OUTPUT_DIR/stage2 \
    trainer.validation_data_dir=$OUTPUT_DIR/stage2/validation_outputs

echo "STAGE 2 Complete!"

# =============================================================================
# STAGE 3: 2 epochs, max_len=40000, temp=0.8, overlong_filtering=False
# =============================================================================

echo "=============================================="
echo "Starting STAGE 3: 2 epochs (~228 iterations)"
echo "  max_response_length=40000"
echo "  temperature=0.8"
echo "  overlong_filtering=False"
echo "  resume_mode=auto (will resume from checkpoint if exists)"
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
    trainer.experiment_name="qwen2.5-1.5b-instruct-math-stage3" \
    trainer.total_epochs=1 \
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
