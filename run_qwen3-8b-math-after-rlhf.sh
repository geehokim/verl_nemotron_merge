# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
source /home/nsml/verl/bin/activate

python -m wandb login 733bd860c7324e3d31ada0288891ee6f19d42c38

export HF_HOME=/mnt/tmp
# Hugging Face Hub(모델/토크나이저 등) 캐시
export HF_HUB_CACHE=/mnt/tmp/hf/hub

# Datasets 캐시
export HF_DATASETS_CACHE=/mnt/tmp/hf/datasets

# (선택) 전반적 cache 기본 경로도 같이 이동
export XDG_CACHE_HOME=/mnt/tmp/.cache

# Transformers verbosity
export TRANSFORMERS_VERBOSITY=error


ENGINE=${1:-vllm}
export VLLM_USE_V1=1

export WANDB_API_KEY="733bd860c7324e3d31ada0288891ee6f19d42c38"
export PROJECT_NAME="vlrvr-debug"
export EXPERIMENT_NAME="test"
export OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/vrlvr_output"
export PYTHONPATH="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

WORLD_SIZE=1
MACHINE_GPU_COUNT=4

train_files_list=(
        "/mnt/ddn/vuvlm/geeho/datasets/Nemotron-Cascade-RL-Math/math_verl_ready.parquet"
    )

val_files_list=(
        "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/aime25/test_verl_ready.parquet"
    )

train_files="[$(printf "'%s'," "${train_files_list[@]}" | sed 's/,$//')]"
val_files="[$(printf "'%s'," "${val_files_list[@]}" | sed 's/,$//')]"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files_list \
    data.val_files=$val_files_list \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=24000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$MACHINE_GPU_COUNT \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=$OUTPUT_DIR/$EXPERIMENT_NAME \
    trainer.val_before_train=False \