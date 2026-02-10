#!/bin/bash
# IF-RL training with Nemotron IF train split + IFEval validation in VERL format.

set -x
set -e
export PYTHONWARNINGS="ignore::UserWarning:megatron"

source /home/nsml/verl/bin/activate

# Ensure official IFEval checker dependencies are available.
python - <<'PY'
import importlib
import subprocess
import sys

required = ["langdetect", "immutabledict", "nltk"]
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import nltk
resources = [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]
for path, pkg in resources:
    try:
        nltk.data.find(path)
    except Exception:
        nltk.download(pkg, quiet=True)
PY

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
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONPATH="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_NAME="nemotron-cascade-parallel"
OUTPUT_DIR="/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-math-if"
BASE_MODEL="Qwen/Qwen3-1.7B"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8
SAVE_FREQ=50
DTYPE=float16
LOSS_AGG_MODE=seq-mean-token-sum-norm

IFRL_INPUT_PARQUET="/mnt/ddn/vuvlm/geeho/datasets/Nemotron-Cascade-RL-IF/ifrl_final_release.parquet"
IFRL_TRAIN_PARQUET="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifrl/train_verl_ready.parquet"
IFRL_SPLIT_VAL_PARQUET="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifrl/val_verl_ready.parquet"
IFRL_CONVERTER="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifrl/convert_ifrl_to_verl_parquet.py"
IFEVAL_INPUT_JSONL="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifeval/input_data.jsonl"
IFEVAL_VAL_PARQUET="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifeval/val_verl_ready.parquet"
IFEVAL_CONVERTER="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/ifeval/convert_ifeval_to_verl_parquet.py"
CUSTOM_REWARD_FN="/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/eval/verl_custom_reward.py"

VAL_ROLLOUT_N=2
IFRL_AUX_VAL_SIZE=256
TOTAL_EPOCHS=1
VAL_ONLY=false

# Regenerate split each run to keep validation size deterministic.
python "${IFRL_CONVERTER}" \
    --input_parquet "${IFRL_INPUT_PARQUET}" \
    --output_train_parquet "${IFRL_TRAIN_PARQUET}" \
    --output_val_parquet "${IFRL_SPLIT_VAL_PARQUET}" \
    --val_samples "${IFRL_AUX_VAL_SIZE}" \
    --seed 42 \
    --data_source nemotron_cascade_rl_if

python "${IFEVAL_CONVERTER}" \
    --input_jsonl "${IFEVAL_INPUT_JSONL}" \
    --output_parquet "${IFEVAL_VAL_PARQUET}" \
    --data_source nemotron_cascade_rl_if

if [ ! -f "${CUSTOM_REWARD_FN}" ]; then
    echo "Missing custom reward function file: ${CUSTOM_REWARD_FN}"
    exit 1
fi

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="${IFRL_TRAIN_PARQUET}"
    data.val_files="${IFEVAL_VAL_PARQUET}"
    data.train_batch_size=256
    data.val_batch_size=256
    data.max_prompt_length=2048
    data.max_response_length=8192
    data.filter_overlong_prompts=True
    data.validation_shuffle=False
    # Dynamic filtering sampler (Yu et al., 2025 style)
    # data.dataloader_num_workers=0
    # +data.sampler.class_path=pkg://verl.experimental.dataset.nemotron_cascade_sampler
    # +data.sampler.class_name=NemotronCascadeCurriculumSampler
    # +data.sampler.hard_resample_prob=0.10
    # +data.sampler.easy_resample_prob=0.01

    actor_rollout_ref.actor.optim.lr=2e-6
    actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=128
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9
    actor_rollout_ref.rollout.max_num_batched_tokens=34816
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    algorithm.use_kl_in_reward=False

    reward_manager.name=naive
    reward_manager.source=register
    custom_reward_function.path="${CUSTOM_REWARD_FN}"
    custom_reward_function.name=compute_score

    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.n_gpus_per_node="${MACHINE_GPU_COUNT}"
    trainer.nnodes="${WORLD_SIZE}"
    trainer.test_freq=10
    trainer.val_before_train=True
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
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4

    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True

    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.actor.fsdp_config.dtype="${DTYPE}"
    actor_rollout_ref.rollout.dtype="${DTYPE}"
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
)

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    trainer.experiment_name="qwen3-1.7b-ifrl-ifeval" \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}"
