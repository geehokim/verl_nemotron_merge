#!/bin/bash
# IFEval RL training (Qwen3-1.7B) with:
# - preflight IFEval JSONL -> VERL parquet conversion (train/val split)
# - custom reward based on evaluation/eval/get_scores_ifeval.py logic
# - in-training periodic validation on IFEval val parquet

set -x
set -e
export PYTHONWARNINGS="ignore::UserWarning:megatron"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

if [ -f /home/nsml/verl/bin/activate ]; then
    source /home/nsml/verl/bin/activate
fi

# Ensure IFEval reward dependencies + parquet tooling are available.
python - <<'PY'
import importlib
import subprocess
import sys

required = {
    "langdetect": "langdetect",
    "immutabledict": "immutabledict",
    "nltk": "nltk",
    "pyarrow": "pyarrow",
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

resources = [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]
for path, pkg in resources:
    try:
        nltk.data.find(path)
    except Exception:
        nltk.download(pkg, quiet=True)
PY

export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=DEBUG
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PROJECT_NAME="nemotron-cascade-ifrl"
OUTPUT_DIR="/131_data/geeho/nemotron_cascade_output/Qwen3-1.7B-ifrl_ifeval"
BASE_MODEL="Qwen/Qwen3-1.7B"

WORLD_SIZE=1
MACHINE_GPU_COUNT=8
SAVE_FREQ=50
DTYPE=float16
LOSS_AGG_MODE=seq-mean-token-sum-norm
VAL_ROLLOUT_N=8
TOTAL_EPOCHS=1
VAL_ONLY=false

SMOKE_MODE="${SMOKE_MODE:-false}"

TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=256
MAX_RESPONSE_LENGTH=8192
PPO_MINI_BATCH_SIZE=256
PPO_MICRO_BATCH_SIZE_PER_GPU=128
ROLLOUT_N=8
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=16
ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
ROLLOUT_MAX_NUM_BATCHED_TOKENS=34816
ULYSSES_SEQUENCE_PARALLEL_SIZE=4
TEST_FREQ=10
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1

if [ "${SMOKE_MODE}" = "true" ]; then
    MACHINE_GPU_COUNT=4
    TRAIN_BATCH_SIZE=4
    VAL_BATCH_SIZE=4
    MAX_RESPONSE_LENGTH=2048
    PPO_MINI_BATCH_SIZE=4
    PPO_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_N=2
    ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
    ULYSSES_SEQUENCE_PARALLEL_SIZE=4
    TEST_FREQ=5
    TRAIN_MAX_SAMPLES=128
    VAL_MAX_SAMPLES=16
    SAVE_FREQ=10
fi

EXPERIMENT_NAME="qwen3-1.7b-ifrl-ifeval"
if [ "${SMOKE_MODE}" = "true" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-smoke2gpu"
fi

echo "SMOKE_MODE=${SMOKE_MODE}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}"
echo "TEST_FREQ=${TEST_FREQ}, TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES}, VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES}"

# --- IFEVAL Training Data (external dataset, data_source="if") ---
IFEVAL_DATASET_ROOT="${IFEVAL_DATASET_ROOT:-/131_data/geeho/data/Nemotron-Cascade-RL-Instruction-Following}"
IFEVAL_TRAIN_PARQUET="${IFEVAL_DATASET_ROOT}/ifrl_if_verl_ready.parquet"

# --- IFEVAL Validation Data (converted from input_data.jsonl) ---
IFEVAL_JSON="${REPO_ROOT}/evaluation/data/ifeval/input_data.jsonl"
IFEVAL_CONVERTER="${REPO_ROOT}/evaluation/data/ifeval/convert_ifeval_to_verl_parquet.py"
IFEVAL_VAL_PARQUET="${REPO_ROOT}/evaluation/data/ifeval/input_data_verl_ready.parquet"

CUSTOM_REWARD_FN="${REPO_ROOT}/evaluation/eval/verl_custom_reward.py"

if [ ! -f "${IFEVAL_TRAIN_PARQUET}" ]; then
    echo "Missing IFEVAL training parquet: ${IFEVAL_TRAIN_PARQUET}"
    exit 1
fi
if [ ! -f "${IFEVAL_CONVERTER}" ]; then
    echo "Missing IFEval converter: ${IFEVAL_CONVERTER}"
    exit 1
fi
if [ ! -f "${CUSTOM_REWARD_FN}" ]; then
    echo "Missing custom reward function file: ${CUSTOM_REWARD_FN}"
    exit 1
fi

# Preflight: build IFEVAL validation parquet (skip if already exists).
if [ ! -f "${IFEVAL_VAL_PARQUET}" ]; then
    if [ ! -f "${IFEVAL_JSON}" ]; then
        echo "Missing IFEVAL input JSONL: ${IFEVAL_JSON}"
        exit 1
    fi
    python "${IFEVAL_CONVERTER}" \
        --input_jsonl "${IFEVAL_JSON}" \
        --output_parquet "${IFEVAL_VAL_PARQUET}" \
        --data_source ifeval \
        --split test \
        --prompt_language en
fi

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    data.train_files="${IFEVAL_TRAIN_PARQUET}"
    data.val_files="${IFEVAL_VAL_PARQUET}"
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
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=20
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    algorithm.use_kl_in_reward=False

    reward_manager.name=naive
    reward_manager.source=register
    custom_reward_function.path="${CUSTOM_REWARD_FN}"
    custom_reward_function.name=compute_score

    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.n_gpus_per_node="${MACHINE_GPU_COUNT}"
    trainer.nnodes="${WORLD_SIZE}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train=True
    trainer.val_only="${VAL_ONLY}"
    trainer.resume_mode=auto
    trainer.save_freq="${SAVE_FREQ}"
    trainer.validation_data_dir="${OUTPUT_DIR}/validation_outputs"
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
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    +reward_model.reward_kwargs.overlong_filtering=False \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}"
