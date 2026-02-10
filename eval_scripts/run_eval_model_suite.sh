#!/usr/bin/env bash
# Run AIME25 + IFEval evaluations for Initial / IF / Math / Merged models.
#
# Usage model:
# - Provide model paths through environment variables.
# - The script skips entries that are empty.
#
# Variables:
#   INITIAL_MODEL (default: Qwen/Qwen3-1.7B)
#   IF_MODEL      (optional)
#   MATH_MODEL    (optional)
#   MERGED_MODEL  (optional)
#
# Shared knobs:
#   OUTPUT_ROOT, N_GPUS_PER_NODE, NNODES, ULYSSES_SEQUENCE_PARALLEL_SIZE, DTYPE, LOSS_AGG_MODE

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INITIAL_MODEL="${INITIAL_MODEL:-Qwen/Qwen3-1.7B}"
IF_MODEL="${IF_MODEL:-}"
MATH_MODEL="${MATH_MODEL:-}"
MERGED_MODEL="${MERGED_MODEL:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ddn/vuvlm/geeho/nemotron_cascade_output}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-float16}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-sum-norm}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

run_one_model() {
    local model_label="$1"
    local model_path="$2"

    # Skip empty model entries so the script can be used incrementally.
    if [[ -z "${model_path}" ]]; then
        echo "[SKIP] ${model_label}: model path is empty."
        return 0
    fi

    MODEL_PATH="${model_path}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    NNODES="${NNODES}" \
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
    ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
    DTYPE="${DTYPE}" \
    LOSS_AGG_MODE="${LOSS_AGG_MODE}" \
    TIMESTAMP="${TIMESTAMP}" \
    bash "${SCRIPT_DIR}/run_eval_aime25_ifeval.sh"
}

# Fixed order to match requested reporting order:
# Initial model -> IF model -> Math model -> merged model.
run_one_model "initial" "${INITIAL_MODEL}"
run_one_model "if" "${IF_MODEL}"
run_one_model "math" "${MATH_MODEL}"
run_one_model "merged" "${MERGED_MODEL}"
