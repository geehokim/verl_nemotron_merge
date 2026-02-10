#!/usr/bin/env bash
# Run both AIME25 and IFEval validation-only evaluations for a single model.
#
# This wrapper keeps decoding settings aligned across benchmarks so the resulting
# scores are directly comparable for one model checkpoint/path.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model and run naming defaults.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ddn/vuvlm/geeho/nemotron_cascade_output}"

# Shared runtime parameters for fair benchmark-to-benchmark comparison.
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
DTYPE="${DTYPE:-float16}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-sum-norm}"

MODEL_PATH="${MODEL_PATH}" \
NNODES="${NNODES}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
DTYPE="${DTYPE}" \
LOSS_AGG_MODE="${LOSS_AGG_MODE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
TIMESTAMP="${TIMESTAMP}" \
PROJECT_NAME="nemotron-eval-aime25" \
bash "${SCRIPT_DIR}/run_eval_aime25.sh"

MODEL_PATH="${MODEL_PATH}" \
NNODES="${NNODES}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" \
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE}" \
DTYPE="${DTYPE}" \
LOSS_AGG_MODE="${LOSS_AGG_MODE}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
TIMESTAMP="${TIMESTAMP}" \
PROJECT_NAME="nemotron-eval-ifeval" \
bash "${SCRIPT_DIR}/run_eval_ifeval.sh"
