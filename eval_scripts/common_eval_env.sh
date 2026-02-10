#!/usr/bin/env bash
# Common environment bootstrap for VERL validation-only evaluation scripts.
#
# This script is intended to be sourced by benchmark-specific scripts so that
# cache directories, Python path, and runtime environment stay consistent.

set -euo pipefail

# Resolve repository root once so all scripts can use absolute paths reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate the shared VERL virtual environment when available.
# This keeps behavior consistent with existing training/evaluation scripts.
if [[ -f "/home/nsml/verl/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "/home/nsml/verl/bin/activate"
fi

# Keep warning volume low to focus logs on evaluation metrics and failures.
export PYTHONWARNINGS="ignore::UserWarning:megatron"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

# Reuse the same temporary cache layout used by existing scripts to avoid
# filling home-directory caches and to keep model load behavior predictable.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/tmp/nsml/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/tmp/nsml/pip-cache}"
export HF_HOME="${HF_HOME:-/mnt/tmp/nsml/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/tmp/nsml/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/tmp/nsml/huggingface/datasets}"
export TORCH_HOME="${TORCH_HOME:-/mnt/tmp/nsml/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/tmp/nsml/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/tmp/nsml/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/tmp/nsml/cuda-cache}"
export TMPDIR="${TMPDIR:-/mnt/tmp/nsml/tmp}"

# Ensure local custom reward modules remain importable in all subprocesses.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Export repo path for downstream scripts so they can build file paths safely.
export NEMOTRON_REPO_ROOT="${REPO_ROOT}"
