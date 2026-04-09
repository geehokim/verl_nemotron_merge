#!/bin/bash
# Example script for running inference on evaluation benchmarks
# 
# Usage: bash run.sh
# 
# Before running:
# 1. Update MODEL_FOLDER, MODEL_NAME, TOKENIZER_FOLDER, and TOKENIZER_NAME
# 2. Update BENCHMARK_FOLDER to point to your benchmark data directory
# 3. Update EVAL_DATASET to the desired benchmark
# 4. Adjust inference parameters as needed (temperature, top-p, etc.)

# Model configuration (REQUIRED)
MODEL_FOLDER="/path/to/models"
MODEL_NAME="your-model-name"
TOKENIZER_FOLDER="/path/to/tokenizers"
TOKENIZER_NAME="your-tokenizer-name"

# Data configuration (REQUIRED)
BENCHMARK_FOLDER="/path/to/benchmarks"
EVAL_DATASET="aime25"  # See README for all supported datasets

# Inference parameters (OPTIONAL - defaults shown)
TEMPERATURE=0.6         # 0 for greedy decoding
TOP_P=0.95              # Top-p sampling threshold
MAX_OUTPUT_LEN=32768    # Maximum output length in tokens
BATCH_SIZE=1024         # Batch size for inference
TENSOR_PARALLEL_SIZE=1  # Number of GPUs for tensor parallelism
YARN_FACTOR=2           # YaRN RoPE scaling factor for extended context for 64k context suiable for long reasoning generation

# Other options
SEED=42              # Random seed
# DEVICE_ID="0,1,2,3"  # Uncomment to specify GPU devices
# USE_R1_FLAG="--use_r1"  # Uncomment for R1-style prompting
# NO_THINK_FLAG="--no-think"  # Uncomment to disable thinking mode

# Run inference
python inference.py \
    --model-folder "${MODEL_FOLDER}" \
    --model-name "${MODEL_NAME}" \
    --tokenizer-folder "${TOKENIZER_FOLDER}" \
    --tokenizer-name "${TOKENIZER_NAME}" \
    --benchmark-folder "${BENCHMARK_FOLDER}" \
    --eval-dataset "${EVAL_DATASET}" \
    --temperature ${TEMPERATURE} \
    --topp ${TOP_P} \
    --max-output-len ${MAX_OUTPUT_LEN} \
    --batch-size ${BATCH_SIZE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --yarn-factor ${YARN_FACTOR} \
    --seed ${SEED}
    # ${DEVICE_ID:+--device-id "${DEVICE_ID}"} \
    # ${USE_R1_FLAG} \
    # ${NO_THINK_FLAG}

