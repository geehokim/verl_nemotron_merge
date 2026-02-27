# [2026-02-23 17:28] Skip Existing Train Parquet Conversion

## Changes
- **File**: `run_qwen3_1.7b_coding.sh`
    - Wrapped competitive coding train parquet conversion in a file-existence guard.
    - Added an explicit skip message when `train_verl_ready.parquet` already exists.

## Rationale
- Avoid unnecessary repeated dataset conversion on every training script run.
- Reduce startup latency and prevent redundant I/O when train parquet is already prepared.

## Technical Details
- Added `if [ ! -f "${CODING_TRAIN_PARQUET}" ]; then ... else ... fi` around the converter invocation.
- Verified shell syntax with `bash -n run_qwen3_1.7b_coding.sh`.
