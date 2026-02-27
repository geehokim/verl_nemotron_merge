# [2026-02-18 22:31] Wandb Experiment Name: Use Last Folder from Model Path

## Changes
- **File**: `eval_scripts/run_eval_ifeval.sh`
    - Replaced verbose `ifeval-valonly-${MODEL_DIR_NAME}-${TIMESTAMP}` experiment name format
    - New format: `{MODEL_DIR_NAME}_{TASK_LABEL}` for merged models (e.g., `ties_merged_real_ifeval`)
    - New format: `{MODEL_DIR_NAME}_{TASK_LABEL}_{basename}` for checkpoint models (e.g., `Qwen3-1.7B_ifeval_actor`)
    - Added dedup logic: when MODEL_DIR_NAME equals the basename of MODEL_PATH (typical for merged models), the basename suffix is omitted to avoid redundancy like `ties_merged_real_ifeval_ties_merged_real`

## Rationale
- User requested wandb run names to use the last folder of the model path for easy identification
- Previous timestamp-based naming made it hard to distinguish runs at a glance
- Format `{모델}_{task_이름}_{마지막폴더}` provides clear, human-readable experiment identifiers

## Technical Details
- `_BASENAME_OF_PATH` extracts the last folder via `basename "${MODEL_PATH%/}"`
- `MODEL_DIR_NAME` is parsed by `parse_model_dir_from_path()` which prefers Qwen3 tokens, falling back to basename
- Conditional logic avoids duplication when both values are identical (merged model case)
- `EXPERIMENT_NAME` is still overridable via environment variable
