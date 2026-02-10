# [2026-02-10 20:11] Include parsed model name in IFEval experiment name

## Changes
- **File**: `eval_scripts/run_eval_ifeval.sh`
    - Updated default `EXPERIMENT_NAME` format to include parsed `MODEL_DIR_NAME`.
    - New default format: `ifeval-valonly-<MODEL_DIR_NAME>-<YYYYMMDD_HHMMSS>`.
    - Added inline comments explaining why model-aware experiment naming improves traceability.
- **File**: `eval_scripts/README.md`
    - Documented that `run_eval_ifeval.sh` now embeds parsed model directory name in default experiment name.

## Rationale
- When multiple checkpoints/models are evaluated, experiment names without model identifiers are hard to track.
- Embedding parsed model name allows faster run identification in logs and output artifacts.

## Technical Details
- Reused existing depth-agnostic `MODEL_DIR_NAME` parsing from `MODEL_PATH`.
- Kept full backward compatibility:
    - If `EXPERIMENT_NAME` is explicitly provided, user value still takes precedence.
- Validation performed:
    - `bash -n eval_scripts/run_eval_ifeval.sh` (syntax check passed).
    - Manual shell expansion test confirmed output like:
      `ifeval-valonly-Qwen3-1.7B-math-YYYYMMDD_HHMMSS`.
