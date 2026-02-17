# [2026-02-13 22:03] Add Rollout Cache And Force Flag

## Changes
- **File**: `scripts/run_fisher_merge_pipeline.py`
    - Added a new CLI flag `--force` to control rollout artifact regeneration behavior.
    - Added `load_or_prepare_rollout_dataset(...)` to reuse existing `rollout_input_verl_ready.parquet` by default.
    - Updated `run_rollout_with_verl(...)` to skip VERL rollout when cached validation JSONL files already exist.
    - Added force-mode cleanup logic in rollout stage to remove stale `validation_data_dir` and `verl_run` before rerun.
    - Updated `process_single_task(...)` to reuse cached `rollout_records.parquet` when available.
    - Wired force option through main orchestration and added explicit force-mode logging.

## Rationale
- Pipeline retries were redoing expensive rollout work for completed tasks (e.g., IF) even when only later tasks failed.
- Caching rollout artifacts allows fast resume behavior.
- `--force` preserves reproducibility/control when full regeneration is explicitly needed.

## Technical Details
- Cache-first behavior now applies to these rollout artifacts:
    - `tasks/<task>/rollout_input_verl_ready.parquet`
    - `tasks/<task>/validation_data_dir/*.jsonl`
    - `tasks/<task>/rollout_records.parquet`
- Force behavior removes prior rollout directories before rerunning subprocess rollout to avoid mixing old/new outputs.
- Validation performed with:
    - `python -m py_compile scripts/run_fisher_merge_pipeline.py`
    - `python scripts/run_fisher_merge_pipeline.py --help`
    - `python scripts/run_fisher_merge_pipeline.py --config ... --dry-run`
