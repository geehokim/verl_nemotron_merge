# [2026-02-23 05:55] LiveCodeBench Validation Pipeline Split (VERL + get_scores_code)

## Changes
- **File**: `nemotron_evaluation/data/livecodebench/convert_livecodebench24_to_verl_parquet.py`
    - Added CLI args: `--data_source`, `--split`, `--lcb_version`, `--prompt_language`.
    - Added validation metadata fields in `extra_info`: `lcb_version`, `prompt_language`, `num_tests`.
    - Kept testcase serialization format unchanged: `base64(zlib(pickle(json.dumps(dict))))`.
- **File**: `nemotron_evaluation/eval/verl_custom_reward.py`
    - Split coding reward routing by `data_source`.
    - Training source `nemotron_cascade_rl_coding` uses AceReason verifier + code-switching zero-reward override.
    - Validation sources (`livecodebench/code_generation_lite_v5`, `livecodebench/code_generation_lite_v6`, and legacy LCB sources) use `get_scores_code.py` checker.
- **File**: `run_qwen3_1.7b_coding.sh`
    - Added preflight conversion for both LCB v5 and v6 parquet files.
    - Updated `data.val_files` to include both val parquets as a Hydra list string.
    - Added `trainer.validation_data_dir` for validation output dumps.
- **File**: `nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py`
    - Added end-to-end validation script for schema, row count, ground-truth decode, reward behavior, and optional RLHFDataset compatibility.
- **File**: `tasks/todo.md`
    - Added a dedicated section for this validation split task, with checklist and verification evidence.

## Rationale
- Validation scoring must be benchmark-faithful to LiveCodeBench evaluation logic, so it is separated from training-time verifier logic.
- Distinct `data_source` identifiers for v5/v6 ensure automatic metric separation in VERL/W&B validation logs.
- Preflight conversion in the training script prevents missing-file failures and standardizes reproducible validation setup.

## Technical Details
- Validation verifier import uses file-path loading of `nemotron_evaluation/eval/get_scores_code.py` and calls `check_coding_correctness(problem_to_check, timeout, debug)`.
- Validation reward decision: boolean correctness -> `score = acc = answer_reward = 1.0/0.0`.
- Training reward path remains unchanged in behavior: AceReason subprocess execution with timeout formula and code-switching penalty enforced as final reward override to `0.0`.
- Generated artifacts:
    - `nemotron_evaluation/data/livecodebench/test_aug2024tojan2025_verl_ready_v5.parquet` (279 rows)
    - `nemotron_evaluation/data/livecodebench/test_feb2025toApr2025_verl_ready_v6.parquet` (175 rows)
