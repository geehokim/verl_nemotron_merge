# RLVR Coding Pipeline (AceReason Verifier + VERL)

## Goal
- Build an end-to-end RLVR coding pipeline for VERL using:
  - competitive coding dataset conversion to VERL parquet
  - AceReason LiveCodeBench verifier logic as reward function core
  - coding code-switching penalty policy (force reward to 0)
  - a runnable training script following IF-RL style
  - a verification script proving data load + reward compute

## Acceptance Criteria
- [x] Competitive coding dataset is converted into VERL-format parquet at `/mnt/ddn/vuvlm/geeho/datasets/Nemotron-RL-coding-competitive_coding/train_verl_ready.parquet`.
- [x] Reward function uses AceReason verifier logic (same correctness decision rule as `evaluate_livecodebench.py`).
- [x] `nemotron_cascade_rl_coding` data source is routed to coding verifier reward.
- [x] Code-switching detection is applied for coding reward; detected switch forces final reward to `0.0`.
- [x] Existing IFRL and SWE reward paths remain functional.
- [x] New run script `run_qwen3_1.7b_ifrl_coding.sh` exists and includes preflight data + dependency checks.
- [x] Verification script validates schema, decodeability, reward behavior (correct/incorrect/code-switch), and RLHFDataset compatibility.
- [x] Task update log file under `updates/` is created.

## Working Notes
- Competitive coding source is a HuggingFace `load_from_disk` dataset directory (Arrow shards), not raw parquet files.
- LiveCodeBench val parquet is already available at `nemotron_evaluation/data/livecodebench/test_aug2024tojan2025_verl_ready.parquet`.
- Current custom reward imports local `tools.code_verifier_utils`; this will be changed for coding path to file-path import of AceReason verifier.

## Implementation Checklist
- [x] Create converter: `nemotron_evaluation/data/coding/convert_competitive_coding_to_verl_parquet.py`.
- [x] Update reward: `nemotron_evaluation/eval/verl_custom_reward.py`.
- [x] Add code-switching helper and coding score override behavior.
- [x] Add verification script: `nemotron_evaluation/eval/verify_coding_rlvr_pipeline.py`.
- [x] Add training script: `run_qwen3_1.7b_ifrl_coding.sh`.

## Verification Checklist
- [x] Run converter and confirm output row count / schema.
- [x] Run verification script end-to-end.
- [x] Confirm reward path for coding source returns expected values for synthetic cases.
- [x] Confirm no regressions for IFRL/SWE routing at import/runtime level.

## Results
- Added `nemotron_evaluation/data/coding/convert_competitive_coding_to_verl_parquet.py`.
- Added `nemotron_evaluation/eval/verify_coding_rlvr_pipeline.py`.
- Updated `nemotron_evaluation/eval/verl_custom_reward.py` to use AceReason verifier for coding data sources and apply code-switching reward override (`score=0` when detected).
- Added `run_qwen3_1.7b_ifrl_coding.sh`.
- Recorded one prevention lesson in `tasks/lessons.md` (quoted heredoc for markdown writes).
- Generated parquet:
  - `/mnt/ddn/vuvlm/geeho/datasets/Nemotron-RL-coding-competitive_coding/train_verl_ready.parquet`
  - rows: 16083
- Verification commands and outcomes:
  - `python nemotron_evaluation/eval/verify_coding_rlvr_pipeline.py ...` -> schema/row_count/sample_decode/reward OK
  - `source /home/nsml/verl/bin/activate && python nemotron_evaluation/eval/verify_coding_rlvr_pipeline.py ...` -> RLHFDataset sample load OK
  - IFRL/SWE smoke checks via `compute_score(...)` -> both returned dict outputs without runtime failure

---

# LiveCodeBench Validation Split Pipeline (VERL + get_scores_code)

## Goal
- Build separate validation pipeline for coding RL:
  - Convert LiveCodeBench v5/v6 JSON into VERL parquet with distinct validation `data_source`.
  - Route training reward and validation reward to different verifiers.
  - Wire v5+v6 validation into `run_qwen3_1.7b_coding.sh`.
  - Add a dedicated verification script covering load + reward routing + RLHFDataset compatibility.

## Acceptance Criteria
- [x] `convert_livecodebench24_to_verl_parquet.py` supports `--data_source`, `--split`, `--lcb_version`, `--prompt_language`.
- [x] LiveCodeBench v5 parquet generated at `nemotron_evaluation/data/livecodebench/test_aug2024tojan2025_verl_ready_v5.parquet`.
- [x] LiveCodeBench v6 parquet generated at `nemotron_evaluation/data/livecodebench/test_feb2025toApr2025_verl_ready_v6.parquet`.
- [x] Validation sources (`livecodebench/code_generation_lite_v5|v6` + legacy) use `get_scores_code.py` verifier path.
- [x] Training source (`nemotron_cascade_rl_coding`) remains on AceReason verifier + code-switching penalty path.
- [x] `run_qwen3_1.7b_coding.sh` validates on v5+v6 together and keeps `trainer.test_freq=10`.
- [x] New validation pipeline verifier script exists and passes end-to-end checks.

## Working Notes
- v5 conversion uses the same streaming converter but takes significantly longer with `batch_size=256` because first parquet flush is triggered only after 256 records.
- Distinct validation `data_source` values are required for separate W&B metric namespaces in VERL validation aggregation.

## Implementation Checklist
- [x] Extend `nemotron_evaluation/data/livecodebench/convert_livecodebench24_to_verl_parquet.py` CLI and row mapping.
- [x] Update `nemotron_evaluation/eval/verl_custom_reward.py` with training/validation coding verifier split.
- [x] Update `run_qwen3_1.7b_coding.sh` to preflight-convert v5/v6 and pass both val files.
- [x] Add `nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py`.

## Verification Checklist
- [x] `python -m py_compile ...` for modified python files.
- [x] Convert v5 JSON -> v5 VERL parquet.
- [x] Convert v6 JSON -> v6 VERL parquet.
- [x] Run validation pipeline verifier (core checks).
- [x] Run validation pipeline verifier with RLHFDataset checks.

## Results
- Updated `nemotron_evaluation/data/livecodebench/convert_livecodebench24_to_verl_parquet.py`:
  - Added CLI args: `data_source/split/lcb_version/prompt_language`
  - Added `extra_info` fields: `lcb_version`, `prompt_language`, `num_tests`, plus existing metadata.
- Updated `nemotron_evaluation/eval/verl_custom_reward.py`:
  - Added source split:
    - training: `nemotron_cascade_rl_coding` -> AceReason verifier.
    - validation: `livecodebench/code_generation_lite_v5`, `livecodebench/code_generation_lite_v6`, and legacy LCB sources -> `get_scores_code.py`.
  - Validation returns `{score, acc, answer_reward}` and does not apply code-switching penalty.
- Updated `run_qwen3_1.7b_coding.sh`:
  - Added v5/v6 preflight conversion.
  - Changed `data.val_files` to both val parquets as Hydra list string.
  - Added `trainer.validation_data_dir`.
- Added `nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py`.
- Generated validation parquets:
  - `nemotron_evaluation/data/livecodebench/test_aug2024tojan2025_verl_ready_v5.parquet` (279 rows)
  - `nemotron_evaluation/data/livecodebench/test_feb2025toApr2025_verl_ready_v6.parquet` (175 rows)
- Verification commands and outcomes:
  - `python -m py_compile nemotron_evaluation/data/livecodebench/convert_livecodebench24_to_verl_parquet.py nemotron_evaluation/eval/verl_custom_reward.py nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py` -> OK
  - `python nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py ... --sample_size 16`
    - schema/row/decode/reward-route checks passed (`v5=279`, `v6=175`, total `454`, validation v5/v6 + legacy routing OK)
  - `source /home/nsml/verl/bin/activate && python nemotron_evaluation/eval/verify_livecodebench_validation_pipeline.py ... --check_rlhf_dataset`
    - RLHFDataset sample load passed for both v5 and v6.
  - `bash -n run_qwen3_1.7b_coding.sh` -> shell syntax OK
