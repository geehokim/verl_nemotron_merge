# [2026-02-13 22:06] Create ValMath128 And Update Config

## Changes
- **File**: `merging_analysis/merging_analysis/artifacts/jwcm_v2/validation_data/val_math_128.parquet`
    - Created a new parquet dataset by selecting 128 rows from `val_math_512.parquet`.
    - Preserved original schema (`source`, `prompt`, `reward_model`, `data_source`) for compatibility.
- **File**: `scripts/configs/fisher_merge_pipeline.local.yaml`
    - Updated math task `input_parquet` from `val_math_512.parquet` to `val_math_128.parquet`.

## Rationale
- Reduced rollout workload and turnaround time by lowering the math validation subset size from 512 to 128 samples.
- Kept data format and pipeline wiring unchanged to avoid downstream compatibility issues.

## Technical Details
- Used pandas parquet I/O to load and subset the dataset deterministically (`head(128)`).
- Verified the new parquet row count is exactly 128.
- Verified config path now references the newly created 128-sample parquet.
