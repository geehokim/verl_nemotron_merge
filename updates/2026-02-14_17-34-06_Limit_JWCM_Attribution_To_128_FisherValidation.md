# [2026-02-14 17:34] Limit JWCM Attribution To 128 Fisher Validation Rows

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Updated `ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK` from `None` to `128`.
    - Updated inline config comment to explicitly state first 128 Fisher validation rows are used.
    - Updated notes section to document `ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK=128` behavior.

## Rationale
- Reduce attribution compute load and runtime by using a bounded validation subset.
- Keep behavior deterministic and consistent across IF/Math tasks.

## Technical Details
- Selection logic remains unchanged:
  - when `ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK` is set, the notebook applies `head(ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK)`.
- Effective behavior after this change:
  - each task uses first 128 rows from Fisher validation parquet.
