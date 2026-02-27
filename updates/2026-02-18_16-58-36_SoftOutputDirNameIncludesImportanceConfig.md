# [2026-02-18 16:58] soft output dir includes importance config

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Updated soft checkpoint output directory naming logic in merge cell.
    - Replaced fixed directory name (`jwcm_v2_soft`) with config-derived name:
      `jwcm_v2_soft_{score_mode}_{selection_side}{selection_percent}`.
    - Kept sparsified directory naming rule unchanged and aligned with the same config-token convention.

## Rationale
- To prevent artifact ambiguity and make soft-merge outputs directly traceable to the selected precomputed importance variant.

## Technical Details
- Reused existing helper functions:
    - `_normalize_selection_percent(...)`
    - `_format_selection_percent_token(...)`
- Built `soft_output_dir_name` from `PRECOMPUTED_IMPORTANCE_CFG` before saving model artifacts.
