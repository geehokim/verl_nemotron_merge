# [2026-02-18 16:55] sparsified output dir includes importance config

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Updated sparse checkpoint output directory naming logic in merge cell.
    - Replaced fixed directory name (`jwcm_v2_soft_sparsified`) with config-derived name:
      `jwcm_v2_soft_sparsified_{score_mode}_{selection_side}{selection_percent}`.
    - Added explicit code comments explaining why directory naming now includes selection config.

## Rationale
- To distinguish sparse-merge outputs across importance scoring modes (`absolute/positive/negative`) and selection variants (`top/bottom + p`), preventing artifact overwrite and ambiguity.

## Technical Details
- Reused existing helper functions:
    - `_normalize_selection_percent(...)`
    - `_format_selection_percent_token(...)`
- Built `sparse_output_dir_name` dynamically from `PRECOMPUTED_IMPORTANCE_CFG` fields before saving artifacts.
