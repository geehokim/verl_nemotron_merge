# [2026-02-14 15:44] Add Fisher Layer/Head Diagnostics and Delta-Merge Safeguard

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Updated the merge implementation to explicit task-vector (delta-space) Fisher merge with base fallback on zero-precision coordinates.
    - Added additional merge diagnostics (`exact_zero_precision_elements`, `exact_zero_precision_ratio`, `max_abs_delta_on_zero_precision_before_fallback`, `merge_space`).
    - Added a new analysis cell to identify dominant layers, module groups, and attention heads from Fisher mass.

## Rationale
- To prevent zero-precision collapse where low/zero Fisher coordinates can destroy pretrained weights.
- To localize which layers, matrices, and heads dominate Fisher concentration when Gini is high.

## Technical Details
- Merge formula uses `theta* = theta_base + weighted_delta`, where `delta = theta_task - theta_base`.
- Coordinates with `denominator <= epsilon` are explicitly set to `theta_base`.
- Layer/module/head analysis aggregates Fisher mass via tensor sums and Q/K/V head reshaping using `num_attention_heads` from `AutoConfig`.
