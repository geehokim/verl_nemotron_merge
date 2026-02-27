# [2026-02-17 11:48] fix(merging_analysis): switch sparse masking from thresholding to exact global top-k

## Changes
- **File**: `merging_analysis/12_if_task_vector_gradnorm_topk_sparse_update.ipynb`
    - Added new exact top-k helper functions (with docstrings and inline rationale comments):
      - `compute_exact_topk_threshold_info`
      - `promote_first_true_entries_inplace`
      - `apply_sparse_if_update_exact_topk_inplace`
    - Updated grad-based sparse execution cell to use exact global top-k masking instead of sampled quantile thresholding.
    - Updated fisher-based sparse execution cell to use exact global top-k masking instead of sampled quantile thresholding.
    - Added explicit `selection_mode='exact_global_topk'` metadata and included `target_keep_elements` in run rows/metadata.
    - Updated threshold diagnostics table builder in visualization cell to report exact top-k thresholds and target keep counts.
- **File**: `updates/2026-02-17_11-48-28_Switch_SparseSelection_To_Exact_TopK.md`
    - Added task update log for this masking-behavior fix.

## Rationale
- User observed mismatch between requested `keep_ratio` and `realized_keep_ratio` from threshold-based selection.
- Threshold-based masking can overshoot/undershoot due to sampling approximation and tied values at the threshold.
- Exact global top-k selection with deterministic tie handling guarantees that kept-element count matches the requested ratio (up to integer rounding on total coordinate count).

## Technical Details
- Exact selection strategy:
  - Compute global target keep count: `target_keep = round(keep_ratio * total_numel)`.
  - Compute exact threshold using global `kthvalue` on flattened scores.
  - Keep all coordinates with `score > threshold`.
  - Keep deterministic first-seen subset of `score == threshold` to fill the exact remaining budget.
- Tie handling implementation:
  - Chunked in-place promotion via `promote_first_true_entries_inplace` to avoid allocating full tie index vectors for very large tensors.
- Metadata/reporting updates:
  - `target_keep_elements`, `selection_mode`, and exact threshold are now persisted and displayed to verify ratio fidelity.
