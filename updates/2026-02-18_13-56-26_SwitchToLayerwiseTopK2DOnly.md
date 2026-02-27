# [2026-02-18 13:56] Switch Notebook14 to Layer-wise Top-K on 2D Parameters Only

## Changes
- **File**: `merging_analysis/14_if_importance_configurable_top_p_sparse_update.ipynb`
    - **Markdown cell**: Updated title and description to reflect layer-wise 2D-only behavior.
    - **Utility cell**: Major refactor of sparse update logic:
        - **Removed** global top-k functions: `compute_exact_topk_threshold_info`, `promote_first_true_entries_inplace`, `count_total_elements`.
        - **Added** `is_sparse_candidate(param)`: Returns True for 2D+ floating-point parameters eligible for sparsification.
        - **Added** `compute_layerwise_topk_mask(score, keep_ratio)`: Per-tensor exact top-k using `torch.topk`, deterministic tie-breaking by index.
        - **Renamed** `apply_sparse_task_update_exact_topk_inplace` → `apply_sparse_task_update_layerwise_inplace`.
        - **Modified** `validate_importance_score_compatibility`: Now only validates 2D+ parameters (1D params skip importance check).
    - **Main execution cell**:
        - Calls `apply_sparse_task_update_layerwise_inplace` instead of old global function.
        - Reports 1D vs 2D parameter split count at startup.
        - Updated metadata keys: `formula_2d`, `formula_1d`, `selection_mode=layerwise_topk_2d_only`.
        - Summary CSV/JSON columns updated to show `kept_2d_elements`, `total_2d_elements`, `realized_2d_keep_ratio`, `total_1d_elements`, etc.

## Rationale
- **1D parameters** (normalization layers like RMSNorm/LayerNorm scales) should not be sparsified because they control global activation scaling. Zeroing out parts of these parameters can destabilize model outputs disproportionately to the number of coordinates modified.
- **Layer-wise top-k** instead of global top-k ensures each weight matrix contributes proportionally to the sparse update, preventing scenarios where a few large-score layers dominate the global budget while other layers get zero updates.

## Technical Details
- `torch.topk(flat_score, k=k, largest=True, sorted=False)` is used per parameter for exact, deterministic top-k selection.
- 1D params receive `theta = theta_task` (full task-vector delta), equivalent to `theta_base + 1.0 * delta`.
- 2D params receive `theta = theta_base + mask_topk_layerwise(importance) * delta` where mask is binary {0, 1}.
