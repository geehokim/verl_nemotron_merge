# [2026-02-17 11:32] feat(merging_analysis): append task-vector score diagnostics visualization cells

## Changes
- **File**: `merging_analysis/12_if_task_vector_gradnorm_topk_sparse_update.ipynb`
    - Appended a new markdown section titled `Task-Vector Score Diagnostics (Visualization)`.
    - Appended a new visualization code cell without modifying existing grad/fisher sparse-update cells.
    - Added diagnostic functions with docstrings and inline comments:
      - `compute_exact_zero_statistics` for exact `|Δ_if| == 0` ratio
      - `compute_topk_l2_energy_curve` for top-x% coordinate L2-energy capture curve
      - `summarize_topk_energy_anchors` for anchor-point CDF summaries
      - `build_threshold_table_for_keep_ratios` for expected threshold table per keep ratio
    - Added plots/tables covering requested items:
      - Score histogram (including near-zero zoom)
      - Top-x% vs captured norm-energy CDF-style curve
      - Zero-ratio table
      - Keep-ratio threshold table
    - Added standalone model-loading and cleanup logic so the diagnostics cell can run independently.
- **File**: `updates/2026-02-17_11-32-06_Add_TaskVector_Score_Diagnostics_Visualization.md`
    - Added task update log for this visualization extension.

## Rationale
- User requested visual diagnostics for task-vector score concentration and sparsity behavior directly in notebook 12.
- The new cells were appended to preserve the existing experiment flow and keep previously added grad/fisher sparse-update cells intact.
- Sampling-based diagnostics were used for distribution/CDF efficiency on large 1.7B-parameter checkpoints.

## Technical Details
- Libraries used: `PyTorch`, `NumPy`, `pandas`, `matplotlib`, `seaborn`.
- Score definition reused from existing notebook flow: `score = |Δ_if|`.
- CDF metric implemented as top-x% coordinate share of global L2 energy:
  - `energy_j = |Δ_if_j|^2`
  - `share(top-x%) = sum(top energy_j) / sum(all energy_j)`
- Threshold table uses existing quantile estimator (`estimate_global_score_threshold`) over `SPARSE_KEEP_RATIOS`.
- Exact zero ratio is computed in full (not sampled) via tensor-wise streaming counts.
