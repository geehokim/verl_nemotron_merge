# [2026-02-14 15:16] Split LowRank Curve By Task Remove FullRank

## Changes
- **File**: `merging_analysis/06_singular_task_interference_rl.ipynb`
  - Replaced the low-rank curve cell to render IF and Math curves separately on a single figure.
  - Removed full-rank horizontal baseline plotting logic (`axhline`) from the low-rank curve visualization.
  - Removed task-averaged low-rank curve logic and replaced it with explicit task-wise metrics columns (`if_metric`, `math_metric`).
  - Kept dual-mode behavior:
    - Proxy mode from rank-ratio spectral anchors when external accuracy CSV is absent.
    - Accuracy mode using task-wise columns (`if_accuracy`, `math_accuracy`) when external CSV is provided.
  - Added fallback handling for `avg_accuracy` CSVs by duplicating the same values for IF/Math with warning.
- **File**: `updates/2026-02-14_15-16-44_Split_LowRank_Curve_By_Task_Remove_FullRank.md`
  - Added this update log for the task-wise low-rank curve change.

## Rationale
- The requested analysis requires task-wise visibility rather than averaged y-values.
- Full-rank reference line was intentionally removed to focus on low-rank task-specific trajectories only.
- Maintaining both proxy and accuracy modes keeps the notebook usable before and after full evaluation runs.

## Technical Details
- Plot outputs now derive from `rank_curve_plot_df` with columns:
  - `rank_fraction`
  - `if_metric`
  - `math_metric`
- Figure path remains `rank_fraction_low_rank_curve.png` for compatibility with existing summary exports.
- CSV path remains `rank_fraction_low_rank_curve.csv` but schema changed to task-wise metrics.
