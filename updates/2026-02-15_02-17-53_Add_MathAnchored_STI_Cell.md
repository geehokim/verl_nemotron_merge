# [2026-02-15 02:17] Add Math-Anchored STI Cell

## Changes
- **File**: `merging_analysis/06_singular_task_interference_rl.ipynb`
    - Appended a new bottom code cell that computes STI using an alternative task-vector pair:
      - `base->math` (`theta_math - theta_base`)
      - `math->IF` (`theta_if - theta_math`)
    - Added a dedicated function `compute_sti_base_to_math_vs_math_to_if(...)` with detailed docstring and inline comments.
    - Reused existing SVD/STI utilities to keep metric definitions consistent with the original analysis.
    - Saved separate artifacts for the alternative run:
      - `parameter_level_sti_metrics_base_to_math_vs_math_to_if.csv`
      - `layer_level_sti_metrics_base_to_math_vs_math_to_if.csv`
      - `layer_sti_profile_base_to_math_vs_math_to_if.png`
      - `layer_sti_single_bar_2layer_mavg_base_to_math_vs_math_to_if.png`

## Rationale
- The original notebook computes interference between `base->IF` and `base->math` vectors.
- This update adds the requested viewpoint that isolates incremental IF adaptation after Math tuning by measuring interference between `base->math` and `math->IF`.

## Technical Details
- Uses existing utilities already defined in the notebook:
  - `adaptive_truncated_svd`, `compute_sti_two_tasks`, `compute_uv_overlap`, `rank_at_energy`
  - `should_use_parameter`, `weighted_mean`, `compute_layer_sti_blockdiag_equivalent`
- Preserves the original dataframe schema so existing plot functions can be reused without additional refactoring.
