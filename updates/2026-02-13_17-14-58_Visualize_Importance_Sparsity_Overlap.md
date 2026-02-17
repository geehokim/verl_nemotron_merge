# [2026-02-13 17:14] Visualize Importance Sparsity and Overlap

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Added a new analysis section (`Importance Sparsity and Overlap Analysis`) to inspect saved `importance_{task}.pt` tensors.
    - Added helper functions with docstrings/comments for:
      - Parameter-to-layer bucketing for interpretable aggregation.
      - Deterministic task-wise high-importance threshold estimation via sampled quantiles.
    - Added per-parameter sparsity/scale statistics export (`importance_parameter_stats.csv`).
    - Added per-layer sparsity statistics export and heatmap visualization.
    - Added IF-vs-Math high-importance overlap computation (global + per-parameter + per-layer).
    - Added visualization outputs for:
      - Parameter sparsity map
      - Layer sparsity map
      - Global overlap ratio bar chart
      - Top-overlap parameter bar chart
      - Layer overlap map
    - Configured all analysis artifacts to be saved under `RUNTIME.output_root / "analysis"`.

## Rationale
- User requested direct visibility into how sparse JWCM importance tensors are at parameter level.
- User also requested confirmation of whether high-importance regions overlap between math and IF tasks.
- The added analysis section provides both quantitative CSV outputs and visual summaries to support ablation/debug decisions.

## Technical Details
- Uses CPU-resident `importance_by_task` tensors already produced by the notebook pipeline.
- Uses sampled quantile thresholds (`q=0.995`, `8192` samples/tensor) to define high-importance coordinates without requiring full tensor concatenation in RAM.
- Computes overlap metrics including `if_only`, `math_only`, `both`, `neither`, and per-parameter/per-layer Jaccard ratios.
- Saves PNG figures with Matplotlib and tabular summaries with pandas CSV export.
