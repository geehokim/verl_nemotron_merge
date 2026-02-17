# [2026-02-13 17:38] Add Layerwise Importance Distribution Analysis

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
  - Added a new analysis section: `Importance Distribution By Layer (IF vs Math)`.
  - Added helper functions (with docstrings) for:
    - Layer bucket parsing from parameter names.
    - Tensor coordinate sampling for memory-safe distribution analysis.
    - Gini coefficient calculation to quantify per-layer parameter concentration.
    - Quantile summarization for robust range analysis.
  - Added full parameter-level summary export:
    - `importance_distribution_parameter_stats.csv`
  - Added sampled global distribution summary export:
    - `importance_distribution_global_summary.csv`
  - Added sampled layer-level distribution export:
    - `importance_distribution_layer_stats.csv`
  - Added per-layer top-parameter concentration export:
    - `importance_distribution_layer_top_parameters.csv`
  - Added visualizations comparing IF vs Math:
    - `importance_distribution_global_hist_log10.png`
    - `importance_distribution_layer_hist_grid.png`
    - `importance_distribution_layer_range_log10.png`
    - `importance_distribution_layer_concentration.png`
  - Added preview table of top parameters (by share of layer importance) for quick outlier inspection.

## Rationale
- User requested direct visibility into importance value distributions, specifically:
  - global and per-layer histograms,
  - whether layer-wise importance is uniform across parameters,
  - whether a few parameters dominate,
  - and IF-vs-Math side-by-side comparison.
- The new section answers all four questions from saved importance tensors without rerunning attribution.

## Technical Details
- Uses both exact full-tensor reductions (sum/mean/max/nonzero) and sampled coordinate histograms to balance fidelity and runtime.
- Distribution plots operate on `log10(nonzero importance)` and report zero ratios separately to avoid distortion from mass at zero.
- Concentration diagnostics use:
  - `top1_param_share` and `top3_param_share` (dominance of largest parameters),
  - `gini_param_sum` (uniformity vs concentration by layer).
- All artifacts are written under `RUNTIME.output_root / "analysis"`.
