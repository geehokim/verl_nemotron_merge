# [2026-02-13 17:19] Switch Importance Heatmaps To Layer Histograms

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Replaced heatmap-centered importance visualization logic with layer-axis histogram/bar plots.
    - Added helper `parse_layer_index()` to extract numeric decoder layer index from `layer_<idx>` buckets.
    - Kept per-parameter importance statistics export and overlap statistics export unchanged for downstream reuse.
    - Expanded overlap row schema to include `if_only_count/math_only_count` and corresponding ratios.
    - Added histogram outputs:
        - `importance_layer_hist_high_ratio.png`
        - `importance_layer_hist_overlap_stacked.png`
        - `importance_layer_hist_parameter_presence.png`
    - Preserved global overlap ratio bar plot and top-overlap parameter table for summary + drill-down.
    - Added explicit non-decoder bucket summary table output.

## Rationale
- User requested x-axis to be decoder layer index and to inspect where high-importance parameters are located via histogram-style plots instead of heatmaps.
- Layer histograms make per-layer concentration and IF/Math overlap structure immediately interpretable.

## Technical Details
- Layer aggregation is derived from saved `importance_by_task` tensors and sampled-quantile high-importance thresholds.
- Histograms include both coordinate-level ratios and parameter-level activity counts to avoid single-view bias.
- All analysis artifacts are saved under `RUNTIME.output_root / "analysis"`.
