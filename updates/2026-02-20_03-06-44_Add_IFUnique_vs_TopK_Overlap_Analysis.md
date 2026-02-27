# [2026-02-20 03:06] feat(analysis): add IF-unique vs top-k overlap diagnostics with layer/global distribution visualizations

## Changes
- **File**: `merging_analysis/19_ram_ramplus_if_math_fp32.ipynb`
    - Inserted a new analysis section before merge-save steps:
      - `IF unique` vs `top-k` overlap comparison.
    - Added configurable top-k comparison settings:
      - modes: `abs`, `value`
      - percentages: `20%`, `30%`
    - Added detailed helper functions with docstrings and inline comments:
      - `build_topk_mask_for_tensor`
      - `compute_if_unique_vs_topk_overlap`
      - `visualize_if_unique_topk_layerwise`
      - `visualize_if_unique_topk_global`
      - `visualize_if_unique_topk_distribution`
    - Added artifact outputs for the new analysis:
      - `if_unique_vs_topk_layerwise_metrics.csv`
      - `if_unique_vs_topk_global_metrics.csv`
      - `if_unique_vs_topk_global_metrics.json`
      - `if_unique_vs_topk_layerwise_metrics.png`
      - `if_unique_vs_topk_global_metrics.png`
      - `if_unique_vs_topk_abs_distribution.png`

- **File**: `updates/2026-02-20_03-06-44_Add_IFUnique_vs_TopK_Overlap_Analysis.md`
    - Recorded this update log entry in the required workflow format.

## Rationale
- To quantify how much IF-unique coordinates overlap with simple top-k filtering baselines.
- To provide both layer-wise and global evidence, plus abs-delta distribution views, for direct interpretability.

## Technical Details
- Unique mask definition:
  - `unique_if = (|Delta_if| > tau) & ~(|Delta_math| > tau)`
- Top-k ranking modes:
  - `abs`: rank by `|Delta_if|`
  - `value`: rank by raw `Delta_if` (largest positive values)
- Metrics computed (layer-wise and global):
  - overlap ratio over all
  - precision (`overlap/topk`)
  - recall (`overlap/unique_if`)
  - jaccard (`overlap/union`)
  - enrichment over random baseline
- Distribution plots compare `overlap`, `unique_only`, and `topk_only` groups using `log10(abs(Delta_if))` histograms.
- Implemented memory-safe sampling caps for distribution visualization.
