# [2026-02-14 17:52] Add JWCM Importance Concentration And Layer/Head Diagnostics

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
  - Added a new markdown section: `JWCM Importance Concentration Diagnostics`.
  - Added a diagnostics code cell that mirrors Fisher concentration analysis on JWCM importance tensors.
  - Implemented helper functions for JWCM importance diagnostics:
    - `resolve_importance_paths_from_context`
    - `load_importance_dictionary`
    - `collect_tensor_mass_stats`
    - `sample_importance_values`
    - `compute_lorenz_curve`
    - `compute_gini_from_lorenz`
    - `compute_top_mass_share`
  - Added summary table + 2x2 visualization panel:
    - log-histogram, Lorenz curve, top-k mass share bars, tensor cumulative mass curve.
  - Added top-15 parameter tensor table by mass share per task.
  - Added a second diagnostics code cell for dominant component localization:
    - layer-level, layer+module-group, and attention head-level mass analysis.
  - Implemented helper functions for layer/head diagnostics:
    - `load_importance_dictionary_for_task`
    - `extract_layer_index`
    - `classify_module_group`
    - `collect_parameter_mass_dataframe`
    - `build_layer_module_summaries`
    - `resolve_num_attention_heads`
    - `collect_attention_head_mass`
    - `resolve_importance_paths_for_dominance`
  - Added 1x3 visualization panel:
    - top-layer bar chart, layer×module heatmap, layer×head heatmap.
  - Added explicit imports in diagnostics cells (`json`, `numpy`, `pandas`, `torch`) for standalone-cell execution stability.

## Rationale
- The user requested Fisher-style concentration and dominant layer/head diagnostics for JWCM as well.
- Matching the Fisher diagnostic structure enables direct side-by-side interpretation between Fisher and JWCM weighting behavior.

## Technical Details
- Diagnostics consume saved JWCM importance tensors (`importance_*.pt`) from:
  - in-memory `importance_paths`, or
  - metadata fallback: `metadata/jwcm_v2_run_summary.json`.
- Concentration metrics include Gini, Lorenz, and top-fraction mass capture over sampled coordinates.
- Dominance analysis aggregates importance mass by parameter tensor, layer, module group, and attention head.
