# [2026-02-14 17:58] Add Sequential Stage-2 Fisher Overlap Experiment

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Added a new analysis code cell titled `# Sequential stage-2 Fisher activation overlap analysis`.
    - Added robust helper functions (with docstrings/comments) to:
      - Ensure/load sequential stage-2 Fisher artifact.
      - Align Fisher mass-share vectors across granularities.
      - Compute overlap metrics (cosine, Pearson, Spearman, top-fraction Jaccard).
      - Optionally compute stage-2 delta magnitude table `|theta_seq_if - theta_math|`.
    - Added comparison workflow across `math`, `if_single`, and `if_seq_stage2` for parameter/layer/module/head granularity.
    - Added focused summary for the key pair `if_single` vs `if_seq_stage2`.
    - Added layer-profile visualizations for the focus pair.

## Rationale
- To test whether sequential training stage-2 IF Fisher activates the same locations as single-task IF Fisher.
- To quantify overlap rather than relying only on qualitative heatmap inspection.
- To support the requested stage-2 isolation perspective (`sequential_model - math_model`) via optional delta alignment.

## Technical Details
- Reuses existing notebook analysis utilities:
  - `load_fisher_dictionary_for_task`
  - `collect_parameter_mass_dataframe`
  - `build_layer_module_summaries`
  - `collect_attention_head_mass`
- Uses optional distributed Fisher generation for missing sequential artifact via `run_distributed_fisher_for_task`.
- Computes activation-overlap metrics at multiple granularities using normalized mass-share vectors.
- Includes optional heavy path controlled by `RUN_STAGE2_DELTA_ALIGNMENT` for direct delta-vs-Fisher comparison.
