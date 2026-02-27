# [2026-02-18 16:45] precomputed importance fp32 merge notebook update

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Replaced attribution-generation execution flow with precomputed importance loading flow.
    - Added configurable importance filename resolver based on pattern:
      `importance_{task_name}_{score_mode}_{selection_side}{selection_percent}.pt`.
    - Enforced FP32 merge runtime (`torch.float32`) for model loading, merge arithmetic, and save outputs.
    - Updated merge stage to read IF/Math importance `.pt` artifacts, stage them in run output, and run soft merge.
    - Added new diagnostics section for non-zero support ratio, non-zero overlap (ratio/Jaccard), and layer-wise importance mass distribution.
    - Kept concentration diagnostics section so importance distribution analysis (Lorenz/Gini/top-k mass) remains available.

## Rationale
- Eliminate unnecessary in-notebook importance recomputation and directly reuse precomputed IF/Math importance artifacts.
- Make artifact selection reproducible by expressing filename selection as explicit config fields.
- Satisfy strict FP32 requirement for load/merge/save consistency.
- Extend diagnostics to explicitly report overlap and mass patterns requested by the user.

## Technical Details
- Added `PrecomputedImportanceConfig` and filename/path resolution helpers.
- Switched runtime dtype to `torch.float32` and merge execution device to CPU for deterministic FP32 workflow.
- Added precomputed artifact staging (`source -> output_root/importance`) for provenance.
- Added non-zero/mass diagnostics outputs under `RUNTIME.output_root/analysis` as CSV/JSON/PNG artifacts.
