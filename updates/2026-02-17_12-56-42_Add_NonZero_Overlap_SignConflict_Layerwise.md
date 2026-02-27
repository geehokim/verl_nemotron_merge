# [2026-02-17 12:56] Add Non-Zero Overlap and Sign Conflict Layerwise Analysis

## Changes
- **File**: `merging_analysis/01_layer_interference_diagnostics.ipynb`
    - Added layer-wise non-zero overlap counters in base/continual/pairwise conflict collection (`if_nonzero_count`, `math/stage nonzero_count`, `both_nonzero_count`, `either_nonzero_count`).
    - Added new dataframe metrics for element-wise overlap and conflict intensity:
      `nonzero_overlap_ratio`, `nonzero_overlap_jaccard`, `sign_conflict_ratio_nonzero`, `sign_conflict_ratio_all`.
    - Extended `lm_head` state_dict diagnostics to emit the same non-zero overlap/conflict metrics for consistent layer coverage in tied-weight setups.
    - Updated visualization panels to include non-zero overlap lines and retitled the conflict subplot to reflect overlap+conflict interpretation.
    - Updated notebook intro markdown with explicit formulas for non-zero overlap and sign-conflict-on-overlap.

## Rationale
- To quantify not only whether signs conflict, but also whether two task vectors are active on the same coordinates at all.
- To separate absolute overlap density (`/all params`) from support alignment quality (`Jaccard on non-zero support`).
- To keep diagnostics comparable across base, continual-stage, and pairwise analyses.

## Technical Details
- Used PyTorch boolean masks over task-vector deltas (`delta != 0`) to compute support intersection/union and sign mismatch statistics.
- Preserved existing tau-threshold conflict metrics while augmenting with non-thresholded non-zero support diagnostics.
- Maintained compatibility with existing downstream ranking/CSV generation and tied-weight `lm_head` handling via `state_dict()`.
