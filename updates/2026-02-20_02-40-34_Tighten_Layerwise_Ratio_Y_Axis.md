# [2026-02-20 02:40] fix(visualization): tighten ratio plot y-axis ranges for clearer variation

## Changes
- **File**: `merging_analysis/19_ram_ramplus_if_math_fp32.ipynb`
    - Reworked the visualization cell to add a new helper function `compute_tight_ratio_ylim`.
    - Updated `visualize_layerwise_ratios` so both subplots use data-adaptive y-axis limits instead of fixed `[0, 1]`.
    - Updated `visualize_global_ratios` to use the same tight y-axis strategy for ratio bars.
    - Added inline comments describing why adaptive y-limits are used and how edge cases are handled.

- **File**: `updates/2026-02-20_02-40-34_Tighten_Layerwise_Ratio_Y_Axis.md`
    - Recorded this update log entry in the required workflow format.

## Rationale
- Fixed wide fixed y-ranges that made meaningful ratio changes visually hard to detect.
- Improved readability of small but important layer-wise variation while preserving ratio bounds.

## Technical Details
- Added `compute_tight_ratio_ylim(series_list, pad_ratio, min_span)` with:
  - finite-value filtering,
  - fallback behavior when data is empty,
  - adaptive padding around observed min/max,
  - minimum-span guard for near-constant signals,
  - clipping to `[0, 1]` for ratio semantics.
- Layer-wise ratio panels now use tighter limits computed from plotted series.
- Global ratio bar panel now uses tighter limits computed from global ratio values.
