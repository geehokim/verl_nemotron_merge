# [2026-02-14 14:56] Enhance STI Notebook With LowRank And BlockHeatmaps

## Changes
- **File**: `merging_analysis/06_singular_task_interference_rl.ipynb`
  - Added `layer_sti_single_sum` and `sti_l_blockdiag_equiv` to layer-level aggregation.
  - Added a paper-style STI visualization using bars (`layer_sti_single_sum`) plus 2-layer moving-average line.
  - Reworked the low-rank experiment cell to support two modes:
    - Proxy mode (rank-fraction vs retained-energy proxy from spectral anchors)
    - Accuracy mode (rank-fraction vs actual accuracy from optional external CSV)
  - Added concat block heatmap visualization for signed `U^T U` and `V^T V` structure on the highest-single-STI layer.
  - Added metadata export for concat heatmap block composition.
  - Updated interpretation and summary outputs to prioritize single-STI ranking and include new artifact paths.
- **File**: `updates/2026-02-14_14-56-00_Enhance_STI_Notebook_With_LowRank_And_BlockHeatmaps.md`
  - Added task update log for this notebook enhancement.

## Rationale
- To align the notebook analysis with requested STI formulation at the layer-single level rather than only numel-weighted averages.
- To add paper-style visual diagnostics for both interference (bar + moving average) and low-rank behavior (rank-fraction curve).
- To expose the concat-block singular-vector interaction structure (`U^T U`, `V^T V`) that is central to STI interpretation.

## Technical Details
- Layer-single STI uses sum of parameter-level STI terms, with explicit block-diagonal equivalence column.
- Moving-average chart uses a fixed 2-layer rolling window over ordered layer indices.
- Concat heatmap construction:
  - Per-parameter cross-blocks `U_if^T U_math` and `V_if^T V_math`
  - Block-diagonal composition and final concat Gram matrices `[I, C; C^T, I]`
  - Signed colormap (`TwoSlopeNorm`) to retain positive/negative overlap patterns.
- Low-rank curve:
  - Standard rank-fraction grid `[0.01, ..., 0.50]`
  - Optional external accuracy CSV integration for true Figure-2-style reproduction.
