# [2026-02-17 13:34] Switch notebook 13 to IF-only SVD with vector/scalar prune modes

## Changes
- **File**: `merging_analysis/13_svd_nonzero_rank_task_arithmetic_merge.ipynb`
    - Refactored the notebook from IF+Math averaging to **IF-only** update flow.
    - Removed all Math checkpoint loading, compatibility checks, and delta usage.
    - Kept SVD truncation for all `ndim >= 2` parameters using non-zero singular value fractions `{1%, 2%, 5%, 10%, 20%}`.
    - Added explicit `vector_scalar_modes = ('unpruned', 'pruned')` behavior for `ndim < 2` parameters:
        - `unpruned`: apply full IF delta without SVD.
        - `pruned`: apply zero delta (base tensor unchanged).
    - Updated checkpoint save loop to emit one model per `(rank_keep_percentage, vector_scalar_mode)` pair.
    - Updated per-run metadata and summary schema to record IF-only formulas and vector/scalar mode statistics.

## Rationale
- To match the user’s revised experimental goal:
    - Test whether IF-only SVD-truncated updates can recover performance.
    - Separate impact of vector/scalar (LayerNorm-like) parameter updates by saving pruned vs unpruned variants.
- To remove confounding influence from Math deltas entirely.

## Technical Details
- Core decomposition uses `torch.linalg.svd(full_matrices=False)` with device fallback logic.
- Non-zero singular values are determined with threshold `max(nonzero_atol, max_sv * nonzero_rtol)`.
- Retained rank for each matrix tensor is `ceil(nonzero_rank * keep_fraction)` and clipped to `[1, nonzero_rank]`.
- Summary artifacts include both checkpoint-level metadata (`merge_metadata.json`) and aggregate run tables (JSON/CSV).
