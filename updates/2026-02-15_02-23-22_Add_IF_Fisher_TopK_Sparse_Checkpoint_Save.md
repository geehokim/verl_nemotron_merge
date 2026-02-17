# [2026-02-15 02:23] Add IF Fisher Top-K Sparse Checkpoint Save

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Added a new bottom code cell that builds and saves 3 sparse IF checkpoints.
    - Implemented Fisher-thresholded sparse update rule: `theta_sparse = theta_base + 1[F_if >= threshold] * (theta_if - theta_base)`.
    - Added helper utilities with docstrings for loading IF Fisher tensors, sampling global Fisher values, estimating thresholds, applying sparse updates, and saving checkpoint metadata.
    - Configured fixed keep ratios `{0.1%, 1%, 5%}` via `(0.001, 0.01, 0.05)`.
    - Added run summary export to `metadata/if_sparse_topk_fisher_summary.json`.

## Rationale
- Needed to test whether IF performance is retained when only the top Fisher-importance parameters are updated from base to IF.
- Saved 3 sparsified variants so evaluation can be run externally and compared against the full IF baseline.

## Technical Details
- Used existing notebook runtime/model-loading utilities (`load_causal_lm`, runtime config paths).
- Used memory-aware global Fisher threshold estimation via uniform sampling instead of full flatten/concatenation.
- Applied element-wise sparse masking on IF task vector `(theta_if - theta_base)` and saved each resulting model/tokenizer pair.
- Persisted per-run metadata including estimated threshold and realized keep ratio.
