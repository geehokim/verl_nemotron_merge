# [2026-02-15 13:13] Notebook05 Full-Rollout Fisher Script

## Changes
- **File**: `scripts/run_notebook05_fisher_from_rollouts.py`
    - Added a standalone CLI script that ports the Fisher-information computation stage from `merging_analysis/05_fisher_merging_precision_weighted.ipynb`.
    - Added task-aware configuration for IF/Math model path, validation parquet, and rollout parquet inputs.
    - Added notebook-compatible outputs: `fisher_diag_{task}.pt` and `fisher_diag_{task}_summary.json`.
    - Added default behavior to use all rollout rows (no sample cap) and disable response-token truncation by default (`--max-response-tokens -1`).
    - Added two estimators: `sequence` (notebook 05 default behavior) and `tokenwise` (sum of per-token squared gradients), selectable via CLI.
    - Added manifest generation for downstream merge/distribution analysis compatibility.

## Rationale
- The notebook-based Fisher workflow is difficult to run reliably as a long background job.
- A standalone script is needed so Fisher can be recomputed with all available trajectories (e.g., IF 3294 rows) and directly reused by merge/distribution notebooks without manual notebook execution.
- Removing default response-token truncation aligns with the request to compute Fisher from full rollout responses rather than filtered token segments.

## Technical Details
- Uses PyTorch autograd over HF causal LM forward pass to accumulate diagonal Fisher estimates.
- Keeps Fisher accumulators on CPU (`float32`) to reduce persistent GPU memory pressure during long runs.
- Uses cached prompt tokenization keyed by `sample_index` for repeated rollout rows from the same prompt.
- Writes JSON summaries/manifest with run metadata for traceability and reproducibility.
