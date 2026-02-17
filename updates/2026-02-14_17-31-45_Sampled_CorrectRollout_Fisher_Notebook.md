# [2026-02-14 17:31] refactor(notebook): recompute fisher from sampled correct rollouts without torchrun

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Replaced the front-side distributed `torchrun` Fisher workflow with direct Fisher recomputation from precomputed `correct_rollout_trajectories.parquet` files.
    - Added `TaskSpec.correct_rollout_parquet`, `RuntimeConfig.fisher_device`, and sampled-correct-rollout Fisher configuration fields.
    - Added helper logic to align `sample_index` with source validation row positions and reconstruct prompt messages for teacher-forced log-prob gradients.
    - Added deterministic subsampling (`max_correct_trajectories`) so a small subset of correct rollouts can be used for faster Fisher recomputation.
    - Updated notebook pipeline/output documentation to reflect the new non-`torchrun` Fisher path.

## Rationale
- The requested workflow was to reuse already-filtered correct rollout parquet files and recompute Fisher from a small sampled subset, instead of launching distributed rollout-based Fisher jobs.
- This reduces iteration cost and avoids rerunning the earlier rollout/Fisher generation stage when correct trajectory artifacts already exist.

## Technical Details
- Used PyTorch teacher-forcing sequence log-probability gradients on prompt+response token sequences and accumulated squared gradients as empirical diagonal Fisher estimates.
- Kept Fisher accumulation tensors on CPU to reduce persistent GPU memory pressure.
- Preserved downstream merge compatibility variables (`FISHER_OUTPUT_PATHS`, `FISHER_SUMMARY_PATHS`, `NORMALIZED_VALIDATION_PATHS`) so later merge/diagnostic cells keep working.
