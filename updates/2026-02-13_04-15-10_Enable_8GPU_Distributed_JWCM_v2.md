# [2026-02-13 04:15] Enable 8-GPU distributed execution for JWCM v2 notebook

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Added distributed runtime support using `torch.distributed` with `torchrun` environment detection (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`).
    - Added distributed helper functions with docstrings:
      - `init_distributed_context`
      - `barrier_if_needed`
      - `finalize_distributed_context`
      - `is_rank_zero`
      - `rank_zero_print`
      - `shard_dataframe_by_rank`
      - `load_json`
    - Updated runtime config to include distributed settings:
      - `distributed_timeout_minutes`
      - `expected_world_size`
    - Updated pipeline so each rank:
      - Loads base/task models on rank-local GPU (`cuda:{LOCAL_RANK}`).
      - Processes deterministic shard of validation data.
      - Writes rank-local token delta/critical CSV files.
      - Writes rank-local importance tensor + metadata.
    - Added rank-0 aggregation stage:
      - Merges rank-local token/critical CSV files into global CSV artifacts.
      - Aggregates rank-local importance tensors into global `importance_{task}.pt`.
      - Applies global token-count normalization once after cross-rank summation.
    - Kept final merge/save stage on rank 0 to avoid duplicate checkpoint writes.
    - Added distributed metadata fields into run summaries and merge metadata.
    - Added notebook notes section explaining multi-GPU usage with `torchrun`.

- **File**: `updates/2026-02-13_04-15-10_Enable_8GPU_Distributed_JWCM_v2.md`
    - Added update log for this distributed execution refactor.

## Rationale
- The previous notebook execution path used only a single GPU in standard notebook mode.
- JWCM attribution is the dominant compute stage, so validation-shard parallelism across 8 GPUs significantly improves throughput.
- Rank-local computation plus rank-0 aggregation keeps implementation robust while preserving exact JWCM-v2 merge behavior.

## Technical Details
- Backend: `nccl` when CUDA is available; falls back to single-process mode when `WORLD_SIZE=1`.
- Data parallel strategy:
    - Validation split is shared; each rank uses row-striding sharding (`df.iloc[rank::world_size]`).
- Aggregation strategy:
    - Rank-local importance tensors are saved to disk and aggregated on rank 0.
    - When local normalization is enabled, tensors are first converted back to raw sums (`* local_processed_tokens`) before global normalization.
- Final artifacts:
    - Global critical/token CSVs per task
    - Global importance tensors per task
    - `jwcm_v2_soft` and `jwcm_v2_soft_sparsified` checkpoints (rank 0 only)
