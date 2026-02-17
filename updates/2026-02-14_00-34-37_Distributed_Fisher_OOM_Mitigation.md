# [2026-02-14 00:34] Distributed Fisher OOM Mitigation

## Changes
- **File**: `scripts/run_fisher_merge_pipeline.py`
    - Refactored Fisher stage from single-GPU in-process computation to a distributed worker orchestration model using `torch.distributed.run`.
    - Added internal worker mode (`--fisher-worker`) and hidden CLI flags for distributed Fisher execution inputs.
    - Implemented rank-sharded Fisher accumulation across all task GPUs (using task `gpu_ids`) with rank-0 aggregation into `fisher_diagonal.pt`.
    - Replaced explicit large `log_softmax` tensor path with `labels`-based loss path to avoid giant intermediate allocations.
    - Added gradient checkpointing support in Fisher settings to reduce activation memory during backward.
    - Added OOM fallback retry logic that halves response token length per sample until a configurable minimum.
    - Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for Fisher subprocess environment to reduce fragmentation risk.
    - Extended `FisherSettings` with:
        - `grad_checkpointing`
        - `oom_retry_min_response_tokens`
    - Updated task orchestration to pass runtime/repo context into Fisher orchestrator.

## Rationale
- Fisher backward on long math responses was exhausting GPU memory on a single device.
- The previous implementation materialized very large token-vocab tensors in float32, causing avoidable memory spikes.
- Distributing trajectories across 8 GPUs and reducing per-sample peak memory is required for stable completion.

## Technical Details
- Distributed execution command:
    - `python -m torch.distributed.run --standalone --nproc_per_node=<len(task.gpu_ids)> ... --fisher-worker ...`
- Rank behavior:
    - Each rank loads the same rollout/prepared parquet metadata.
    - Each rank processes deterministic strided shard (`iloc[rank::world_size]`).
    - Each rank stores local Fisher shard and metadata; rank-0 aggregates and normalizes by total processed trajectories.
- Memory-focused implementation details:
    - Disabled model cache (`use_cache=False`) during Fisher backward.
    - Optional `gradient_checkpointing_enable()`.
    - OOM retry loop with response-length halving.
- Validation commands executed:
    - `python -m py_compile scripts/run_fisher_merge_pipeline.py`
    - `python scripts/run_fisher_merge_pipeline.py --help`
    - `python scripts/run_fisher_merge_pipeline.py --config scripts/configs/fisher_merge_pipeline.local.yaml --dry-run`
