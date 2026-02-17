# [2026-02-15 13:45] JWCM08 Distributed Importance-Only

## Changes
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - Added a separate distributed execution script for JWCM08 importance computation.
    - Implemented torchrun-compatible distributed runtime initialization and teardown.
    - Implemented rollout-row deterministic filtering followed by rank-based sharding.
    - Implemented global threshold computation for critical-token selection using all-rank token-delta aggregation.
    - Implemented distributed importance reduction where each rank computes local partial importance and rank 0 saves final merged importance tensors.
    - Added mode-specific importance artifact naming compatible with downstream merge/analysis usage.

## Rationale
- Single-GPU attribution runs are too slow for practical iteration.
- A separate distributed script avoids changing the single-GPU script while enabling multi-GPU acceleration.
- Keeping output format compatible allows immediate reuse by existing notebooks and merge pipelines.

## Technical Details
- Uses PyTorch distributed (`torch.distributed`) with `nccl`/`gloo` backend.
- Uses rank-strided sharding for rollout rows and rank-local autograd accumulation.
- Reduces large importance tensors to rank 0 with distributed SUM operations.
- Preserves selection modes (`positive`, `negative`, `absolute`, `all`) and default `top_p=0.20`.
