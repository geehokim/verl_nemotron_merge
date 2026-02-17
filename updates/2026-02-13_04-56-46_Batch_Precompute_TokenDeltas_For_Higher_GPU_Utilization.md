# [2026-02-13 04:56] Batch precompute token-delta collection for higher GPU utilization

## Changes
- **File**: `scripts/run_collect_token_deltas_8gpu.py`
    - Refactored `collect_token_delta_records(...)` from per-sample processing to batched processing.
    - Added new argument `inference_batch_size` to drive batched generation and batched log-prob forward passes.
    - Added prompt batching with tokenizer `padding=True` and left-padding configuration for decoder-only generation stability.
    - Added sequence trimming logic:
      - removes left prompt padding
      - trims generated tail at first EOS in generated segment
      - preserves single-sample-compatible `token_position` semantics.
    - Added batched log-prob computation for RL/Base on padded trimmed sequences using explicit attention masks.
    - Added robust metadata mapping for `sample_id`/`dataset_index` through packed batches.
    - Added summary field `inference_batch_size` in local task summary payload.
    - Added CLI flag:
      - `--inference-batch-size` (default: `16`)
    - Passed `--inference-batch-size` from `main()` into `collect_token_delta_records(...)`.

- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Updated precompute command in Notes to include `--inference-batch-size 16`.
    - Added note recommending increase of `--inference-batch-size` to raise GPU memory/utilization.

- **File**: `updates/2026-02-13_04-56-46_Batch_Precompute_TokenDeltas_For_Higher_GPU_Utilization.md`
    - Added update log for this batching/performance refactor.

## Rationale
- Previous token-delta precompute processed one prompt at a time per GPU, underutilizing available memory/throughput.
- Batched generation + batched log-prob forward improves GPU occupancy and reduces wall-clock runtime.
- CLI-level batch-size control enables practical tuning to target desired per-GPU memory usage.

## Technical Details
- Generation step now handles micro-batches per rank via `--inference-batch-size`.
- Forward pass for `log p_rl` and `log p_base` is executed on padded trimmed sequence batches with explicit attention masks.
- Code path preserves downstream notebook compatibility by keeping `sequence_cache` and `token_position` format consistent.
- Validation checks:
  - `python -m py_compile scripts/run_collect_token_deltas_8gpu.py`
  - notebook code-cell compile check for `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
