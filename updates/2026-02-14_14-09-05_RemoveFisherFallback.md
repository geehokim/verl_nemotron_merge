# [2026-02-14 14:09] Remove Fisher OOM Fallback Truncation

## Changes
- **File**: `scripts/run_fisher_merge_pipeline.py`
    - Removed runtime behavior that retried Fisher gradient computation with halved response length on CUDA OOM.
    - Updated Fisher worker loop to run a single exact pass per trajectory using configured `max_response_tokens` only.
    - Added explicit OOM error propagation with contextual details (`task_name`, `sample_index`, `response_tokens`) so failures are actionable.
    - Removed rollout-to-worker CLI wiring for `--fisher-oom-retry-min-response-tokens` to avoid hidden approximation behavior.
    - Removed OOM retry/skip aggregation fields from rank and merged summaries.
    - Kept `oom_retry_min_response_tokens` in config dataclass as a legacy compatibility field, but documented that it is no longer used.

## Rationale
- Hidden token-length fallback changed the effective Fisher objective per failing sample, which can distort parameter sensitivity statistics.
- The pipeline should fail transparently on memory limits so users can deliberately choose new limits (`max_response_tokens`, `max_prompt_tokens`, GPU strategy) rather than applying silent approximation.

## Technical Details
- Fisher gradient path remains based on model `labels` loss with one-shot autograd per trajectory.
- OOM now raises a `RuntimeError` immediately with guidance for manual configuration adjustment.
- Validation: `python -m py_compile scripts/run_fisher_merge_pipeline.py`.
