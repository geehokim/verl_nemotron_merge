# [2026-02-13 04:33] Force load-only mode for precomputed token delta artifacts in JWCM notebook

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Removed optional toggle-based branching for token-delta acquisition in the main task loop.
    - Enforced mandatory precomputed artifact loading from `PRECOMPUTED_TOKEN_ROOT` for each task:
      - `all_token_deltas.csv`
      - `sequence_cache.pt`
      - `token_summary.json`
    - Kept strict fail-fast behavior when any required precompute artifact is missing.
    - Updated runtime metadata writes to fixed values:
      - `used_precomputed_token_deltas: True`
    - Replaced optional precompute config text with explicit load-only mode declaration.
    - Updated notebook notes to mark 8-GPU precompute as required before notebook execution.

- **File**: `updates/2026-02-13_04-33-30_Force_LoadOnly_Precomputed_TokenDeltas.md`
    - Added task update log for this notebook behavior change.

## Rationale
- Prevent accidental single-GPU inline token-delta recomputation inside notebook runs.
- Make workflow deterministic and scalable by separating heavy token-delta collection into the dedicated distributed script.
- Ensure notebook execution always consumes the same precomputed intermediate artifacts for reproducible JWCM attribution/merge experiments.

## Technical Details
- The notebook now always reads token-delta artifacts from:
  - `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-jwcm-v2/distributed_token_precompute/<task>/`
- Sequence cache keys are still coerced to integer `sample_id` values for stable downstream indexing.
- Inline fallback call to `collect_token_delta_records(...)` is removed from the task loop execution path.
