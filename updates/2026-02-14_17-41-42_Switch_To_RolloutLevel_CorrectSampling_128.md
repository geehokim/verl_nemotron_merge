# [2026-02-14 17:41] Switch To Rollout-Level Correct Sampling (128)

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Changed attribution selection policy from prompt-level subset focus to rollout-level correct-trajectory sampling.
    - Added `ATTRIBUTION_CORRECT_ROLLOUT_SAMPLES_PER_TASK: int | None = 128`.
    - Restored validation prompt scope to full Fisher validation set via `ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK=None`.
    - Extended `collect_token_delta_records_from_fisher_rollouts(...)` with:
      - `max_total_rollouts`
      - `sampling_seed`
    - Implemented deterministic global rollout-row sampling after correctness/sample filters.
    - Added run-summary metadata field `attribution_correct_rollout_samples_per_task`.
    - Updated notes to clarify trajectory-level sampling semantics.

## Rationale
- The requested behavior is to sample 128 rows from the entire pool of correct rollouts, not to first limit prompts to 128 and then filter by correctness.
- Rollout-level sampling better matches the intended experimental design for JWCM attribution when multiple trajectories exist per prompt.

## Technical Details
- Sampling is applied on filtered correct rollout rows using:
  - `working_df.sample(n=128, random_state=seed, replace=False)` when enough rows exist.
- For tasks with fewer than 128 correct rollouts, all available rows are used.
- Verified on current artifacts:
  - IF: 3294 correct rows -> 128 sampled rows
  - Math: 127 correct rows -> 127 used rows
