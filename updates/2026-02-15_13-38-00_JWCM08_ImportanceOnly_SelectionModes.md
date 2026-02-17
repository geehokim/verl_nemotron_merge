# [2026-02-15 13:38] JWCM08 Importance-Only Selection Modes

## Changes
- **File**: `scripts/run_jwcm08_importance_only.py`
    - Added a standalone CLI script that extracts the JWCM v2 importance-computation path from `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`.
    - Implemented Fisher-rollout reuse token-delta collection (`delta_logp = logp_rl - logp_base`) and sequence payload construction.
    - Implemented configurable critical-token selection modes:
      - `positive` (top p% of positive tail)
      - `negative` (bottom p% / negative tail)
      - `absolute` (top p% by `|delta_logp|`)
      - `all` (compute all three in one run)
    - Set default `top_p` to `0.20` (20%) per user request.
    - Added mode-specific artifact naming so importance tensors are saved with distinct filenames (e.g., `importance_if_positive_top20.pt`, `importance_if_negative_bottom20.pt`, `importance_if_absolute_top20.pt`).
    - Added run summary metadata JSON for downstream merging/analysis traceability.

## Rationale
- Notebook-only execution is cumbersome for long attribution runs and hard to automate.
- A script is required to run importance computation reproducibly in background jobs and save reusable artifacts.
- Different token-selection criteria are needed for strict comparison against paper-inspired variants and ablation workflows.

## Technical Details
- Uses Hugging Face causal LM forward passes to compute per-token `delta_logp` on reused rollout trajectories.
- Uses JWCM attribution formula accumulation with autograd:
  - exact mode: per-critical-token gradient accumulation
  - approximate mode: sequence-sum single backward
- Stores importance tensors as PyTorch dictionaries on CPU (`float32`) for direct reuse in merge/analysis notebooks.
