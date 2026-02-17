# [2026-02-14 17:29] JWCM v2 Soft Attribution Fisher Reuse

## Changes
- **File**: `merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`
    - Added a new JWCM v2 notebook derived from the prior soft-attribution notebook.
    - Switched validation sourcing to Fisher validation parquet files under `Qwen3-1.7B-fisher-merge/validation`.
    - Added prioritized token-delta sourcing logic:
      - Fisher correct rollout reuse (`correct_rollout_trajectories.parquet`) first.
      - Legacy precomputed token-delta artifacts fallback.
      - Online generation fallback as last resort.
    - Implemented rollout-to-token-delta conversion that supports multiple trajectories per sample via `sequence_id`.
    - Added delta-logp analysis artifact generation and persistence:
      - task-level summary JSON,
      - quantile CSV,
      - per-sample CSV,
      - top critical-token CSV.
    - Kept and reused JWCM importance computation (`exact`/`approx`) and merge stages (`jwcm_v2_soft`, `jwcm_v2_soft_sparsified`).
    - Kept and reused importance sparsity and layer-distribution analysis sections.
    - Updated runtime output root to `Qwen3-1.7B-jwcm-v2-new`.

## Rationale
- Reuse existing Fisher artifacts to avoid repeating expensive rollout generation and to align JWCM attribution with Fisher evaluation conditions.
- Focus attribution on high `Δlog p` tokens and persist rich token statistics for downstream diagnostics.
- Preserve prior JWCM merge and analysis pipeline to keep methodological comparability while changing token attribution source.

## Technical Details
- Libraries: `PyTorch`, `transformers`, `pandas`, `numpy`, `tqdm`.
- Implemented token-level delta computation on rollout trajectories using:
  - `Δlog p = log p_rl - log p_base` on continuation tokens.
- Critical token extraction uses joint thresholding:
  - quantile-based `top_p` and absolute `epsilon` floor.
- Importance scores follow JWCM v2 formulation:
  - `S_j = Σ |∂log p/∂θ_j · Δθ_j|`.
- Merge formulas preserved for both variants:
  - soft merge and sparsified soft merge with sampled quantile kappa.
