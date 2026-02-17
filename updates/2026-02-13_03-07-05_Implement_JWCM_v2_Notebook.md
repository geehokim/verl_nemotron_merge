# [2026-02-13 03:07] Implement JWCM v2 notebook for critical-token attribution merge

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Added a new end-to-end notebook implementing JWCM v2.
    - Added validation sampling pipeline from IF/Math training parquet files (default: 512 samples per task).
    - Added critical-token identification pipeline using `Δlog p = log p_rl - log p_base` on RL-generated trajectories.
    - Added parameter importance scoring implementation for
      `S_j = Σ |∂log p(y_t)/∂θ_j · Δθ_j|` with two modes:
      - `exact_token_abs` (formula-faithful token-level backprop)
      - `sequence_sum_approx` (runtime-oriented approximation)
    - Added JWCM v2 soft attribution merge:
      `θ_base + Σ(S·Δθ)/(ΣS+ε)`.
    - Added JWCM v2 sparsified merge:
      `1[max S > κ] · Σ(S·Δθ)/(ΣS+ε)` with configurable `absolute` and `sampled_quantile` kappa modes.
    - Added artifact saving for both ablations:
      - `jwcm_v2_soft`
      - `jwcm_v2_soft_sparsified`
    - Added structured metadata and intermediate artifact outputs (validation splits, token deltas, critical sets, importance tensors, run summary).
    - Added detailed docstrings and inline comments for all helper functions and major logic blocks.

- **File**: `updates/2026-02-13_03-07-05_Implement_JWCM_v2_Notebook.md`
    - Added per-task update log describing implementation details and rationale.

## Rationale
- Implement the JWCM formulation directly from the critical-token attribution objective, focusing on RLVR-specific token shifts instead of whole-dataset Fisher-style averaging.
- Replace winner-take-all behavior with contribution-proportional soft merge for a less aggressive and more stable parameter composition.
- Provide a sparsified variant for ablation to remove low-evidence parameters and reduce merge noise.

## Technical Details
- Libraries: `torch`, `transformers`, `pandas`, `numpy`, `tqdm`, `nbformat`.
- Critical-token extraction:
  - Deterministic generation (`do_sample=False`) from task RL model.
  - Token-wise `Δlog p` computed against base model on the same generated trajectory.
  - Selection by combined quantile threshold (`top_p`) and minimum epsilon.
- Importance scoring:
  - Uses absolute task vector `|Δθ|` and gradient magnitude accumulation.
  - Supports exact token-wise and approximate sequence-wise gradient accumulation.
- Merge:
  - CPU-side parameter-wise merge for deterministic checkpoint writing.
  - Optional sampled-quantile kappa estimation for sparsification without full flattening of all parameters.
