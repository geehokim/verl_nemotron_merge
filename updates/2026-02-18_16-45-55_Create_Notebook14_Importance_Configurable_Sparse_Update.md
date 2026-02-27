# [2026-02-18 16:45] create notebook 14 for configurable importance sparse update

## Changes
- **File**: `merging_analysis/14_if_importance_configurable_top_p_sparse_update.ipynb`
    - Added a new executable notebook (`14`) for config-driven importance artifact resolution.
    - Implemented filename resolver for artifacts like `importance_{task_name}_{mode}_{tail}{top_p}.pt` with `_exact.pt` fallback handling.
    - Implemented exact global top-k sparse mask selection and tie-safe deterministic coordinate promotion.
    - Implemented sparse merge checkpoint export loop for keep ratios `{0.1, 0.5, 1, 5, 10, 20, 50}%`.
    - Added JSON/CSV summary export and per-checkpoint merge metadata output.

## Rationale
- Needed a dedicated notebook that accepts config inputs and automatically loads the correct importance `.pt` artifact.
- Needed consistent sparse update generation from importance scores across requested top-p keep percentages.

## Technical Details
- Uses `torch`, `transformers`, `pandas`, and `numpy`.
- Reuses exact global top-k thresholding logic with deterministic tie handling for stable reproducibility.
- Uses update rule: `theta_sparse = theta_base + 1[importance in global top-k] * (theta_task - theta_base)`.
