# [2026-02-14 00:32] Add STI RL Analysis Notebook

## Changes
- **File**: `merging_analysis/06_singular_task_interference_rl.ipynb`
  - Added a new end-to-end analysis notebook for RL task-vector interference between IF and Math checkpoints.
  - Implemented adaptive truncated SVD (`torch.svd_lowrank`) with fallback behavior and energy tracking.
  - Implemented parameter-level STI computation (`entrywise L1` form), U/V singular-vector overlap metrics, and 90/95/99 reconstruction-rank diagnostics.
  - Implemented layer-level aggregation using `numel`-weighted statistics and output CSV generation.
  - Added visualization cells for layer STI profile, overlap profile, low-rank profile, and top-layer U/V heatmaps.
  - Added results-interpretation and reproducibility summary sections, including JSON run summary export.
- **File**: `updates/2026-02-14_00-32-06_Add_STI_RL_Analysis_Notebook.md`
  - Added task update log documenting notebook additions and rationale.

## Rationale
- To introduce a dedicated RL validation notebook that operationalizes geometric task-interference analysis (STI) with practical runtime constraints.
- To quantify whether RL task vectors are low-rank and where layer-wise singular-axis overlap may cause destructive interference during merging.
- To produce reproducible artifacts (CSV/PNG/JSON) that can be reused for downstream merge-policy decisions.

## Technical Details
- Core libraries: PyTorch (`torch.svd_lowrank`, `torch.linalg.svd` fallback), pandas, numpy, matplotlib.
- STI implementation: `STI = ||(U^T U - I)Sigma(V^T V - I)||_1` using entrywise absolute-sum norm.
- Overlap implementation: sign-invariant cosine overlap using `abs(dot)` for singular-vector sign ambiguity.
- Rank analysis: cumulative energy thresholds at 0.90 / 0.95 / 0.99 with monotonicity validation.
- Aggregation: parameter-to-layer projection via `numel`-weighted means.
