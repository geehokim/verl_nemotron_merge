# [2026-02-17 13:20] Create 13 notebook for SVD non-zero truncation task-arithmetic merge

## Changes
- **File**: `merging_analysis/13_svd_nonzero_rank_task_arithmetic_merge.ipynb`
    - Added a new notebook based on the `06` analysis context, focused only on SVD truncation + checkpoint generation (no STI plotting pipeline).
    - Implemented layer-wise truncated SVD for all tensor parameters with `ndim >= 2` using top non-zero singular-value fractions `{1%, 2%, 5%, 10%, 20%}`.
    - Implemented fallback Task Arithmetic for vector/scalar parameters (`ndim < 2`), applying direct IF/Math delta averaging without SVD.
    - Added in-place merge routine that applies `theta = theta_base + 0.5 * (Delta_if_trunc + Delta_math_trunc)` and saves one model per truncation ratio.
    - Added output metadata per checkpoint (`merge_metadata.json`) and run-level summary files (JSON/CSV) under output root metadata.

## Rationale
- To create a dedicated experiment notebook requested by the user for low-rank SVD truncation ablation prior to evaluation.
- To align implementation with the paper statement: SVD only on matrix-like layers, while non-matrix layers use standard task arithmetic.
- To produce directly evaluable checkpoint directories for each rank fraction.

## Technical Details
- Libraries/frameworks: PyTorch (`torch.linalg.svd`), Transformers (`AutoModelForCausalLM`, `AutoTokenizer`), pandas, tqdm.
- Non-zero singular-value handling uses thresholding by `max(nonzero_atol, max_sv * nonzero_rtol)`.
- For each matrix tensor, retained rank is `ceil(nonzero_rank * keep_fraction)`, clamped to `[1, nonzero_rank]` when non-zero rank exists.
- SVD executes on configured device (`cuda` preferred when available, CPU fallback on runtime failure).
