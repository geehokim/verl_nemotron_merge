# [2026-02-14 06:26] Fix Fisher Merging: Delta-Space to Prevent Zero-Precision Catastrophe

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - **Cell 0 (markdown)**: Updated title and description to document the delta-space
      variant. Added explanation of the zero-precision catastrophe and why delta-space
      fixes it.
    - **Cell 4 (merge function + execution)**:
        - Modified `merge_with_fisher_precision_inplace` to operate in task-vector
          (delta) space instead of absolute parameter space.
        - Old formula: `theta* = sum(lambda_i * F_i * theta_i) / (sum(lambda_i * F_i) + eps)`
        - New formula: `theta* = theta_base + sum(lambda_i * F_i * Delta_i) / (sum(lambda_i * F_i) + eps)`
          where `Delta_i = theta_i - theta_base`.
        - Updated output directory: `fisher_merge_if_math` → `fisher_merge_delta_if_math`
        - Updated metadata: method name, formula string, formula_note field added.
        - Updated run summary path: `fisher_merge_run_summary.json` → `fisher_merge_delta_run_summary.json`

## Rationale
- The original Fisher merging formula operates on **absolute parameter values**.
  When Fisher information F_{i,j} ≈ 0 for all tasks at parameter j (12.94% of all
  1.72B parameters = 222M elements), the numerator and denominator both approach
  zero, causing the merged parameter to collapse to approximately zero.
- This is catastrophic because these are pretrained parameters (e.g., embedding
  weights) that should be preserved, not zeroed out.
- The delta-space variant fixes this by merging **task vectors** (deltas from base)
  instead of absolute values. When F=0, the merged delta is 0, so `theta* = theta_base`
  — the pretrained value is preserved.
- Naive task arithmetic (which operates in delta space by default) doesn't have
  this problem, explaining why it outperformed the original Fisher merging.

## Technical Details
- The fix requires only 3 lines of logic change in the merge loop:
  1. Save `theta_base = merge_param.data` before modification
  2. Compute `delta_i = theta_i - theta_base` instead of using `theta_i` directly
  3. Compute `theta* = theta_base + merged_delta` instead of just the ratio
- Fisher diagonal computation (cell-3, `run_distributed_fisher_diag.py`) is unchanged —
  the precomputed Fisher tensors are fully reused.
- All validation/compatibility checks remain identical.
