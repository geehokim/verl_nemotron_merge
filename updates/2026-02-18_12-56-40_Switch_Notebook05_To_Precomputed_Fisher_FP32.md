# [2026-02-18 12:56] Switch Notebook05 To Precomputed Fisher FP32

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Updated notebook pipeline description to use precomputed Fisher `.pt` files instead of in-notebook Fisher recomputation.
    - Added fixed precomputed Fisher directory constant:
      `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal`.
    - Switched runtime merge dtype from `bf16` to `fp32`.
    - Rewired `FISHER_OUTPUT_PATHS` / `FISHER_SUMMARY_PATHS` to load `fisher_diag_if.pt` and `fisher_diag_math.pt` from the external directory.
    - Replaced Fisher recomputation execution block with defensive load-validation logic based on notebook 11 Fisher loader pattern.
    - Updated merge execution path to force FP32 for model loading, merging, and saving.
    - Added FP32 guards and explicit FP32 normalization before save.
    - Updated metadata payload and output artifact documentation to record precomputed Fisher mode and FP32 merge dtype.

- **File**: `updates/2026-02-18_12-56-40_Switch_Notebook05_To_Precomputed_Fisher_FP32.md`
    - Added this task update log.

## Rationale
- The requested workflow requires reusing precomputed Fisher diagonals (`if`, `math`) from a fixed directory and avoiding direct Fisher estimation inside notebook 05.
- FP32 end-to-end handling is required to prevent dtype drift during load/merge/save and to keep merged checkpoint precision explicit.

## Technical Details
- Reused Fisher dictionary loading/validation pattern from notebook 11:
  - file existence check
  - mapping type validation
  - CPU loading via `torch.load(..., map_location='cpu')`
- Kept the original delta-space precision merge equation:
  - `theta*_j = theta_base_j + sum_i(lambda_i * F_{i,j} * Delta_{i,j}) / (sum_i(lambda_i * F_{i,j}) + epsilon)`
- Added dtype integrity checks for all floating parameters and an explicit FP32 conversion pass before `save_pretrained`.
