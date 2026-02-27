# [2026-02-17 11:23] feat(merging_analysis): restore grad cells and append fisher sparse-update cell

## Changes
- **File**: `merging_analysis/12_if_task_vector_gradnorm_topk_sparse_update.ipynb`
    - Restored the original grad-based notebook flow as the leading cells (config, helper functions, and grad-score sparse execution cell).
    - Kept grad experiment keep ratios as originally configured: top `1%`, `10%`, `50%`, `100%`.
    - Appended a new Fisher-based execution cell at the end (without replacing the existing grad cells).
    - Added Fisher-specific appended-cell config and validation logic for:
      - IF Fisher path loading
      - Fisher score tensor compatibility checks
      - Fisher top-k sparse update save loop for top `0.1%`, `1%`, `10%`, `100%`.
    - Preserved JSON metadata + checkpoint saving behavior for both grad and fisher experiments.
- **File**: `updates/2026-02-17_11-23-59_Restore_GradCells_Append_FisherCell.md`
    - Added task update log for this notebook restructuring request.

## Rationale
- User explicitly requested that existing grad-based cells remain unchanged and that Fisher logic be added as an appended cell rather than replacing previous cells.
- The notebook was restructured to satisfy this sequencing requirement: grad experiment first, Fisher experiment appended second.
- Separate output namespaces were used for Fisher artifacts to avoid overwriting grad-based experiment outputs.

## Technical Details
- Libraries used: `PyTorch`, `Transformers`, `NumPy`, `pandas`.
- Grad-based sparse update formula:
  - `theta_sparse = theta_base + 1[|Delta_if| >= threshold] * Delta_if`
- Fisher-based appended sparse update formula:
  - `theta_sparse = theta_base + 1[F_if >= threshold] * (theta_if - theta_base)`
- Appended Fisher cell uses:
  - IF Fisher artifact path: `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal/fisher_diag_if.pt`
  - Keep ratios: `(0.001, 0.01, 0.10, 1.00)`
  - Sampled quantile threshold estimation (`sample_size=2,000,000`) to keep memory bounded.
