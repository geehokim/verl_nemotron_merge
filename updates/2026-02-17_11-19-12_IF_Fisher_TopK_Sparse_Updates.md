# [2026-02-17 11:19] feat(merging_analysis): switch notebook 12 to IF Fisher top-k sparse update checkpoints

## Changes
- **File**: `merging_analysis/12_if_task_vector_gradnorm_topk_sparse_update.ipynb`
    - Replaced previous task-vector-magnitude masking logic with IF Fisher-diagonal masking logic.
    - Added explicit IF Fisher artifact path configuration:
      - `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal/fisher_diag_if.pt`
    - Updated keep-ratio set to requested values: `top 0.1%`, `top 1%`, `top 10%`, `top 100%`.
    - Added Fisher-specific helper functions with docstrings and inline rationale comments:
      - Fisher loading and compatibility validation
      - Global Fisher sampling and quantile-threshold estimation
      - Sparse IF update with Fisher top-k masks
    - Updated output artifact namespace and summary metadata paths to Fisher-specific names.
- **File**: `updates/2026-02-17_11-19-12_IF_Fisher_TopK_Sparse_Updates.md`
    - Added task update log for this notebook refactor.

## Rationale
- User requested sparse IF update checkpoints based on precomputed IF Fisher information instead of IF task-vector magnitude.
- The notebook now directly consumes the existing Fisher artifact and applies top-k masking with exact requested ratios to support reconstruction/performance verification experiments.
- Output paths were separated to avoid overwriting prior sparse runs and to keep experiment provenance clear.

## Technical Details
- Libraries used: `PyTorch`, `Transformers`, `NumPy`, `pandas`.
- Masking formula implemented:
  - `theta_sparse = theta_base + 1[F_if >= threshold] * (theta_if - theta_base)`
- Threshold computation:
  - Global threshold estimated via sampled Fisher values (`sample_size=2,000,000`) to reduce memory pressure on 1B+ models.
  - Quantile mapping: `threshold = quantile(F_if, 1 - keep_ratio)`.
- Saved outputs:
  - Checkpoints under `.../Qwen3-1.7B-fisher-merge/if_sparse_topk_fisher_0p1_1_10_100/`
  - Summary JSON at `.../Qwen3-1.7B-fisher-merge/metadata/if_sparse_topk_fisher_summary_0p1_1_10_100.json`
