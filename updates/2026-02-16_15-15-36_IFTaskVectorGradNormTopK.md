# [2026-02-16 15:15] feat(merging_analysis): add IF task-vector gradnorm top-k sparse update notebook

## Changes
- **File**: `merging_analysis/12_if_task_vector_gradnorm_topk_sparse_update.ipynb`
    - Added a new notebook for IF task-vector sparse reconstruction experiments.
    - Added runtime/config cell for base model, IF checkpoint path, output root, keep-ratio list, and threshold sample size.
    - Added utility functions with docstrings for JSON metadata saving, dtype resolution, tokenizer/model loading, parameter compatibility checks, score sampling, quantile-threshold estimation, sparse update application, and checkpoint saving.
    - Added execution cell that builds IF task-vector magnitude scores (`|theta_if - theta_base|`), applies sparse updates at top 1%/10%/50%/100%, saves each checkpoint, and writes summary metadata.
- **File**: `updates/2026-02-16_15-15-36_IFTaskVectorGradNormTopK.md`
    - Added task-specific update log for traceability.

## Rationale
- Implemented a direct reconstruction test requested by the user: keep only high-magnitude IF task-vector coordinates and measure whether IF behavior is recoverable as keep ratio increases.
- Reused structure and saving flow from `05_fisher_merging_precision_weighted.ipynb` so artifact layout and metadata style remain consistent with existing analysis notebooks.
- Used IF delta magnitude as a coordinate-level grad-norm proxy to support global top-k parameter filtering without requiring additional rollout recomputation.

## Technical Details
- Frameworks/libraries used: `PyTorch`, `Transformers`, `NumPy`, `pandas`.
- Sparse update formula:
  - `Delta_if = theta_if - theta_base`
  - `theta_sparse = theta_base + 1[|Delta_if| >= threshold] * Delta_if`
- Threshold strategy:
  - Global threshold estimated from sampled score values (2,000,000 samples) to avoid full flatten materialization overhead.
  - Quantile mapping: threshold at `q = 1 - keep_ratio` for each keep ratio.
- Output artifacts:
  - Checkpoints under `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/if_sparse_topk_task_vector_gradnorm/`
  - Run summary JSON at `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/metadata/if_sparse_topk_task_vector_gradnorm_summary.json`
