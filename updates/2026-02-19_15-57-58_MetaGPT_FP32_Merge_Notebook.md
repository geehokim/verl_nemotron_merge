# [2026-02-19 15:57] MetaGPT FP32 Merge Notebook Creation

## Changes
- **File**: `merging_analysis/17_metagpt_model_exclusive_task_arithmetic_fp32.ipynb`
    - Added a new notebook (index 17) that computes task vectors in the same style as `01_layer_interference_diagnostics.ipynb` (`Delta_t = theta_t - theta_0` in FP32) and performs MetaGPT closed-form scaling merge.
    - Implemented strict FP32 validation for model loading, task-vector construction, merge arithmetic, and checkpoint saving.
    - Added metadata export (`merge_metadata.json` in checkpoint dir and run summary JSON in `metadata/`) and pairwise task-vector alignment diagnostics.
    - Reused save pattern aligned with `08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb` (`save_pretrained(..., safe_serialization=True)` + tokenizer + JSON metadata).
- **File**: `updates/2026-02-19_15-57-58_MetaGPT_FP32_Merge_Notebook.md`
    - Added this task-specific update log.

## Rationale
- Implement MetaGPT (Model-Exclusive Task Arithmetic) scaling without extra data by deriving lambdas from task-vector squared norms.
- Keep notebook workflow consistent with prior analysis notebooks (`01` task-vector definition and `08` artifact-saving pattern).
- Enforce FP32 end-to-end because this task explicitly requires FP32 for load, merge, and save.

## Technical Details
- Libraries: `torch`, `transformers`, `numpy`, `tqdm`, `json`, `dataclasses`.
- Formula implemented:
  - `lambda_t = ||theta_t - theta_0||^2 / sum_k ||theta_k - theta_0||^2`
  - `theta_merge = theta_0 + sum_t(lambda_t * Delta_t)`
- Added compatibility checks for parameter-key/shape mismatch and strict dtype guards.
- Added pairwise task-vector dot/cosine diagnostics to expose orthogonality assumption quality in saved metadata.
