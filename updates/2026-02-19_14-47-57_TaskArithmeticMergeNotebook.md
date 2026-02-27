# [2026-02-19 14:47] Task Arithmetic IF+Math FP32 Notebook Creation

## Changes
- **File**: `merging_analysis/15_if_math_task_arithmetic_fp32.ipynb`
    - Added a new merge notebook that reuses the same task-vector definition as notebook 08 (`Delta_task = theta_task - theta_base`).
    - Implemented task arithmetic merge rule: `theta_merge = theta_base + lambda_if * Delta_if + lambda_math * Delta_math`.
    - Exposed `lambda_if` and `lambda_math` as editable runtime hyperparameters.
    - Added strict FP32 enforcement for model loading, merge arithmetic, and checkpoint saving.
    - Added output-directory naming that includes per-task lambda values.
    - Added merge metadata save path with lambda-aware naming.

## Rationale
- Implemented a lightweight and controllable merge pipeline to compare with importance-weighted merging while keeping task-vector semantics aligned with notebook 08.
- Ensured reproducibility and artifact traceability by embedding lambda values into saved checkpoint paths and metadata.
- Enforced FP32 end-to-end to satisfy numerical precision requirements specified for this experiment.

## Technical Details
- Libraries: `torch`, `transformers`, `tqdm`, `numpy`.
- Method: classic task arithmetic over parameter-wise FP32 task vectors.
- Validation: model parameter compatibility checks (key/shape) and explicit FP32 assertions before saving.
