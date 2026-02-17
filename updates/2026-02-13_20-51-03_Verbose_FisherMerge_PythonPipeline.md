# [2026-02-13 20:51] feat(fisher): implement verbose python pipeline for rollout->fisher->merge

## Changes
- **File**: `scripts/fisher_rollout_reward_wrapper.py`
    - Added a VERL custom reward wrapper that preserves base reward behavior while always returning dict payloads.
    - Added `sample_index` injection from `extra_info["index"]` so rollout JSONL rows can be joined back to source rows deterministically.
    - Added robust fallback handling for reward-function errors and metadata tagging (`data_source`).
- **File**: `scripts/run_fisher_merge_pipeline.py`
    - Added end-to-end verbose pipeline script for:
      - VERL val-only rollout with task-specific sampling (`top_p`, `top_k`, `n`, `temperature`, `max_new_tokens`, `do_sample`).
      - Correct-trajectory extraction from dumped validation JSONL.
      - Source parquet augmentation with `rollout_results` and `rollout_correct_count` columns.
      - Fisher diagonal estimation using one-shot sequence log-probability backward and `g^2` accumulation.
      - Fisher precision-weighted model merge via `theta*=sum(lambda*F*theta)/sum(lambda*F)`.
    - Added detailed logging, sectioned progress output, subprocess log streaming, config serialization, and task/merge summaries.
    - Added `--dry-run` mode for configuration/path validation without executing heavy jobs.
- **File**: `scripts/configs/fisher_merge_pipeline.example.yaml`
    - Added a concrete example config with IF/Math tasks, per-task rollout settings, Fisher settings, merge lambdas, and runtime controls.

## Rationale
- User requested replacing notebook-based implementation with a script-based pipeline and very verbose logging.
- User requested strict three-stage workflow: VERL rollout + correct trajectory persistence, Fisher computation on correct trajectories, and Fisher weighted merging.
- Deterministic row alignment was required to safely write rollout outcomes back to dataset rows.

## Technical Details
- Uses VERL `main_ppo` in `trainer.val_only=True` mode with `trainer.validation_data_dir` to dump JSONL rollouts.
- Uses reward wrapper indirection to support both default reward routing and task-specific custom reward functions while attaching `sample_index`.
- Fisher estimator implementation:
  - Forward: sum token log-probabilities over generated response tokens.
  - Backward: single `torch.autograd.grad` call per trajectory.
  - Accumulation: element-wise squared gradients into diagonal Fisher buffers.
  - Normalization: divide by number of processed correct trajectories.
- Merge implementation:
  - Coordinate-wise precision weighting with task lambdas.
  - Denominator near-zero fallback to anchor parameter value to avoid degenerate coordinates.
