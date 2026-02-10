# [2026-02-10 18:43] Add eval scripts for AIME25 and IFEval

## Changes
- **File**: `eval_scripts/common_eval_env.sh`
    - Added a shared environment bootstrap script for evaluation workflows.
    - Centralized cache paths, venv activation, and `PYTHONPATH` setup.
- **File**: `eval_scripts/run_eval_aime25.sh`
    - Added benchmark-specific VERL `val_only` evaluation script for AIME25.
    - Configured default model path to `Qwen/Qwen3-1.7B` for initial baseline.
- **File**: `eval_scripts/run_eval_ifeval.sh`
    - Added benchmark-specific VERL `val_only` evaluation script for IFEval.
    - Enabled custom IF reward routing and dependency bootstrap for strict rule checks.
- **File**: `eval_scripts/run_eval_aime25_ifeval.sh`
    - Added wrapper script to run AIME25 and IFEval sequentially for one model.
- **File**: `eval_scripts/run_eval_model_suite.sh`
    - Added suite runner for Initial/IF/Math/Merged model comparisons.
- **File**: `eval_scripts/README.md`
    - Added usage documentation and execution examples.

## Rationale
- Implement benchmark-separated evaluation entry points while reusing VERL's native validation pipeline.
- Provide direct baseline measurement for the initial model and an easy path to compare IF/Math/Merged variants.
- Keep evaluation reproducible with explicit script-level configuration and stable environment setup.

## Technical Details
- Evaluation mode uses `python3 -m verl.trainer.main_ppo` with:
    - `trainer.val_before_train=True`
    - `trainer.val_only=True`
- Data inputs:
    - AIME: `nemotron_evaluation/data/aime25/test_verl_ready.parquet`
    - IFEval: `nemotron_evaluation/data/ifeval/val_verl_ready.parquet`
- IFEval scoring path:
    - `custom_reward_function.path=nemotron_evaluation/eval/verl_custom_reward.py`
    - `custom_reward_function.name=compute_score`
- Verified shell syntax using `bash -n` for all newly added scripts.
