# VERL Evaluation Scripts (AIME25 + IFEval)

This directory contains benchmark-specific evaluation scripts that run
`verl.trainer.main_ppo` in validation-only mode:

- `trainer.val_before_train=True`
- `trainer.val_only=True`

This reuses the same generation and reward path used by RL training while
skipping optimization updates.

## Scripts

- `common_eval_env.sh`
  - Shared environment bootstrap (venv activation, cache paths, `PYTHONPATH`).
- `run_eval_aime25.sh`
  - AIME25-only evaluation.
- `run_eval_ifeval.sh`
  - IFEval-only evaluation (includes dependency bootstrap for strict IF scoring).
- `run_eval_aime25_ifeval.sh`
  - Runs both benchmarks for one model.
- `run_eval_model_suite.sh`
  - Runs both benchmarks for Initial / IF / Math / Merged model paths.

## Quick Start (Initial Model Baseline)

```bash
MODEL_PATH="Qwen/Qwen3-1.7B" \
bash eval_scripts/run_eval_aime25_ifeval.sh
```

## Single Benchmark

```bash
MODEL_PATH="Qwen/Qwen3-1.7B" \
bash eval_scripts/run_eval_aime25.sh
```

```bash
MODEL_PATH="Qwen/Qwen3-1.7B" \
bash eval_scripts/run_eval_ifeval.sh
```

## 4-Model Suite

```bash
INITIAL_MODEL="Qwen/Qwen3-1.7B" \
IF_MODEL="/path/to/if_model_checkpoint" \
MATH_MODEL="/path/to/math_model_checkpoint" \
MERGED_MODEL="/path/to/merged_model_checkpoint" \
bash eval_scripts/run_eval_model_suite.sh
```

## Important Notes

- `data.train_files` is intentionally set to the same parquet as `data.val_files`
  because VERL constructs the train dataset object before entering val-only mode.
- In `run_eval_ifeval.sh`, the default `OUTPUT_DIR` is auto-derived from
  `MODEL_PATH` using a depth-agnostic `Qwen3-*` parser.
  - Example:
    - `MODEL_PATH=/.../Qwen3-1.7B-math/stage1/global_step_80/actor`
    - `OUTPUT_DIR=/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-math/evaluation_output/ifeval`
  - You can still override with `OUTPUT_DIR=/your/path`.
- In `run_eval_ifeval.sh`, the default `EXPERIMENT_NAME` includes the parsed
  model directory name (for example, `ifeval-valonly-Qwen3-1.7B-math-...`).
- Default logger is console-only to avoid WANDB login requirements during eval.
- You can override decoding behavior with env vars:
  - `TEMPERATURE`, `TOP_P`, `TOP_K`, `DO_SAMPLE`
