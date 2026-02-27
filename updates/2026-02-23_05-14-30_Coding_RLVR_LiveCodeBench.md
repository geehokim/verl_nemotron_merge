# [2026-02-23 05:14] Coding RLVR LiveCodeBench Pipeline

## Changes
- **File**: `tasks/todo.md`
    - Added task checklist, acceptance criteria, verification checklist, and results summary for this implementation.
- **File**: `tasks/lessons.md`
    - Created initial lessons log template.
- **File**: `nemotron_evaluation/data/coding/convert_competitive_coding_to_verl_parquet.py`
    - Added converter from HF `load_from_disk` competitive coding dataset to VERL-format parquet.
    - Implemented batch parquet writing with `pyarrow.parquet.ParquetWriter`.
    - Added LiveCodeBench-compatible `ground_truth` encoding (`base64 + zlib + pickle(json.dumps(...))`).
- **File**: `nemotron_evaluation/eval/verl_custom_reward.py`
    - Switched coding verifier path to AceReason `tools/code_verifier.py` via file-path dynamic import.
    - Added `nemotron_cascade_rl_coding` reward routing.
    - Added code-switching detection helper (copied logic pattern from math reward) and coding reward override to `0.0` when code-switching is detected.
    - Preserved IFRL/SWE reward routing behavior.
- **File**: `nemotron_evaluation/eval/verify_coding_rlvr_pipeline.py`
    - Added verification script for schema, row-count parity, ground-truth decoding, synthetic reward behavior, forced code-switching override, and optional RLHFDataset compatibility.
- **File**: `run_qwen3_1.7b_ifrl_coding.sh`
    - Added coding RL training launcher mirroring IF-RL script style.
    - Added preflight conversion for competitive coding train parquet and conditional LiveCodeBench val parquet generation.
    - Added dependency checks including `fasttext-langdetect`.

## Rationale
- Implement end-to-end coding RLVR pipeline with minimal divergence from existing VERL integration patterns.
- Keep verifier behavior aligned with AceReason LiveCodeBench evaluation logic.
- Apply coding-specific code-switching rule (reward forced to 0) as requested, while preserving existing non-coding reward paths.

## Technical Details
- AceReason verifier execution is isolated in subprocesses to avoid side effects from reliability guards that patch builtins/os.
- Correctness judgment uses `np.all(np.array(res) > 0)` exactly as in LiveCodeBench evaluation flow.
- Prompt language selection for code-switching uses `extra_info.prompt_language -> extra_info.language -> extra_info.lang -> "en"`.
- Verification evidence collected by running:
  - converter CLI for competitive coding dataset
  - pipeline verification script (base python and verl venv)
  - IFRL/SWE reward smoke checks through `compute_score(...)`
