# [2026-02-10 20:10] Auto-parse model path for IFEval output directory

## Changes
- **File**: `eval_scripts/run_eval_ifeval.sh`
    - Added `parse_model_dir_from_path()` helper to extract model directory names using a depth-agnostic `Qwen3-*` regex.
    - Added fallback parsing logic for non-Qwen3 identifiers using the last path segment.
    - Updated default output path construction to derive `OUTPUT_DIR` from `MODEL_PATH` as:
      `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/<parsed_model_dir>/evaluation_output/ifeval`.
    - Added detailed inline comments explaining parsing priority, fallback behavior, and why depth-agnostic parsing is required.
- **File**: `eval_scripts/README.md`
    - Documented the new `MODEL_PATH`-based auto-output-dir behavior for `run_eval_ifeval.sh`.
    - Added concrete example mapping from deep checkpoint path to normalized evaluation output path.

## Rationale
- Ensure consistent evaluation output location at model-family level even when checkpoint depth differs across benchmarks and training stages.
- Avoid manual `OUTPUT_DIR` setup for common stage/checkpoint paths such as `.../Qwen3-1.7B-math/stage1/global_step_80/actor`.

## Technical Details
- Parsing rule prefers regex match: `Qwen3[^/[:space:]]*`.
- Fallback rule uses shell basename extraction for broader model-path compatibility.
- The default root was aligned to `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output`.
- Validation:
    - `bash -n eval_scripts/run_eval_ifeval.sh` passed.
    - Regex mapping sanity-check confirmed expected output for the provided checkpoint path.
