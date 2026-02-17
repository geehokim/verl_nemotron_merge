# [2026-02-13 04:24] Revert notebook flow and add 8-GPU precompute script for token delta collection

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Reverted the notebook execution flow back to the original single-process structure.
    - Removed distributed in-notebook orchestration from the main execution path.
    - Added optional precompute loading configuration:
      - `USE_PRECOMPUTED_TOKEN_DELTAS`
      - `PRECOMPUTED_TOKEN_ROOT`
    - Added `load_json` helper function to load precomputed metadata.
    - Updated task loop logic:
      - If precompute is enabled, load
        - `all_token_deltas.csv`
        - `sequence_cache.pt`
        - `token_summary.json`
        from `<PRECOMPUTED_TOKEN_ROOT>/<task>/`.
      - If precompute is disabled, fallback to in-notebook `collect_token_delta_records(...)` computation.
    - Updated run metadata to explicitly store whether precomputed token deltas were used.
    - Updated Notes section with concrete `torchrun` command for external 8-GPU precompute.

- **File**: `scripts/run_collect_token_deltas_8gpu.py`
    - Added a new standalone distributed precompute script dedicated to
      `collect_token_delta_records` workload.
    - Implemented `torchrun`-compatible distributed execution with rank-based validation sharding.
    - For each task (`if`, `math`), each rank computes local token deltas and sequence cache.
    - Added rank-local artifact saving and rank-0 aggregation into notebook-consumable files:
      - `<output_root>/<task>/all_token_deltas.csv`
      - `<output_root>/<task>/sequence_cache.pt`
      - `<output_root>/<task>/token_summary.json`
    - Included comprehensive docstrings and inline comments for all core functions.

- **File**: `updates/2026-02-13_04-24-22_Revert_Notebook_Add_8GPU_Precompute_Script.md`
    - Added update log for this refactor.

## Rationale
- Keep the notebook simple and stable for iterative analysis/ablation.
- Isolate the heavy distributed token-delta generation into a dedicated Python script for 8-GPU execution.
- Preserve reproducibility by writing deterministic intermediate artifacts that the notebook can load directly.

## Technical Details
- Distributed precompute is launched with `torchrun` and uses row-stride sharding of validation prompts.
- The script computes `Δlog p = log p_rl - log p_base` on RL-generated trajectories, matching notebook semantics.
- The notebook now supports two execution modes:
  - precompute-load mode (default)
  - inline-compute fallback mode
- Validation compatibility is maintained through shared `sample_id` keys and persisted `sequence_cache.pt` tensors.
