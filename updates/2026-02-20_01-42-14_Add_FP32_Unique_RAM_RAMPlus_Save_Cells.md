# [2026-02-20 01:42] feat(merging_analysis): add FP32 save cells for unique-only, RAM, and RAM+ merges

## Changes
- **File**: `merging_analysis/19_ram_ramplus_if_math_fp32.ipynb`
    - Updated the integrity-check cell to keep task vectors available for downstream save operations.
    - Added a new markdown section documenting the requested FP32 model-saving extensions.
    - Added FP32 merge/save helper functions with detailed docstrings and inline comments:
        - `build_unique_task_vectors_two_tasks`
        - `apply_single_task_vector_update_inplace`
        - `apply_ram_merge_inplace`
        - `compute_ram_plus_rescales_v2`
        - `apply_ram_plus_merge_v2_inplace`
        - `save_fp32_model_and_metadata`
    - Added an execution cell that performs and saves all requested models:
        - IF unique-only reconstruction (`base + unique_if * 1.0`)
        - Math unique-only reconstruction (`base + unique_math * 1.0`)
        - RAM merged model (FP32)
        - RAM+ (`arm-r-v2`) merged model (FP32)
    - Added a post-save smoke-check cell for saved directories and metadata files.

- **File**: `updates/2026-02-20_01-42-14_Add_FP32_Unique_RAM_RAMPlus_Save_Cells.md`
    - Recorded this update log entry in the required workflow format.

## Rationale
- To directly satisfy the request to extend notebook 19 with concrete FP32 model-saving steps for reconstruction and merge outputs.
- To keep analysis and save logic in one notebook flow so the same extracted task vectors are used consistently.

## Technical Details
- All load/merge/save paths enforce FP32 (`torch.float32`) and run on CPU.
- Unique reconstruction uses thresholded exclusivity masks:
  - `unique_if = (|Delta_if| > tau) & ~(|Delta_math| > tau)`
  - `unique_math = (|Delta_math| > tau) & ~(|Delta_if| > tau)`
- RAM merge matches active-coordinate averaging from the provided ARM baseline.
- RAM+ merge follows the provided `arm-r-v2` logic:
  - overlap coordinates use average active delta
  - non-overlap coordinates use rescaled weighted sum
- Saved model metadata is written to `merge_metadata.json` per output directory.
