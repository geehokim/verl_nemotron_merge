# [2026-02-13 04:52] Limit JWCM attribution to 128 validation samples per task

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb`
    - Added an explicit attribution sample budget config:
      - `ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK = 128`
    - Updated validation loading stage to actually enforce the budget using:
      - `selected_validation_df = validation_df.head(ATTRIBUTION_VALIDATION_SAMPLES_PER_TASK)`
    - Added runtime logging that reports selected vs source row counts per task.
    - Updated precomputed token-delta loading path to filter artifacts by selected `sample_id` set:
      - Filters `token_records_df` to selected sample ids.
      - Filters `sequence_cache` to selected sample ids.
    - Added fail-fast guard for empty post-filter token records.
    - Extended `token_summary` metadata with selected-subset counters:
      - `selected_validation_samples_for_attribution`
      - `selected_token_records_for_attribution`
      - `selected_sequence_count_for_attribution`
    - Updated notebook notes to state that attribution currently uses first 128 validation samples per task.

- **File**: `updates/2026-02-13_04-52-32_Limit_JWCM_Attribution_To_128_Per_Task.md`
    - Added update log for this attribution-budget change.

## Rationale
- Reduce attribution runtime by constraining task-level validation coverage from 512 to 128 prompts.
- Keep precompute workflow intact while ensuring notebook-side attribution strictly respects the smaller budget.
- Avoid accidental full-512 processing by enforcing sample-id-based filtering after precomputed artifact load.

## Technical Details
- Selection strategy is deterministic (`head(128)`), preserving original `sample_id` mapping.
- Filter logic is applied to both token-level and sequence-level precomputed artifacts.
- Notebook code-cell compilation was verified after the update.
