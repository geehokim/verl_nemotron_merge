# [2026-02-20 13:14] fix(notebook): show all IF unique-vs-topk layer-wise rows instead of head(12)

## Changes
- **File**: `merging_analysis/19_ram_ramplus_if_math_fp32.ipynb`
    - Replaced truncated output `display(if_unique_topk_layerwise_df.head(12))` with full-table display.
    - Added `print` for total row count to make output size explicit.
    - Added `pd.option_context("display.max_rows", None, "display.max_columns", None)` so all layer-wise rows/columns are visible.

- **File**: `updates/2026-02-20_13-14-36_Show_All_Layerwise_Rows_For_IFUnique_TopK.md`
    - Recorded this update log entry in the required workflow format.

## Rationale
- The previous `head(12)` display only showed the first 12 rows (commonly up to around layer_11), which hid remaining layers/config rows.
- Full display is required for complete layer-wise inspection.

## Technical Details
- Updated final output block in the IF unique vs top-k analysis cell to:
  - print total row count,
  - render full `if_unique_topk_layerwise_df` without truncation,
  - keep `if_unique_topk_global_df` display unchanged.
