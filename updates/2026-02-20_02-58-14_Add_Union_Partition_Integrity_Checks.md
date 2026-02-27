# [2026-02-20 02:58] fix(analysis): add explicit union-partition and sum-to-one checks

## Changes
- **File**: `merging_analysis/19_ram_ramplus_if_math_fp32.ipynb`
    - Extended the integrity-check cell with explicit partition validation:
      - `shared + unique_if + unique_math == either_changed` (layer-wise and global).
    - Added explicit ratio-sum validation for active union layers:
      - `shared/union + unique_if/union + unique_math/union == 1` within tolerance.
    - Added diagnostic printout that reports:
      - number of active-union layers,
      - number of zero-union layers,
      - maximum layer-wise union-sum error,
      - global union-ratio sum.

- **File**: `updates/2026-02-20_02-58-14_Add_Union_Partition_Integrity_Checks.md`
    - Recorded this update log entry in the required workflow format.

## Rationale
- To directly address user concern about whether union-normalized ratios are computed correctly.
- To make ratio consistency failures fail fast with clear error messages instead of silent plotting confusion.

## Technical Details
- Added numeric checks with tolerance `1e-8` for union-ratio sum validation.
- Kept zero-union layers as explicit edge cases where union-normalized ratios are defined as 0.
- Verified the updated integrity cell compiles successfully.
