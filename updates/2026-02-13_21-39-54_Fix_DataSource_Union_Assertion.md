# [2026-02-13 21:39] Fix DataSource Union Assertion

## Changes
- **File**: `scripts/fisher_rollout_reward_wrapper.py`
    - Updated module-level documentation to document a critical compatibility constraint with VERL `DataProto.union()`.
    - Removed duplicate `data_source` emission in reward-extra payload to prevent non-tensor key collisions.
    - Added a non-conflicting debug key `reward_data_source` for traceability.
    - Added explicit inline comments explaining the dtype mismatch failure mode (`object` vs string array) and why the key rename is required.

## Rationale
- VERL validation unions original batch non-tensor fields with generated output non-tensor fields.
- The custom wrapper added `data_source` again through reward metadata, which collided with the existing `data_source` field and caused:
  - `AssertionError: \\`data_source\\` in tensor_dict1 and tensor_dict2 are not the same object.`
- Renaming/removing the duplicate field resolves the conflict while preserving observability.

## Technical Details
- Root-cause location: `verl/protocol.py` -> `union_numpy_dict()` performs strict deep-equality checks for overlapping non-tensor keys.
- Async rollout path carries reward-extra metadata into output non-tensor fields, so duplicated keys can trigger union failures.
- Fix strategy: keep `sample_index` for Fisher join logic, avoid reserved non-tensor keys (`data_source`) in reward-extra outputs.
