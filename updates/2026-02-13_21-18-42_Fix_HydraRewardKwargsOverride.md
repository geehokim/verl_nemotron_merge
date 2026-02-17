# [2026-02-13 21:18] fix(fisher): use Hydra append syntax for custom reward kwargs

## Changes
- **File**: `scripts/run_fisher_merge_pipeline.py`
  - Updated rollout command construction to append custom reward kwargs using `+custom_reward_function.reward_kwargs...` syntax.
  - Added inline explanation comment describing why `+` is mandatory for missing struct keys.
  - Updated optional task-specific reward function path/name kwargs to also use `+` append syntax.

## Rationale
- VERL/Hydra config runs in struct mode, and `custom_reward_function.reward_kwargs` is not a predefined key in the base schema.
- Plain overrides caused `ConfigKeyError` and `ConfigCompositionException` during rollout stage.

## Technical Details
- Replaced:
  - `custom_reward_function.reward_kwargs.fail_on_base_error=False`
  - `custom_reward_function.reward_kwargs.base_reward_function_path=...`
  - `custom_reward_function.reward_kwargs.base_reward_function_name=...`
- With:
  - `+custom_reward_function.reward_kwargs.fail_on_base_error=False`
  - `+custom_reward_function.reward_kwargs.base_reward_function_path=...`
  - `+custom_reward_function.reward_kwargs.base_reward_function_name=...`
