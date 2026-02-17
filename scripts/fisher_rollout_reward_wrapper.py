#!/usr/bin/env python3
"""Reward wrapper for verbose Fisher-merging rollout pipelines.

This wrapper keeps compatibility with VERL's `custom_reward_function` interface
while adding rollout-trace metadata that the Fisher pipeline needs:

1. Preserve original reward behavior (default or task-specific custom reward).
2. Always return a dictionary payload (including score/acc when available).
3. Attach `sample_index` from `extra_info["index"]` so dumped JSONL can be
   joined back to dataset rows without prompt-text ambiguity.

Important compatibility rule:
- Do NOT write `data_source` into reward-extra metadata. VERL already carries
  `data_source` in `non_tensor_batch`, and duplicate keys can break
  `DataProto.union()` during validation when dtypes differ.

Expected callable signature by VERL reward manager:
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict, Optional

from verl.utils.reward_score import default_compute_score


@lru_cache(maxsize=16)
def _load_external_reward_function(
    module_path: str,
    function_name: str,
) -> Callable[..., Any]:
    """Load external reward function once and cache it.

    Args:
        module_path: Python file path that exports the reward function.
        function_name: Function name inside `module_path`.

    Returns:
        Loaded callable object.
    """

    from verl.utils.import_utils import load_extern_object

    return load_extern_object(module_path=module_path, object_name=function_name)


def _normalize_reward_output(raw_reward: Any) -> Dict[str, Any]:
    """Normalize arbitrary reward output to dictionary form.

    Args:
        raw_reward: Reward output returned by base reward function.

    Returns:
        Dictionary payload containing at least `score`.
    """

    if isinstance(raw_reward, dict):
        # Keep the original rich payload when base reward already returns dict.
        normalized = dict(raw_reward)
        if "score" not in normalized:
            # Defensive fallback for non-standard reward dict outputs.
            normalized["score"] = 0.0
        return normalized

    # Scalar rewards are converted into a standard dict for consistent logging.
    return {
        "score": float(raw_reward),
        "acc": float(raw_reward),
    }


def _get_sample_index(extra_info: Optional[dict[str, Any]]) -> Optional[int]:
    """Extract integer sample index from reward `extra_info`.

    Args:
        extra_info: Optional metadata dictionary passed by VERL reward manager.

    Returns:
        Integer sample index when available; otherwise None.
    """

    if not isinstance(extra_info, dict):
        return None

    candidate = extra_info.get("index", None)
    if candidate is None:
        return None

    try:
        return int(candidate)
    except Exception:
        return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    base_reward_function_path: Optional[str] = None,
    base_reward_function_name: str = "compute_score",
    fail_on_base_error: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute reward with optional task-specific base function and metadata injection.

    Args:
        data_source: Dataset source key used by reward routing.
        solution_str: Model output string.
        ground_truth: Ground-truth object from `reward_model.ground_truth`.
        extra_info: Optional metadata dictionary passed by VERL.
        base_reward_function_path: Optional file path for a task-specific base
            reward function (e.g., IF strict verifier module).
        base_reward_function_name: Callable name in `base_reward_function_path`.
        fail_on_base_error: If True, re-raise base reward errors. If False,
            fallback to `score=0.0` with error metadata.
        **kwargs: Additional reward kwargs forwarded by VERL.

    Returns:
        Dictionary reward payload that always includes:
            - `score` (float)
            - `sample_index` (int or None)
    """

    try:
        if base_reward_function_path:
            base_reward_fn = _load_external_reward_function(
                module_path=base_reward_function_path,
                function_name=base_reward_function_name,
            )
            raw_reward = base_reward_fn(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        else:
            # Default VERL reward router supports Nemotron math data source.
            raw_reward = default_compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **kwargs,
            )
        reward_dict = _normalize_reward_output(raw_reward)
    except Exception as reward_error:
        if fail_on_base_error:
            raise

        # Fallback keeps rollout alive while surfacing failure in dumped JSONL.
        reward_dict = {
            "score": 0.0,
            "acc": 0.0,
            "reward_error": repr(reward_error),
        }

    # Keep per-sample lineage for post-rollout joins.
    reward_dict["sample_index"] = _get_sample_index(extra_info=extra_info)

    # NOTE:
    # We intentionally avoid setting `reward_dict["data_source"]` here.
    # VERL already stores `data_source` in `DataProto.non_tensor_batch`, and
    # re-inserting the same key through reward-extra payload can produce a
    # dtype mismatch (`object` vs `str`) in `DataProto.union()` during val
    # rollout, causing:
    #   AssertionError: `data_source` in tensor_dict1 and tensor_dict2 ...
    #
    # If dataset-level diagnostics need this field, use a non-conflicting key.
    reward_dict["reward_data_source"] = str(data_source)
    return reward_dict
