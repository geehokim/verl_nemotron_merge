"""IFEval-based instruction-following reward for data_source='if'.

Reward logic:
  1.0 if the response follows ALL instructions (strict mode)
  0.0 otherwise

Uses the IFEval evaluation library vendored at verl/utils/reward_score/ifeval/.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Optional, Union

_EVALUATION_LIB = None


def _ensure_runtime():
    """Ensure runtime dependencies and load IFEval evaluation library."""
    global _EVALUATION_LIB
    if _EVALUATION_LIB is not None:
        return _EVALUATION_LIB

    # Check runtime dependencies
    for pkg in ("nltk", "langdetect"):
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Missing required package '{pkg}' for IF reward. "
                f"Install with: pip install nltk langdetect"
            )

    # Ensure NLTK punkt tokenizer data
    import nltk

    for path, resource in [("tokenizers/punkt", "punkt"), ("tokenizers/punkt_tab", "punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

    from verl.utils.reward_score.ifeval import evaluation_lib

    _EVALUATION_LIB = evaluation_lib
    return evaluation_lib


def _parse_ground_truth(
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """Parse IFEval ground_truth from dict or JSON string."""
    payload = ground_truth
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Cannot parse ground_truth: {str(payload)[:200]}")

    if not isinstance(payload, dict):
        raise ValueError(f"ground_truth must be dict, got {type(payload)}")

    # Extract prompt
    prompt = payload.get("prompt", "")
    if not prompt and isinstance(extra_info, dict):
        prompt = extra_info.get("prompt", "")
    if not prompt:
        raise ValueError("ground_truth must include non-empty 'prompt'")

    instruction_id_list = payload.get("instruction_id_list", [])
    if hasattr(instruction_id_list, "tolist"):
        instruction_id_list = instruction_id_list.tolist()
    instruction_id_list = list(instruction_id_list)
    if not instruction_id_list:
        raise ValueError("ground_truth must include non-empty 'instruction_id_list'")

    kwargs = payload.get("kwargs", [])
    if hasattr(kwargs, "tolist"):
        kwargs = kwargs.tolist()
    kwargs = list(kwargs)

    # Pad kwargs to match instruction_id_list length
    if len(kwargs) < len(instruction_id_list):
        kwargs.extend({} for _ in range(len(instruction_id_list) - len(kwargs)))
    kwargs = kwargs[: len(instruction_id_list)]
    kwargs = [k if isinstance(k, dict) else {} for k in kwargs]

    key = payload.get("key", 0)
    if isinstance(extra_info, dict):
        key = extra_info.get("key", key)
    try:
        key = int(key)
    except (TypeError, ValueError):
        key = 0

    return {
        "key": key,
        "prompt": str(prompt),
        "instruction_id_list": instruction_id_list,
        "kwargs": kwargs,
    }


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """Compute IFEval instruction-following reward.

    Returns:
        dict with 'score': 1.0 if all instructions followed (strict), 0.0 otherwise.
    """
    evaluation_lib = _ensure_runtime()
    gt = _parse_ground_truth(ground_truth, extra_info)

    # Filter out None values from kwargs (data may include all 23 fields with unused=None)
    cleaned_kwargs = [
        {k: v for k, v in kw.items() if v is not None}
        for kw in gt["kwargs"]
    ]

    inp = evaluation_lib.InputExample(
        key=gt["key"],
        instruction_id_list=gt["instruction_id_list"],
        prompt=gt["prompt"],
        kwargs=cleaned_kwargs,
    )
    prompt_to_response = {inp.prompt: solution_str}

    strict_out = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose_out = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

    strict_follow_list = [bool(x) for x in strict_out.follow_instruction_list]
    loose_follow_list = [bool(x) for x in loose_out.follow_instruction_list]
    strict_follow_all = bool(strict_out.follow_all_instructions)
    loose_follow_all = bool(loose_out.follow_all_instructions)

    return {
        "score": 1.0 if strict_follow_all else 0.0,
        "acc": strict_follow_all,
        "answer_reward": 1.0 if strict_follow_all else 0.0,
        "strict_follow_all": strict_follow_all,
        "loose_follow_all": loose_follow_all,
        "strict_follow_instruction_list": strict_follow_list,
        "loose_follow_instruction_list": loose_follow_list,
        "instruction_count": len(gt["instruction_id_list"]),
    }
