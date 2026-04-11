#!/usr/bin/env python3
"""Custom reward router for Nemotron RL tasks.

Coding reward paths are handled here and use the same correctness judgment core
as `evaluation/eval/get_scores_code.py` (check_coding_correctness,
which is used by evaluate_livecodebench).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import pickle
import re
import sys
import traceback
import zlib
from pathlib import Path
from typing import Any

try:
    from ftlangdetect import detect as ft_detect

    FASTTEXT_AVAILABLE = True
except Exception:
    ft_detect = None
    FASTTEXT_AVAILABLE = False

CODING_TRAIN_SOURCES = {
    "nemotron_cascade_rl_coding",
}

CODING_VAL_SOURCES = {
    "livebench/coding",
    "livebench/code_generation_lite",
    "livebench/code_generation",
    "livebench",
}

# Keep backward compatibility with older validation rows.
CODING_COMPAT_SOURCES = {
    "livecodebench/code_generation_lite",
    "livecodebench/code_generation",
    "livecodebench",
}

_ALL_CODING_SOURCES = CODING_TRAIN_SOURCES | CODING_VAL_SOURCES | CODING_COMPAT_SOURCES

IFEVAL_SOURCES = {
    "nemotron_cascade_rl_if",
    "ifeval",
    "ifeval/train",
    "ifeval/val",
    "ifeval/validation",
}

_EVAL_MODULE = None
_IFEVAL_MODULE = None
_IFEVAL_EVALUATION_LIB = None

# Track which (kind, error_message) pairs we have already logged so a noisy
# failure mode does not flood stderr across thousands of reward calls.
_LOGGED_REWARD_ERRORS: set[tuple[str, str]] = set()


def _log_reward_exception(kind: str, data_source: str, exc: BaseException) -> None:
    """Print first occurrence of a reward-fn exception to stderr.

    The reward router otherwise swallows exceptions and returns score=0.0,
    which can mask decoding/runtime bugs as legitimate zero scores. We log the
    first time each unique error message appears so the failure is visible in
    the trainer logs without spamming.
    """
    key = (kind, str(exc))
    if key in _LOGGED_REWARD_ERRORS:
        return
    _LOGGED_REWARD_ERRORS.add(key)
    print(
        f"[verl_custom_reward] {kind} score error (data_source={data_source}): {exc}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc(file=sys.stderr)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_compute_score_lazy(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    **kwargs,
):
    from verl.utils.reward_score import default_compute_score

    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def _load_code_eval_module():
    global _EVAL_MODULE
    if _EVAL_MODULE is not None:
        return _EVAL_MODULE

    module_path = _repo_root() / "evaluation" / "eval" / "get_scores_code.py"
    if not module_path.exists():
        raise FileNotFoundError(f"get_scores_code.py not found: {module_path}")

    eval_dir = module_path.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    spec = importlib.util.spec_from_file_location("_verl_get_scores_code", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _EVAL_MODULE = module
    return module


def _load_ifeval_eval_module():
    global _IFEVAL_MODULE
    if _IFEVAL_MODULE is not None:
        return _IFEVAL_MODULE

    module_path = _repo_root() / "evaluation" / "eval" / "get_scores_ifeval.py"
    if not module_path.exists():
        raise FileNotFoundError(f"get_scores_ifeval.py not found: {module_path}")

    eval_dir = module_path.parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    spec = importlib.util.spec_from_file_location("_verl_get_scores_ifeval", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _IFEVAL_MODULE = module
    return module


def _ensure_ifeval_runtime():
    global _IFEVAL_EVALUATION_LIB
    if _IFEVAL_EVALUATION_LIB is not None:
        return _IFEVAL_EVALUATION_LIB

    module = _load_ifeval_eval_module()
    nltk_module = module._ensure_runtime_dependencies()
    module._ensure_nltk_resources(nltk_module)
    _IFEVAL_EVALUATION_LIB = module._ensure_vendor_imports()
    return _IFEVAL_EVALUATION_LIB


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _decode_payload_text(payload_text: str) -> Any:
    """Decode base64+zlib+pickle(json.dumps(...)) payload."""
    raw = base64.b64decode(payload_text.encode("utf-8"))
    decomp = zlib.decompress(raw)
    obj = pickle.loads(decomp)
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        parsed = _safe_json_loads(obj)
        return obj if parsed is None else parsed
    return obj


def _normalize_ground_truth(ground_truth: Any) -> dict[str, Any]:
    payload = ground_truth

    if isinstance(payload, str):
        parsed = _safe_json_loads(payload)
        if parsed is not None:
            payload = parsed
        else:
            try:
                payload = _decode_payload_text(payload)
            except Exception as exc:
                raise ValueError(f"Failed to decode coding ground_truth payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported coding ground_truth type: {type(payload)}")

    if "inputs" in payload and "outputs" in payload:
        inputs = payload.get("inputs", [])
        outputs = payload.get("outputs", [])
        fn_name = _as_str(payload.get("fn_name", "")).strip()
    elif "input_output" in payload:
        io_pairs = payload.get("input_output", [])
        if isinstance(io_pairs, str):
            parsed = _safe_json_loads(io_pairs)
            io_pairs = parsed if parsed is not None else []
        if not isinstance(io_pairs, list):
            io_pairs = []
        inputs = [_as_str(x.get("input", "")) for x in io_pairs if isinstance(x, dict)]
        outputs = [_as_str(x.get("output", "")) for x in io_pairs if isinstance(x, dict)]
        fn_name = _as_str(payload.get("fn_name", "")).strip()
    else:
        raise ValueError("coding ground_truth must include inputs/outputs or input_output")

    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("coding ground_truth inputs/outputs must be lists")

    input_output = [
        {"input": _as_str(inp), "output": _as_str(out)}
        for inp, out in zip(inputs, outputs, strict=False)
    ]
    if not input_output:
        raise ValueError("Empty coding testcases in ground_truth")

    return {
        "input_output": input_output,
        "fn_name": fn_name,
        "num_tests": len(input_output),
    }


def _get_prompt_language(extra_info: dict[str, Any] | None) -> str:
    if isinstance(extra_info, dict):
        for key in ("prompt_language", "language", "lang"):
            value = extra_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "en"


def _extract_reasoning_text(solution_str: str) -> str:
    text = solution_str
    for marker in ("</think>", "<\\/think>", "</Think>"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Exclude code blocks from language detection.
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"<code>[\s\S]*?</code>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_code_switching(solution_str: str, prompt_language: str) -> bool:
    if not FASTTEXT_AVAILABLE:
        return False

    reasoning = _extract_reasoning_text(solution_str)
    if len(reasoning) < 20:
        return False

    try:
        result = ft_detect(text=reasoning, low_memory=False)
        detected = result.get("lang", prompt_language)
        return bool(detected and detected != prompt_language)
    except Exception:
        return False


def _resolve_timeout(extra_info: dict[str, Any] | None, num_tests: int) -> int:
    default_timeout = 6
    if isinstance(extra_info, dict):
        for key in ("test_time_limit", "timeout", "time_limit"):
            value = extra_info.get(key)
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            if isinstance(value, int):
                default_timeout = value
                break

    # Keep runtime bounded for reward loop stability.
    per_test = max(2, min(20, int(default_timeout)))
    if num_tests > 20:
        per_test = min(per_test, 8)
    return per_test


def _evaluate_coding_correctness(problem_to_check: dict[str, Any], timeout: int) -> bool:
    module = _load_code_eval_module()

    # `check_coding_correctness` is the per-example core used by
    # `evaluate_livecodebench` in get_scores_code.py.
    checker = getattr(module, "check_coding_correctness", None)
    if checker is None:
        raise AttributeError("check_coding_correctness not found in get_scores_code.py")

    return bool(checker(problem_to_check, timeout=timeout))


def _normalize_ifeval_ground_truth(
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = ground_truth
    if isinstance(payload, str):
        parsed = _safe_json_loads(payload)
        if parsed is not None:
            payload = parsed

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported IFEval ground_truth type: {type(payload)}")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        if isinstance(extra_info, dict):
            extra_prompt = extra_info.get("prompt")
            if isinstance(extra_prompt, str) and extra_prompt.strip():
                prompt = extra_prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IFEval ground_truth must include non-empty prompt")

    instruction_id_list = payload.get("instruction_id_list", [])
    if hasattr(instruction_id_list, "tolist"):
        instruction_id_list = instruction_id_list.tolist()
    if not isinstance(instruction_id_list, list):
        instruction_id_list = list(instruction_id_list) if instruction_id_list else []
    if not instruction_id_list:
        raise ValueError("IFEval ground_truth must include non-empty instruction_id_list")
    instruction_id_list = [_as_str(x).strip() for x in instruction_id_list if _as_str(x).strip()]
    if not instruction_id_list:
        raise ValueError("IFEval instruction_id_list became empty after normalization")

    kwargs = payload.get("kwargs", [])
    if isinstance(kwargs, str):
        parsed_kwargs = _safe_json_loads(kwargs)
        kwargs = parsed_kwargs if isinstance(parsed_kwargs, list) else []
    if hasattr(kwargs, "tolist"):
        kwargs = kwargs.tolist()
    if not isinstance(kwargs, list):
        kwargs = list(kwargs) if kwargs else []

    normalized_kwargs: list[dict[str, Any]] = []
    for value in kwargs[: len(instruction_id_list)]:
        if isinstance(value, dict):
            normalized_kwargs.append({k: v for k, v in value.items() if v is not None})
        else:
            normalized_kwargs.append({})
    if len(normalized_kwargs) < len(instruction_id_list):
        normalized_kwargs.extend({} for _ in range(len(instruction_id_list) - len(normalized_kwargs)))

    key_value = payload.get("key")
    if key_value is None and isinstance(extra_info, dict):
        key_value = extra_info.get("key", extra_info.get("id", 0))
    try:
        key_int = int(key_value)
    except Exception:
        key_int = 0

    return {
        "key": key_int,
        "prompt": prompt,
        "instruction_id_list": instruction_id_list,
        "kwargs": normalized_kwargs,
    }


def _safe_mean_bools(values: list[bool]) -> float:
    return float(sum(1 for x in values if x) / len(values)) if values else 0.0


def _compute_coding_score(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    gt = _normalize_ground_truth(ground_truth)

    timeout = _resolve_timeout(extra_info, gt["num_tests"])
    problem = {
        "input_output": gt["input_output"],
        "starter_code": gt["fn_name"] if gt["fn_name"] else "",
        "generation": solution_str,
    }

    is_correct = _evaluate_coding_correctness(problem, timeout=timeout)
    answer_reward = 1.0 if is_correct else 0.0

    apply_code_switch_override = data_source in CODING_TRAIN_SOURCES
    prompt_language = _get_prompt_language(extra_info)
    has_code_switching = _detect_code_switching(solution_str, prompt_language) if apply_code_switch_override else False

    if apply_code_switch_override and has_code_switching:
        final_score = 0.0
        code_switching_penalty = -1.0
    else:
        final_score = answer_reward
        code_switching_penalty = 0.0

    return {
        "score": float(final_score),
        "acc": bool(is_correct),
        "answer_reward": float(answer_reward),
        "code_switching": bool(has_code_switching),
        "code_switching_penalty": float(code_switching_penalty),
        "timeout": int(timeout),
        "num_tests": int(gt["num_tests"]),
    }


def _compute_ifeval_score(
    *,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    evaluation_lib = _ensure_ifeval_runtime()
    gt = _normalize_ifeval_ground_truth(ground_truth, extra_info)

    inp = evaluation_lib.InputExample(
        key=int(gt["key"]),
        instruction_id_list=list(gt["instruction_id_list"]),
        prompt=str(gt["prompt"]),
        kwargs=list(gt["kwargs"]),
    )
    prompt_to_response = {inp.prompt: solution_str}

    strict_out = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    loose_out = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

    strict_follow_list = [bool(x) for x in strict_out.follow_instruction_list]
    loose_follow_list = [bool(x) for x in loose_out.follow_instruction_list]
    strict_follow_all = bool(strict_out.follow_all_instructions)
    loose_follow_all = bool(loose_out.follow_all_instructions)

    strict_instruction_acc = _safe_mean_bools(strict_follow_list)
    loose_instruction_acc = _safe_mean_bools(loose_follow_list)

    return {
        "score": float(1.0 if strict_follow_all else 0.0),
        "acc": bool(strict_follow_all),
        "answer_reward": float(1.0 if strict_follow_all else 0.0),
        "strict_follow_all": bool(strict_follow_all),
        "loose_follow_all": bool(loose_follow_all),
        "strict_instruction_accuracy": float(strict_instruction_acc),
        "loose_instruction_accuracy": float(loose_instruction_acc),
        "strict_follow_instruction_list": strict_follow_list,
        "loose_follow_instruction_list": loose_follow_list,
        "instruction_count": int(len(strict_follow_list)),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
):
    if data_source in IFEVAL_SOURCES:
        try:
            return _compute_ifeval_score(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except Exception as exc:
            _log_reward_exception("ifeval", data_source, exc)
            return {
                "score": 0.0,
                "acc": False,
                "answer_reward": 0.0,
                "strict_follow_all": False,
                "loose_follow_all": False,
                "strict_instruction_accuracy": 0.0,
                "loose_instruction_accuracy": 0.0,
                "instruction_count": 0,
                "error": str(exc),
            }

    if data_source in _ALL_CODING_SOURCES:
        try:
            return _compute_coding_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except Exception as exc:
            _log_reward_exception("coding", data_source, exc)
            return {
                "score": 0.0,
                "acc": False,
                "answer_reward": 0.0,
                "code_switching": False,
                "code_switching_penalty": 0.0,
                "error": str(exc),
            }

    return _default_compute_score_lazy(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
