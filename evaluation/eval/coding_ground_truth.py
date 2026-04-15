"""Decode + normalize helpers for coding-task ground truth payloads.

Extracted into a standalone, regularly-importable module so that the verifier
child process (spawned via the ``forkserver`` start method by
``get_scores_code.py``) can re-import it cleanly. Keeping these helpers out of
``verl_custom_reward.py`` is required because that file is loaded dynamically
by verl with a hash-based module name and is therefore not importable by a
fresh forkserver worker.

The functions here are intentionally side-effect free: the heavy work
(``base64 -> zlib -> pickle.loads`` and the resulting dict/list expansion) is
the *one* call the parent reward router avoids by deferring decode into the
child. See ``check_coding_correctness_from_raw`` for the integration point.
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def as_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def decode_payload_text(payload_text: str) -> Any:
    """Decode base64+zlib+pickle(json.dumps(...)) payload."""
    raw = base64.b64decode(payload_text.encode("utf-8"))
    decomp = zlib.decompress(raw)
    obj = pickle.loads(decomp)
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8")
    if isinstance(obj, str):
        parsed = safe_json_loads(obj)
        return obj if parsed is None else parsed
    return obj


def normalize_ground_truth(ground_truth: Any) -> dict[str, Any]:
    """Decode + normalize a coding ground-truth payload into the verifier shape.

    Returns a dict with keys: ``input_output`` (list of {"input","output"}),
    ``fn_name`` (str), ``num_tests`` (int).
    """
    payload = ground_truth

    if isinstance(payload, str):
        parsed = safe_json_loads(payload)
        if parsed is not None:
            payload = parsed
        else:
            try:
                payload = decode_payload_text(payload)
            except Exception as exc:
                raise ValueError(f"Failed to decode coding ground_truth payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported coding ground_truth type: {type(payload)}")

    if "inputs" in payload and "outputs" in payload:
        inputs = payload.get("inputs", [])
        outputs = payload.get("outputs", [])
        fn_name = as_str(payload.get("fn_name", "")).strip()
    elif "input_output" in payload:
        io_pairs = payload.get("input_output", [])
        if isinstance(io_pairs, str):
            parsed = safe_json_loads(io_pairs)
            io_pairs = parsed if parsed is not None else []
        if not isinstance(io_pairs, list):
            io_pairs = []
        inputs = [as_str(x.get("input", "")) for x in io_pairs if isinstance(x, dict)]
        outputs = [as_str(x.get("output", "")) for x in io_pairs if isinstance(x, dict)]
        fn_name = as_str(payload.get("fn_name", "")).strip()
    else:
        raise ValueError("coding ground_truth must include inputs/outputs or input_output")

    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("coding ground_truth inputs/outputs must be lists")

    input_output = [
        {"input": as_str(inp), "output": as_str(out)}
        for inp, out in zip(inputs, outputs, strict=False)
    ]
    if not input_output:
        raise ValueError("Empty coding testcases in ground_truth")

    return {
        "input_output": input_output,
        "fn_name": fn_name,
        "num_tests": len(input_output),
    }
