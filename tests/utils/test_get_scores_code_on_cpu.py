import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation" / "eval"

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import get_scores_code  # noqa: E402


def _stdio_problem(code_block: str, *, with_think_close: bool = True) -> dict:
    """Build a verifier-ready problem dict.

    AceReason's ``run_test`` requires the generation to contain ``</think>``
    so an unfinished thinking trace cannot be silently treated as code.
    Real model rollouts always emit ``</think>`` before the answer block,
    so test fixtures mirror that contract by default. Set
    ``with_think_close=False`` to specifically exercise the
    incomplete-generation guard.
    """
    if with_think_close:
        generation = f"<think>reasoning</think>\n{code_block}"
    else:
        generation = code_block
    return {
        "input_output": [{"input": "3\n", "output": "3\n"}],
        "starter_code": "",
        "generation": generation,
    }


def test_check_coding_correctness_does_not_use_manager(monkeypatch):
    def _manager_should_not_be_called(*args, **kwargs):
        raise AssertionError("multiprocessing.Manager() should not be used in check_coding_correctness")

    monkeypatch.setattr(get_scores_code.multiprocessing, "Manager", _manager_should_not_be_called, raising=True)

    ok = get_scores_code.check_coding_correctness(
        _stdio_problem("```python\nprint(input())\n```"),
        timeout=1,
    )
    bad = get_scores_code.check_coding_correctness(
        _stdio_problem("```python\nprint(0)\n```"),
        timeout=1,
    )

    assert ok is True
    assert bad is False


def test_check_coding_correctness_timeout_and_missing_code_return_false():
    timeout_result = get_scores_code.check_coding_correctness(
        _stdio_problem("```python\nwhile True:\n    pass\n```"),
        timeout=1,
    )
    no_code_result = get_scores_code.check_coding_correctness(
        _stdio_problem("No code block here."),
        timeout=1,
    )

    assert timeout_result is False
    assert no_code_result is False


def test_check_coding_correctness_rejects_incomplete_thinking():
    """Generations missing ``</think>`` must score 0 immediately.

    This is the regression guard for the OOM scenario where a collapsed
    model emits an unfinished thinking trace until ``MAX_RESPONSE_LENGTH``;
    AceReason's verifier short-circuits on the missing token instead of
    trying to compile/run whatever fragment the regex happens to match.
    """
    incomplete = get_scores_code.check_coding_correctness(
        _stdio_problem(
            "```python\nprint(input())\n```",
            with_think_close=False,
        ),
        timeout=1,
    )
    assert incomplete is False
