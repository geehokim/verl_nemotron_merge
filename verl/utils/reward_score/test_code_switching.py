#!/usr/bin/env python3
"""Test code-switching detection across both reward paths.

Covers:
  1. FastText availability check
  2. detect_code_switching() in nemotron_cascade_rl_math.py
  3. _detect_code_switching() / _compute_coding_score() gate in verl_custom_reward.py

Run:
    cd /home2/geeho/tmp/verl_nemotron_merge
    python -m pytest verl/utils/reward_score/test_code_switching.py -v
  or directly:
    python verl/utils/reward_score/test_code_switching.py
"""

import sys
import os
import unittest

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ---------------------------------------------------------------------------
# Helper: check FastText availability once upfront
# ---------------------------------------------------------------------------
try:
    from ftlangdetect import detect as _ft_detect
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    _ft_detect = None


# ---------------------------------------------------------------------------
# Suite A: FastText sanity check
# ---------------------------------------------------------------------------
class TestFastTextAvailability(unittest.TestCase):
    """A.1 – FastText must be importable; without it detection is always False."""

    def test_fasttext_is_importable(self):
        """FastText library must be installed in this environment."""
        self.assertTrue(
            FASTTEXT_AVAILABLE,
            "ftlangdetect is NOT installed → code-switching detection is silently disabled. "
            "Install with: pip install fasttext-langdetect",
        )

    def test_fasttext_detects_english(self):
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")
        result = _ft_detect(text="This is a simple English sentence for testing.", low_memory=False)
        self.assertEqual(result.get("lang"), "en", f"Expected 'en', got {result}")

    def test_fasttext_detects_chinese(self):
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")
        result = _ft_detect(text="这是一个中文句子，用于测试目的。", low_memory=False)
        self.assertEqual(result.get("lang"), "zh", f"Expected 'zh', got {result}")

    def test_fasttext_detects_korean(self):
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")
        result = _ft_detect(text="이것은 한국어 문장입니다 테스트 목적으로 사용합니다.", low_memory=False)
        self.assertEqual(result.get("lang"), "ko", f"Expected 'ko', got {result}")


# ---------------------------------------------------------------------------
# Suite B: detect_code_switching() in nemotron_cascade_rl_math.py
# ---------------------------------------------------------------------------
class TestNemotronMathCodeSwitching(unittest.TestCase):
    """B – detect_code_switching() in nemotron_cascade_rl_math.py must work end-to-end."""

    @classmethod
    def setUpClass(cls):
        from verl.utils.reward_score.nemotron_cascade_rl_math import (
            detect_code_switching,
            FASTTEXT_AVAILABLE as FT_AVAIL,
        )
        cls.detect_code_switching = staticmethod(detect_code_switching)
        cls.ft_avail = FT_AVAIL

    def _skip_if_no_ft(self):
        if not self.ft_avail:
            self.skipTest("FastText not available – detection is always False")

    def test_pure_english_no_switch(self):
        """English reasoning on English prompt → no code-switching."""
        self._skip_if_no_ft()
        solution = (
            "<think>Let me solve this step by step. "
            "First I need to find the common denominator and then simplify the fraction.</think>"
            "\\boxed{1/2}"
        )
        result = self.detect_code_switching(solution, "en")
        self.assertFalse(result, "Pure English should NOT trigger code-switching")

    def test_chinese_in_english_prompt_detected(self):
        """Chinese characters in an English-prompted reasoning → code-switching."""
        self._skip_if_no_ft()
        solution = (
            "<think>让我来解决这个问题。首先需要找到公分母，然后化简分数。</think>"
            "\\boxed{1/2}"
        )
        result = self.detect_code_switching(solution, "en")
        self.assertTrue(result, "Chinese reasoning on English prompt MUST be detected as code-switching")

    def test_korean_in_english_prompt_detected(self):
        """Korean characters in an English-prompted reasoning → code-switching."""
        self._skip_if_no_ft()
        solution = (
            "<think>이 문제를 단계별로 풀어보겠습니다. 먼저 공통 분모를 찾아야 합니다.</think>"
            "\\boxed{1/2}"
        )
        result = self.detect_code_switching(solution, "en")
        self.assertTrue(result, "Korean reasoning on English prompt MUST be detected as code-switching")

    def test_too_short_reasoning_skipped(self):
        """Reasoning shorter than 20 chars → no penalty (edge case)."""
        self._skip_if_no_ft()
        result = self.detect_code_switching("<think>짧다</think>\\boxed{1}", "en")
        self.assertFalse(result, "Too-short reasoning must not be penalized")

    def test_latex_only_reasoning_not_penalized(self):
        """LaTeX-heavy reasoning (after cleanup) that's too short → no penalty."""
        self._skip_if_no_ft()
        solution = "<think>\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}</think>\\boxed{5/6}"
        # After LaTeX stripping the text may be too short; should not crash
        result = self.detect_code_switching(solution, "en")
        # Don't assert True/False — just ensure no exception
        self.assertIsInstance(result, bool)

    def test_no_think_tag_uses_full_text(self):
        """When there's no </think> tag the full output is scanned."""
        self._skip_if_no_ft()
        solution = "这是没有think标签的中文推理内容，用于测试代码切换检测是否正常工作。"
        result = self.detect_code_switching(solution, "en")
        self.assertTrue(result, "Chinese text without think tag should also be detected")

    # ------------------------------------------------------------------
    # Regression: stale TODO comment must not hide an early `return False`
    # ------------------------------------------------------------------
    def test_stale_disabled_comment_does_not_skip_detection(self):
        """BUG REGRESSION: lines 213-214 have a 'Temporarily disabled' comment but
        the function must still run detection when FastText IS available.
        If someone accidentally adds `return False` there, this test will catch it."""
        self._skip_if_no_ft()
        chinese_solution = (
            "<think>首先，我们需要找到两个整数，使得它们的和等于目标值。"
            "我们可以使用哈希表来存储每个元素的索引。</think>\\boxed{42}"
        )
        result = self.detect_code_switching(chinese_solution, "en")
        self.assertTrue(
            result,
            "detect_code_switching() returned False for obvious Chinese-in-English input. "
            "Check if an early `return False` was accidentally added near the TODO comment "
            "at nemotron_cascade_rl_math.py:213.",
        )


# ---------------------------------------------------------------------------
# Suite C: _compute_coding_score() gate in verl_custom_reward.py
# This is where the REAL silent disabling happens.
# ---------------------------------------------------------------------------
class TestCustomRewardCodeSwitchingGate(unittest.TestCase):
    """C – verl_custom_reward._compute_coding_score() silently skips detection
    for data sources NOT in CODING_TRAIN_SOURCES.

    CODING_TRAIN_SOURCES = {"nemotron_cascade_rl_coding"}
    LiveBench val data uses  "livebench/coding" → detection is SILENTLY SKIPPED.
    """

    @classmethod
    def setUpClass(cls):
        # We need a fake ground truth to call _compute_coding_score without
        # actually running code execution. We mock _evaluate_coding_correctness.
        import importlib
        spec = importlib.util.spec_from_file_location(
            "verl_custom_reward",
            os.path.join(REPO_ROOT, "evaluation/eval/verl_custom_reward.py"),
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def _make_gt(self):
        """Minimal ground truth dict that passes _normalize_ground_truth."""
        import json
        # _normalize_ground_truth accepts {"inputs": [...], "outputs": [...]} format
        return json.dumps({
            "inputs": ["1"],
            "outputs": ["1"],
            "fn_name": "",
        })

    def _call_score(self, data_source, solution_str, extra_info=None):
        """Call _compute_coding_score with a mocked code evaluator."""
        original = getattr(self.module, "_evaluate_coding_correctness", None)
        # Monkeypatch to avoid actually running the sandbox
        self.module._evaluate_coding_correctness = lambda problem, timeout: True
        try:
            result = self.module._compute_coding_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=self._make_gt(),
                extra_info=extra_info or {"prompt_language": "en"},
            )
        finally:
            if original is not None:
                self.module._evaluate_coding_correctness = original
        return result

    # ---- sub-test A: detection runs for CODING_TRAIN_SOURCES ----
    def test_train_source_detects_code_switching(self):
        """For 'nemotron_cascade_rl_coding', Chinese reasoning triggers penalty."""
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")
        chinese_solution = (
            "<think>我来分析这道编程题。首先需要定义一个哈希表来存储键值对，"
            "然后遍历数组找到目标和。</think>\n```python\nreturn [0, 1]\n```"
        )
        result = self._call_score("nemotron_cascade_rl_coding", chinese_solution)
        self.assertTrue(
            result["code_switching"],
            "Training source 'nemotron_cascade_rl_coding' must detect code-switching",
        )
        self.assertEqual(result["code_switching_penalty"], -1.0)
        self.assertEqual(result["score"], 0.0, "Correct but code-switching → score=0.0")

    def test_train_source_no_penalty_for_english(self):
        """For 'nemotron_cascade_rl_coding', English reasoning gets no penalty."""
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")
        en_solution = (
            "<think>Let me analyze this problem. I need to use a hash map to store "
            "key-value pairs and then iterate through the array to find the target sum.</think>"
            "\n```python\nreturn [0, 1]\n```"
        )
        result = self._call_score("nemotron_cascade_rl_coding", en_solution)
        self.assertFalse(result["code_switching"])
        self.assertEqual(result["code_switching_penalty"], 0.0)

    # ---- sub-test B: THE BUG — detection silently skipped for val/other sources ----
    def test_livebench_val_source_silently_skips_detection(self):
        """BUG: 'livebench/coding' is NOT in CODING_TRAIN_SOURCES.
        Even with obvious Chinese code-switching, detection is bypassed.
        code_switching is always False for LiveBench validation data."""
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available – can't distinguish the bug from missing lib")

        chinese_solution = (
            "<think>我来分析这道编程题。首先需要定义一个哈希表来存储键值对，"
            "然后遍历数组找到目标和。</think>\n```python\nreturn [0, 1]\n```"
        )
        result = self._call_score("livebench/coding", chinese_solution)

        # Document the current (buggy) behavior:
        self.assertFalse(
            result["code_switching"],
            "CONFIRMED BUG: 'livebench/coding' silently returns code_switching=False "
            "even with clear Chinese-in-English code-switching. "
            "Fix: extend apply_code_switch_override to also cover CODING_VAL_SOURCES.",
        )
        self.assertEqual(result["code_switching_penalty"], 0.0)

    def test_other_data_source_silently_skips_detection(self):
        """Any unknown/math data source also gets detection silently skipped."""
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")

        chinese_solution = (
            "<think>使用哈希表来解决两数之和问题，时间复杂度为O(n)。</think>"
            "\n```python\nreturn [0, 1]\n```"
        )
        for source in ["nemotron_cascade_rl_math", "aime", "livebench", "livecodebench"]:
            result = self._call_score(source, chinese_solution)
            self.assertFalse(
                result["code_switching"],
                f"Source '{source}' silently skips detection (not in CODING_TRAIN_SOURCES)",
            )

    # ---- sub-test C: verify the internal detector itself works ----
    def test_internal_detect_function_works_correctly(self):
        """_detect_code_switching() itself works when called directly.
        The bug is only in the gate, not the function."""
        if not FASTTEXT_AVAILABLE:
            self.skipTest("FastText not available")

        chinese_solution = (
            "<think>我来分析这道编程题。首先需要定义一个哈希表来存储键值对，"
            "然后遍历数组找到目标和。</think>\n```python\nreturn [0, 1]\n```"
        )
        result = self.module._detect_code_switching(chinese_solution, "en")
        self.assertTrue(
            result,
            "_detect_code_switching() itself correctly detects Chinese-in-English code-switching. "
            "The bug is in the gate: `apply_code_switch_override = data_source in CODING_TRAIN_SOURCES`.",
        )


# ---------------------------------------------------------------------------
# Suite D: CODING_TRAIN_SOURCES membership sanity
# ---------------------------------------------------------------------------
class TestCodingTrainSourcesMembership(unittest.TestCase):
    """D – Document which sources get detection and which don't."""

    @classmethod
    def setUpClass(cls):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "verl_custom_reward2",
            os.path.join(REPO_ROOT, "evaluation/eval/verl_custom_reward.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.CODING_TRAIN_SOURCES = mod.CODING_TRAIN_SOURCES

    def test_nemotron_cascade_rl_coding_in_train_sources(self):
        self.assertIn("nemotron_cascade_rl_coding", self.CODING_TRAIN_SOURCES)

    def test_livebench_NOT_in_train_sources(self):
        """LiveBench val data is NOT guarded by code-switching detection."""
        for src in ["livebench/coding", "livebench", "livebench/code_generation"]:
            self.assertNotIn(
                src, self.CODING_TRAIN_SOURCES,
                f"'{src}' is in CODING_TRAIN_SOURCES — detection would apply; "
                "update this test if the fix was applied.",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
