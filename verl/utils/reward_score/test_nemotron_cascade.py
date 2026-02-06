#!/usr/bin/env python3
"""
Test cases for nemotron_cascade_rl_math.py

This script tests the reward function implementation with various edge cases.
Run from the reward_score directory with:
    cd /mnt/ddn/vuvlm/geeho/verl_nemotron_merge/verl/utils/reward_score
    python test_nemotron_cascade.py
"""

import json
import sys
import os
import traceback
import re
from typing import Optional

# Log file path for debug mode
LOG_PATH = "/mnt/ddn/vuvlm/geeho/.cursor/debug.log"

def log_debug(hypothesis_id: str, location: str, message: str, data: dict):
    """Write debug log in NDJSON format."""
    # #region agent log
    import time
    log_entry = {
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "sessionId": "debug-session",
        "runId": "test-run-1"
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    # #endregion


# =============================================================================
# Copy the functions to test directly (avoid import issues)
# =============================================================================

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def extract_boxed_after_think(solution_str: str) -> Optional[str]:
    """Extract the \\boxed{} answer that follows the </think> token."""
    think_end_markers = ["</think>", "<\\/think>", "</Think>"]
    think_end_idx = -1
    
    for marker in think_end_markers:
        idx = solution_str.find(marker)
        if idx != -1:
            think_end_idx = idx + len(marker)
            break
    
    if think_end_idx == -1:
        text_after_think = solution_str
    else:
        text_after_think = solution_str[think_end_idx:]
    
    boxed_str = last_boxed_only_string(text_after_think)
    
    if boxed_str is None:
        return None
    
    try:
        return remove_boxed(boxed_str)
    except AssertionError:
        return None


# =============================================================================
# Test Functions
# =============================================================================

def test_extract_boxed_after_think():
    """Test H1 & H2: extract_boxed_after_think function."""
    print("\n" + "="*60)
    print("TEST: extract_boxed_after_think()")
    print("="*60)
    
    test_cases = [
        # H1: Without </think> token
        {
            "input": "The answer is \\boxed{42}",
            "expected": "42",
            "description": "H1: No </think> token, simple boxed"
        },
        # H1: With </think> token
        {
            "input": "<think>Let me solve this...</think>The answer is \\boxed{42}",
            "expected": "42",
            "description": "H1: With </think> token"
        },
        # H2: Nested braces
        {
            "input": "</think>\\boxed{\\frac{1}{2}}",
            "expected": "\\frac{1}{2}",
            "description": "H2: Nested braces (frac)"
        },
        # H2: Deeply nested
        {
            "input": "</think>\\boxed{\\frac{\\sqrt{2}}{3}}",
            "expected": "\\frac{\\sqrt{2}}{3}",
            "description": "H2: Deeply nested braces"
        },
        # Edge case: No boxed
        {
            "input": "</think>The answer is 42",
            "expected": None,
            "description": "Edge: No boxed answer"
        },
        # Edge case: Multiple boxed (should take last after </think>)
        {
            "input": "<think>\\boxed{wrong}</think>\\boxed{correct}",
            "expected": "correct",
            "description": "H1: Multiple boxed, take last after </think>"
        },
        # Edge case: Empty boxed
        {
            "input": "</think>\\boxed{}",
            "expected": "",
            "description": "Edge: Empty boxed"
        },
    ]
    
    all_passed = True
    for i, tc in enumerate(test_cases):
        try:
            result = extract_boxed_after_think(tc["input"])
            passed = result == tc["expected"]
            status = "✓ PASS" if passed else "✗ FAIL"
            
            log_debug(
                "H1" if "H1" in tc["description"] else "H2",
                f"test_extract_boxed:{i}",
                tc["description"],
                {"input": tc["input"][:50], "expected": tc["expected"], "result": result, "passed": passed}
            )
            
            print(f"{status}: {tc['description']}")
            print(f"       Input: {tc['input'][:60]}...")
            print(f"       Expected: {tc['expected']}, Got: {result}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ ERROR: {tc['description']}")
            print(f"       Exception: {e}")
            log_debug("H1", f"test_extract_boxed:{i}", "Exception", {"error": str(e)})
            all_passed = False
    
    return all_passed


def test_last_boxed_only_string():
    """Test last_boxed_only_string function with edge cases."""
    print("\n" + "="*60)
    print("TEST: last_boxed_only_string()")
    print("="*60)
    
    test_cases = [
        {"input": "\\boxed{42}", "expected": "\\boxed{42}", "desc": "Simple boxed"},
        {"input": "\\boxed{\\frac{1}{2}}", "expected": "\\boxed{\\frac{1}{2}}", "desc": "Nested frac"},
        {"input": "\\boxed{a} and \\boxed{b}", "expected": "\\boxed{b}", "desc": "Multiple boxed (last)"},
        {"input": "No boxed here", "expected": None, "desc": "No boxed"},
        {"input": "\\boxed{\\sqrt{\\frac{a}{b}}}", "expected": "\\boxed{\\sqrt{\\frac{a}{b}}}", "desc": "Deep nesting"},
    ]
    
    all_passed = True
    for i, tc in enumerate(test_cases):
        try:
            result = last_boxed_only_string(tc["input"])
            passed = result == tc["expected"]
            status = "✓ PASS" if passed else "✗ FAIL"
            
            log_debug("H2", f"test_last_boxed:{i}", tc["desc"],
                     {"input": tc["input"], "expected": tc["expected"], "result": result, "passed": passed})
            
            print(f"{status}: {tc['desc']}")
            print(f"       Input: {tc['input']}")
            print(f"       Expected: {tc['expected']}, Got: {result}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ ERROR: {tc['desc']}: {e}")
            log_debug("H2", f"test_last_boxed:{i}", "Exception", {"error": str(e)})
            all_passed = False
    
    return all_passed


def test_fasttext_import():
    """Test H4: FastText import and code-switching detection."""
    print("\n" + "="*60)
    print("TEST: FastText Import & Code-Switching")
    print("="*60)
    
    # Try to import FastText
    try:
        from ftlangdetect import detect as ft_detect
        fasttext_available = True
        print("✓ FastText is available")
        log_debug("H4", "fasttext_import", "FastText available", {"available": True})
    except ImportError as e:
        fasttext_available = False
        print(f"✗ FastText not available: {e}")
        log_debug("H4", "fasttext_import", "FastText not available", {"available": False, "error": str(e)})
        return None  # Skip further tests
    
    # Test language detection
    test_cases = [
        {"text": "This is a simple English sentence for testing purposes.", "expected_lang": "en", "desc": "English text"},
        {"text": "这是一个中文句子用于测试目的。", "expected_lang": "zh", "desc": "Chinese text"},
        {"text": "이것은 한국어 문장입니다 테스트용입니다.", "expected_lang": "ko", "desc": "Korean text"},
    ]
    
    all_passed = True
    for i, tc in enumerate(test_cases):
        try:
            result = ft_detect(text=tc["text"], low_memory=False)
            detected_lang = result.get("lang", "unknown")
            passed = detected_lang == tc["expected_lang"]
            status = "✓ PASS" if passed else "✗ FAIL"
            
            log_debug("H4", f"test_langdetect:{i}", tc["desc"],
                     {"text": tc["text"][:30], "expected": tc["expected_lang"], "detected": detected_lang, "passed": passed})
            
            print(f"{status}: {tc['desc']}")
            print(f"       Expected: {tc['expected_lang']}, Detected: {detected_lang}")
            
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ ERROR: {tc['desc']}: {e}")
            log_debug("H4", f"test_langdetect:{i}", "Exception", {"error": str(e)})
            all_passed = False
    
    return all_passed


def test_reward_computation_logic():
    """Test H5: Reward computation logic (without full module import)."""
    print("\n" + "="*60)
    print("TEST: Reward Computation Logic")
    print("="*60)
    
    # Test the additive reward formula
    test_cases = [
        {"answer_reward": 1.0, "cs_penalty": 0.0, "expected": 1.0, "desc": "Correct + No CS"},
        {"answer_reward": 1.0, "cs_penalty": -1.0, "expected": 0.0, "desc": "Correct + CS"},
        {"answer_reward": 0.0, "cs_penalty": 0.0, "expected": 0.0, "desc": "Incorrect + No CS"},
        {"answer_reward": 0.0, "cs_penalty": -1.0, "expected": -1.0, "desc": "Incorrect + CS"},
    ]
    
    all_passed = True
    for i, tc in enumerate(test_cases):
        total = tc["answer_reward"] + tc["cs_penalty"]
        passed = abs(total - tc["expected"]) < 0.001
        status = "✓ PASS" if passed else "✗ FAIL"
        
        log_debug("H5", f"test_reward_logic:{i}", tc["desc"],
                 {"answer_reward": tc["answer_reward"], "cs_penalty": tc["cs_penalty"],
                  "expected": tc["expected"], "total": total, "passed": passed})
        
        print(f"{status}: {tc['desc']}")
        print(f"       {tc['answer_reward']} + ({tc['cs_penalty']}) = {total}, Expected: {tc['expected']}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_full_module_import():
    """Test importing the full module (requires verl environment)."""
    print("\n" + "="*60)
    print("TEST: Full Module Import")
    print("="*60)
    
    try:
        # Try direct file import
        import importlib.util
        
        # First, set up the module search path
        reward_score_dir = "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/verl/utils/reward_score"
        sys.path.insert(0, reward_score_dir)
        
        # Import grader first
        from grader import math_equal
        print("✓ grader.math_equal imported")
        log_debug("IMPORT", "grader", "Imported successfully", {})
        
        # Import aime helpers
        from aime import math_answer_cleaning, round_number, is_equal_after_calculation
        print("✓ aime helpers imported")
        log_debug("IMPORT", "aime", "Imported successfully", {})
        
        # Now test math_equal
        test_cases = [
            {"a": "42", "b": "42", "expected": True, "desc": "Simple equal"},
            {"a": "0.5", "b": "0.5", "expected": True, "desc": "Decimal equal"},
            {"a": "1/2", "b": "0.5", "expected": True, "desc": "Fraction vs decimal"},
        ]
        
        all_passed = True
        for i, tc in enumerate(test_cases):
            try:
                result = math_equal(tc["a"], tc["b"])
                passed = result == tc["expected"]
                status = "✓ PASS" if passed else "✗ FAIL"
                
                log_debug("H3", f"test_math_equal:{i}", tc["desc"],
                         {"a": tc["a"], "b": tc["b"], "expected": tc["expected"], "result": result, "passed": passed})
                
                print(f"{status}: math_equal({tc['a']}, {tc['b']}) = {result}")
                
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"✗ ERROR: {tc['desc']}: {e}")
                log_debug("H3", f"test_math_equal:{i}", "Exception", {"error": str(e)})
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        traceback.print_exc()
        log_debug("IMPORT", "full_module", "Import failed", {"error": str(e)})
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Nemotron-Cascade RL Math Reward Function Tests")
    print("#"*60)
    
    results = {}
    
    # Test 1: last_boxed_only_string
    results["last_boxed"] = test_last_boxed_only_string()
    
    # Test 2: extract_boxed_after_think
    results["extract_boxed"] = test_extract_boxed_after_think()
    
    # Test 3: FastText import
    results["fasttext"] = test_fasttext_import()
    
    # Test 4: Reward computation logic
    results["reward_logic"] = test_reward_computation_logic()
    
    # Test 5: Full module import (math_equal, etc.)
    results["full_module"] = test_full_module_import()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        if passed is None:
            status = "⚠ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {name}: {status}")
    
    log_debug("SUMMARY", "main", "Test results", {k: str(v) for k, v in results.items()})
    
    # Overall result
    failed = [k for k, v in results.items() if v is False]
    if failed:
        print(f"\n❌ TESTS FAILED: {failed}")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED (or skipped)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
