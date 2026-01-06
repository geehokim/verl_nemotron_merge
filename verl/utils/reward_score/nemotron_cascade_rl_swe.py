# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from Nemotron-Cascade paper for SWE RL implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Nemotron-Cascade RL SWE Reward Function

This module implements the execution-free reward function for Software Engineering
(SWE) Reinforcement Learning as described in the Nemotron-Cascade paper Section 4.7.

Reward Logic (4 Cases):
    1. lexical_similarity == 1.0 → reward = 1.0 (exact match)
    2. Patch identical to original code → reward = 0.0 (no change)
    3. Patch cannot be parsed → reward = -1.0 (invalid patch)
    4. Otherwise → reward = lexical_similarity (partial reward)

Note: Semantic similarity (LLM-based) is not implemented in this version.
      Future versions may add semantic similarity using Kimi-Dev-72B or similar.

Reference:
    Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for
    General-Purpose Reasoning Models (Section 4.7 - SWE RL)

    "We adopt an execution-free reward function with 4 distinct cases:
    1. If the lexical similarity between the generated patch and the ground
       truth is exactly 1, the reward is 1.
    2. If the generated patch is identical to the original code (i.e., no
       changes), the reward is 0.
    3. If the generated patch cannot be parsed, the reward is -1.
    4. Otherwise, the reward is the semantic similarity score obtained by
       querying an LLM..."
"""

import re
from typing import Optional, Union

# Import SWE utility functions
from .swe_utils import (
    parse_patch,
    validate_patch,
    is_empty_patch,
    compute_lexical_similarity,
)
from .swe_utils.patch_parser import extract_patch_from_response


# =============================================================================
# Patch Extraction Functions
# =============================================================================

def extract_patch_after_think(solution_str: str) -> Optional[str]:
    """Extract the patch from the model response after the </think> token.

    Similar to the math reward function, this extracts content after the
    reasoning chain ends (marked by </think>).

    The function attempts to find a unified diff patch in the response,
    either within code blocks or as plain text.

    Args:
        solution_str: Model's full output string (reasoning + patch)

    Returns:
        Extracted patch string, or None if no valid patch found

    Example:
        >>> response = '''<think>Let me analyze the bug...</think>
        ... Here's the fix:
        ... ```diff
        ... --- a/file.py
        ... +++ b/file.py
        ... @@ -1 +1 @@
        ... -buggy_code()
        ... +fixed_code()
        ... ```
        ... '''
        >>> patch = extract_patch_after_think(response)
        >>> patch is not None
        True
    """
    # Find </think> token position
    think_end_markers = ["</think>", "<\/think>", "</Think>"]
    think_end_idx = -1

    for marker in think_end_markers:
        idx = solution_str.find(marker)
        if idx != -1:
            think_end_idx = idx + len(marker)
            break

    if think_end_idx == -1:
        # If </think> is not found, search the entire string
        text_after_think = solution_str
    else:
        # Extract text after </think>
        text_after_think = solution_str[think_end_idx:]

    # Extract patch from the text after </think>
    return extract_patch_from_response(text_after_think)


# =============================================================================
# Semantic Similarity (Placeholder for Future Implementation)
# =============================================================================

def compute_semantic_similarity(
    generated_patch: str,
    ground_truth_patch: str,
    problem_description: Optional[str] = None,
) -> float:
    """Compute semantic similarity between two patches using an LLM.

    NOTE: This is a placeholder for future implementation.

    From the Nemotron-Cascade paper:
    "The semantic similarity is computed by prompting Kimi-Dev-72B (Team, 2025)
    with the question 'Does this patch fix the same issue as the ground truth?',
    and using P(YES) as the reward score."

    Args:
        generated_patch: The model-generated patch
        ground_truth_patch: The ground truth patch
        problem_description: Optional problem description for context

    Returns:
        Semantic similarity score in [0, 1]

    TODO:
        - Implement LLM-based semantic similarity
        - Add support for different LLM backends (vLLM, OpenAI, etc.)
        - Consider caching for repeated queries
    """
    # Placeholder: Fall back to lexical similarity
    # Future implementation should use LLM to compute P(YES) for semantic match
    return compute_lexical_similarity(generated_patch, ground_truth_patch)


# =============================================================================
# Main Reward Computation Function
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    return_dict: bool = True,
) -> Union[float, dict]:
    """Compute reward score for Nemotron-Cascade SWE RL training.

    This is the main reward function implementing the Nemotron-Cascade paper's
    SWE reward logic (Section 4.7).

    Reward Cases:
        Case 1: lexical_similarity == 1.0 → reward = 1.0
        Case 2: Patch identical to original (empty patch) → reward = 0.0
        Case 3: Patch cannot be parsed → reward = -1.0
        Case 4: Otherwise → reward = lexical_similarity (semantic placeholder)

    Args:
        solution_str: Model's full output string (reasoning + patch)
        ground_truth: Ground truth patch string in unified diff format
        extra_info: Optional dict containing:
            - "original_code": Original code before the fix (for empty patch check)
            - "problem_id": Identifier for the problem (for logging)
            - "overlong_filtering": Whether to filter overlong responses
            - "response_length": Actual response length in tokens
            - "max_response_length": Maximum allowed response length
            - "use_semantic_similarity": Whether to use semantic similarity (default False)
        return_dict: If True, return dict with score and metadata

    Returns:
        If return_dict=True:
            dict with score, acc, pred, valid, empty, etc.
        If return_dict=False:
            float: Reward score in range [-1.0, 1.0]

    Example:
        >>> solution = "<think>Fix the bug</think>```diff\\n--- a/f.py\\n...```"
        >>> gt = "--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"
        >>> result = compute_score(solution, gt)
        >>> isinstance(result, dict)
        True
        >>> -1.0 <= result['score'] <= 1.0
        True
    """
    # ==========================================================================
    # Step 0: Parse extra_info
    # ==========================================================================

    original_code = None
    problem_id = None
    overlong_filtering = False
    response_length = 0
    max_response_length = float("inf")
    use_semantic_similarity = False

    if extra_info and isinstance(extra_info, dict):
        original_code = extra_info.get("original_code", None)
        problem_id = extra_info.get("problem_id", None)
        overlong_filtering = extra_info.get("overlong_filtering", False)
        response_length = extra_info.get("response_length", 0)
        max_response_length = extra_info.get("max_response_length", float("inf"))
        use_semantic_similarity = extra_info.get("use_semantic_similarity", False)

    # ==========================================================================
    # Step 0.5: Check for overlong filtering
    # ==========================================================================

    is_overlong = (
        (response_length >= max_response_length)
        if max_response_length != float("inf")
        else False
    )

    if overlong_filtering and is_overlong:
        # Return skip marker - these samples won't contribute to policy gradient
        if return_dict:
            return {
                "score": 0.0,
                "acc": None,
                "pred": "",
                "valid": False,
                "empty": False,
                "lexical_similarity": 0.0,
                "skip": True,
                "overlong": True,
                "problem_id": problem_id,
            }
        return 0.0

    # ==========================================================================
    # Step 1: Extract patch from model response
    # ==========================================================================

    generated_patch = extract_patch_after_think(solution_str)

    if generated_patch is None:
        # No patch found in response -> treat as invalid
        # Case 3: Cannot parse -> reward = -1.0
        if return_dict:
            return {
                "score": -1.0,
                "acc": False,
                "pred": "",
                "valid": False,
                "empty": False,
                "lexical_similarity": 0.0,
                "extraction_failed": True,
                "skip": False,
                "overlong": is_overlong,
                "problem_id": problem_id,
            }
        return -1.0

    # ==========================================================================
    # Step 2: Check if patch can be parsed (Case 3)
    # ==========================================================================
    # From paper: "If the generated patch cannot be parsed, the reward is -1"

    is_valid = validate_patch(generated_patch)

    if not is_valid:
        # Case 3: Patch cannot be parsed -> reward = -1.0
        if return_dict:
            return {
                "score": -1.0,
                "acc": False,
                "pred": generated_patch,
                "valid": False,
                "empty": False,
                "lexical_similarity": 0.0,
                "extraction_failed": False,
                "skip": False,
                "overlong": is_overlong,
                "problem_id": problem_id,
            }
        return -1.0

    # ==========================================================================
    # Step 3: Check if patch is empty (Case 2)
    # ==========================================================================
    # From paper: "If the generated patch is identical to the original code
    # (i.e., no changes), the reward is 0"

    is_empty = is_empty_patch(generated_patch, original_code)

    if is_empty:
        # Case 2: Empty patch (no changes) -> reward = 0.0
        if return_dict:
            return {
                "score": 0.0,
                "acc": False,
                "pred": generated_patch,
                "valid": True,
                "empty": True,
                "lexical_similarity": 0.0,
                "extraction_failed": False,
                "skip": False,
                "overlong": is_overlong,
                "problem_id": problem_id,
            }
        return 0.0

    # ==========================================================================
    # Step 4: Compute lexical similarity
    # ==========================================================================
    # From paper: "If the lexical similarity between the generated patch and
    # the ground truth is exactly 1, the reward is 1"

    lexical_sim = compute_lexical_similarity(generated_patch, ground_truth)

    if lexical_sim == 1.0:
        # Case 1: Exact match -> reward = 1.0
        if return_dict:
            return {
                "score": 1.0,
                "acc": True,
                "pred": generated_patch,
                "valid": True,
                "empty": False,
                "lexical_similarity": 1.0,
                "extraction_failed": False,
                "skip": False,
                "overlong": is_overlong,
                "problem_id": problem_id,
            }
        return 1.0

    # ==========================================================================
    # Step 5: Compute semantic similarity or use lexical similarity (Case 4)
    # ==========================================================================
    # From paper: "Otherwise, the reward is the semantic similarity score..."
    #
    # NOTE: Semantic similarity is not implemented in this version.
    #       Using lexical similarity as fallback.

    if use_semantic_similarity:
        # Future: Use LLM-based semantic similarity
        problem_desc = extra_info.get("problem_description", None) if extra_info else None
        reward = compute_semantic_similarity(
            generated_patch, ground_truth, problem_desc
        )
    else:
        # Use lexical similarity as the reward for partial matches
        reward = lexical_sim

    # Determine if this is considered "correct" (threshold = 1.0 for strict matching)
    is_correct = lexical_sim == 1.0

    if return_dict:
        return {
            "score": reward,
            "acc": is_correct,
            "pred": generated_patch,
            "valid": True,
            "empty": False,
            "lexical_similarity": lexical_sim,
            "extraction_failed": False,
            "skip": False,
            "overlong": is_overlong,
            "problem_id": problem_id,
        }

    return reward


# =============================================================================
# Utility Functions for Testing
# =============================================================================

def test_reward_function():
    """Test the SWE reward function with example patches.

    This function demonstrates the 4 cases of the reward function.
    Run this to verify the implementation is working correctly.
    """
    # Test Case 1: Exact match (reward = 1.0)
    gt_patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
 hello()
"""
    solution_exact = f"<think>Let me fix this...</think>```diff\n{gt_patch}```"
    result = compute_score(solution_exact, gt_patch)
    print(f"Case 1 (Exact match): score={result['score']}, expected=1.0")
    assert result['score'] == 1.0, "Case 1 failed"

    # Test Case 2: Empty patch (reward = 0.0)
    empty_patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
     print("Hello")
 hello()
"""
    solution_empty = f"<think>No changes needed</think>```diff\n{empty_patch}```"
    result = compute_score(solution_empty, gt_patch)
    print(f"Case 2 (Empty patch): score={result['score']}, expected=0.0")
    assert result['score'] == 0.0, "Case 2 failed"

    # Test Case 3: Invalid patch (reward = -1.0)
    solution_invalid = "<think>Let me try...</think>This is not a valid patch"
    result = compute_score(solution_invalid, gt_patch)
    print(f"Case 3 (Invalid patch): score={result['score']}, expected=-1.0")
    assert result['score'] == -1.0, "Case 3 failed"

    # Test Case 4: Partial match (0 < reward < 1)
    partial_patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def hello():
-    print("Hello")
+    print("Hi, World!")
 hello()
"""
    solution_partial = f"<think>Let me fix...</think>```diff\n{partial_patch}```"
    result = compute_score(solution_partial, gt_patch)
    print(f"Case 4 (Partial match): score={result['score']:.4f}, expected in (0, 1)")
    assert 0 < result['score'] < 1, "Case 4 failed"

    print("\nAll test cases passed!")


if __name__ == "__main__":
    test_reward_function()
