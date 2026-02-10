# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Adapted for VERL from SWE-RL (https://github.com/facebookresearch/swe-rl)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
SWE-RL Reward Function for VERL

This module implements the execution-free reward function from the SWE-RL paper.
Unlike the previous implementation which compared patch text directly, this version:

1. Uses Search/Replace block format (not unified diff)
2. Applies search/replace edits to original code
3. Computes similarity between resulting diffs (not patch text)

Expected format:
    <think>
    ... reasoning ...
    </think>
    <solution>
    ```path/to/file.py
    ### path/to/file.py
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE
    ```
    </solution>

Reference:
    SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Engineering Tasks
"""

import difflib
import re
import warnings
import json
from typing import Any, Optional, TypedDict, Union

try:
    from unidiff import PatchedFile, PatchSet
    from unidiff.errors import UnidiffParseError
except ImportError:
    PatchSet = None
    PatchedFile = None
    UnidiffParseError = Exception

# =============================================================================
# Constants
# =============================================================================

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<solution>"
ANSWER_END = "</solution>"

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)?\n<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE\n```"


# =============================================================================
# Exceptions
# =============================================================================

class FormatError(Exception):
    """Raised when the model output format is invalid."""
    pass


# =============================================================================
# Input Normalization
# =============================================================================

def _normalize_code_context(raw: Any) -> dict[str, str]:
    """Normalize code_context into dict[path, content].

    Supported input types:
    - dict[path, content]
    - list[{"file_path": ..., "content": ...}]
    - JSON string for either of the above
    """
    if raw is None:
        return {}

    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out

    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = item.get("file_path")
            content = item.get("content")
            if isinstance(path, str) and isinstance(content, str):
                out[path] = content
        return out

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            return _normalize_code_context(json.loads(text))
        except Exception:
            return {}

    return {}


# =============================================================================
# Extraction Functions
# =============================================================================

def extract_thought_solution(output: str) -> tuple[str, str]:
    """
    Extract the thought and solution from the output.

    Expected format:
        <think>
        ...
        </think>
        <solution>
        ...
        </solution>

    Args:
        output: Model's full output string

    Returns:
        Tuple of (thought, solution)

    Raises:
        FormatError: If tags are missing or malformed
    """
    for tag in [THINK_START, THINK_END, ANSWER_START, ANSWER_END]:
        if output.count(tag) != 1:
            raise FormatError(f"count of {tag} is not 1")

    thought = output.split(THINK_START)[1].split(THINK_END)[0].strip()
    answer = output.split(ANSWER_START)[1].split(ANSWER_END)[0].strip()
    if len(thought) == 0:
        raise FormatError("Thought is empty")
    return thought, answer


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Args:
        text: The solution text containing search/replace blocks

    Returns:
        A dictionary where the key is the file path and the value is a list
        of (search, replace) tuples.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


# =============================================================================
# Diff Generation Functions
# =============================================================================

def generate_unified_diff(
    old_code: str,
    new_code: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code strings.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff (without file headers).
    """
    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)  # Skip "--- old"
        next(diff)  # Skip "+++ new"
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


# =============================================================================
# Code Change Application
# =============================================================================

def apply_code_change(
    code_context: dict[str, str],
    search_replace_dict: dict[str, list[tuple[str, str]]],
    silent: bool = False,
) -> dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.

    Raises:
        FormatError: If search block not found or search equals replace
    """
    new_content_dict = dict[str, str]()
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            # Ensure search block can be matched
            # "\n" + search to ensure the indentations are correct
            if not silent and len(search) == len(replace) and search == replace:
                raise FormatError("Search and replace blocks are identical")
            search = "\n" + search
            replace = "\n" + replace
            if not silent and search not in new_content:
                raise FormatError(f"Search block not found in the code: {search}")
            new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        new_content_dict[path] = new_content[1:]
    return new_content_dict


# =============================================================================
# Patch Normalization
# =============================================================================

def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """
    Generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        if patch:
            patch_dict[path] = patch
    return patch_dict


# =============================================================================
# Similarity Computation
# =============================================================================

class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: dict[str, str],
    oracle_patch: dict[str, str],
) -> list[ChangeSimilarity]:
    """
    Compute per-file similarity between predicted and oracle patches.

    Args:
        pred_patch: Dictionary of file path -> predicted diff
        oracle_patch: Dictionary of file path -> oracle diff

    Returns:
        List of ChangeSimilarity dicts with per-file similarity scores
    """
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # One has changes, the other doesn't -> penalize
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )
    return similarities


# =============================================================================
# Main Reward Calculation Functions
# =============================================================================

def calculate_reward(
    code_context: dict[str, str],
    oracle_new_content: dict[str, str],
    pred_new_content: dict[str, str],
) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given the code context and new contents.

    This is the general version that takes already-computed new content dicts.

    The return value is always within the range of [0, 1].

    Args:
        code_context: path -> original content of the file.
        oracle_new_content: path -> oracle new content of the file after change.
        pred_new_content: path -> predicted new content of the file after change.

    Returns:
        A tuple of (reward, metadata dict)
    """
    # Obtain a unified diff for each file
    oracle_patch = get_normalized_patch(code_context, oracle_new_content)
    pred_patch = get_normalized_patch(code_context, pred_new_content)

    # Calculate the reward based on the similarity between patches
    similarities = compute_change_similarities(pred_patch, oracle_patch)

    # If both patches are empty, they are identical -> reward 1.0
    if len(similarities) == 0:
        assert len(oracle_patch) == 0 and len(pred_patch) == 0
        return 1.0, dict(similarities=[])

    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


def calculate_search_replace_reward(
    code_context: dict[str, str],
    ground_truth: str,
    output: str,
) -> tuple[float, dict]:
    """
    The search/replace version of the reward calculation.

    Expected format:
        <think>
        ...
        </think>
        <solution>
        ...search/replace blocks...
        </solution>

    Args:
        code_context: path -> original content of the file.
        ground_truth: Golden patch in unified diff format (git diff output).
        output: The output from the model containing the thought and solution.

    Returns:
        A tuple of (reward, metadata dict).
        Returns -1.0 for format errors.
    """
    try:
        # Extract the thought and solution from the output
        thought, answer = extract_thought_solution(output)
        # Parse the search/replace edits from the solution
        pred_search_replaces = parse_search_replace(answer)
        if len(pred_search_replaces) == 0:
            raise FormatError("No valid search blocks found")
        # Get the new content of each file after applying the search/replace edits
        pred_new_content = apply_code_change(code_context, pred_search_replaces)

        # Generate predicted patch (unified diff from code changes)
        pred_patch = get_normalized_patch(code_context, pred_new_content)

        # Parse golden patch from ground_truth (already unified diff format)
        oracle_patch = get_filelevel_diff(ground_truth)

        # Calculate the reward based on the similarity between patches
        similarities = compute_change_similarities(pred_patch, oracle_patch)

        # If both patches are empty, they are identical -> reward 1.0
        if len(similarities) == 0:
            assert len(oracle_patch) == 0 and len(pred_patch) == 0
            reward = 1.0
            metadata = dict(similarities=[])
        else:
            reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
            metadata = dict(similarities=similarities)

        metadata["thought"] = thought
        metadata["answer"] = answer
        return reward, metadata
    except FormatError as e:
        return -1.0, dict(error=str(e))


# =============================================================================
# Unified Diff Based Reward (Alternative)
# =============================================================================

def get_filelevel_diff(patch_text: str) -> dict[str, str]:
    """
    Convert a unified diff text into a dictionary of file patches.

    Args:
        patch_text: Unified diff string

    Returns:
        Dictionary of file path -> normalized diff content
    """
    if PatchSet is None:
        warnings.warn("unidiff not installed, returning empty dict")
        return {}

    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        warnings.warn(f"Unexpected unidiff parsing error: {str(e)}")
        return {}

    result = dict[str, str]()
    for patchfile in patch:
        patchfile: PatchedFile = patchfile
        if patchfile.is_binary_file:
            continue
        if patchfile.is_rename:
            source_file = patchfile.source_file
            target_file = patchfile.target_file
            if source_file.startswith("a/"):
                source_file = source_file[2:]
            if target_file.startswith("b/"):
                target_file = target_file[2:]
            header = f"rename from {source_file} to {target_file}"
            path = source_file
        else:
            header = ""
            path = patchfile.path
        body = "\n".join(str(hunk).strip() for hunk in patchfile)
        content = header + "\n" + body
        content = content.strip()
        result[path] = content
    return result


def calculate_reward_unidiff(
    oracle_patches: list[str],
    pred_patches: list[str]
) -> tuple[float, dict]:
    """
    Compute the SWE-RL reward given two sets of unified diffs.

    The return value is always within the range of [0, 1].

    Args:
        oracle_patches: A list of oracle diffs.
        pred_patches: A list of predicted diffs.

    Returns:
        A tuple of (reward, metadata dict)
    """
    pred_patch_dict = dict[str, str]()
    oracle_patch_dict = dict[str, str]()

    for patch_text in oracle_patches:
        oracle_patch_dict.update(get_filelevel_diff(patch_text))

    for patch_text in pred_patches:
        pred_patch_dict.update(get_filelevel_diff(patch_text))

    similarities = compute_change_similarities(pred_patch_dict, oracle_patch_dict)
    if len(similarities) == 0:
        assert len(pred_patch_dict) == 0 and len(oracle_patch_dict) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


# =============================================================================
# VERL-Compatible Interface
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    return_dict: bool = True,
) -> Union[float, dict]:
    """
    Compute SWE-RL reward score for VERL training.

    This function provides a VERL-compatible interface to the SWE-RL reward.

    IMPORTANT: extra_info MUST contain:
        - "code_context": dict[str, str] - Original file contents (path -> content)

    Args:
        solution_str: Model's full output (with <think>/<solution> tags)
        ground_truth: Golden patch in unified diff format (git diff output)
        extra_info: Must contain "code_context"
        return_dict: If True, return dict with score and metadata

    Returns:
        If return_dict=True: dict with score, acc, valid, etc.
        If return_dict=False: float reward score in [-1.0, 1.0]
    """
    # Extract required data from extra_info
    if not extra_info:
        if return_dict:
            return {
                "score": -1.0,
                "acc": False,
                "pred": "",
                "valid": False,
                "error": "extra_info is required with code_context",
            }
        return -1.0

    code_context = _normalize_code_context(extra_info.get("code_context", {}))
    problem_id = extra_info.get("problem_id", None)

    # Handle overlong filtering
    overlong_filtering = extra_info.get("overlong_filtering", False)
    response_length = extra_info.get("response_length", 0)
    max_response_length = extra_info.get("max_response_length", float("inf"))

    is_overlong = (
        (response_length >= max_response_length)
        if max_response_length != float("inf")
        else False
    )

    if overlong_filtering and is_overlong:
        if return_dict:
            return {
                "score": 0.0,
                "acc": None,
                "pred": "",
                "valid": False,
                "skip": True,
                "overlong": True,
                "problem_id": problem_id,
            }
        return 0.0

    # Calculate reward using SWE-RL method
    # ground_truth contains the golden patch (unified diff)
    reward, metadata = calculate_search_replace_reward(
        code_context, ground_truth, solution_str
    )

    # Determine accuracy (exact match = 1.0)
    is_correct = reward == 1.0
    is_valid = reward >= 0  # -1.0 means format error

    if return_dict:
        return {
            "score": reward,
            "acc": is_correct,
            "pred": metadata.get("answer", ""),
            "valid": is_valid,
            "thought": metadata.get("thought", ""),
            "error": metadata.get("error", None),
            "similarities": metadata.get("similarities", []),
            "skip": False,
            "overlong": is_overlong,
            "problem_id": problem_id,
        }

    return reward


# =============================================================================
# Testing
# =============================================================================

def test_reward_function():
    """Test the SWE-RL reward function with example data."""

    # Original code
    code_context = {
        "src/utils.py": '''def hello():
    print("Hello")
    return True

def goodbye():
    print("Goodbye")
'''
    }

    # Golden patch (unified diff format)
    golden_patch = '''diff --git a/src/utils.py b/src/utils.py
index 1234567..abcdefg 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,6 +1,6 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True

 def goodbye():
     print("Goodbye")
'''

    # Test Case 1: Exact match
    solution_exact = '''<think>
I need to fix the hello function to print "Hello, World!" instead of "Hello".
</think>
<solution>
```python
### src/utils.py
<<<<<<< SEARCH
def hello():
    print("Hello")
    return True
=======
def hello():
    print("Hello, World!")
    return True
>>>>>>> REPLACE
```
</solution>'''

    result = compute_score(
        solution_exact,
        golden_patch,
        extra_info={
            "code_context": code_context,
        }
    )
    print(f"Case 1 (Exact match): score={result['score']:.4f}, expected=1.0")
    assert result['score'] == 1.0, f"Case 1 failed: got {result['score']}"

    # Test Case 2: Partial match (different fix)
    solution_partial = '''<think>
I'll fix the hello function with a slightly different message.
</think>
<solution>
```python
### src/utils.py
<<<<<<< SEARCH
def hello():
    print("Hello")
    return True
=======
def hello():
    print("Hi, World!")
    return True
>>>>>>> REPLACE
```
</solution>'''

    result = compute_score(
        solution_partial,
        golden_patch,
        extra_info={
            "code_context": code_context,
        }
    )
    print(f"Case 2 (Partial match): score={result['score']:.4f}, expected in (0, 1)")
    assert 0 < result['score'] < 1, f"Case 2 failed: got {result['score']}"

    # Test Case 3: Format error (missing tags)
    solution_invalid = "This is not a valid format"

    result = compute_score(
        solution_invalid,
        golden_patch,
        extra_info={
            "code_context": code_context,
        }
    )
    print(f"Case 3 (Format error): score={result['score']:.4f}, expected=-1.0")
    assert result['score'] == -1.0, f"Case 3 failed: got {result['score']}"

    # Test Case 4: Search block not found
    solution_wrong_search = '''<think>
I'll try to fix code that doesn't exist.
</think>
<solution>
```python
### src/utils.py
<<<<<<< SEARCH
def nonexistent():
    pass
=======
def nonexistent():
    return None
>>>>>>> REPLACE
```
</solution>'''

    result = compute_score(
        solution_wrong_search,
        golden_patch,
        extra_info={
            "code_context": code_context,
        }
    )
    print(f"Case 4 (Search not found): score={result['score']:.4f}, expected=-1.0")
    assert result['score'] == -1.0, f"Case 4 failed: got {result['score']}"

    print("\nAll test cases passed!")


if __name__ == "__main__":
    test_reward_function()
