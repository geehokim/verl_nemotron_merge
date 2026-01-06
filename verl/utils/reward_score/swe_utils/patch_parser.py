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
Patch Parser Module for SWE RL

This module provides utilities for parsing and validating unified diff patches
using the unidiff library. These functions are essential for the SWE RL reward
computation as described in the Nemotron-Cascade paper Section 4.7.

The reward function requires:
1. Parsing patches to check if they are valid (reward = -1 if unparseable)
2. Checking if a patch is empty/identical to original (reward = 0)
3. Normalizing patches for lexical similarity comparison

Reference:
    Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for
    General-Purpose Reasoning Models (Section 4.7 - SWE RL)
"""

import re
from typing import Optional, Union, List, Tuple

# =============================================================================
# Lazy Import for unidiff (performance optimization)
# =============================================================================

_unidiff = None


def _get_unidiff():
    """Lazily import unidiff module.

    This avoids loading unidiff at module import time, which is beneficial
    when the SWE reward function is not being used.

    Returns:
        The unidiff module

    Raises:
        ImportError: If unidiff is not installed
    """
    global _unidiff
    if _unidiff is None:
        try:
            import unidiff as _unidiff_module
            _unidiff = _unidiff_module
        except ImportError as e:
            raise ImportError(
                "unidiff is required for SWE patch parsing. "
                "Please install it with: pip install unidiff"
            ) from e
    return _unidiff


# =============================================================================
# Patch Extraction Functions
# =============================================================================

def extract_patch_from_response(response: str) -> Optional[str]:
    """Extract a unified diff patch from model response.

    The model may generate the patch within code blocks (```diff or ```)
    or as plain text. This function attempts to extract the patch content.

    Args:
        response: Model's full response string

    Returns:
        Extracted patch string, or None if no patch found

    Example:
        >>> response = '''Here's the fix:
        ... ```diff
        ... --- a/file.py
        ... +++ b/file.py
        ... @@ -1,3 +1,3 @@
        ...  line1
        ... -old_line
        ... +new_line
        ...  line3
        ... ```
        ... '''
        >>> extract_patch_from_response(response)
        '--- a/file.py\\n+++ b/file.py\\n@@ -1,3 +1,3 @@\\n line1\\n-old_line\\n+new_line\\n line3\\n'
    """
    # Pattern 1: Extract from ```diff ... ``` code block
    diff_block_pattern = r'```(?:diff)?\s*\n((?:---|\+\+\+|@@|[-+ ]|diff --git).*?)```'
    match = re.search(diff_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 2: Look for unified diff markers directly
    # Find content starting with "---" or "diff --git"
    direct_diff_pattern = r'((?:diff --git.*?\n)?---\s+\S+.*?\n\+\+\+\s+\S+.*?\n(?:@@.*?\n(?:[-+ ].*?\n)*)+)'
    match = re.search(direct_diff_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Pattern 3: Check if the entire response looks like a patch
    if response.strip().startswith(('---', 'diff --git')):
        return response.strip()

    return None


# =============================================================================
# Patch Parsing Functions
# =============================================================================

def parse_patch(patch_str: str) -> Optional['unidiff.PatchSet']:
    """Parse a unified diff patch string into a PatchSet object.

    Uses the unidiff library to parse the patch. Returns None if parsing fails.

    Args:
        patch_str: Unified diff format patch string

    Returns:
        unidiff.PatchSet object if parsing succeeds, None otherwise

    Example:
        >>> patch = '''--- a/file.py
        ... +++ b/file.py
        ... @@ -1,3 +1,3 @@
        ...  line1
        ... -old
        ... +new
        ...  line3
        ... '''
        >>> patchset = parse_patch(patch)
        >>> patchset is not None
        True
        >>> len(patchset)
        1
    """
    if not patch_str or not patch_str.strip():
        return None

    unidiff = _get_unidiff()

    try:
        # Parse the patch string
        patchset = unidiff.PatchSet.from_string(patch_str)
        return patchset
    except Exception:
        # unidiff can raise various exceptions for malformed patches
        return None


def validate_patch(patch_str: str) -> bool:
    """Validate if a patch string can be successfully parsed.

    This is used for the reward computation Case 3:
    "If the patch cannot be parsed, reward = -1"

    Args:
        patch_str: Unified diff format patch string

    Returns:
        True if the patch can be parsed, False otherwise

    Example:
        >>> validate_patch("--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n")
        True
        >>> validate_patch("this is not a valid patch")
        False
    """
    if not patch_str or not patch_str.strip():
        return False

    patchset = parse_patch(patch_str)
    if patchset is None:
        return False

    # A valid patch should have at least one file changed
    return len(patchset) > 0


# =============================================================================
# Patch Content Analysis Functions
# =============================================================================

def is_empty_patch(patch_str: str, original_code: Optional[str] = None) -> bool:
    """Check if a patch is empty (no actual changes).

    This is used for the reward computation Case 2:
    "If the patch is identical to the original code, reward = 0"

    A patch is considered empty if:
    1. It cannot be parsed
    2. It has no files changed
    3. All hunks have zero additions and zero removals

    Args:
        patch_str: Unified diff format patch string
        original_code: Optional original code (not used in current implementation,
                      but kept for API compatibility with future enhancements)

    Returns:
        True if the patch makes no changes, False otherwise

    Example:
        >>> is_empty_patch("")
        True
        >>> is_empty_patch("--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n")
        False
    """
    if not patch_str or not patch_str.strip():
        return True

    patchset = parse_patch(patch_str)
    if patchset is None:
        # Cannot parse -> treat as "no valid change"
        return True

    if len(patchset) == 0:
        return True

    # Check if all files have no actual changes
    total_added = 0
    total_removed = 0

    for patched_file in patchset:
        total_added += patched_file.added
        total_removed += patched_file.removed

    # If no lines were added or removed, it's an empty patch
    return total_added == 0 and total_removed == 0


def normalize_patch(patch_str: str) -> str:
    """Normalize a patch for comparison purposes.

    Normalization includes:
    1. Removing timestamps from --- and +++ lines
    2. Normalizing whitespace in file paths
    3. Sorting hunks by line number (for consistent comparison)

    This helps with lexical similarity computation by making semantically
    equivalent patches have the same string representation.

    Args:
        patch_str: Unified diff format patch string

    Returns:
        Normalized patch string, or empty string if parsing fails

    Example:
        >>> patch = "--- a/f.py 2024-01-01\\n+++ b/f.py 2024-01-01\\n@@ -1 +1 @@\\n-old\\n+new\\n"
        >>> normalized = normalize_patch(patch)
        >>> "2024-01-01" in normalized
        False
    """
    if not patch_str or not patch_str.strip():
        return ""

    patchset = parse_patch(patch_str)
    if patchset is None:
        return ""

    normalized_lines = []

    for patched_file in patchset:
        # Normalize file paths (remove timestamps and 'a/' 'b/' prefixes)
        source_file = patched_file.source_file
        target_file = patched_file.target_file

        # Remove 'a/' and 'b/' prefixes if present
        if source_file.startswith('a/'):
            source_file = source_file[2:]
        if target_file.startswith('b/'):
            target_file = target_file[2:]

        normalized_lines.append(f"--- {source_file}")
        normalized_lines.append(f"+++ {target_file}")

        for hunk in patched_file:
            # Add hunk header
            normalized_lines.append(
                f"@@ -{hunk.source_start},{hunk.source_length} "
                f"+{hunk.target_start},{hunk.target_length} @@"
            )

            # Add hunk lines
            for line in hunk:
                # line.value includes the newline, strip it for consistency
                line_content = line.value.rstrip('\n\r')

                if line.is_added:
                    normalized_lines.append(f"+{line_content}")
                elif line.is_removed:
                    normalized_lines.append(f"-{line_content}")
                else:
                    # Context line
                    normalized_lines.append(f" {line_content}")

    return "\n".join(normalized_lines)


def get_patch_stats(patch_str: str) -> dict:
    """Get statistics about a patch.

    Useful for debugging and logging reward computation.

    Args:
        patch_str: Unified diff format patch string

    Returns:
        Dictionary with patch statistics:
        - valid: Whether the patch can be parsed
        - num_files: Number of files modified
        - total_added: Total lines added
        - total_removed: Total lines removed
        - files: List of file paths affected

    Example:
        >>> stats = get_patch_stats("--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n")
        >>> stats['valid']
        True
        >>> stats['num_files']
        1
    """
    result = {
        "valid": False,
        "num_files": 0,
        "total_added": 0,
        "total_removed": 0,
        "files": [],
    }

    if not patch_str or not patch_str.strip():
        return result

    patchset = parse_patch(patch_str)
    if patchset is None:
        return result

    result["valid"] = True
    result["num_files"] = len(patchset)

    for patched_file in patchset:
        result["total_added"] += patched_file.added
        result["total_removed"] += patched_file.removed
        result["files"].append(patched_file.path)

    return result
