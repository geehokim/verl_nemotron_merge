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
Lexical Similarity Module for SWE RL

This module computes lexical similarity between two patches using the unidiff
library as described in the Nemotron-Cascade paper Section 4.7.

From the paper:
"For our SWE reward, we leverage the unidiff library to compute lexical
similarity between the generated patch and the ground truth patch."

The lexical similarity score is used in the reward function:
- If lexical_similarity == 1.0, reward = 1.0 (perfect match)
- Otherwise, the lexical similarity score is used as a partial reward

Reference:
    Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for
    General-Purpose Reasoning Models (Section 4.7 - SWE RL)
"""

import difflib
from typing import Optional, List, Tuple, Set

from .patch_parser import parse_patch, normalize_patch, validate_patch


# =============================================================================
# Core Lexical Similarity Computation
# =============================================================================

def compute_lexical_similarity(
    generated_patch: str,
    ground_truth_patch: str,
    normalize: bool = True,
) -> float:
    """Compute lexical similarity between two patches.

    This implements the lexical similarity computation described in the
    Nemotron-Cascade paper using the unidiff library.

    The similarity is computed by:
    1. Normalizing both patches (removing timestamps, sorting hunks)
    2. Comparing the normalized patches using sequence matching
    3. Returning a score in [0, 1] where 1.0 means identical patches

    Args:
        generated_patch: The model-generated patch string
        ground_truth_patch: The ground truth patch string
        normalize: Whether to normalize patches before comparison (default True)

    Returns:
        Similarity score in range [0.0, 1.0]
        - 1.0: Patches are identical (lexically equivalent)
        - 0.0: Patches have no similarity
        - (0, 1): Partial similarity

    Example:
        >>> gen = "--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"
        >>> gt = "--- a/f.py\\n+++ b/f.py\\n@@ -1 +1 @@\\n-old\\n+new\\n"
        >>> compute_lexical_similarity(gen, gt)
        1.0
    """
    # Handle edge cases
    if not generated_patch and not ground_truth_patch:
        return 1.0  # Both empty -> identical
    if not generated_patch or not ground_truth_patch:
        return 0.0  # One empty, one not -> no similarity

    # Normalize patches for fair comparison
    if normalize:
        gen_normalized = normalize_patch(generated_patch)
        gt_normalized = normalize_patch(ground_truth_patch)
    else:
        gen_normalized = generated_patch.strip()
        gt_normalized = ground_truth_patch.strip()

    # Handle case where normalization produces empty strings
    if not gen_normalized and not gt_normalized:
        return 1.0  # Both normalize to empty -> treat as identical
    if not gen_normalized or not gt_normalized:
        return 0.0

    # Exact match check (fast path)
    if gen_normalized == gt_normalized:
        return 1.0

    # Use SequenceMatcher for similarity computation
    # This provides a ratio in [0, 1] based on the longest contiguous matching
    # subsequence
    matcher = difflib.SequenceMatcher(None, gen_normalized, gt_normalized)
    return matcher.ratio()


def compute_hunk_level_similarity(
    generated_patch: str,
    ground_truth_patch: str,
) -> float:
    """Compute similarity at the hunk level.

    This provides a more granular similarity measure by comparing individual
    hunks (change blocks) rather than the entire patch string.

    The algorithm:
    1. Parse both patches into hunks
    2. For each hunk in the generated patch, find the best matching hunk
       in the ground truth
    3. Average the best match scores

    Args:
        generated_patch: The model-generated patch string
        ground_truth_patch: The ground truth patch string

    Returns:
        Similarity score in range [0.0, 1.0]

    Note:
        This is a more robust similarity measure when patches modify the
        same code but with slight differences in context lines.
    """
    gen_patchset = parse_patch(generated_patch)
    gt_patchset = parse_patch(ground_truth_patch)

    # Handle parsing failures
    if gen_patchset is None or gt_patchset is None:
        # Fall back to string-level comparison
        return compute_lexical_similarity(generated_patch, ground_truth_patch)

    if len(gen_patchset) == 0 and len(gt_patchset) == 0:
        return 1.0
    if len(gen_patchset) == 0 or len(gt_patchset) == 0:
        return 0.0

    # Extract all hunks from both patches
    gen_hunks = _extract_hunks(gen_patchset)
    gt_hunks = _extract_hunks(gt_patchset)

    if not gen_hunks and not gt_hunks:
        return 1.0
    if not gen_hunks or not gt_hunks:
        return 0.0

    # Compute similarity using best matching
    total_similarity = 0.0
    matched_gt_indices = set()

    for gen_hunk in gen_hunks:
        best_match_score = 0.0
        best_match_idx = -1

        for i, gt_hunk in enumerate(gt_hunks):
            if i in matched_gt_indices:
                continue

            score = _compute_hunk_similarity(gen_hunk, gt_hunk)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = i

        if best_match_idx >= 0:
            matched_gt_indices.add(best_match_idx)
        total_similarity += best_match_score

    # Penalize for unmatched ground truth hunks
    num_hunks = max(len(gen_hunks), len(gt_hunks))
    return total_similarity / num_hunks


def compute_change_set_similarity(
    generated_patch: str,
    ground_truth_patch: str,
) -> float:
    """Compute similarity based on the set of changed lines.

    This focuses on what lines were added and removed, ignoring the
    structure of the diff (hunk boundaries, context lines).

    The algorithm:
    1. Extract all added lines from both patches
    2. Extract all removed lines from both patches
    3. Compute Jaccard similarity for added lines
    4. Compute Jaccard similarity for removed lines
    5. Return weighted average

    Args:
        generated_patch: The model-generated patch string
        ground_truth_patch: The ground truth patch string

    Returns:
        Similarity score in range [0.0, 1.0]

    Note:
        This is useful when the exact formatting of the patch differs
        but the actual changes are the same.
    """
    gen_patchset = parse_patch(generated_patch)
    gt_patchset = parse_patch(ground_truth_patch)

    if gen_patchset is None or gt_patchset is None:
        return compute_lexical_similarity(generated_patch, ground_truth_patch)

    # Extract change sets
    gen_added, gen_removed = _extract_change_sets(gen_patchset)
    gt_added, gt_removed = _extract_change_sets(gt_patchset)

    # Handle empty change sets
    if not gen_added and not gen_removed and not gt_added and not gt_removed:
        return 1.0

    # Compute Jaccard similarity for added lines
    added_similarity = _jaccard_similarity(gen_added, gt_added)

    # Compute Jaccard similarity for removed lines
    removed_similarity = _jaccard_similarity(gen_removed, gt_removed)

    # Weighted average (equal weight for adds and removes)
    return (added_similarity + removed_similarity) / 2.0


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_hunks(patchset) -> List[str]:
    """Extract all hunks from a patchset as strings.

    Args:
        patchset: unidiff.PatchSet object

    Returns:
        List of hunk strings
    """
    hunks = []
    for patched_file in patchset:
        for hunk in patched_file:
            hunk_lines = []
            for line in hunk:
                hunk_lines.append(line.value.rstrip('\n\r'))
            hunks.append("\n".join(hunk_lines))
    return hunks


def _compute_hunk_similarity(hunk1: str, hunk2: str) -> float:
    """Compute similarity between two hunks.

    Args:
        hunk1: First hunk string
        hunk2: Second hunk string

    Returns:
        Similarity score in [0, 1]
    """
    if hunk1 == hunk2:
        return 1.0
    matcher = difflib.SequenceMatcher(None, hunk1, hunk2)
    return matcher.ratio()


def _extract_change_sets(patchset) -> Tuple[Set[str], Set[str]]:
    """Extract sets of added and removed lines from a patchset.

    Args:
        patchset: unidiff.PatchSet object

    Returns:
        Tuple of (added_lines_set, removed_lines_set)
    """
    added = set()
    removed = set()

    for patched_file in patchset:
        for hunk in patched_file:
            for line in hunk:
                line_content = line.value.strip()
                if not line_content:
                    continue

                if line.is_added:
                    added.add(line_content)
                elif line.is_removed:
                    removed.add(line_content)

    return added, removed


def _jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Jaccard similarity = |intersection| / |union|

    Args:
        set1: First set
        set2: Second set

    Returns:
        Similarity in [0, 1], or 1.0 if both sets are empty
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0
