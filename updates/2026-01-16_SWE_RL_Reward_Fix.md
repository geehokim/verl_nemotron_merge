# SWE-RL Reward Function Fix

**Date**: 2026-01-16
**Author**: Claude
**Status**: Completed

## Summary

Fixed the SWE RL reward calculation to match the original SWE-RL paper implementation. The previous implementation was incorrectly comparing patch text directly instead of applying edits and comparing resulting diffs.

## Problem

The existing `nemotron_cascade_rl_swe.py` had fundamental issues:

| Issue | Previous Implementation | Correct Implementation (SWE-RL) |
|-------|------------------------|--------------------------------|
| **Input Format** | `<think>` + unified diff block | `<think>` + `<solution>` tags with Search/Replace blocks |
| **Patch Format** | Unified diff (```diff) | Search/Replace blocks (<<<<<<< SEARCH / ======= / >>>>>>> REPLACE) |
| **Similarity Calculation** | Direct comparison of patch text | Apply edits to code, then compare resulting diffs |
| **Required Data** | `ground_truth` patch only | `code_context` + `oracle_new_content` |

## Solution

Created new file `swe_rl_reward.py` that implements the correct SWE-RL reward logic from the original paper.

### Key Functions

1. **`extract_thought_solution()`**: Extracts content from `<think>...</think><solution>...</solution>` tags
2. **`parse_search_replace()`**: Parses Search/Replace blocks using regex
3. **`apply_code_change()`**: Applies search/replace edits to original code
4. **`get_normalized_patch()`**: Generates unified diff from code changes
5. **`compute_change_similarities()`**: Computes per-file similarity using `difflib.SequenceMatcher`
6. **`calculate_search_replace_reward()`**: Main reward calculation function

### Expected Model Output Format

```
<think>
... reasoning about the bug fix ...
</think>
<solution>
```python
### path/to/file.py
<<<<<<< SEARCH
def buggy_function():
    return wrong_value
=======
def buggy_function():
    return correct_value
>>>>>>> REPLACE
```
</solution>
```

### Usage

```python
from verl.utils.reward_score.swe_rl_reward import compute_score

result = compute_score(
    solution_str=model_output,
    ground_truth="",  # Not used (API compatibility)
    extra_info={
        "code_context": {"path/to/file.py": "original code content"},
        "oracle_new_content": {"path/to/file.py": "expected code after fix"},
    }
)

# result['score'] is in range [-1.0, 1.0]
# -1.0: Format error (missing tags, search block not found, etc.)
# 0.0 ~ 1.0: Similarity score (average of per-file diff similarities)
```

## Files Changed

| File | Change |
|------|--------|
| `verl/utils/reward_score/swe_rl_reward.py` | **NEW** - SWE-RL reward implementation |
| `verl/utils/reward_score/swe_utils/__init__.py` | Updated exports |

## Testing

```bash
source ~/verl/bin/activate
python verl/utils/reward_score/swe_rl_reward.py
```

Output:
```
Case 1 (Exact match): score=1.0000, expected=1.0
Case 2 (Partial match): score=0.9774, expected in (0, 1)
Case 3 (Format error): score=-1.0000, expected=-1.0
Case 4 (Search not found): score=-1.0000, expected=-1.0

All test cases passed!
```

## Important Notes

1. **Dataset Requirements**: Your dataset must provide `code_context` and `oracle_new_content` in `extra_info`
2. **Model Training**: Model must be trained to output in the `<think>/<solution>` format with Search/Replace blocks
3. **Backward Compatibility**: The old `nemotron_cascade_rl_swe.py` is preserved for reference

## Reference

- [SWE-RL Paper](https://arxiv.org/abs/2502.18449): SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Engineering Tasks
- Original implementation: `swe-rl/src/swerl/core/reward.py`
