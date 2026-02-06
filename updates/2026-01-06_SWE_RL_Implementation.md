# [2026-01-06] Nemotron-Cascade SWE RL Implementation

## Summary
Implemented the SWE (Software Engineering) RL training pipeline based on Nemotron-Cascade paper Section 4.7.

## Changes

### New Files Created

- **File**: `verl/utils/reward_score/swe_utils/__init__.py`
    - Package initialization for SWE utility functions
    - Exports: `parse_patch`, `validate_patch`, `is_empty_patch`, `normalize_patch`, `compute_lexical_similarity`

- **File**: `verl/utils/reward_score/swe_utils/patch_parser.py`
    - Unified diff patch parsing utilities using `unidiff` library
    - Functions:
        - `extract_patch_from_response()`: Extract patch from model response
        - `parse_patch()`: Parse unified diff string into PatchSet
        - `validate_patch()`: Check if patch can be parsed
        - `is_empty_patch()`: Check if patch makes no changes
        - `normalize_patch()`: Normalize patch for comparison
        - `get_patch_stats()`: Get statistics about a patch

- **File**: `verl/utils/reward_score/swe_utils/lexical_similarity.py`
    - Lexical similarity computation between patches
    - Functions:
        - `compute_lexical_similarity()`: Main similarity function using SequenceMatcher
        - `compute_hunk_level_similarity()`: Hunk-level granular comparison
        - `compute_change_set_similarity()`: Set-based comparison (Jaccard similarity)

- **File**: `verl/utils/reward_score/nemotron_cascade_rl_swe.py`
    - Main SWE reward function implementing 4-case logic from paper
    - Reward cases:
        1. `lexical_similarity == 1` → reward = 1.0 (exact match)
        2. Patch identical to original → reward = 0.0 (no change)
        3. Patch cannot be parsed → reward = -1.0 (invalid)
        4. Otherwise → lexical_similarity (partial reward)
    - Note: Semantic similarity (LLM-based) is placeholder for future implementation

### Modified Files

- **File**: `requirements.txt`
    - Added `unidiff==0.7.5` dependency for patch parsing

- **File**: `verl/utils/reward_score/__init__.py`
    - Registered `nemotron_cascade_rl_swe` reward function in dispatcher
    - Routes requests with `data_source="nemotron_cascade_rl_swe"`

- **File**: `run_nemotron_cascade_8b_swe.sh`
    - Complete rewrite for SWE RL training
    - Hyperparameters from paper:
        - `batch_size=128`
        - `learning_rate=2.5e-6`
        - `rollouts=16`
        - `temperature=1.0`
        - `max_response_length=16384` (16K)
        - `max_prompt_length=16384` (16K)
    - Single stage training (user choice)

## Rationale
- Implements Nemotron-Cascade paper Section 4.7 (SWE RL)
- Execution-free reward function enables training without code execution sandbox
- Lexical similarity using `unidiff` library as described in paper
- Semantic similarity (LLM-based P(YES)) deferred to future implementation

## Technical Details

### Reward Function Logic
```python
def compute_reward(generated_patch, ground_truth_patch):
    # Case 1: Exact match
    if lexical_similarity == 1.0:
        return 1.0

    # Case 2: Empty patch (no changes)
    if is_empty_patch(generated_patch):
        return 0.0

    # Case 3: Invalid patch
    if not validate_patch(generated_patch):
        return -1.0

    # Case 4: Partial match
    return lexical_similarity  # or semantic_similarity (future)
```

### Dataset Format
Parquet file should contain:
- `prompt`: Problem description and buggy code
- `data_source`: `"nemotron_cascade_rl_swe"` (required)
- `ground_truth`: Ground truth patch in unified diff format
- `extra_info` (optional): `{"original_code": "...", "problem_id": "..."}`

## Future Extensions
- [ ] Semantic similarity using LLM (Kimi-Dev-72B or alternative)
- [ ] Multi-stage training (16K → 24K context expansion)
- [ ] SWE-bench dataset integration
