# [2025-01-01 00:00] AIME Reward Function Integration

## Changes

- **File**: `verl/utils/reward_score/aime.py` (NEW)
    - Created new AIME-specific reward function adapted from Nemotron-Cascade evaluation code
    - Implements 5 regex patterns for answer extraction: `\boxed{}`, `**...**`, `\[...\]`, `is \(...\)`, `\[\n...\n\]`
    - Includes `math_answer_cleaning()` for answer normalization
    - Includes `extract_answer()` for multi-pattern answer extraction
    - Includes `compute_score()` as main reward function with fallback comparison chain:
        1. `math_equal()` - symbolic + numeric comparison
        2. `round_number()` - small decimal rounding
        3. `is_equal_after_calculation()` - fraction evaluation

- **File**: `verl/utils/reward_score/grader.py` (NEW)
    - Ported math grading utilities from `nemotron_evaluation/eval/tools/grader.py`
    - `math_equal()` - comprehensive mathematical equivalence checking
    - `symbolic_equal()` - SymPy-based symbolic comparison
    - `numeric_equal()` - float comparison with tolerance
    - Graceful fallback if sympy or latex2sympy unavailable

- **File**: `verl/utils/reward_score/__init__.py`
    - Added AIME data source routing (lines 63-67)
    - Supports: `"aime25"`, `"aime24"`, `"aime"`, and any `data_source.startswith("aime")`

## Rationale

- To align RL training reward function with Nemotron-Cascade's evaluation metric
- Ensures consistency between training reward and evaluation accuracy
- The original `math_dapo.py` only did simple string comparison; AIME requires robust mathematical equivalence checking

## Technical Details

- Uses 5 regex patterns in priority order (boxed > markdown > display math > inline math > escaped)
- Answer extraction uses `[-1]` to get last match (final answer)
- Comparison chain provides fallback: symbolic → rounded → calculated
- SymPy integration for symbolic math comparison (optional, graceful fallback)
- Compatible with verl's `default_compute_score()` interface

## Dependencies

- `sympy` (optional but recommended for symbolic comparison)
- `regex` (optional, falls back to `re`)

## Usage

```python
# Automatic routing via data_source
from verl.utils.reward_score import default_compute_score

result = default_compute_score(
    data_source="aime25",
    solution_str="The answer is \\boxed{42}",
    ground_truth="42"
)
# Returns: {"score": 1.0, "acc": True, "pred": "42"}
```

