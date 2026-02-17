# [2026-02-15 13:52] Notebook05 Defaults To Unlimited

## Changes
- **File**: `scripts/run_notebook05_fisher_from_rollouts.py`
    - Changed default value of `--max-prompt-tokens` from `1536` to `-1`.
    - This aligns default Fisher run behavior with unlimited-token settings (`-1` means no truncation limit).

## Rationale
- User requested that all relevant cap defaults be set to `-1` so runs are full/unbounded by default.
- `--max-response-tokens` and `--max-rollout-rows` were already `-1`; prompt truncation default remained capped and is now aligned.

## Technical Details
- The script already converts negative limits to `None` via `_int_to_optional_limit`, so tokenizer calls run with `truncation=False` when defaults are `-1`.
