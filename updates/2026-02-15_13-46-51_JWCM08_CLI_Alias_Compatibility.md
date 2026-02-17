# [2026-02-15 13:46] JWCM08 CLI Alias Compatibility

## Changes
- **File**: `scripts/run_jwcm08_importance_only.py`
    - Added CLI alias `--max-response-tokens` for `--max-new-tokens`.
    - Added CLI alias `--max-rollout-rows` for `--max-total-rollouts`.
    - Added support for `--max-response-tokens -1` (no response truncation) by making response tokenization truncation optional.
    - Added support for `--max-prompt-tokens -1` (no prompt truncation) for consistency.
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - Added CLI alias `--max-response-tokens` for `--max-new-tokens`.
    - Added CLI alias `--max-rollout-rows` for `--max-total-rollouts`.
    - Updated config parsing to pass optional truncation limits (`-1` -> no truncation).

## Rationale
- User workflows were already using `--max-rollout-rows` and `--max-response-tokens` naming from related scripts.
- Alias support removes friction and keeps backward-compatible CLI behavior.
- `-1` semantics are required for untruncated response processing in attribution runs.

## Technical Details
- Implemented argparse aliases via shared destination (`dest`) fields.
- Converted truncation limits with sentinel handling (`-1` to `None`) before tokenization.
- Updated rollout tokenization logic to switch between truncated and non-truncated tokenizer calls based on optional limits.
