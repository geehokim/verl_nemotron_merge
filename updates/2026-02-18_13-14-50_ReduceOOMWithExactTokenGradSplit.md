# [2026-02-18 13:14] reduce oom with exact token gradient split

## Changes
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - Added CLI flag `--exact-token-grad-split` to enable memory-safe gradient splitting for `exact_token_abs`.
    - Added helper functions to compute scalar token log-probabilities without materializing full `log_softmax` tensors.
    - Added prefix-only exact-token forward path that preserves causal conditional semantics while reducing peak graph/logit memory.
    - Set `use_cache=False` during attribution forwards to avoid unnecessary KV-cache memory usage.
    - Threaded the new split flag from runtime config through attribution calls and metadata summaries.
- **File**: `scripts/run_jwcm08_importance_only.py`
    - Added helper to gather target token log-probs from logits via `logsumexp` identity.
    - Refactored rollout delta collection to compute RL/Base log-probs sequentially (instead of holding both full logits simultaneously).
    - Set `use_cache=False` in rollout delta forwards to reduce memory pressure without changing outputs.

## Rationale
- Long response sequences caused OOM from large retained graphs and large logits/log-softmax tensors.
- The new exact-token split path keeps the same causal conditional objective (`log p(y_t | prefix)`) while reducing peak memory.
- Sequential RL/Base log-prob extraction removes avoidable concurrent memory pressure in token-delta collection.

## Technical Details
- PyTorch autograd with per-token scalar objective and `retain_graph=False` in split mode.
- Stable scalar log-prob computation: `logit[target] - logsumexp(logits)`.
- Optional model API detection for last-logits-only output (`num_logits_to_keep` / `logits_to_keep`) via forward signature introspection.
