# [2026-02-18 14:30] Speed Up Attribution Backward Pass

## Changes
- **File**: `scripts/run_jwcm08_importance_only.py`
    - Modified `load_causal_lm` to accept optional `attn_implementation` parameter
      (e.g., `"flash_attention_2"`, `"sdpa"`, `"eager"`), passed through to
      `AutoModelForCausalLM.from_pretrained`. Backwards compatible (defaults to `None`).

- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - **New CLI flags**:
        - `--attn-implementation {flash_attention_2,sdpa,eager}`: Select attention
          backend at model load time. Flash Attention 2 gives 2-4x speedup on both
          forward and backward passes.
        - `--gradient-checkpointing`: Enable gradient checkpointing during attribution.
          Trades ~30% extra compute for significantly less activation memory, enabling
          non-split mode on longer sequences.
        - `--freeze-zero-delta-params`: Freeze parameters whose `|Δθ|` is exactly zero.
          Skips gradient computation for unchanged parameters (e.g., frozen embeddings
          or LM head during RL fine-tuning).
    - **New helper functions**:
        - `freeze_zero_delta_parameters()`: Selectively sets `requires_grad_(False)` on
          parameters with zero abs_delta and logs the frozen count/percentage.
        - `unfreeze_all_parameters()`: Restores all parameters to `requires_grad=True`
          after attribution completes.
    - **`compute_importance_scores_distributed` optimizations**:
        - Added `freeze_zero_delta` and `enable_gradient_checkpointing` parameters.
        - Sequences are sorted by critical token count (ascending) so the progress bar
          updates more frequently and early stopping via `--max-backprop-tokens` is
          more effective.
        - Added inner per-token progress bar (`tqdm`) to show real progress within
          sequences, since the outer bar only updates per-sequence.
        - Outer progress bar now shows `tokens=N, seq_len=M, done_tok=K` postfix.
        - Model state (grad checkpointing, requires_grad) is cleaned up after
          attribution to avoid side effects.
    - **`main()` updates**:
        - New optimization flags are parsed, logged, recorded in run_summary, and
          passed to `load_causal_lm` and `compute_importance_scores_distributed`.

## Rationale
- The `exact_token_abs` attribution mode requires one backward pass per critical
  token. For N critical tokens across 16 sequences, this means N full backward
  passes through the entire model. With a 1.7B+ parameter model and hundreds of
  critical tokens, this can take hours.
- Flash Attention 2 speeds up the attention computation (dominant cost in both
  forward and backward) by 2-4x via hardware-aware memory tiling.
- Freezing zero-delta parameters eliminates gradient computation for parameters
  that contribute nothing to importance (`|grad| * 0 = 0`), reducing backward cost
  proportionally to the fraction of frozen parameters.
- Gradient checkpointing reduces activation memory, enabling non-split mode (one
  forward + N backward with retain_graph) which avoids N redundant forward passes.
- Sorting sequences by token count and adding inner progress bars improves UX and
  allows users to monitor real progress.

## Technical Details
- Flash Attention 2 requires the `flash_attn` pip package; `sdpa` uses PyTorch's
  built-in SDPA kernel and requires no additional dependencies.
- `gradient_checkpointing_enable(use_reentrant=False)` is used for compatibility
  with modern PyTorch versions.
- The freeze optimization only freezes parameters with exactly zero abs_delta sum;
  near-zero but nonzero parameters are kept to preserve mathematical correctness.
- All optimizations are opt-in via CLI flags to maintain backwards compatibility.

## Recommended Usage
```bash
torchrun --nproc_per_node=4 scripts/run_jwcm08_importance_only_distributed.py \
    --task if \
    --attn-implementation flash_attention_2 \
    --gradient-checkpointing \
    --freeze-zero-delta-params \
    --exact-token-grad-split \
    --max-backprop-tokens 500
```

For maximum speed (at cost of per-token exactness):
```bash
    --attribution-mode sequence_sum_approx
```
