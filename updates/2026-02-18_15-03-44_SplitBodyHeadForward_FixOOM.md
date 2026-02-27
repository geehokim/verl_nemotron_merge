# [2026-02-18 15:03] Split Body/Head Forward to Eliminate 22 GB Logits+Gradient OOM

## Changes
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - **New helper `_resolve_transformer_body_and_head()`**: Identifies the
      transformer body (`model.model` or `model.transformer`) and LM head
      (`model.lm_head`) sub-modules for HuggingFace CausalLM models.
    - **Split forward in `sequence_sum_approx` mode**: Instead of calling
      `model(input_ids)` which produces full `[1, seq_len, vocab_size]` logits,
      the forward is split into:
      1. `model.model(input_ids)` → hidden_states `[1, seq_len, hidden_dim]`
      2. Slice hidden_states at critical positions → `[1, N_crit, hidden_dim]`
      3. `model.lm_head(critical_hidden)` → `[1, N_crit, vocab_size]`
      This avoids materializing full logits AND their backward gradient.
    - **Fallback path**: If the model structure isn't recognized (no `model.model`
      or `model.lm_head`), falls back to the original full-forward path.
    - **Per-sequence GC**: Changed from every 32 sequences to every sequence
      (the old threshold never fired with only 16 sequences).
    - **Eager `del` statements**: All GPU tensors (`grads`, `scalar_terms`,
      `summed_scalar`, `outputs`, `logits`, `full_ids`) are explicitly deleted
      after use in all three attribution paths (split, non-split exact_token,
      sequence_sum_approx fallback).

## Rationale
- **Root cause of OOM**: In the old `sequence_sum_approx` code, all
  `scalar_terms` are sliced from the same `logits [1, seq_len, vocab_size]`
  tensor.  During backward, PyTorch's autograd creates a dense gradient for
  `logits` of shape `[1, seq_len, vocab_size]` in fp32.  For 24k tokens ×
  150k vocab, this gradient alone is **~14.4 GB**.  Combined with the logits
  tensor itself (~7.2 GB bf16), the logits pathway consumes **~22 GB**.
- **Why per-position log_softmax didn't help**: The per-position optimization
  (`logit[target] - logsumexp`) avoids the full *forward* log_softmax tensor,
  but all scalar terms still depend on the same `logits` tensor, so backward
  still creates the full dense gradient.
- **Why body/head split works**: By applying `lm_head` only to `N_crit`
  positions (typically 1000-2000), the logits tensor shrinks from
  `[1, 24000, 150000]` to `[1, 2000, 150000]` — a 12× reduction.  More
  importantly, the gradient flows back through `hidden_states` which is
  `[1, seq_len, hidden_dim=2048]` — only ~188 MB vs 14.4 GB.

## Memory Budget (1.7B model, 24k seq, 150k vocab, 2000 critical positions)
| Component                           | Before (full fwd) | After (split fwd) |
|-------------------------------------|--------------------|--------------------|
| task_model (bf16)                   | 3.4 GB             | 3.4 GB             |
| logits [1, 24k, 150k] bf16         | 7.2 GB             | 0 GB               |
| critical_logits [1, 2k, 150k] bf16 | —                  | 0.6 GB             |
| hidden_states [1, 24k, 2048] bf16  | —                  | 0.1 GB             |
| logits grad [1, 24k, 150k] fp32    | 14.4 GB            | 0 GB               |
| critical_logits grad fp32           | —                  | 1.2 GB             |
| hidden_states grad fp32             | —                  | 0.2 GB             |
| Param gradients (fp32)             | 6.8 GB             | 6.8 GB             |
| Activations (w/ checkpointing)     | ~3 GB              | ~3 GB              |
| **Total GPU peak**                 | **~35 GB**         | **~15 GB**         |

## Technical Details
- `model.model(input_ids, use_cache=False)` returns `BaseModelOutputWithPast`
  with `last_hidden_state` that already includes the final layer norm (for
  Qwen2, LLaMA, Mistral families).  This is the exact input that `lm_head`
  expects — no intermediate normalization is skipped.
- Gradient checkpointing still works: it's enabled on the transformer body
  (`model.model`), which is what we forward through.  The LM head is a
  single `nn.Linear` and doesn't need checkpointing.
- The mathematical result is identical: `lm_head(body(ids)[:, pos, :])` gives
  the same logits as `model(ids).logits[:, pos, :]` for causal LMs, because
  the LM head is a position-independent linear projection.
- Compatible with: Qwen2, LLaMA, Mistral, Gemma (`model.model` + `model.lm_head`)
  and GPT-2, GPT-J (`model.transformer` + `model.lm_head`).
