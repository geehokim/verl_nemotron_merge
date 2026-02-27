# [2026-02-18 15:10] Fix Attribution OOM: CPU Offload + Eliminate Full log_softmax

## Changes
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - **CPU offload for abs_delta + importance**: Both tensors now stay on CPU
      instead of GPU, saving ~13 GB VRAM (for 1.7B model). Gradients are moved
      GPU → CPU after each backward pass for accumulation. Overhead is ~2% since
      forward+backward dominates runtime.
    - **Eliminated full log_softmax tensor in `sequence_sum_approx`**: The naive
      `torch.log_softmax(logits[:, :-1, :].float())` materializes a full
      `[1, seq_len, vocab_size]` fp32 tensor (~14.6 GB for 24k seq × 150k vocab).
      Replaced with per-position `logit[target] - logsumexp(logits)` computation
      which is mathematically identical and uses negligible memory.
    - **Base model offload to CPU during attribution**: After `build_abs_task_vector_cpu`,
      `base_model` is moved to CPU since attribution only needs `task_model`.
      Saves additional ~model_size × 2 bytes of GPU VRAM.
    - **New helper `_accumulate_grads_to_importance_cpu`**: Extracts gradient
      accumulation logic into a reusable function that handles GPU → CPU transfer.
    - **Distributed reduction adapted for CPU buffers**: Importance tensors are
      moved to GPU one-at-a-time for NCCL reduction, then back to CPU. This uses
      only ~one parameter tensor's worth of GPU memory at a time.
    - **Reduced GC frequency**: Changed from every 64 sequences to every 32.

## Rationale
- OOM was caused by three memory hogs during attribution:
  1. `abs_delta_gpu` + `importance_gpu` on GPU: ~13 GB
  2. Full `log_softmax` tensor: ~14.6 GB for 24k-token sequences
  3. `base_model` still on GPU: ~3.4 GB (for 1.7B model)
- Combined savings: ~31 GB of GPU VRAM freed
- For 24k-token sequences with 150k vocab on A100 80GB, this is critical to
  avoid OOM while keeping the full sequence length (no truncation needed).

## Technical Details
- `log p(target) = logit[target] - logsumexp(logits)` is mathematically identical
  to indexing into `log_softmax(logits)`. This identity avoids materializing the
  full `[seq_len, vocab_size]` log_softmax tensor while preserving exact gradients.
- CPU accumulation uses `grad.detach().to(dtype=float32, device='cpu').abs_()`
  followed by in-place `mul_` and `add_` for zero-allocation accumulation.
- Distributed reduction temporarily moves each importance tensor to GPU for NCCL,
  then moves it back. Peak GPU usage for reduction: ~one parameter tensor.
- `base_model` is automatically moved back to GPU at the start of the next task
  iteration if needed for delta logp collection.

## Memory Budget (1.7B model, 24k seq, 150k vocab)
| Component                | Before  | After  |
|--------------------------|---------|--------|
| base_model (bf16)        | 3.4 GB  | 0 GB   |
| task_model (bf16)        | 3.4 GB  | 3.4 GB |
| abs_delta_gpu (fp32)     | 6.8 GB  | 0 GB   |
| importance_gpu (fp32)    | 6.8 GB  | 0 GB   |
| log_softmax (fp32)       | 14.6 GB | 0 GB   |
| logits (bf16)            | 7.2 GB  | 7.2 GB |
| Activations (w/ ckpt)    | ~3 GB   | ~3 GB  |
| Gradients (fp32)         | 6.8 GB  | 6.8 GB |
| **Total**                | **52 GB** | **20.4 GB** |
