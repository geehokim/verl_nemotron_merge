# [2026-02-18 14:44] Fix Memory Leak: Per-Sequence GPU Cleanup + Eager Tensor Deletion

## Changes
- **File**: `scripts/run_jwcm08_importance_only_distributed.py`
    - **Fixed GC never triggering**: The previous `processed_sequences_local % 32 == 0`
      condition never fired because there are only 16 sequences (none is a multiple
      of 32). Changed to run `gc.collect()` + `torch.cuda.empty_cache()` after
      **every** sequence.
    - **Added explicit `del` for GPU tensors in `sequence_sum_approx` mode**: After
      backward pass and CPU accumulation, `scalar_terms`, `summed_scalar`, and `grads`
      are now explicitly deleted before `outputs`/`logits`. This releases Python
      references to intermediate computation graph nodes, allowing PyTorch's CUDA
      allocator to reclaim the underlying GPU storage.
    - **Added explicit `del grads` in both `exact_token_abs` modes**: In both the
      split and non-split paths, gradients are now deleted immediately after CPU
      accumulation to free GPU memory sooner.

## Rationale
- GPU memory was monotonically climbing from sequence to sequence (68 GB → 80+ GB)
  because:
  1. `gc.collect()` + `torch.cuda.empty_cache()` never executed (16 sequences < 32
     threshold), so PyTorch's CUDA allocator retained all freed blocks in its cache.
  2. Python variables (`scalar_terms`, `summed_scalar`, `grads`) held references to
     dead GPU tensors until the next loop iteration reassigned them, preventing the
     allocator from reusing that storage.
- With 24k-token sequences and 150k vocab, each forward pass allocates ~7+ GB of
  logits alone. Without per-sequence cleanup, memory grows by several GB per
  iteration.

## Technical Details
- `torch.cuda.empty_cache()` tells PyTorch's CUDA memory allocator to return all
  unused cached blocks back to CUDA. Without this call, the allocator keeps blocks
  in a free-list for potential reuse, but fragmentation prevents effective reuse
  when allocation sizes vary across sequences.
- `gc.collect()` ensures Python's garbage collector runs cycle detection to break
  any reference cycles involving GPU tensors (e.g., closures in autograd graph).
- The overhead of per-sequence cleanup is negligible (~ms) compared to the forward
  + backward pass (~25 seconds per sequence).
