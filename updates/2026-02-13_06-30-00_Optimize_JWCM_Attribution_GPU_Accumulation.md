# [2026-02-13 06:30] Optimize JWCM v2 Attribution: GPU-Resident Accumulation

## Changes
- **File**: `merging_analysis/04_jwcm_v2_soft_attribution_merge.ipynb` (cell `dc075c50`)
    - Rewrote `compute_importance_scores()` with 4 performance optimizations while preserving mathematical equivalence.

## Rationale
- `exact_token_abs` mode was extremely slow due to per-token CPU↔GPU gradient transfers.
- Each backward pass triggered ~310 synchronous `grad.cpu()` calls (one per parameter), creating thousands of CUDA synchronization barriers per sequence.

## Technical Details

### Optimization 1: GPU-Resident Accumulation Buffers (biggest impact)
- **Before**: `abs_delta` and `importance` on CPU → every backward requires 310× `grad.detach().float().cpu()` (sync GPU→CPU transfer)
- **After**: Both buffers on GPU → accumulation is pure GPU ops, single CPU transfer at the very end
- **Why it matters**: Each `.cpu()` is a CUDA sync point. For N critical tokens × 310 params = 310N sync barriers eliminated.

### Optimization 2: `torch.autograd.grad` replaces `.backward()` + `.zero_grad()`
- **Before**: `zero_grad(set_to_none=True)` iterates 310 params + `backward()` + read `.grad` attributes
- **After**: `autograd.grad()` returns gradient tensors directly as a tuple, no `.grad` population needed

### Optimization 3: Selective log_softmax at Critical Positions Only
- **Before**: `log_softmax(logits[:, :-1, :])` over full sequence (seq_len × vocab)
- **After**: `log_softmax(logits[0, shifted_positions, :])` only at N_critical positions
- Reduces retained graph size for `retain_graph=True` backward passes

### Optimization 4: In-Place Operations
- `g.to(float32).abs_().mul_(abs_delta_gpu[name])` uses in-place ops on transient tensors
- Only one temporary tensor per parameter per backward (instead of 2-3)

### Mathematical Equivalence Proof
- `S_j = Σ_t |grad_j^(t)| * |Δθ_j|` is computed identically:
  - Same per-token backward passes with `retain_graph`
  - Same `|grad| * |Δθ|` accumulation formula
  - Same normalization by token count at the end
  - Only the device placement (GPU vs CPU) differs for intermediate accumulation

### GPU Memory Budget (Qwen3-1.7B)
- Model (bf16): ~3.4 GB
- abs_delta_gpu (fp32): ~6.8 GB
- importance_gpu (fp32): ~6.8 GB
- Gradients (fp32, temp): ~6.8 GB
- Forward activations: ~2-4 GB
- **Total peak: ~26-28 GB** (fits on 40GB+ GPUs)
