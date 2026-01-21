# [2026-01-21 07:30] Add tqdm logging for gradient accumulation steps

## Changes
- **File**: `verl/workers/actor/dp_actor.py`
    - Added `from tqdm import tqdm` import
    - Wrapped micro_batch loop with tqdm progress bar for gradient accumulation visualization
    - Added `set_postfix` to display current loss and pg_loss values in real-time
    - Progress bar only shown on rank 0 to avoid cluttered output in multi-GPU training

## Rationale
- Gradient accumulation steps can take a significant amount of time, especially with large models
- Without progress indication, it's difficult to monitor training progress during each mini-batch update
- tqdm provides real-time feedback on gradient accumulation progress with loss values

## Technical Details
- tqdm progress bar format: `Grad Accum (epoch X, batch Y): XX%|████| M/N [time, loss=X.XXXX, pg_loss=X.XXXX]`
- `disable=not is_rank_zero` ensures only rank 0 process shows the progress bar
- `leave=False` keeps the terminal clean by removing progress bars after completion
- `ncols=100` sets a fixed width for consistent display
