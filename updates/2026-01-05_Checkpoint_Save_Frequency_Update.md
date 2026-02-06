# [2026-01-05] Checkpoint Save Frequency Update

## Changes
- **File**: `run_nemotron_cascade_8b_math_debug.sh`
    - Changed `trainer.save_freq` from 100 to 5
    - Now saves checkpoint every 5 iterations instead of 100

## Rationale
- Enable more frequent checkpointing for easier resume and debugging
- Checkpoints are saved with iteration number in folder name (`global_step_X`)
- Each checkpoint includes actor weights and optimizer state for full resumability

## Technical Details
- VERL framework saves checkpoints to: `$OUTPUT_DIR/stageX/global_step_N/`
- Each checkpoint contains:
  - `actor/` - Model weights
  - Optimizer state (automatically saved by FSDP)
- Resume capability: Use `actor_rollout_ref.model.path=$CKPT_DIR/actor` to resume from any checkpoint

