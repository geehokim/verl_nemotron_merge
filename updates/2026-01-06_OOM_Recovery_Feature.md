# [2026-01-06] OOM Recovery and Auto-Resume Feature for Training Script

## Changes
- **File**: `run_nemotron_cascade_8b_math_debug.sh`
    - Added automatic OOM recovery and checkpoint resume functionality
    - Added new configuration variables for OOM recovery:
        - `MAX_RETRIES=10`: Maximum retry attempts per stage on OOM/crash
        - `RETRY_DELAY_SECONDS=30`: Delay between retries to allow GPU memory clearing
        - `SAVE_FREQ=1`: Checkpoint save frequency (every iteration for OOM safety)
        - `MAX_CKPT_TO_KEEP=3`: Maximum checkpoints to keep per stage
    - Added verl trainer settings for checkpoint management:
        - `trainer.resume_mode=auto`: Automatically resume from latest checkpoint
        - `trainer.save_freq=$SAVE_FREQ`: Save checkpoint every N iterations
        - `trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP`: Limit checkpoint storage
        - `trainer.max_critic_ckpt_to_keep=$MAX_CKPT_TO_KEEP`: Limit checkpoint storage
    - Implemented helper functions:
        - `is_oom_error()`: Detects OOM errors from exit codes and log files
        - `clear_gpu_memory()`: Clears GPU cache before retry attempts
        - `get_latest_checkpoint()`: Finds the most recent checkpoint for a stage
        - `run_stage_with_retry()`: Main retry logic wrapper for each training stage
    - Refactored training stages to use the new retry mechanism
    - Added comprehensive logging to track training progress and failures
    - Removed `set -e` to handle errors manually for retry logic

## Rationale
- OOM (Out-Of-Memory) errors are common during large-scale RL training, especially with long context lengths
- Without automatic recovery, training would need to be manually restarted, losing progress
- The Nemotron-Cascade training uses progressively longer response lengths (24K → 32K → 40K tokens), increasing OOM risk in later stages
- `save_freq=1` ensures minimal loss of progress (at most 1 iteration) on any crash
- `resume_mode=auto` leverages verl's built-in checkpoint loading for seamless recovery

## Technical Details
- **Checkpoint Contents Saved**:
    - Model state (sharded across ranks)
    - Optimizer state (including momentum buffers)
    - LR scheduler state
    - Dataloader state (for exact data position resumption)
    - Iteration number (global_step)
- **OOM Detection**:
    - Exit code 137 (SIGKILL from OOM killer)
    - Log pattern matching: "CUDA out of memory", "OutOfMemoryError", etc.
- **Resume Mechanism**:
    - verl's `_load_checkpoint()` restores all training state
    - `find_latest_ckpt_path()` finds the most recent `global_step_*` folder
    - StatefulDataLoader resumes from exact batch position
- **GPU Memory Cleanup**:
    - `torch.cuda.empty_cache()` on all devices
    - `torch.cuda.synchronize()` to ensure completion
    - Additional 5-second delay for memory deallocation








