# [2026-01-04] Nemotron-Cascade Math RL Training Recipe Implementation

## Overview

Implemented the complete Nemotron-Cascade paper's Math RL training recipe with:
- 3-stage curriculum (24K→32K→40K) via **sequential bash script execution**
- Dynamic filtering (epoch-based accuracy-aware sampling)
- Overlong response filtering (skip flag in reward function)

## Design Decision: Sequential Bash Execution

Instead of creating a custom trainer with dynamic stage switching,
we use a simpler approach: **run 3 sequential training stages via bash script**.

**Advantages:**
1. ✅ No code modification to existing trainer (uses proven RayPPOTrainer)
2. ✅ More stable and predictable behavior
3. ✅ Each stage can be independently monitored and debugged
4. ✅ Easy to resume from any stage checkpoint
5. ✅ Clear separation of concerns

## Files Created

### 1. `verl/experimental/dataset/nemotron_cascade_sampler.py`
- `NemotronCascadeCurriculumSampler` class
- Tracks per-problem accuracy across rollouts
- At epoch end: filters 100%/0% acc problems, resamples with probability
  - Hard problems (0% acc): 10% probability of resampling
  - Easy problems (100% acc): 1% probability of resampling

### 2. `run_nemotron_cascade_8b_math.sh`
- **3-stage sequential training script**
- Stage 1: Steps 0-190, max_len=24000, temp=1.0, overlong_filtering=True
- Stage 2: Steps 190-430, max_len=32000, temp=1.0, overlong_filtering=False  
- Stage 3: Steps 430-500, max_len=40000, temp=0.8, overlong_filtering=False
- Each stage resumes from previous stage's checkpoint
- batch_size=128, n=8, lr=2e-6, betas=[0.9,0.95]

## Files Modified

### 1. `verl/workers/reward_manager/naive.py`
- Added `overlong_filtering` and `max_response_length` parameters to `__init__`
- Added `response_length`, `overlong_filtering`, `max_response_length` to `extra_info`
- Added `index` for curriculum sampler tracking

### 2. `verl/utils/reward_score/nemotron_cascade_rl_math.py`
- Added overlong filtering support with `skip` flag
- Added `overlong` flag to track response length status
- Samples with `skip=True` won't contribute to policy gradient

### 3. `verl/trainer/ppo/ray_trainer.py`
- Added skip mask handling after advantage computation
- Zeros out advantages and returns for skipped samples
- Logs `nemotron_cascade/skipped_overlong_samples` and `nemotron_cascade/skip_ratio`

### 4. `verl/experimental/dataset/__init__.py`
- Added `NemotronCascadeCurriculumSampler` to exports

## Stage Configuration (8B Model)

| Stage | Steps | max_response_length | temperature | top_p | overlong_filtering |
|-------|-------|---------------------|-------------|-------|-------------------|
| 1 | 0-190 | 24000 | 1.0 | 0.95 | True |
| 2 | 190-430 | 32000 | 1.0 | 0.95 | False |
| 3 | 430-500 | 40000 | 0.8 | 0.95 | False |

## Execution Flow

```bash
# Stage 1: Train from base model
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    data.max_response_length=24000 \
    trainer.total_training_steps=190

# Stage 2: Resume from Stage 1 checkpoint
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=$STAGE1_CKPT/actor \
    data.max_response_length=32000 \
    trainer.total_training_steps=240

# Stage 3: Resume from Stage 2 checkpoint
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=$STAGE2_CKPT/actor \
    data.max_response_length=40000 \
    trainer.total_training_steps=70
```

## Reference

Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models

