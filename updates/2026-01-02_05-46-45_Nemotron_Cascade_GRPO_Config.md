# [2026-01-02 05:46] feat: Add Nemotron-Cascade GRPO training config and reward function

## Summary

Implemented GRPO training configuration following Nemotron-Cascade paper Appendix D.4 (Table 17) hyperparameters and added corresponding reward function for math reasoning evaluation.

## Changes

### 1. **File**: `run_qwen3-8b-math-after-rlhf.sh`
- **Added**: Complete GRPO training script with Nemotron-Cascade Appendix D.4 hyperparameters
- **Hyperparameters Applied**:
  | Parameter | Value | Notes |
  |-----------|-------|-------|
  | Algorithm | GRPO (On-policy) | `algorithm.adv_estimator=grpo` |
  | Global Batch Size | 128 → 512 | Adjusted for multi-GPU |
  | Rollout | 8 responses/prompt | `actor_rollout_ref.rollout.n=8` |
  | Learning Rate | 2e-6 | With cosine scheduler |
  | Optimizer | AdamW | betas=(0.9, 0.95) |
  | KL Coefficient | 0.0 | Strictly zero |
  | Entropy Coefficient | 0.0 | Strictly zero |
  | Max Response Length | 24,000 tokens | For long reasoning chains |
  | Precision | bf16 | Default FSDP dtype |
- **Environment Setup**: Added HuggingFace cache paths, wandb integration
- **Model Path**: `/mnt/ddn/vuvlm/geeho/models/Nemotron-Cascade-8B-Intermediate-ckpts/Nemotron-Cascade-8B-RLHF`

### 2. **File**: `verl/utils/reward_score/nemotron_cascade_rl_math.py` (NEW - 458 lines)
- **Added**: Complete reward function implementation following the paper's specification
- **Key Functions**:
  - `extract_boxed_after_think()`: Extracts `\boxed{}` answer after `</think>` token
  - `detect_code_switching()`: FastText-based language detection for penalty
  - `verify_answer()`: 3-stage verification using AceMath rule-based verifier
  - `compute_score()`: Main reward computation function

- **Reward Logic** (Additive):
  ```
  total_reward = answer_reward + code_switching_penalty
  
  answer_reward: 1.0 (correct) or 0.0 (incorrect)
  code_switching_penalty: 0.0 (no code-switching) or -1.0 (detected)
  
  Examples:
    Correct + No code-switching: 1.0 + 0.0 = 1.0
    Correct + Code-switching:    1.0 + (-1.0) = 0.0
    Incorrect + No code-switching: 0.0 + 0.0 = 0.0
    Incorrect + Code-switching:    0.0 + (-1.0) = -1.0
  ```

### 3. **File**: `verl/utils/reward_score/__init__.py`
- **Added**: Registration of `nemotron_cascade` data source
- Mapped to `compute_score` function from `nemotron_cascade_rl_math.py`

### 4. **File**: `verl/utils/reward_score/aime.py`
- **Modified**: Minor updates for helper function compatibility

### 5. **File**: `verl/experimental/agent_loop/agent_loop.py`
- **Modified**: Improved handling for agent loop processing

## Rationale

1. **Paper Reproduction**: To exactly reproduce the Nemotron-Cascade paper's RL Math training stage, we need the precise hyperparameters from Appendix D.4 (Table 17).

2. **Reward Function Design**: The paper specifies:
   - Extract answer from `\boxed{}` after `</think>` token
   - Use AceMath-style rule-based verifier (1 for correct, 0 for incorrect)
   - Apply -1.0 penalty for code-switching (detecting tokens from a different language)

3. **3-Stage Verification**: The verification uses three methods in sequence:
   - `math_equal`: Symbolic + numeric comparison via SymPy
   - `round_number`: Decimal rounding comparison for floating-point
   - `is_equal_after_calculation`: LaTeX fraction evaluation

## Technical Details

### Dependencies
- `fasttext-langdetect`: For code-switching detection (FastText language identification)
- `sympy`: For symbolic math comparison in `math_equal`
- `verl.utils.reward_score.grader`: AceMath-style math equality checker

### GRPO Configuration
```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: False  # KL strictly 0

actor_rollout_ref:
  actor:
    use_kl_loss: False
    kl_loss_coef: 0.0
    entropy_coeff: 0.0
    optim:
      lr: 2e-6
      lr_scheduler_type: cosine
      betas: [0.9, 0.95]
  rollout:
    n: 8  # 8 responses per prompt
    name: vllm

data:
  train_batch_size: 128  # Global batch size
  max_response_length: 24000  # For extended reasoning
```

### Reward Function Return Format
```python
{
    "score": float,  # Total reward (answer + penalty)
    "acc": bool,     # Answer correctness
    "pred": str,     # Extracted answer
    "code_switching": bool,  # Whether code-switching detected
    "extraction_failed": bool,  # Whether \boxed{} extraction failed
    "answer_reward": float,  # 1.0 or 0.0
    "code_switching_penalty": float  # 0.0 or -1.0
}
```

## References

- Nemotron-Cascade Paper: Appendix D.4, Table 17
- AceMath Verifier: Rule-based mathematical equality checking
- FastText Language Detection: For code-switching penalty

