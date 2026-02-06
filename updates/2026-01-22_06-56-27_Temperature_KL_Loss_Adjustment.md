# [2026-01-22 06:56] Temperature and KL Loss Adjustment

## Changes
- **File**: `run_nemotron_distill_1,5b_math.sh`
  - Enabled KL loss regularization: `use_kl_loss=True`, `kl_loss_coef=0.001`
  - Adjusted temperature for Stage 2: `1.0` → `0.8` (more deterministic sampling)
  - Adjusted temperature for Stage 3: `0.8` → `0.7` (further refinement)
  - Updated documentation comments and echo messages to reflect new settings

## Rationale
- **KL Loss Regularization**: Adding KL divergence penalty (0.001) helps prevent the actor model from deviating too far from the reference model, improving training stability and preventing mode collapse in RL training.
- **Temperature Reduction**: Lowering temperature makes the sampling more deterministic and focused, which is beneficial for math reasoning tasks where precision is important. The curriculum approach (0.8 → 0.7) gradually refines the model's output quality.

## Technical Details
- KL loss coefficient: 0.001 (standard value for PPO/GRPO training)
- Temperature schedule: Stage 2 (0.8), Stage 3 (0.7)
- KL loss is computed between actor and reference model distributions
- Lower temperature reduces entropy in the output distribution, leading to more confident and consistent responses
