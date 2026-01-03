# [2026-01-04 15:30] Nemotron-Cascade Overlong Filtering Implementation

## Changes
- **File**: `verl/trainer/ppo/ray_trainer.py`
    - skip==True인 샘플의 advantage를 0으로 설정하여 policy gradient에 기여하지 않게 함
    - 메트릭 기록: `nemotron_cascade/skipped_overlong_samples`, `nemotron_cascade/skip_ratio`

## Rationale
Nemotron-Cascade 논문의 Stage 1에서는 max_response_length에 도달한 (truncated) 응답을 
policy gradient 업데이트에서 제외합니다. 이를 구현하기 위해:

```python
# 1. 모든 샘플로 advantage 계산
batch = compute_advantage(batch, ...)

# 2. skip된 샘플의 advantage를 0으로 설정
if "skip" in batch.non_tensor_batch:
    batch.batch["advantages"][skip_mask] = 0.0
    batch.batch["returns"][skip_mask] = 0.0
```

**Note**: skip된 샘플의 reward는 여전히 GRPO mean/std 계산에 포함됩니다.
하지만 advantage=0이면 policy gradient에 기여하지 않으므로 실제 학습에는 영향이 없습니다.

## Technical Details
- reward function에서 `skip=True`를 반환하면 해당 샘플은 policy gradient에서 제외됨
- 배치 사이즈는 변경되지 않음 (다른 코드 부분에 영향 없음)
- 메트릭으로 skip된 샘플 수와 비율을 WandB에 기록

