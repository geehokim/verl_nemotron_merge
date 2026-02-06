# Nemotron-Cascade Overlong Filtering 구현

**작성일**: 2026-01-05  
**작성자**: 자동 생성  
**관련 논문**: Nemotron-Cascade Technical Report

---

## 📋 개요

Nemotron-Cascade 논문의 **Stage 1 RL 학습**에서는 `max_response_length`에 도달하여 truncate된 응답을 policy gradient 업데이트에서 제외합니다. 이를 **Overlong Filtering**이라고 합니다.

### 왜 필요한가?

1. **Truncated 응답의 불완전성**: vLLM이 `max_response_length`에서 강제로 잘라낸 응답은 완전한 추론 결과가 아님
2. **잘못된 gradient 방지**: 불완전한 응답에 reward를 부여하면 policy가 잘못된 방향으로 학습될 수 있음
3. **Stage 1 전용**: 이후 Stage에서는 max_response_length를 줄이면서 모델이 짧은 응답을 생성하도록 유도

---

## 🏗️ 구현 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Rollout (vLLM)                             │
│  - response 생성                                                    │
│  - response_length 측정                                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Reward Manager (naive)                           │
│  - compute_score() 호출                                              │
│  - overlong 체크: response_length >= max_response_length            │
│  - skip=True 반환 (overlong인 경우)                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Ray Trainer (PPO/GRPO)                          │
│  - compute_advantage() 실행                                          │
│  - skip된 샘플의 advantage = 0으로 마스킹                             │
│  - policy gradient에서 자동 제외 (gradient = 0)                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 파일별 구현 상세

### 1. Reward Function (`verl/utils/reward_score/nemotron_cascade_rl_math.py`)

```python
def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    return_dict: bool = True,
) -> Union[float, dict]:
    
    # extra_info에서 overlong filtering 설정 읽기
    overlong_filtering = False
    response_length = 0
    max_response_length = float("inf")
    
    if extra_info and isinstance(extra_info, dict):
        overlong_filtering = extra_info.get("overlong_filtering", False)
        response_length = extra_info.get("response_length", 0)
        max_response_length = extra_info.get("max_response_length", float("inf"))
    
    # Overlong 체크: response가 max_response_length에 도달했는지 확인
    is_overlong = (response_length >= max_response_length) if max_response_length != float("inf") else False
    
    if overlong_filtering and is_overlong:
        # skip=True를 반환하여 이 샘플을 policy gradient에서 제외
        if return_dict:
            return {
                "score": 0.0,
                "acc": None,              # accuracy 계산 안 함
                "pred": "",
                "code_switching": False,
                "extraction_failed": False,
                "answer_reward": 0.0,
                "code_switching_penalty": 0.0,
                "skip": True,             # ⭐ 핵심: policy gradient 제외 마커
                "overlong": True,
            }
        return 0.0
    
    # ... 정상 reward 계산 로직 ...
```

**핵심 포인트**:
- `response_length >= max_response_length` 조건으로 truncate 여부 판단
- `skip=True`를 반환하여 downstream에서 처리

---

### 2. Ray Trainer (`verl/trainer/ppo/ray_trainer.py`)

```python
# compute_advantage() 이후에 skip 마스킹 적용

# =====================================================================
# Nemotron-Cascade Overlong Filtering
# skip==True인 샘플의 advantage를 0으로 설정하여 policy gradient에 기여하지 않게 함
# Note: skip된 샘플의 reward는 여전히 GRPO mean/std 계산에 포함됨
# =====================================================================
if "skip" in batch.non_tensor_batch:
    skip_flags = batch.non_tensor_batch["skip"]
    if not isinstance(skip_flags, np.ndarray):
        skip_flags = np.array(skip_flags)
    skip_mask = torch.tensor(
        skip_flags, dtype=torch.bool, device=batch.batch["advantages"].device
    )
    
    if skip_mask.any():
        # skip된 샘플의 advantage/returns를 0으로 설정
        batch.batch["advantages"][skip_mask] = 0.0
        batch.batch["returns"][skip_mask] = 0.0
        
        # 메트릭 기록
        skipped_count = skip_mask.sum().item()
        metrics["nemotron_cascade/skipped_overlong_samples"] = skipped_count
        metrics["nemotron_cascade/skip_ratio"] = skipped_count / len(skip_mask)
```

**핵심 포인트**:
- advantage = 0이면 `policy_gradient = advantage * log_prob_ratio = 0`
- 배치 크기는 유지하면서 gradient 기여도만 0으로 만듦
- WandB에 skip된 샘플 수와 비율 기록

---

### 3. 설정 (`run_nemotron_cascade_8b_math_debug.sh`)

```bash
# Stage 1: overlong_filtering=True
STAGE1_ARGS=(
    "${COMMON_ARGS[@]}"
    data.max_response_length=24000
    actor_rollout_ref.rollout.temperature=1.0
    +data.reward_fn.extra_info.overlong_filtering=True   # ⭐ 활성화
    +data.reward_fn.extra_info.max_response_length=24000
    trainer.total_epochs=2
    trainer.experiment_name=$EXPERIMENT_NAME_S1
)

# Stage 2, 3: overlong_filtering=False
STAGE2_ARGS=(
    "${COMMON_ARGS[@]}"
    data.max_response_length=18000
    actor_rollout_ref.rollout.temperature=0.6
    +data.reward_fn.extra_info.overlong_filtering=False  # ⭐ 비활성화
    ...
)
```

---

## 📊 모니터링 메트릭

WandB에서 확인할 수 있는 메트릭:

| 메트릭 | 설명 |
|--------|------|
| `nemotron_cascade/skipped_overlong_samples` | 해당 batch에서 skip된 샘플 수 |
| `nemotron_cascade/skip_ratio` | skip된 샘플 비율 (0.0 ~ 1.0) |

**정상 범위**:
- Stage 1 초반: skip_ratio가 높을 수 있음 (긴 응답 생성)
- Stage 1 후반: skip_ratio가 점점 감소해야 함 (짧은 응답 학습)

---

## 🔄 DAPO Overlong Reward Shaping과의 비교

| 특성 | Overlong Filtering (Nemotron) | Overlong Reward Shaping (DAPO) |
|------|------------------------------|--------------------------------|
| **방식** | Policy gradient에서 완전 제외 | 음수 reward 페널티 |
| **영향** | Gradient = 0 | Gradient ≠ 0 (음수 방향) |
| **설정** | `overlong_filtering=True` | `overlong_buffer.enable=True` |
| **적용 시점** | max_length 도달 시 | max_length - buffer ~ max_length 구간 |
| **사용 단계** | Stage 1 only | 모든 Stage 가능 |

DAPO 스타일 Overlong Reward Shaping 설정 (참고용):
```bash
reward_model.overlong_buffer.enable=True
reward_model.overlong_buffer.len=4096
reward_model.overlong_buffer.penalty_factor=1.0
```

---

## ⚠️ 주의사항

1. **Stage 1에서만 사용**: Stage 2, 3에서는 `overlong_filtering=False`로 설정
2. **GRPO 통계에는 포함**: skip된 샘플의 reward는 여전히 baseline 계산에 포함됨
3. **메모리 영향 없음**: 배치 크기는 동일하게 유지됨

---

## 🔗 관련 파일

- `verl/utils/reward_score/nemotron_cascade_rl_math.py`: Reward function 및 skip 로직
- `verl/trainer/ppo/ray_trainer.py`: Advantage 마스킹 로직
- `run_nemotron_cascade_8b_math_debug.sh`: 학습 스크립트 설정

