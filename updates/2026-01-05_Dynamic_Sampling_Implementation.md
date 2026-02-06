# Nemotron-Cascade Dynamic Filtering (Curriculum Sampler) 구현

**작성일**: 2026-01-05  
**작성자**: 자동 생성  
**관련 논문**: Nemotron-Cascade Technical Report

---

## 📋 개요

Nemotron-Cascade 논문의 **Dynamic Filtering** 전략을 구현한 Curriculum Sampler입니다.  
매 epoch 후 문제별 정확도를 분석하여 너무 쉽거나 너무 어려운 문제를 동적으로 필터링합니다.

### 핵심 아이디어

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Dynamic Filtering Logic                         │
├─────────────────────────────────────────────────────────────────────┤
│ After each epoch:                                                   │
│   • 100% accuracy problems → Filter (1% resample probability)       │
│   • 0% accuracy problems   → Filter (10% resample probability)      │
│   • 중간 난이도 문제       → Keep (active set)                      │
└─────────────────────────────────────────────────────────────────────┘
```

**왜 필요한가?**
1. **학습 효율성**: 이미 완벽히 푸는 문제에 compute 낭비 방지
2. **난이도 균형**: 너무 어려운 문제에서 잘못된 패턴 학습 방지
3. **적응형 커리큘럼**: 모델 능력에 맞게 데이터셋이 동적으로 변화

---

## 🏗️ 구현 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Epoch N Training                               │
│  - 문제 샘플링 (active_indices에서)                                  │
│  - Rollout 생성 (n=8)                                               │
│  - Reward 계산 (acc 값 포함)                                         │
│  - sampler.update(batch) 호출 → accuracy 추적                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ (epoch_size만큼 샘플 봤으면)
┌─────────────────────────────────────────────────────────────────────┐
│                      _on_epoch_end()                                │
│  1. 문제별 평균 정확도 계산                                          │
│  2. 0% acc → filtered_hard (10% 확률로 resample)                    │
│  3. 100% acc → filtered_easy (1% 확률로 resample)                   │
│  4. 중간 → new_active                                               │
│  5. active_indices 업데이트                                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Epoch N+1 Training                             │
│  - 필터링된 active_indices에서 샘플링                                │
│  - 더 적절한 난이도의 문제에 집중!                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 파일 구조

```
verl/experimental/dataset/
├── __init__.py                       # NemotronCascadeCurriculumSampler export
├── sampler.py                        # AbstractCurriculumSampler base class
└── nemotron_cascade_sampler.py       # 🎯 핵심 구현
```

---

## 💻 핵심 코드

### 1. 초기화 (`__init__`)

```python
class NemotronCascadeCurriculumSampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Sized, data_config: DictConfig):
        # Resample 확률 (논문 기준)
        self.hard_resample_prob = data_config.get("hard_resample_prob", 0.10)  # 10%
        self.easy_resample_prob = data_config.get("easy_resample_prob", 0.01)  # 1%
        
        # 문제별 정확도 추적: {problem_idx: [acc_1, acc_2, ...]}
        self.problem_accuracy: dict[int, list[float]] = defaultdict(list)
        
        # Active indices (처음엔 전체 데이터셋)
        self.active_indices: list[int] = list(range(len(data_source)))
        self.filtered_hard: list[int] = []
        self.filtered_easy: list[int] = []
```

### 2. 정확도 업데이트 (`update`)

```python
def update(self, batch: DataProto) -> None:
    """매 training step 후 호출되어 정확도 추적"""
    acc_values = batch.non_tensor_batch.get("acc", None)
    indices = batch.non_tensor_batch.get("index", None)
    skip_flags = batch.non_tensor_batch.get("skip", None)
    
    for i in range(len(acc_values)):
        # Overlong으로 skip된 샘플은 제외
        if skip_flags is not None and skip_flags[i]:
            continue
        
        problem_idx = int(indices[i])
        acc = float(acc_values[i]) if acc_values[i] is not None else None
        
        if acc is not None:
            self.problem_accuracy[problem_idx].append(acc)
    
    # Epoch 완료 체크
    self.samples_seen_in_epoch += len(acc_values)
    if self.samples_seen_in_epoch >= self.epoch_size:
        self._on_epoch_end()
```

### 3. Epoch 종료 필터링 (`_on_epoch_end`)

```python
def _on_epoch_end(self) -> None:
    """Epoch 끝에서 dynamic filtering 수행"""
    
    # 1. 문제별 평균 정확도 계산
    problem_mean_acc = {
        idx: np.mean(acc_list) 
        for idx, acc_list in self.problem_accuracy.items()
    }
    
    # 2. 난이도별 분류
    new_filtered_hard = []  # 0% acc
    new_filtered_easy = []  # 100% acc
    new_active = []         # 중간 난이도
    
    for problem_idx, mean_acc in problem_mean_acc.items():
        if mean_acc == 0.0:
            new_filtered_hard.append(problem_idx)
        elif mean_acc == 1.0:
            new_filtered_easy.append(problem_idx)
        else:
            new_active.append(problem_idx)
    
    # 3. 확률적 resample
    hard_resampled = [idx for idx in new_filtered_hard 
                      if self.rng.random() < self.hard_resample_prob]  # 10%
    easy_resampled = [idx for idx in new_filtered_easy 
                      if self.rng.random() < self.easy_resample_prob]  # 1%
    
    # 4. Active set 업데이트
    self.active_indices = new_active + hard_resampled + easy_resampled + untracked
    
    # 5. 정확도 추적 리셋
    self.problem_accuracy.clear()
```

---

## ⚙️ 설정 방법

### `run_nemotron_cascade_8b_math_debug.sh`

```bash
COMMON_ARGS=(
    # Curriculum Sampler 활성화 (num_workers=0 필수!)
    data.dataloader_num_workers=0
    +data.sampler.class_path=pkg://verl.experimental.dataset.nemotron_cascade_sampler
    +data.sampler.class_name=NemotronCascadeCurriculumSampler
    +data.sampler.hard_resample_prob=0.10    # 0% acc 문제 10% 확률로 재포함
    +data.sampler.easy_resample_prob=0.01    # 100% acc 문제 1% 확률로 재포함
)
```

### ⚠️ 주의: `num_workers=0` 필수

```bash
data.dataloader_num_workers=0  # 반드시 0이어야 함!
```

**이유**: 멀티프로세스 DataLoader에서는 sampler 상태 동기화가 안 됨

---

## 📊 모니터링 로그

### 콘솔 출력 예시

```
NemotronCascadeCurriculumSampler initialized with 50000 samples.
Hard resample prob: 0.1, Easy resample prob: 0.01

Starting epoch 0 with 50000 active samples (filtered_hard: 0, filtered_easy: 0)

Epoch 0 filtering complete:
  - Total problems tracked: 48523
  - Hard (0% acc): 2341 -> filtered 2107, resampled 234
  - Easy (100% acc): 8912 -> filtered 8823, resampled 89
  - Appropriate difficulty: 37270
  - Untracked (new): 1477
  - New active set size: 39070

Starting epoch 1 with 39070 active samples (filtered_hard: 2107, filtered_easy: 8823)
```

---

## 🔄 상태 저장/복원

체크포인트에서 sampler 상태를 저장하고 복원할 수 있습니다:

```python
# 저장
state = sampler.get_state_dict()
# 반환값: {
#     "active_indices": [...],
#     "filtered_hard": [...],
#     "filtered_easy": [...],
#     "current_epoch": 2,
#     "samples_seen_in_epoch": 1500,
#     "problem_accuracy": {...},
#     "rng_state": {...}
# }

# 복원
sampler.load_state_dict(state)
```

---

## 📈 효과

| 메트릭 | 효과 |
|--------|------|
| **학습 효율** | 쉬운/어려운 문제 필터링으로 compute 절약 |
| **수렴 속도** | 적절한 난이도 문제에 집중하여 빠른 학습 |
| **일반화** | 과적합 방지 (너무 쉬운 문제 반복 학습 방지) |
| **탐색-활용 균형** | 확률적 resample로 다양성 유지 |

---

## 🔗 관련 파일

| 파일 | 설명 |
|------|------|
| `verl/experimental/dataset/nemotron_cascade_sampler.py` | 핵심 구현 |
| `verl/experimental/dataset/sampler.py` | AbstractCurriculumSampler 베이스 클래스 |
| `verl/workers/reward_manager/naive.py` | `index`, `acc` 필드 제공 |
| `run_nemotron_cascade_8b_math_debug.sh` | 설정 예시 |

