# [2026-01-07] SWE RL Reward Function Implementation

## Summary
Nemotron-Cascade 논문의 Section 4.7 (SWE RL)을 기반으로 Software Engineering RL 학습을 위한 
execution-free reward function을 구현했습니다.

## Changes

### New Files Created

- **File**: `verl/utils/reward_score/swe_utils/__init__.py`
    - SWE Utils 패키지 초기화
    - patch_parser, lexical_similarity 모듈 export

- **File**: `verl/utils/reward_score/swe_utils/patch_parser.py`
    - `parse_patch()`: unidiff 라이브러리를 사용한 패치 파싱
    - `validate_patch()`: 패치 유효성 검증 (Case 3용)
    - `is_empty_patch()`: 빈 패치 확인 (Case 2용)
    - `normalize_patch()`: 패치 정규화 (비교용)
    - `extract_patch_from_response()`: 모델 응답에서 패치 추출

- **File**: `verl/utils/reward_score/swe_utils/lexical_similarity.py`
    - `compute_lexical_similarity()`: 두 패치 간 렉시컬 유사도 계산
    - `compute_hunk_level_similarity()`: Hunk 레벨 유사도 (optional)
    - `compute_change_set_similarity()`: Change set 기반 유사도 (optional)

- **File**: `verl/utils/reward_score/nemotron_cascade_rl_swe.py`
    - 4-case reward function 구현:
        1. lexical_similarity == 1 → reward = 1.0
        2. patch == original (empty) → reward = 0.0
        3. patch parse error → reward = -1.0
        4. otherwise → lexical_similarity (semantic은 나중에)
    - `extract_patch_after_think()`: </think> 토큰 이후 패치 추출
    - `test_reward_function()`: 내장 테스트 함수

### Modified Files

- **File**: `verl/utils/reward_score/__init__.py`
    - `nemotron_cascade_rl_swe` data_source 등록 (line 56-60)

- **File**: `run_nemotron_cascade_8b_swe.sh`
    - SWE RL 학습 스크립트 (이미 구성됨)
    - Hyperparameters: batch_size=128, lr=2.5e-6, rollouts=16, temp=1.0
    - Context: max_prompt=16K, max_response=16K

## Rationale
Nemotron-Cascade 논문 Section 4.7에서 설명한 execution-free reward function을 구현하여
SWE-bench 스타일의 코드 수정 문제에 대한 RL 학습을 가능하게 함.

주요 특징:
- 실행 없이 패치 품질 평가 (execution-free)
- unidiff 라이브러리를 활용한 정규화된 패치 비교
- 4단계 reward logic으로 점진적 학습 신호 제공

## Technical Details

### Reward Function Logic (4 Cases)
```python
def compute_reward(generated_patch, ground_truth_patch):
    lexical_sim = compute_lexical_similarity(generated_patch, ground_truth_patch)
    
    # Case 1: Exact match
    if lexical_sim == 1.0:
        return 1.0
    
    # Case 2: Empty patch (no change)
    if is_empty_patch(generated_patch):
        return 0.0
    
    # Case 3: Invalid patch
    if not validate_patch(generated_patch):
        return -1.0
    
    # Case 4: Partial match (semantic similarity placeholder)
    return lexical_sim
```

### Dependencies
- `unidiff==0.7.5` (already in requirements.txt)
- Standard library: `difflib`, `re`

### Test Results
```
Case 1 (Exact match): score=1.0 ✓
Case 2 (Empty patch): score=0.0 ✓
Case 3 (Invalid patch): score=-1.0 ✓
Case 4 (Partial match): score=0.9770 ✓
```

## Future Extensions
- [ ] Semantic similarity using LLM (Kimi-Dev-72B style)
- [ ] Multi-stage training (16K → 24K)
- [ ] SWE-bench dataset integration

