# [2026-02-14 00:31] SVD-Based Task Interference Analysis Notebook

## Changes
- **File**: `merging_analysis/07_svd_task_interference_analysis.ipynb`
    - Created new Jupyter notebook implementing SVD-based task interference analysis
    - 15 cells: 2 markdown (title + interpretation guide), 13 code cells
    - Implements Singular Task Interference (STI) metric from geometric perspective paper
    - Covers 4 analysis dimensions: rank analysis, cosine similarity, STI per layer, cumulative energy

## Rationale
- RL fine-tuned IF/Math 모델 병합 시 발생하는 파괴적 간섭을 기하학적 관점에서 정량화
- 기존 프로젝트에 SVD 기반 분석이 없었으므로 완전히 새로 구현
- Layer별 간섭 패턴 파악으로 최적 merging 전략 수립 지원

## Technical Details
- **STI 공식**: `|| (U^T U - I) Σ (V^T V - I) ||_1`
- **메모리 최적화**: safetensors 직접 로드 (AutoModelForCausalLM 미사용, ~12GB 절약)
- **GPU SVD**: `torch.linalg.svd` with OOM fallback to CPU
- **분석 대상**: 7 weight types × 28 layers = 196 parameters
- **시각화**: singular value spectrum, cross-cosine heatmap, STI profile, cumulative energy, summary dashboard
- **출력 artifacts**: CSV (metrics), PNG (plots), JSON (metadata), NPZ (singular values)
