# [2026-02-19 16:19] Create Notebook 18 Hessian Task Vector Alignment

## Changes
- **File**: `merging_analysis/18_hessian_task_vector_alignment.ipynb`
    - Created a new analysis notebook with the `18_` prefix for Hessian-task vector independence analysis.
    - Added notebook-level methodology notes for the second-order metric `rho(t,k) = (tau_k^T H_t tau_k) / (tau_t^T H_t tau_t)`.
    - Added reusable utilities for model/tokenizer loading, prompt parquet ingestion, task-vector construction, and parameter-direction alignment.
    - Added HVP-based quadratic-form functions using double autograd to estimate `v^T H v` without constructing the full Hessian.
    - Added end-to-end execution flow for both directions (`IF on Math` and `Math on IF`) with CSV/JSON artifact export.

## Rationale
- To provide a dedicated, numbered notebook (`18`) focused on the theoretically grounded Hessian-task-vector alignment criterion discussed in the analysis.
- To enable practical large-model second-order independence estimation via HVP, which is feasible unlike full Hessian construction.

## Technical Details
- Used PyTorch second-order autodiff (`torch.autograd.grad` twice) to compute Hessian-vector products and directional curvature.
- Used Hugging Face `AutoModelForCausalLM` and tokenizer-based generation + teacher-forcing objective construction for sequence log-probability curvature probes.
- Saved outputs under `merging_analysis/artifacts/hessian_task_vector_alignment/` as `rho_alignment_results.csv` and `rho_alignment_summary.json`.
