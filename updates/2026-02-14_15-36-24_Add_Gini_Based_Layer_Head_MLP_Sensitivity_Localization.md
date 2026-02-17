# [2026-02-14 15:36] Add Gini Based Layer Head MLP Sensitivity Localization

## Changes
- **File**: `merging_analysis/06_singular_task_interference_rl.ipynb`
  - Added a new Gini-based sensitivity concentration analysis section that localizes dominant contributors at three granularities:
    - Layer level (`sti_l_single_sum`)
    - Parameter/matrix level (`sti_param`)
    - Attention-head level (per-head STI on top attention projection matrices)
  - Added utility functions for Gini computation, contribution-share tracking, attention projection parsing, and head-dimension inference.
  - Added per-head STI decomposition workflow for attention projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) using block SVD and STI.
  - Added dominance visualization figure with top-K bars for layers, matrices, and heads.
  - Added artifact exports:
    - `gini_layer_dominance.csv`
    - `gini_matrix_dominance.csv`
    - `gini_mlp_matrix_dominance.csv`
    - `gini_attention_matrix_dominance.csv`
    - `gini_attention_head_dominance.csv`
    - `gini_sensitivity_dominance_overview.png`
    - `gini_sensitivity_dominance_summary.json`
  - Extended final run summary JSON payload to include new Gini-dominance artifact paths and structured summary.
- **File**: `updates/2026-02-14_15-36-24_Add_Gini_Based_Layer_Head_MLP_Sensitivity_Localization.md`
  - Added this update log for the Gini-based localization enhancement.

## Rationale
- A high Gini value indicates sensitivity concentration; this requires explicit localization of dominant layer, matrix, and attention head contributors.
- Existing notebook outputs summarized global trends but did not identify which concrete modules caused concentration.
- The added analysis directly supports actionable intervention targets for RL task-vector merging and robustness diagnostics.

## Technical Details
- Gini is computed on non-negative sensitivity vectors using the sorted discrete closed-form expression.
- Layer concentration uses `sti_l_single_sum` shares and cumulative shares.
- Matrix concentration uses `sti_param` shares and separate slices for MLP and attention matrices.
- Head concentration computes per-head STI by slicing attention projection rows into head blocks, running adaptive truncated SVD, and applying the same STI function.
- Summary JSON now includes both scalar Gini diagnostics and top-contributor records for downstream reporting.
