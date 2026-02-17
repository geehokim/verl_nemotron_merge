# [2026-02-16 15:05] Create IF Fisher Diag Analysis Notebook

## Changes
- **File**: `merging_analysis/11_fisher_diag_if_analysis.ipynb`
    - Added a new standalone notebook (`11` series) dedicated to IF Fisher-diagonal analysis.
    - Reused the Fisher concentration diagnostics logic from `05_fisher_merging_precision_weighted.ipynb`.
    - Reused the Fisher dominant layer/module/head diagnostics logic from `05_fisher_merging_precision_weighted.ipynb`.
    - Fixed standalone execution by binding analysis input directly to:
      - `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal/fisher_diag_if.pt`
    - Added robust `num_attention_heads` resolution fallback via local `config.json` when `transformers` is unavailable.
    - Added artifact export paths for concentration and dominant layer/head tables.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/concentration/tensor_mass_stats.csv`
    - Generated from runtime validation execution of the new notebook concentration diagnostics.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/concentration/summary.json`
    - Generated concentration summary (Gini/top-k mass share) for IF Fisher.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/dominant_layer_head/layer_df.csv`
    - Generated layer-level Fisher mass summary.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/dominant_layer_head/layer_module_df.csv`
    - Generated layer-module Fisher mass summary.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/dominant_layer_head/parameter_df_sorted.csv`
    - Generated parameter-level Fisher mass ranking.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/dominant_layer_head/head_df.csv`
    - Generated attention-head Fisher mass summary.

- **File**: `merging_analysis/artifacts/fisher_diag_if_analysis/if/dominant_layer_head/summary.json`
    - Generated dominant layer/head summary metadata.

## Rationale
- Isolate Fisher-diagonal diagnostics into a lightweight, reproducible notebook focused on the precomputed IF Fisher artifact.
- Preserve analysis parity with notebook `05` while removing unnecessary pipeline dependencies.
- Support environments where `transformers` may not be installed by adding config fallback for head analysis.

## Technical Details
- Uses PyTorch tensor loading and aggregation (`torch.load`, absolute Fisher mass accumulation).
- Uses NumPy sampling + Lorenz/Gini concentration metrics.
- Uses pandas groupby aggregation for layer/module/head decomposition.
- Uses matplotlib for histogram/Lorenz/bar/heatmap visualization panels.
- Output artifact persistence implemented with CSV and JSON export in `merging_analysis/artifacts/fisher_diag_if_analysis/`.
