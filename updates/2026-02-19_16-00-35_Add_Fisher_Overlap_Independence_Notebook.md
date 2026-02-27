# [2026-02-19 16:00] add fisher overlap independence notebook

## Changes
- **File**: `merging_analysis/16_fisher_information_overlap_independence.ipynb`
    - Added a new standalone notebook (`16`) to measure IF vs Math Fisher independence via scalar-level global top-k overlap.
    - Implemented deterministic global-index top-k extraction without full-parameter flattening to keep memory practical.
    - Implemented random baseline comparison (`k/N`) and hypergeometric z-score diagnostics.
    - Added artifact persistence (CSV/JSON/PNG) for reproducible reporting.

- **File**: `merging_analysis/artifacts/fisher_overlap_independence/scalar_topk_overlap_results.csv`
    - Saved computed overlap metrics for requested top-k percentages.

- **File**: `merging_analysis/artifacts/fisher_overlap_independence/scalar_topk_overlap_summary.json`
    - Saved run metadata, configuration, and full numeric results.

- **File**: `merging_analysis/artifacts/fisher_overlap_independence/scalar_topk_overlap_plot.png`
    - Saved visualization comparing observed overlap against random baseline.

## Rationale
- Needed a dedicated analysis notebook to directly test whether IF and Math tasks are independent in Fisher-important parameter locations.
- Needed explicit overlap-vs-random diagnostics so independence can be judged quantitatively rather than qualitatively.

## Technical Details
- Loaded precomputed Fisher diagonals from:
  - `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal/fisher_diag_if.pt`
  - `/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/fisher_diagonal/fisher_diag_math.pt`
- Used NumPy/Torch top-k candidate merging with deterministic value/index tie ordering.
- Computed overlap as `|TopK_if ∩ TopK_math| / k` and baseline as `k / N`.
- Computed hypergeometric standard deviation and z-score (`(observed - baseline) / std`).
- Executed notebook pipeline once and persisted artifacts under `merging_analysis/artifacts/fisher_overlap_independence/`.
