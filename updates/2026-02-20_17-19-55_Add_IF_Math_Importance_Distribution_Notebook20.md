# [2026-02-20 17:19] Add IF/Math Importance Distribution Notebook20

## Changes
- **File**: `merging_analysis/20_if_math_attribute_importance_distribution_analysis.ipynb`
    - Created a new analysis notebook (`20`) dedicated to IF/Math precomputed attribute importance distribution analysis.
    - Added experiment-separated cells for setup, file resolution, utility functions, raw-score analysis, sqrt-score analysis, and raw-vs-sqrt comparison.
    - Implemented threshold-based active parameter ratio and active mass ratio analysis for thresholds `1e-8`, `1e-7`, `1e-6`, `1e-5`.
    - Added visualizations with log-scaled x-axis and zoomed y-axis ranges to improve trend readability.
    - Added sampled value distribution visualization (log-domain) with threshold marker lines.
    - Added compatibility fallback to matplotlib-only plotting when `seaborn` is unavailable.

## Rationale
- To provide a focused notebook for analyzing IF/Math importance sparsity and mass concentration behavior at specific thresholds.
- To compare how value transformation (`sqrt`) changes activation ratios and mass concentration patterns under identical threshold settings.
- To keep the analysis interpretable by separating experiments into clear notebook cells and controlling plot ranges for trend visibility.

## Technical Details
- Uses PyTorch (`torch.load`) to load precomputed importance tensor dictionaries.
- Uses bucketized threshold aggregation (`torch.bucketize` + `torch.bincount`) for efficient active-count and mass-ratio computation.
- Uses proportional random sampling for large-tensor distribution plots to avoid full flatten/concat memory spikes.
- Uses matplotlib (and seaborn when available) for line/bar/distribution visualizations.
