# [2026-02-14 15:00] Add Fisher Diagonal Concentration Visualization

## Changes
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
    - Added a new analysis code cell that visualizes Fisher diagonal concentration with a 2x2 panel.
    - Added tensor-level and sampled element-level concentration metrics (Lorenz curve, top-k mass share, Gini).
    - Added helper functions with docstrings and inline comments for loading Fisher tensors, sampling, and concentration statistics.
- **File**: `updates/2026-02-14_15-00-26_Add_Fisher_Diagonal_Concentration_Visualization.md`
    - Added this update log entry for traceability.

## Rationale
- To determine whether Fisher diagonal scale is concentrated in a small subset of parameters (sparse) or distributed relatively uniformly across parameters.
- To provide both visual and numeric diagnostics that can be compared across IF and Math tasks.

## Technical Details
- Uses `torch.load` to load Fisher diagonal dictionaries from `FISHER_OUTPUT_PATHS`.
- Computes per-parameter-tensor mass statistics with `pandas`.
- Performs memory-conscious global sampling over Fisher elements with NumPy index sampling instead of full flatten-concatenation.
- Visualizes diagnostics using `matplotlib` (histogram, Lorenz curve, top-fraction mass bars, cumulative tensor-mass curve).
