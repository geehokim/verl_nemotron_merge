# [2026-02-12 18:43] Logit Calibration Last-Layer Diagnostics Notebook

## Changes
- **File**: `merging_analysis/03_logit_calibration_last_layer_diagnostics.ipynb`
  - Added a new end-to-end analysis notebook focused on merge-induced logit calibration and consistency degradation.
  - Implemented JSONL loading and schema normalization for n-sample validation outputs (`input`, `output`, `acc`, `score`, `pred`, etc.).
  - Added teacher-forced token-level metric extraction functions for entropy, confidence, top-1 probability, margin, NLL, and PPL.
  - Added calibration metrics (`ECE`, reliability bins, Brier score) and prompt-level consistency summaries (`acc/mean`, `acc/best`, `acc/worst`, mixed prompt rate, confidence inversion).
  - Added last-layer diagnostics around `lm_head` and `final_norm` with drift size/alignment/sign-flip metrics relative to a reference model.
  - Added visualization functions for reliability diagram, uncertainty distributions, prompt instability scatter, mean/best/worst bars, and last-layer drift bars.
  - Added CSV/PNG artifact export paths under `merging_analysis/artifacts/logit_calibration/`.

- **File**: `updates/2026-02-12_18-43-05_Logit_Calibration_Last_Layer_Diagnostics.md`
  - Added this task update log entry.

## Rationale
- To validate the hypothesis that naive model merging mainly hurts worst-case behavior through reduced sampling consistency and calibration mismatch rather than through simple top-line collapse.
- To focus diagnostics on the final logit production path (`final_norm` -> `lm_head`) where confidence and entropy behavior is most directly expressed.

## Technical Details
- Frameworks/libraries used in notebook code:
  - `PyTorch` and `transformers` for teacher-forced logit extraction and last-layer tensor comparisons.
  - `pandas` / `numpy` for aggregation and summary-table construction.
  - `matplotlib` / `seaborn` for visualization.
- Core algorithms and metrics:
  - Token-level uncertainty metrics via `log_softmax` distributions.
  - Calibration via reliability bins, ECE, and Brier score.
  - Prompt-level n-sample consistency statistics for mean/best/worst and inversion diagnostics.
  - Last-layer drift quantification using delta norm ratio, cosine similarity, and thresholded sign-flip rate.
