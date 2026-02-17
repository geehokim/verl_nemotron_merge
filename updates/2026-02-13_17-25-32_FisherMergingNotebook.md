# [2026-02-13 17:25] feat(merging): add Fisher merging notebook and distributed Fisher worker

## Changes
- **File**: `scripts/run_distributed_fisher_diag.py`
  - Added a distributed empirical Fisher diagonal worker script for one task model.
  - Implemented sequence-level squared-gradient Fisher estimation using sampled continuations.
  - Added torchrun-compatible rank sharding, all-reduce aggregation, and metadata export.
  - Added detailed function docstrings and inline implementation comments for reproducibility.
- **File**: `merging_analysis/05_fisher_merging_precision_weighted.ipynb`
  - Added a new notebook implementing end-to-end Fisher merge pipeline.
  - Added validation parquet normalization into deterministic `sample_id` + `prompt_text` tables.
  - Added 4-GPU distributed execution cell using `CUDA_VISIBLE_DEVICES=4,5,6,7` and torchrun.
  - Added Fisher precision-weighted merge implementation and merged checkpoint/metadata saving.

## Rationale
- Implement Fisher Merging (Matena & Raffel, 2022) directly from squared-gradient Fisher diagonal definition.
- Reuse provided IF/Math validation parquet files while maximizing gradient throughput with multi-GPU sharding.
- Keep merge outputs reproducible with explicit manifests and summary JSON files.

## Technical Details
- Libraries: `torch`, `torch.distributed`, `transformers`, `pandas`, `numpy`.
- Fisher estimator: one sampled continuation per prompt, sequence log-probability gradient squared, globally averaged.
- Merge formula: `theta*=sum_i(lambda_i*F_i*theta_i)/(sum_i(lambda_i*F_i)+epsilon)` applied per parameter coordinate.
