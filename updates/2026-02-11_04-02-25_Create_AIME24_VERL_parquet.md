# [2026-02-11 04:02] Create AIME24 VERL parquet with system prompt

## Changes
- **File**: `nemotron_evaluation/data/aime24/dummy.ipynb`
    - Added a full conversion notebook that loads `test.jsonl`, injects the same AIME25 instruction system prompt, and writes VERL-format parquet.
    - Added documented helper functions for JSONL loading, answer normalization, year inference, prompt construction, and dataframe conversion.
    - Added runtime validation cells that assert prompt role order (`system` then `user`) and exact system prompt match.
- **File**: `nemotron_evaluation/data/aime24/test_verl_ready_with_instruction.parquet`
    - Generated a new VERL-ready evaluation parquet from AIME24 JSONL with schema aligned to AIME25 instruction parquet.
- **File**: `updates/2026-02-11_04-02-25_Create_AIME24_VERL_parquet.md`
    - Added this task update log for reproducibility and traceability.

## Rationale
- Match AIME24 evaluation data format to AIME25 instruction dataset so both can be used consistently by existing VERL math evaluation pipelines.
- Ensure prompt formatting explicitly enforces `<think>...</think>` reasoning tags and `\boxed{your_answer}` final answer style.

## Technical Details
- Used **pandas** for dataframe and parquet serialization.
- Used **NumPy object arrays** for `prompt` messages to mirror the observed AIME25 parquet prompt representation.
- Preserved AIME25-compatible output column order:
  `id, problem, answer, solution, url, year, __index_level_0__, prompt, reward_model, data_source`.
- Set reward payload as rule-based dict: `{"ground_truth": <answer_as_string>, "style": "rule"}`.
