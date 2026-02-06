# [2026-01-16] User Prompt Suffix Feature

## Changes
- **File**: `verl/utils/dataset/rl_dataset.py`
  - Added `user_prompt_suffix` configuration option in `__init__` method
  - Modified `_build_messages` method to append suffix to user messages
  - Supports both string and multimodal (list) content formats
  
- **File**: `run_nemotron_distill_1,5b_math.sh`
  - Added `data.user_prompt_suffix` configuration to COMMON_ARGS
  - Set to " please reason step by step. answer with \\boxed{}"

## Rationale
- To add instructions to user prompts without modifying dataset files
- Allows dynamic prompt modification at training time
- Useful for instructing models to provide step-by-step reasoning and boxed answers
- Follows the Nemotron-Cascade paper's requirement for structured reasoning output

## Technical Details
- The `user_prompt_suffix` is read from config in `RLHFDataset.__init__`
- In `_build_messages`, the suffix is appended to user message content
- Handles both string content (simple text) and list content (multimodal with type/text dicts)
- For multimodal content, finds the last text segment and appends the suffix
- The suffix is added before image/video processing to ensure it's included in the final prompt
