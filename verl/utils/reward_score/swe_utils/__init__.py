# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from Nemotron-Cascade paper for SWE RL implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SWE Utils Package

This package provides utility functions for SWE RL (Software Engineering
Reinforcement Learning) reward computation.

Modules:
    - patch_parser: Unified diff patch parsing and validation
    - lexical_similarity: Lexical similarity computation between patches
    - search_replace: Search/Replace block parsing (SWE-RL format)

Reference:
    SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Engineering Tasks
"""

from .patch_parser import (
    parse_patch,
    validate_patch,
    is_empty_patch,
    normalize_patch,
    extract_patch_from_response,
)
from .lexical_similarity import compute_lexical_similarity

__all__ = [
    "parse_patch",
    "validate_patch",
    "is_empty_patch",
    "normalize_patch",
    "compute_lexical_similarity",
    "extract_patch_from_response",
]
