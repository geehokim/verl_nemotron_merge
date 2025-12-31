# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from nemotron_evaluation/eval/get_scores_math.py and tools/grader.py
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
AIME (American Invitational Mathematics Examination) Reward Function

This module provides reward computation for AIME-style math problems,
adapted from Nemotron-Cascade's evaluation code for use in RL training.

The reward function:
1. Extracts answers from model output using 5 regex patterns (boxed, markdown bold, etc.)
2. Cleans and normalizes both extracted and ground truth answers
3. Compares using multiple methods: math_equal, round_number, is_equal_after_calculation
"""

import re
from typing import Optional, Union

# Import math_equal from local grader module
from .grader import math_equal


# =============================================================================
# Answer Extraction Patterns
# =============================================================================
# These 5 patterns are used to extract answers from model outputs.
# Priority order: boxed > markdown bold > display math > inline math > escaped newline

# Pattern 1: LaTeX \boxed{...} - handles nested braces up to 3 levels
PATTERN_BOXED = re.compile(
    r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
    re.DOTALL
)

# Pattern 2: Markdown bold **...** 
PATTERN_MARKDOWN_BOLD = re.compile(
    r"\*\*(.*?)\*\*",
    re.DOTALL
)

# Pattern 3: LaTeX display math \[...\] with actual newlines
PATTERN_DISPLAY_MATH_NEWLINE = re.compile(
    r"\\\[\n(.*?)\n\\\]",
    re.DOTALL
)

# Pattern 4: "is \(...\)" inline math pattern
PATTERN_IS_INLINE_MATH = re.compile(
    r'is \\\((.*?)\\\)',
    re.DOTALL
)

# Pattern 5: LaTeX display math with escaped newlines \[\n...\n\]
PATTERN_DISPLAY_MATH_ESCAPED = re.compile(
    r"\\\[\\n(.*?)\\n\\\]",
    re.DOTALL
)


# =============================================================================
# Helper Functions
# =============================================================================

def is_completely_wrapped_by_text(input_string: str) -> Optional[str]:
    """Check if input string is completely wrapped by LaTeX \\text{}.
    
    Args:
        input_string: LaTeX string to check
    
    Returns:
        str or None: Extracted content if wrapped, None otherwise
    
    Example:
        >>> is_completely_wrapped_by_text("\\text{42}")
        '42'
        >>> is_completely_wrapped_by_text("42")
        None
    """
    pattern = r'^\\text{(.*)}$'
    match = re.match(pattern, input_string)
    if match:
        extracted_content = match.group(1)
        # Remove parentheses and commas from extracted content
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    return None


def math_answer_cleaning(answer: str) -> str:
    """Clean and normalize math answer for comparison.
    
    Performs various cleaning operations:
    - Remove LaTeX formatting (\\text, \\quad, etc.)
    - Normalize fractions and scientific notation
    - Remove units and special characters
    - Convert to lowercase
    
    Args:
        answer: Raw answer string
    
    Returns:
        str: Cleaned answer string
    
    Example:
        >>> math_answer_cleaning("\\frac{1}{2}")
        'frac{1}{2}'
        >>> math_answer_cleaning("42^\\circ")
        '42'
    """
    # Extract content if wrapped by \text{}
    extracted_content = is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    # Remove various LaTeX formatting
    answer = answer.replace(",\!", "").replace("{,}", "").replace("\$", "")
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    answer = answer.replace("^\circ", "").replace("^{\circ}", "")
    answer = answer.replace("\quad", "")
    
    # Remove \text{...} patterns
    answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    
    # Remove negative exponents like ^{-1}
    answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
    
    # Remove whitespace and newlines
    answer = answer.replace(" ", "").replace("\n", "").replace("\\n", "")
    
    # Normalize scientific notation: 1.5\times10^{3} -> 1.5e3
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}', r'\1e\2', answer)
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
    
    # Normalize exponents: 2^{3} -> 2^3
    answer = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', answer)
    
    # Convert 10^{n} to 1en format
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    
    # Remove commas and convert to lowercase
    answer = answer.replace(",", "").lower()

    # Remove trailing backslash
    if answer.endswith("\\"):
        answer = answer[:-1]
    
    # If answer contains "=" and left side is a function or short variable, take right side
    # e.g., "f(x)=42" -> "42", "x=5" -> "5"
    func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0]) <= 3):
        answer = answer.split("=", 1)[1]

    return answer


def round_number(answer: str) -> str:
    """Round very small numbers to 2 significant figures.
    
    Args:
        answer: Answer string
    
    Returns:
        str: Rounded answer if applicable, otherwise original answer
    
    Example:
        >>> round_number("0.333333")
        '0.33'
        >>> round_number("5.5")
        '5.5'
    """
    def _is_float(string):
        try:
            float(string)
            return True
        except:
            return False

    if _is_float(answer) and float(answer) < 1:
        return f"{float(answer):.2g}"
    
    return answer


def calculate_numbers(input_string: str) -> Optional[float]:
    """Safely evaluate mathematical expression string.
    
    Args:
        input_string: Mathematical expression as string
    
    Returns:
        Result of evaluation, or None if error
    
    Warning:
        Uses eval() - only use with trusted input!
    """
    try:
        result = eval(input_string)
        return result
    except:
        return None


def is_equal_after_calculation(extracted_answer: str, gold: str) -> bool:
    """Check if answers are equal after converting fractions and evaluating.
    
    Converts LaTeX fractions \\frac{a}{b} to (a/b) and evaluates numerically.
    
    Args:
        extracted_answer: Extracted answer string
        gold: Gold standard answer string
    
    Returns:
        bool: True if answers are mathematically equal
    
    Example:
        >>> is_equal_after_calculation("\\frac{3}{4}", "0.75")
        True
    """
    # Convert LaTeX fractions to evaluable format
    gold = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', gold)
    extracted_answer = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', extracted_answer)
    
    gold_result = calculate_numbers(gold)
    extracted_answer_result = calculate_numbers(extracted_answer)

    if gold_result is not None and gold_result == extracted_answer_result:
        return True
    return False


# =============================================================================
# Main Answer Extraction Function
# =============================================================================

def extract_answer(solution_str: str) -> Optional[str]:
    """Extract answer from model solution using 5 regex patterns.
    
    Patterns are tried in priority order:
    1. \\boxed{...} - LaTeX boxed (most common for math models)
    2. **...** - Markdown bold
    3. \\[\\n...\\n\\] - LaTeX display math with newlines
    4. is \\(...\\) - Inline math after "is"
    5. \\[\\n...\\n\\] - Display math with escaped newlines
    
    Args:
        solution_str: Model's solution/output string
    
    Returns:
        Extracted answer string, or None if no pattern matches
    
    Note:
        Uses [-1] to get the LAST match, assuming final answer appears last.
    """
    # Try each pattern in priority order
    matches1 = PATTERN_BOXED.findall(solution_str)
    if matches1:
        return matches1[-1]
    
    matches2 = PATTERN_MARKDOWN_BOLD.findall(solution_str)
    if matches2:
        return matches2[-1]
    
    matches3 = PATTERN_DISPLAY_MATH_NEWLINE.findall(solution_str)
    if matches3:
        return matches3[-1]
    
    matches4 = PATTERN_IS_INLINE_MATH.findall(solution_str)
    if matches4:
        return matches4[-1]
    
    matches5 = PATTERN_DISPLAY_MATH_ESCAPED.findall(solution_str)
    if matches5:
        return matches5[-1]
    
    return None


# =============================================================================
# Main Reward Computation Function
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: str,
    return_dict: bool = True,
) -> Union[float, dict]:
    """Compute reward score for AIME-style math problems.
    
    This function adapts Nemotron-Cascade's evaluation logic for RL training:
    1. Extract answer from solution using regex patterns
    2. Clean both extracted and ground truth answers
    3. Compare using multiple methods (math_equal, round_number, calculation)
    
    Args:
        solution_str: Model's solution/output string
        ground_truth: Ground truth answer string
        return_dict: If True, return dict with score and metadata; else return float
    
    Returns:
        If return_dict=True:
            dict: {"score": float, "acc": bool, "pred": str or None}
        Else:
            float: Reward score (1.0 for correct, 0.0 for incorrect)
    
    Example:
        >>> compute_score("The answer is \\boxed{42}", "42")
        {"score": 1.0, "acc": True, "pred": "42"}
    """
    # Step 1: Extract answer from solution
    extracted_answer = extract_answer(solution_str)
    
    # If no answer found, return incorrect
    if extracted_answer is None:
        if return_dict:
            return {
                "score": 0.0,
                "acc": False,
                "pred": None,
                "extraction_failed": True,
            }
        return 0.0
    
    # Step 2: Clean both answers
    extracted_answer_cleaned = math_answer_cleaning(extracted_answer)
    ground_truth_cleaned = math_answer_cleaning(ground_truth)
    
    # Step 3: Compare using multiple methods (fallback chain)
    is_correct = False
    
    # Method 1: math_equal (symbolic + numeric comparison)
    try:
        if math_equal(extracted_answer_cleaned, ground_truth_cleaned):
            is_correct = True
    except Exception:
        pass  # math_equal can fail on some inputs
    
    # Method 2: round_number comparison (for small decimals)
    if not is_correct:
        if round_number(extracted_answer_cleaned) == round_number(ground_truth_cleaned):
            is_correct = True
    
    # Method 3: is_equal_after_calculation (evaluate fractions)
    if not is_correct:
        if is_equal_after_calculation(extracted_answer_cleaned, ground_truth_cleaned):
            is_correct = True
    
    # Compute reward (1.0 for correct, 0.0 for incorrect)
    reward = 1.0 if is_correct else 0.0
    
    if return_dict:
        return {
            "score": reward,
            "acc": is_correct,
            "pred": extracted_answer_cleaned,
        }
    return reward

