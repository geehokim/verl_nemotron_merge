# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted from Nemotron-Cascade paper reward function implementation
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
Nemotron-Cascade RL Math Reward Function

This module implements the reward function described in the Nemotron-Cascade paper:

Reward Logic:
1. Rewards are assigned strictly based on answer correctness
2. Answer is extracted from \boxed{} that follows the </think> token
3. Verification uses AceMath-style rule-based verifier (1 for correct, 0 for incorrect)
4. Code-switching penalty: -1.0 if tokens from a different language are detected

Reference:
    Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models
    https://arxiv.org/abs/2505.00000  # TODO: Update with actual arxiv link
"""

import re
from typing import Optional, Union

# Import math_equal from local grader module (AceMath-style verifier)
from .grader import math_equal

# Import helper functions from aime.py for 3-stage verification
from .aime import math_answer_cleaning, round_number, is_equal_after_calculation

# Try to import FastText language detection for code-switching penalty
try:
    from ftlangdetect import detect as ft_detect
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    ft_detect = None


# =============================================================================
# Answer Extraction Functions
# =============================================================================

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    This function finds the last occurrence of \\boxed{...} in the input string,
    properly handling nested braces.
    
    Adapted from math_dapo.py for Nemotron-Cascade reward function.
    
    Args:
        string: Input string containing LaTeX code
    
    Returns:
        The last boxed expression (including \\boxed{}) or None if not found
    
    Example:
        >>> last_boxed_only_string("The answer is \\boxed{42}")
        '\\boxed{42}'
        >>> last_boxed_only_string("\\boxed{1} and \\boxed{2}")
        '\\boxed{2}'
    """
    # Find the last occurrence of \boxed{
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    # Track brace depth to handle nested braces
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.
    
    Extracts the content inside \\boxed{...}.
    
    Args:
        s: String with format "\\boxed{content}"
    
    Returns:
        The content inside the boxed command
    
    Raises:
        AssertionError: If the string doesn't match expected boxed format
    
    Example:
        >>> remove_boxed("\\boxed{42}")
        '42'
        >>> remove_boxed("\\boxed{\\frac{1}{2}}")
        '\\frac{1}{2}'
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def extract_boxed_after_think(solution_str: str) -> Optional[str]:
    """Extract the \\boxed{} answer that follows the </think> token.
    
    This implements the Nemotron-Cascade paper's requirement:
    "extracting the boxed answer (\\boxed{}) that follows the </think> token"
    
    The function:
    1. Finds the </think> token position
    2. Extracts text after </think>
    3. Finds the last \\boxed{} in that text
    4. Returns the content inside the boxed
    
    Args:
        solution_str: Model's full output string (reasoning + answer)
    
    Returns:
        Extracted answer string, or None if no valid boxed answer found
    
    Example:
        >>> extract_boxed_after_think("<think>Let me solve...</think>The answer is \\boxed{42}")
        '42'
        >>> extract_boxed_after_think("No think token \\boxed{42}")
        '42'
    """
    # Find </think> token position
    think_end_markers = ["</think>", "<\/think>", "</Think>"]
    think_end_idx = -1
    
    for marker in think_end_markers:
        idx = solution_str.find(marker)
        if idx != -1:
            think_end_idx = idx + len(marker)
            break
    
    if think_end_idx == -1:
        # If </think> is not found, search the entire string
        # This handles cases where the model doesn't use the think token
        text_after_think = solution_str
    else:
        # Extract text after </think>
        text_after_think = solution_str[think_end_idx:]
    
    # Find the last \boxed{} in the text after </think>
    boxed_str = last_boxed_only_string(text_after_think)
    
    if boxed_str is None:
        return None
    
    # Remove the \boxed{} wrapper and return the content
    try:
        return remove_boxed(boxed_str)
    except AssertionError:
        return None


# =============================================================================
# Code-Switching Detection (FastText Language Detection)
# =============================================================================

def detect_code_switching(solution_str: str, prompt_language: str = "en") -> bool:
    """Detect code-switching in the reasoning chain using FastText.
    
    This implements the Nemotron-Cascade paper's code-switching penalty:
    "we apply a code-switching penalty by assigning a reward of -1 whenever
    tokens from a language (e.g., Chinese) different from the original prompt's
    language (e.g., English) are detected in the reasoning chain."
    
    Uses FastText language detection (ftlangdetect) which is the same approach
    used in NVIDIA's Nemotron implementations.
    
    Args:
        solution_str: Model's full output string (reasoning + answer)
        prompt_language: Language code of the original prompt (e.g., "en", "zh", "ko")
                        Default is "en" (English)
    
    Returns:
        True if code-switching is detected (different language found),
        False otherwise
    
    Note:
        - If FastText is not available, returns False (no penalty applied)
        - Only checks the reasoning chain (text before </think>)
        - Requires: pip install fasttext-langdetect
    
    Example:
        >>> detect_code_switching("Let me 计算一下...", "en")
        True  # Chinese detected in English prompt
        >>> detect_code_switching("Let me solve this problem...", "en")
        False  # Only English detected
    """
    if not FASTTEXT_AVAILABLE:
        # FastText is required for code-switching detection
        # Raise error if not installed to ensure proper reward computation
        raise ImportError(
            "fasttext-langdetect is required for code-switching detection. "
            "Please install it with: pip install fasttext-langdetect"
        )
    
    # Extract reasoning chain (text before </think> if exists)
    think_end_markers = ["</think>", "<\/think>", "</Think>"]
    reasoning_chain = solution_str
    
    for marker in think_end_markers:
        idx = solution_str.find(marker)
        if idx != -1:
            reasoning_chain = solution_str[:idx]
            break
    
    # Skip detection if reasoning chain is too short
    if len(reasoning_chain.strip()) < 20:
        return False
    
    # Clean the text for better language detection
    # Remove LaTeX math expressions which can confuse the detector
    cleaned_text = re.sub(r'\$[^$]+\$', ' ', reasoning_chain)  # Remove $...$
    cleaned_text = re.sub(r'\\\[.*?\\\]', ' ', cleaned_text, flags=re.DOTALL)  # Remove \[...\]
    cleaned_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', cleaned_text)  # Remove \cmd{...}
    cleaned_text = re.sub(r'\\[a-zA-Z]+', ' ', cleaned_text)  # Remove \cmd
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
    
    # Skip if cleaned text is too short
    if len(cleaned_text) < 10:
        return False
    
    try:
        # Detect language using FastText
        # low_memory=False for better accuracy
        result = ft_detect(text=cleaned_text, low_memory=False)
        detected_lang = result.get("lang", prompt_language)
        
        # Check for code-switching
        # If detected language is different from prompt language, it's code-switching
        if detected_lang != prompt_language:
            return True
            
    except Exception:
        # If detection fails for any reason, don't penalize
        pass
    
    return False


# =============================================================================
# Answer Verification (AceMath-style Rule-Based Verifier)
# =============================================================================

def verify_answer(extracted_answer: str, ground_truth: str) -> bool:
    """Verify if the extracted answer matches the ground truth.
    
    This implements the AceMath-style rule-based verifier described in the
    Nemotron-Cascade paper. Uses a 3-stage verification approach:
    
    1. math_equal: Symbolic + numeric comparison using SymPy
       - Handles LaTeX expressions, fractions, equations
       - Performs numerical comparison with tolerance
    
    2. round_number: Decimal rounding comparison
       - Useful for floating-point answers
       - Rounds small numbers to 2 significant figures
    
    3. is_equal_after_calculation: Fraction evaluation
       - Converts LaTeX fractions to evaluable format
       - Compares numerical results
    
    Args:
        extracted_answer: Answer extracted from model output
        ground_truth: Correct answer to compare against
    
    Returns:
        True if answers are mathematically equivalent, False otherwise
    
    Example:
        >>> verify_answer("0.5", "1/2")
        True
        >>> verify_answer("\\frac{1}{2}", "0.5")
        True
        >>> verify_answer("42", "43")
        False
    """
    # Clean both answers using aime.py's cleaning function
    extracted_cleaned = math_answer_cleaning(extracted_answer)
    gt_cleaned = math_answer_cleaning(ground_truth)
    
    # Method 1: math_equal (symbolic + numeric comparison)
    # This is the primary verification method from AceMath
    try:
        if math_equal(extracted_cleaned, gt_cleaned):
            return True
    except Exception:
        # math_equal can fail on some malformed inputs
        pass
    
    # Method 2: round_number comparison (for floating-point answers)
    # Useful for answers like 0.333... vs 1/3
    try:
        if round_number(extracted_cleaned) == round_number(gt_cleaned):
            return True
    except Exception:
        pass
    
    # Method 3: is_equal_after_calculation (fraction evaluation)
    # Converts \frac{a}{b} to (a/b) and evaluates numerically
    try:
        if is_equal_after_calculation(extracted_cleaned, gt_cleaned):
            return True
    except Exception:
        pass
    
    return False


# =============================================================================
# Main Reward Computation Function
# =============================================================================

def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    return_dict: bool = True,
) -> Union[float, dict]:
    """Compute reward score for Nemotron-Cascade RL math training.
    
    This is the main reward function implementing the Nemotron-Cascade paper's
    reward logic:
    
    Reward Computation (Additive):
        total_reward = answer_reward + code_switching_penalty
    
    Where:
        - answer_reward: 1.0 (correct) or 0.0 (incorrect/extraction failed)
        - code_switching_penalty: 0.0 (no code-switching) or -1.0 (code-switching detected)
    
    Result Examples:
        - Correct + No code-switching: 1.0 + 0.0 = 1.0
        - Correct + Code-switching:    1.0 + (-1.0) = 0.0
        - Incorrect + No code-switching: 0.0 + 0.0 = 0.0
        - Incorrect + Code-switching:    0.0 + (-1.0) = -1.0
    
    The function:
    1. Detects code-switching to compute penalty
    2. Extracts \boxed{} answer after </think> token
    3. Verifies answer using 3-stage verification
    4. Returns sum of answer_reward and code_switching_penalty
    
    Args:
        solution_str: Model's full output string (reasoning + answer)
        ground_truth: Ground truth answer string
        extra_info: Optional dict containing:
            - "language": Prompt language code (default: "en")
            - Other metadata (ignored)
        return_dict: If True, return dict with score and metadata;
                     If False, return float score only
    
    Returns:
        If return_dict=True:
            dict: {
                "score": float (answer_reward + code_switching_penalty),
                "acc": bool (whether answer is correct),
                "pred": str (extracted answer or empty string),
                "code_switching": bool (whether code-switching detected),
                "extraction_failed": bool (whether answer extraction failed),
                "answer_reward": float (1.0 or 0.0),
                "code_switching_penalty": float (0.0 or -1.0)
            }
        If return_dict=False:
            float: Total reward score (answer_reward + code_switching_penalty)
    
    Example:
        >>> compute_score("<think>...</think>\\boxed{42}", "42")
        {"score": 1.0, "acc": True, "pred": "42", ...}  # 1.0 + 0.0 = 1.0
        
        >>> compute_score("Let me 计算一下...</think>\\boxed{42}", "42", {"language": "en"})
        {"score": 0.0, "acc": True, "pred": "42", "code_switching": True, ...}  # 1.0 + (-1.0) = 0.0
    """
    # ==========================================================================
    # Step 0: Parse extra_info for prompt language
    # ==========================================================================
    
    prompt_language = "en"  # Default to English
    if extra_info and isinstance(extra_info, dict):
        # Support multiple possible keys for language
        prompt_language = extra_info.get("language", 
                         extra_info.get("lang",
                         extra_info.get("prompt_language", "en")))
    
    # ==========================================================================
    # Step 1: Detect code-switching to compute penalty
    # ==========================================================================
    # From the paper: "we apply a code-switching penalty by assigning a reward
    # of -1 whenever tokens from a language different from the original prompt's
    # language are detected in the reasoning chain."
    has_code_switching = detect_code_switching(solution_str, prompt_language)
    code_switching_penalty = -1.0 if has_code_switching else 0.0
    
    # ==========================================================================
    # Step 2: Extract \boxed{} answer after </think> token
    # ==========================================================================
    # From the paper: "extracting the boxed answer (\boxed{}) that follows
    # the </think> token"
    extracted_answer = extract_boxed_after_think(solution_str)
    
    extraction_failed = False
    if extracted_answer is None:
        # No valid boxed answer found -> answer_reward = 0.0
        extraction_failed = True
        is_correct = False
        answer_reward = 0.0
        extracted_answer = ""
    else:
        # ==========================================================================
        # Step 3: Verify answer using AceMath-style rule-based verifier
        # ==========================================================================
        # From the paper: "verifying it using the AceMath rule-based verifier
        # (1 for correct and 0 for incorrect)"
        is_correct = verify_answer(extracted_answer, ground_truth)
        answer_reward = 1.0 if is_correct else 0.0
    
    # ==========================================================================
    # Step 4: Compute total reward (answer_reward + code_switching_penalty)
    # ==========================================================================
    total_reward = answer_reward + code_switching_penalty
    
    if return_dict:
        return {
            "score": total_reward,
            "acc": is_correct,
            "pred": extracted_answer,
            "code_switching": has_code_switching,
            "extraction_failed": extraction_failed,
            "answer_reward": answer_reward,
            "code_switching_penalty": code_switching_penalty,
        }
    return total_reward

