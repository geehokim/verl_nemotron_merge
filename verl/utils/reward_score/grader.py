"""
Math answer grading utilities.

This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/eval/eval_utils.py

Adapted for verl reward function usage.
"""

import re
import multiprocessing
from math import isclose
from typing import Union

# Try to import regex, fallback to re if not available
try:
    import regex
except ImportError:
    regex = re  # Use standard re as fallback

# Try to import sympy for symbolic comparison
try:
    from sympy import simplify, N
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    parse_expr = None
    parse_latex = None
    simplify = None
    N = None

# Try to import latex2sympy for enhanced LaTeX parsing
try:
    from nemotron_evaluation.eval.tools.latex2sympy.latex2sympy2 import latex2sympy
    LATEX2SYMPY_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        import sys
        sys.path.insert(0, '/mnt/ddn/vuvlm/geeho/nemotron_evaluation/eval/tools')
        from latex2sympy.latex2sympy2 import latex2sympy
        LATEX2SYMPY_AVAILABLE = True
    except ImportError:
        LATEX2SYMPY_AVAILABLE = False
        latex2sympy = None


def choice_answer_clean(pred: str) -> str:
    """Clean multiple choice answer prediction.
    
    Args:
        pred: Raw prediction string
    
    Returns:
        Cleaned answer (single letter A-E or original)
    """
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num) -> float:
    """Parse a number string to float, handling percentages.
    
    Args:
        num: Number string (can include commas, percentages)
    
    Returns:
        float or None if parsing fails
    """
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def is_digit(num) -> bool:
    """Check if string can be parsed as a number.
    
    Args:
        num: String to check
    
    Returns:
        True if parseable as number
    """
    return parse_digits(num) is not None


def str_to_pmatrix(input_str: str) -> str:
    """Convert matrix notation to LaTeX pmatrix format.
    
    Args:
        input_str: String with {a,b} style matrix notation
    
    Returns:
        LaTeX pmatrix formatted string
    """
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def numeric_equal(prediction: float, reference: float) -> bool:
    """Check if two floats are numerically equal within tolerance.
    
    Args:
        prediction: Predicted value
        reference: Reference/ground truth value
    
    Returns:
        True if values are close (rel_tol=1e-4)
    """
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b) -> bool:
    """Check if two expressions are symbolically equal using SymPy.
    
    Tries multiple parsing methods and comparison techniques.
    
    Args:
        a: First expression (string or sympy)
        b: Second expression (string or sympy)
    
    Returns:
        True if expressions are symbolically equivalent
    """
    if not SYMPY_AVAILABLE:
        return False
    
    def _parse(s):
        """Try multiple parsers to convert string to sympy expression."""
        parsers = [parse_latex, parse_expr]
        if LATEX2SYMPY_AVAILABLE and latex2sympy is not None:
            parsers.append(latex2sympy)
        
        for f in parsers:
            if f is None:
                continue
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    # numeric equal after symbolic evaluation
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix equal
    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    """Process function for timeout-protected symbolic comparison."""
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    """Call a function with timeout protection using multiprocessing.
    
    Args:
        func: Function to call
        *args: Arguments to pass
        timeout: Timeout in seconds
        **kwargs: Keyword arguments
    
    Returns:
        Function result or False if timeout
    """
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        return output_queue.get_nowait()
    except:
        return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Check if prediction equals reference mathematically.
    
    Performs multiple types of comparison:
    1. Exact string match (case-insensitive)
    2. Numerical equality (with percentage handling)
    3. Symbolic equality (using SymPy)
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        include_percentage: Whether to try percentage variations (x, x/100, x*100)
        is_close: Whether to use approximate numerical comparison
        timeout: Whether to use timeout protection for symbolic comparison
    
    Returns:
        True if answers are mathematically equivalent
    
    Example:
        >>> math_equal("0.5", "1/2")
        True
        >>> math_equal("\\frac{1}{2}", "0.5")
        True
    """
    # Handle None inputs
    if prediction is None or reference is None:
        return False
    
    # Convert to string for comparison
    prediction = str(prediction)
    reference = str(reference)
    
    # Exact string match (case-insensitive)
    if prediction.strip().lower() == reference.strip().lower():
        return True
    
    # Multiple choice answer handling
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    # 1. Numerical equality
    try:
        if is_digit(prediction) and is_digit(reference):
            pred_num = parse_digits(prediction)
            ref_num = parse_digits(reference)
            
            # Try percentage variations
            if include_percentage:
                gt_result = [ref_num / 100, ref_num, ref_num * 100]
            else:
                gt_result = [ref_num]
            
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(pred_num, item):
                            return True
                    else:
                        if item == pred_num:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    # Empty prediction check
    if not prediction and prediction not in [0, False]:
        return False

    # 2. Symbolic equality
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix handling
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # Handle brackets [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    
    if pred_str.lower() == ref_str.lower():
        return True

    # List/tuple comparison [a, b] vs [c, d]
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return True

    # Matrix comparison
    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    # Equation comparison (x = 5 vs x = 5)
    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    # Symbolic equality with SymPy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    """Process function for parallel math_equal evaluation."""
    return math_equal(param[-2], param[-1])

