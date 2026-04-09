"""GPQA (Graduate-Level Google-Proof Q&A) benchmark evaluation utilities.

This module provides evaluation functions for the GPQA benchmark, a challenging
multiple-choice science question dataset. It includes answer extraction from various
formats and mathematical equivalence checking.
"""

import argparse
import glob
import json
import os
import re
import signal

import numpy as np
from sympy import simplify
from sympy.parsing.latex import parse_latex

from tools.grader import math_equal


def get_option_char(s: str):
    """Extract single-letter option from LaTeX \\boxed{} construct.
    
    Handles multiple formats:
    - \\boxed{B}
    - \\boxed{\\text{D}}
    - \\boxed{\\text{(E)}}
    - \\boxed{{A}}
    - \\boxed{(A)}
    
    Args:
        s: String containing potential boxed answer
    
    Returns:
        str or None: Extracted letter (A-D), or None if no match found
    """
    pattern = r"""
        \\boxed\{                   # \boxed{
          \s*
          (?:                        # one of:
            \\text\{                 #   \text{…}
              \s*\(?([A-Za-z])\)?\s*
            \}
          |                          # or
            \{([A-Za-z])\}           #   {A}
          |                          # or
            \(\s*([A-Za-z])\s*\)     #   (A)
          |                          # or
            ([A-Za-z])               #   B
          )
          \s*
        \}                           # }
    """
    m = re.search(pattern, s, re.VERBOSE)
    if not m:
        return None
    return m.group(1) or m.group(2) or m.group(3) or m.group(4)


def read_text_data(datapath):
    """Read model outputs from JSONL file.
    
    Args:
        datapath: Path to JSONL file containing model outputs
    
    Returns:
        list: List of output strings
    """
    print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_list.append(json.loads(line.strip())['output'])

    return data_list


def read_jsonl_data(datapath):
    """Read model outputs from JSONL file (alternative method).
    
    Args:
        datapath: Path to JSONL file
    
    Returns:
        list: List of output strings
    """
    print("reading from %s" % datapath)
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_item = json.loads(line.strip())
            data_list.append(data_item['output'])

    return data_list


def read_json_data(datapath):
    """Read JSON data file.
    
    Args:
        datapath: Path to JSON file
    
    Returns:
        Data structure from JSON file
    """
    print("reading from %s" % datapath)
    with open(datapath, "r") as f:
        data_list = json.load(f)

    return data_list


def evaluate_gsm8k_zeroshot(input_datapath, test_datapath):
    def _maybe_remove_comma(x: str):
        # Example: 5,600 -> 5600
        return x.replace(',', '')

    output_list = read_text_data(input_datapath)
    gold_list = read_json_data(test_datapath)

    # pattern = r"\\boxed\{(.*?)\}"
    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    num_samples = len(gold_list)
    # print(len(output_list), len(gold_list))
    assert len(output_list) == len(gold_list) == num_samples

    count_none = 0
    correct = 0
    for output, gold in zip(output_list, gold_list):
        gold = gold['answer'].split("#### ")[-1]

        matches1 = pattern1_re.findall(output)
        matches2 = pattern2_re.findall(output)
        matches3 = pattern3_re.findall(output)
        matches4 = pattern4_re.findall(output)
        matches5 = pattern5_re.findall(output)

        if len(matches1) >= 1:
            extracted_answer = matches1[-1]
        elif len(matches2) >= 1:
            extracted_answer = matches2[-1]
        elif len(matches3) >= 1:
            extracted_answer = matches3[-1]
        elif len(matches4) >= 1:
            extracted_answer = matches4[-1]
        elif len(matches5) >= 1:
            extracted_answer = matches5[-1]
        else:
            extracted_answer = None

        if extracted_answer is None:
            count_none += 1
            continue

        # extracted_answer = _maybe_remove_comma(extracted_answer)
        extracted_answer = math_answer_cleaning(extracted_answer)
        # gold = _maybe_remove_comma(gold)
        gold = math_answer_cleaning(gold)

        # if extracted_answer == gold:
        if math_equal(extracted_answer, gold):
            correct += 1
        elif is_equal_after_calculation(extracted_answer, gold):
            correct += 1

    acc = correct / len(gold_list)
    print("count_none:", count_none)
    print("accuracy:", acc)
    return acc


def is_completely_wrapped_by_text(input_string):
    pattern = r'^\\text{(.*)}$'
    match = re.match(pattern, input_string)
    if match:
        ## input_string is completely wrapped by \text{}
        extracted_content = match.group(1)
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    else:
        return None


def math_answer_cleaning(answer):
    ## remove irrelevant text and space to see whether it is exact match
    extracted_content = is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    ## convert 5,\!460 into 5460; convert 14{,}916 into 14916; convert \$4 into 4; convert 50\\\% into 50
    answer = answer.replace(",\!", "").replace("{,}", "").replace("\$", "")
    ## convert \dfrac{3}{2} into frac{3}{2}
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    ## convert 121^\circ into 121
    answer = answer.replace("^\circ", "")
    answer = answer.replace("^{\circ}", "")
    ## remove \quad
    answer = answer.replace("\quad", "")
    ## convert 558\,\text{nm} into 558
    answer = re.sub(r'\\,\\text\{.*?\}', '', answer)
    ## convert 558\text{nm} into 558
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    ## convert 2.45e6^{-1} into 2.45e6; "15000^{-2}^{-1}" into "15000"
    answer = re.sub(r'(\s\^\{-\d+\})', '', answer)
    ## remove space
    answer = answer.replace(" ", "")
    ## remove \n
    answer = answer.replace("\n", "").replace("\\n", "")
    ## convert 3.54\times10^{10} into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}', r'\1e\2', answer)
    ## convert 3.54\times10^10 into 3.54e10
    answer = re.sub(r'([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)', r'\1e\2', answer)
    ## convert 2^{10} into 2^10
    answer = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', answer)
    ## convert 10^{-5} into 1e-5; 10^{5} into 1e5
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    ## remove comma
    answer = answer.replace(",", "")
    ## lowercase
    answer = answer.lower()

    ## convert 7.04e5\ into 7.04e5
    if answer.endswith("\\"):
        answer = answer[:-1]

    ## convert f(x)=ax+b into ax+b; convert z=123 into 123; convert t_r=123 into 123
    func_pattern = r'^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$'
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0])<=3):
        answer = answer.split("=", 1)[1]

    return answer


def round_number(answer):
    def _is_float(string):
        try:
            float(string)
            return True
        except:
            return False

    if _is_float(answer) and float(answer) < 1:
        ## to consider the case like 5.56e-10 (convert 5.56e-10 into 5.6e-10)
        ## still return a string type
        return f"{float(answer):.2g}"

    return answer


def evaluate_math500_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    print("reading from %s" % test_datapath)

    # suppress_prints()
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            answer = item['answer']
            gold_list.append(answer)

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            # match = re.search(pattern1, line)
            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    # restore_prints()

    acc = correct / len(gold_list)
    print("len(gold_list):", len(gold_list))
    print("count_output_none:", count_output_none)
    print("count_timeout:", count_timeout)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc


def evaluate_minerva_math_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    # pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    # pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    solution_list = []
    print("reading from %s" % test_datapath)

    # suppress_prints()
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            problem = item['problem']
            solution = item['solution']
            matches1 = pattern1_re.findall(solution)
            if len(matches1) == 0:
                extracted_answer = None
            else:
                extracted_answer = matches1[-1]
            gold_list.append(extracted_answer)
            solution_list.append(solution)

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            # match = re.search(pattern1, line)
            matches1 = pattern1_re.findall(line)
            # matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            # elif len(matches2) >= 1:
            #     extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            ## remove unit
            unit_list = ["\\hbar^{4}"]
            for unit in unit_list:
                if extracted_answer.endswith(unit):
                    extracted_answer = extracted_answer[:-len(unit)]

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)
                if math_equal(extracted_answer, gold):
                    correct += 1
                elif round_number(extracted_answer) == round_number(gold):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    # restore_prints()

    acc = correct / len(gold_list)
    print("len(gold_list):", len(gold_list))
    print("count_output_none:", count_output_none)
    print("count_timeout:", count_timeout)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc


def calculate_numbers(input_string):
    try:
        result = eval(input_string)
        return result
    except:
        return None


def is_equal_after_calculation(extracted_answer, gold):
    ## convert \frac{3}{2} into 3/2
    gold = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', gold)
    extracted_answer = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1/\2)', extracted_answer)
    gold_result = calculate_numbers(gold)
    extracted_answer_result = calculate_numbers(extracted_answer)

    if gold_result and gold_result == extracted_answer_result:
        return True
    else:
        return False


def is_math_formula_equal(extracted_answer, gold):

    try:
        extracted_answer_expr = parse_latex(extracted_answer)
        gold_expr = parse_latex(gold)

        return simplify(extracted_answer_expr - gold_expr) == 0

    except Exception as e:
        print("error:", e)
        return False


def check_after_fraction_mapping(extracted_answer, gold):
    return re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', extracted_answer) == re.sub(r'\\frac{(.*?)}{(.*?)}', r'\1/\2', gold)


def evaluate_gaokao2023en_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    unit_strings = ['students', 'dollars', 'boxes', 'feet', 'kilometers', 'meters', 'degreesontheBreadusscale', '$', 'a.m.', 'am', 'minutes']

    gold_list = []
    question_list = []
    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item['question'].strip()
            question_list.append(question)
            answer = item['answer']
            answer = re.sub(r'^\$(.*)\$$', r'\1', answer)
            gold_list.append(answer)

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0
    print("reading from %s" % input_datapath)

    # suppress_prints()
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]
            question = question_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            if len(gold) == 1 and gold.isalpha():
                ## gold is a option like A, B, C, D
                ## need to extract the content of this option
                option = "(" + gold + ")"
                assert option in question
                start_option_idx = question.index(option)
                next_option = "(" + chr(ord(gold)+1) + ")"
                next_option2 = "(" + chr(ord(gold)+2) + ")"
                if next_option in question:
                    end_option_idx = question.index(next_option)
                    gold2 = question[start_option_idx: end_option_idx]
                elif next_option2 in question:
                    end_option_idx = question.index(next_option2)
                    gold2 = question[start_option_idx: end_option_idx]
                else:
                    gold2 = question[start_option_idx:]

                gold2 = gold2.replace(option, "").strip().replace("\\n", "")
            else:
                gold2 = None

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)
            if gold2:
                gold2 = math_answer_cleaning(gold2)

            for unit in unit_strings:
                extracted_answer = extracted_answer.replace(unit, "")
                gold = gold.replace(unit, "")
                if gold2:
                    gold2 = gold2.replace(unit, "")

            if "=" in extracted_answer and not gold2:
                ## convert x=3 into 3
                extracted_answer = extracted_answer.split("=", 1)[1]

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1
                elif gold2 and math_equal(extracted_answer, gold2):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    # restore_prints()

    acc = correct / len(gold_list)
    print("benchmark size:", len(gold_list))
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc


def evaluate_olympiadbench_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    question_list = []

    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            assert len(item['final_answer']) == 1
            answer = item['final_answer'][0]
            answer = re.sub(r'^\$(.*)\$$', r'\1', answer)
            gold_list.append(answer)
            question_list.append(item['question'])

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0

    print("reading from %s" % input_datapath)

    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1

                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    acc = correct / len(gold_list)
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("count_timeout:", count_timeout)
    print("accuracy:", acc)

    return acc


def evaluate_collegemath_zeroshot(input_datapath, test_datapath):

    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    # pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"
    pattern6 = r"\\\[\s*(.*?)\s*\\\]"

    def _extra_content_from_gold(input_string):
        pattern_extra = r'\$(.*?)\$'
        matches = re.findall(pattern_extra, input_string)
        return matches

    def _extra_cleaning(input_string):
        ## convert 558\mathrm{ft} into 558
        input_string = re.sub(r'\\mathrm\{.*?\}', '', input_string)
        input_string = re.sub(r'\$\([^)]*\)', '', input_string)
        return input_string

    def _check_after_equal(extracted_answer_ori, gold_ori, matches):

        if extracted_answer_ori.replace(",", "").replace("$", "") == gold_ori.replace(",", "").replace("$", ""):
            return True

        if not extracted_answer_ori.split("=")[-1] == gold_ori.split("=")[-1].replace("$", ""):
            return False

        if "," in gold_ori or "or" in gold_ori or "and" in gold_ori:
            ## there are multiple answers
            if len(matches) <= 1:
                return False

            answer1 = extracted_answer_ori.split("=")[-1]
            answer2 = math_answer_cleaning(matches[-2].split("=")[-1])
            gold_ori = gold_ori.replace("$", "")
            if "or" in gold_ori:
                gold_list = gold_ori.split("or", 1)
            elif "and" in gold_ori:
                gold_list = gold_ori.split("and", 1)
            else:
                gold_list = gold_ori.split(",", 1)

            gold_ori1 = gold_list[-1].split("=")[-1]
            gold_ori2 = gold_list[-2].split("=")[-1]

            if math_equal(answer1, gold_ori1) and math_equal(answer2, gold_ori2):
                return True
            elif math_equal(answer2, gold_ori1) and math_equal(answer1, gold_ori2):
                return True

            return False

        else:
            return True

    pattern1_re = re.compile(pattern1, re.DOTALL)
    # pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)
    pattern6_re = re.compile(pattern6, re.DOTALL)

    gold_list = []
    question_list = []
    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            answer = item['answer']
            answer = re.sub(r'^\$(.*)\$$', r'\1', answer)
            gold_list.append(answer)
            question_list.append(item['question'])

    count_output_none = 0
    count_answer_none = 0
    count_timeout = 0
    correct = 0
    print("reading from %s" % input_datapath)
    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            # matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)
            matches6 = pattern6_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            # elif len(matches2) >= 1:
            #     extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            elif len(matches6) >= 1:
                extracted_answer = matches6[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)
            extracted_answer = _extra_cleaning(extracted_answer)
            gold = _extra_cleaning(gold)
            if gold.endswith("-"):
                gold = gold[:-1]
            if gold.endswith("."):
                gold = gold[:-1]
            if gold.endswith("hours"):
                gold = gold[:-len("hours")]
            if extracted_answer.endswith("."):
                extracted_answer = extracted_answer[:-1]

            extracted_answer_ori = extracted_answer
            gold_ori = gold

            if "=" in gold:
                gold = gold.split("=", 1)[1]
            if ":" in gold:
                gold = gold.split(":", 1)[1]

            if "=" in extracted_answer:
                extracted_answer = extracted_answer.split("=", 1)[1]
            if ":" in extracted_answer:
                extracted_answer = extracted_answer.split(":", 1)[1]

            ## \emptyset and \oslash both reprsent empty set in latex
            extracted_answer = extracted_answer.replace("\\emptyset", "\\oslash")

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1
                elif _check_after_equal(extracted_answer_ori, gold_ori, matches1):
                    correct += 1
                elif _check_after_equal(extracted_answer, gold, matches1):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1
                elif round_number(extracted_answer) == round_number(gold):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1
                else:
                    # restore_prints()
                    gold_matches = _extra_content_from_gold(gold)

                    correctflag = False
                    if len(gold_matches) >= 1:
                        gold2 = "".join(gold_matches)
                        if "\\approx" in gold2:
                            gold2 = gold2.split("\\approx", 1)[1]
                        ## convert 1,500 into 1500
                        gold2 = gold2.replace(",", "")
                        extracted_answer2 = extracted_answer.replace(",", "")
                        if gold2 != "" and extracted_answer2 == gold2:
                            correct += 1
                            correctflag = True

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    acc = correct / len(gold_list)
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("count_timeout:", count_timeout)
    print("accuracy:", acc)

    return acc


def evaluate_amc23_or_aime24_zeroshot(input_datapath, test_datapath):

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    question_list = []
    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            # print(item)
            # print(item.keys())
            # exit(1)
            answer = str(item['answer'])
            gold_list.append(answer)
            question_list.append(item['problem'])

    count_output_none = 0
    count_answer_none = 0
    correct = 0
    print("reading from %s" % input_datapath)

    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            if math_equal(extracted_answer, gold):
                correct += 1
            elif round_number(extracted_answer) == round_number(gold):
                correct += 1
            elif is_equal_after_calculation(extracted_answer, gold):
                correct += 1

    acc = correct / len(gold_list)
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc


def evaluate_omnimath_zeroshot(input_datapath, test_datapath):
    class _TimeoutException(Exception):
        pass
    def _timeout_handler(signum, frame):
        # raise Exception("Function took too long to complete.")
        raise _TimeoutException

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    gold_list = []
    question_list = []
    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            answer = str(item['answer'])
            gold_list.append(answer)
            question_list.append(item['problem'])

    count_output_none = 0
    count_answer_none = 0
    correct = 0
    print("reading from %s" % input_datapath)

    with open(input_datapath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            line = json.loads(line)['output']

            matches1 = pattern1_re.findall(line)
            matches2 = pattern2_re.findall(line)
            matches3 = pattern3_re.findall(line)
            matches4 = pattern4_re.findall(line)
            matches5 = pattern5_re.findall(line)

            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            elif len(matches3) >= 1:
                extracted_answer = matches3[-1]
            elif len(matches4) >= 1:
                extracted_answer = matches4[-1]
            elif len(matches5) >= 1:
                extracted_answer = matches5[-1]
            else:
                extracted_answer = None
            gold = gold_list[i]

            if extracted_answer is None:
                # print("="*80)
                # print(line)
                count_output_none += 1
                continue
            if gold is None:
                count_answer_none += 1
                continue

            gold_ori = gold
            extracted_answer = math_answer_cleaning(extracted_answer)
            gold = math_answer_cleaning(gold)

            try:
                # raise exception after 5 sections
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(5)

                if math_equal(extracted_answer, gold):
                    correct += 1
                elif check_after_fraction_mapping(extracted_answer, gold):
                    correct += 1
                elif round_number(extracted_answer) == round_number(gold):
                    correct += 1
                elif is_equal_after_calculation(extracted_answer, gold):
                    correct += 1

                ## Disable the alarm
                signal.alarm(0)

            except:
                ## Disable the alarm
                signal.alarm(0)
                count_timeout += 1

    acc = correct / len(gold_list)
    print("count_output_none:", count_output_none)
    print("count_answer_none:", count_answer_none)
    print("accuracy:", acc)

    return acc


def get_answer_by_marjority_voting(output_list):

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern2 = r"\*\*(.*?)\*\*"
    pattern3 = r"\\\[\n(.*?)\n\\\]"
    pattern4 = r'is \\\((.*?)\\\)'
    pattern5 = r"\\\[\\n(.*?)\\n\\\]"

    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(pattern2, re.DOTALL)
    pattern3_re = re.compile(pattern3, re.DOTALL)
    pattern4_re = re.compile(pattern4, re.DOTALL)
    pattern5_re = re.compile(pattern5, re.DOTALL)

    answer_dict = {}
    for output in output_list:
        matches1 = pattern1_re.findall(output)
        matches2 = pattern2_re.findall(output)
        matches3 = pattern3_re.findall(output)
        matches4 = pattern4_re.findall(output)
        matches5 = pattern5_re.findall(output)

        if len(matches1) >= 1:
            extracted_answer = matches1[-1]
        elif len(matches2) >= 1:
            extracted_answer = matches2[-1]
        elif len(matches3) >= 1:
            extracted_answer = matches3[-1]
        elif len(matches4) >= 1:
            extracted_answer = matches4[-1]
        elif len(matches5) >= 1:
            extracted_answer = matches5[-1]
        else:
            extracted_answer = None

        if extracted_answer is None:
            continue

        extracted_answer = math_answer_cleaning(extracted_answer)

        has_found = False
        for prev_ans in answer_dict:
            if extracted_answer == prev_ans:
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif math_equal(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif check_after_fraction_mapping(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif round_number(extracted_answer) == round_number(prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break
            elif is_equal_after_calculation(extracted_answer, prev_ans):
                answer_dict[prev_ans]['count'] += 1
                has_found = True
                break

        if not has_found:
            answer_dict[extracted_answer] = {"count": 1, "original_output": output}

    ## rank the answer based on count
    sorted_answers = sorted(answer_dict, key=lambda x: answer_dict[x]["count"], reverse=True)

    return answer_dict[sorted_answers[0]]


def evaluate_gpqa(input_datapath, test_datapath):
    """Evaluate GPQA (Graduate-Level Google-Proof Q&A) benchmark.
    
    Extracts answers from multiple formats and compares against correct choices
    using both exact matching and mathematical equivalence checking.
    
    Args:
        input_datapath: Path to model output JSONL file
        test_datapath: Path to GPQA test JSON file
    
    Returns:
        float: Accuracy score
    """
    class _TimeoutException(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _TimeoutException

    output_list = read_text_data(input_datapath)
    gold_list = read_json_data(test_datapath)

    num_samples = len(gold_list)
    assert len(output_list) == len(gold_list) == num_samples

    pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    pattern1_re = re.compile(pattern1, re.DOTALL)
    pattern2_re = re.compile(r'\b(?:Answer|Final Answer|ANSWER)\b[:\s\*]*\(?([A-D])\)?')

    count_none = 0
    count_timeout = 0
    correct = 0

    for output, gold in zip(output_list, gold_list):
        choices = [gold['choice_A'], gold['choice_B'], gold['choice_C'], gold['choice_D']]
        correct_answer = gold["correct_answer"]
        correct_index = choices.index(correct_answer)
        correct_choice = "ABCD"[correct_index]

        extracted_answer = get_option_char(output)
        if extracted_answer is None:
            matches1 = pattern1_re.findall(output)
            matches2 = pattern2_re.findall(output)
            if len(matches1) >= 1:
                extracted_answer = matches1[-1]
            elif len(matches2) >= 1:
                extracted_answer = matches2[-1]
            else:
                extracted_answer = None

        if extracted_answer is None:
            count_none += 1
            continue

        correct_answer = math_answer_cleaning(correct_answer)
        extracted_answer = math_answer_cleaning(extracted_answer)

        try:
            # raise exception after 5 sections
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(5)

            if extracted_answer.lower() == correct_choice.lower():
                correct += 1
            elif math_equal(extracted_answer, correct_answer):
                correct += 1
            elif "("+correct_choice+")" in extracted_answer:
                correct += 1

            signal.alarm(0)

        except:
            signal.alarm(0)
            count_timeout += 1

    acc = correct / num_samples
    print("num_samples:", num_samples)
    print("count_none:", count_none)
    print("accuracy:", acc)

    return acc

def get_args():
    """Parse command-line arguments for GPQA evaluation script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="GPQA Benchmark Evaluation")
    parser.add_argument("--modelfolder", type=str, required=True, 
                       help="Path to model output folder")
    parser.add_argument("--testfolder", type=str, required=True,
                       help="Path to test data folder")
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function for GPQA benchmark."""
    args = get_args()

    model_folder = args.modelfolder
    test_datafolder = args.testfolder

    gpqa_accs = []

    input_datapaths = glob.glob(model_folder+"/outputs_*/gpqa_diamond.jsonl")
    
    if not input_datapaths:
        print(f"No GPQA output files found in {model_folder}")
        return
    
    acc_tmp = 0
    for input_datapath in input_datapaths:
        test_datapath = os.path.join(test_datafolder, "gpqa/gpqa_diamond.json")
        print(f"\nEvaluating: {input_datapath}")
        acc = evaluate_gpqa(input_datapath, test_datapath)
        gpqa_accs.append(acc)
        acc_tmp += acc

    gpqa_acc = acc_tmp / len(input_datapaths)
    gpqa_std = np.std(gpqa_accs) if len(gpqa_accs) > 1 else 0
    
    print("="*80)
    print(f"Average accuracy for GPQA: {gpqa_acc:.4f} ± {gpqa_std:.4f}")
    print(f"Number of runs evaluated: {len(gpqa_accs)}")


if __name__ == "__main__":
    main()