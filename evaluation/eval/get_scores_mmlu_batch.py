"""MMLU benchmark evaluation utilities.

This module provides evaluation functions for MMLU (Massive Multitask Language Understanding)
benchmark variants including MMLU, MMLU Pro, and MMLU Redux. It supports boxed answer
extraction and multiple-choice answer formats.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas


def read_jsonl_data(datapath):
    """Read model outputs from JSONL file.
    
    Args:
        datapath: Path to JSONL file containing model outputs
    
    Returns:
        list: List of output strings
    """
    data_list = []
    with open(datapath, "r") as f:
        for line in f:
            data_item = json.loads(line.strip())
            data_list.append(data_item['output'])
    return data_list


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
        str or None: Extracted letter, or None if no match found
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


def evaluate_mmlu_boxed(input_datapath, test_datapath, print_per_category_acc=False):
    """Evaluate MMLU with boxed answer format.
    
    Supports batch split files for large-scale evaluation.
    
    Args:
        input_datapath: Path to model output JSONL file (or base path for splits)
        test_datapath: Path to MMLU test CSV file
        print_per_category_acc: Whether to print per-category accuracy (default: False)
    
    Returns:
        float: Accuracy score
    """
    print("reading from %s" % test_datapath)
    df = pandas.read_csv(test_datapath)

    if os.path.exists(input_datapath):
        output_list = read_jsonl_data(input_datapath)
        test_list = [row.to_dict() for _, row in df.iterrows()]
    else:
        test_list_full = [row.to_dict() for _, row in df.iterrows()]
        test_list = []

        output_list = []
        for i in range(32):
            input_datapath_i = os.path.join('/'.join(input_datapath.split("/")[:-1]), f"mmlu_{i*439}to{((i+1)*439)}_thinking.jsonl")
            if os.path.exists(input_datapath_i):
                output_list.extend(read_jsonl_data(input_datapath_i))
                test_list += test_list_full[i*439:(i+1)*439]

    print("#Test_samples:", len(test_list), "#Output_samples:", len(output_list))
    assert len(output_list) == len(test_list)

    correct = 0
    count_not_found = 0

    by_subject_len = {}
    by_subject_correct = {}

    for i, (output, gold) in enumerate(zip(output_list, test_list)):
        subject = gold['Subject']
        if subject not in by_subject_len:
            by_subject_len[subject] = 0
            by_subject_correct[subject] = 0
        by_subject_len[subject] += 1

        output_ans = get_option_char(output)
        if output_ans is None:
            pattern2 = r'\((A|B|C|D)\)'
            match2 = re.search(pattern2, output)
            output_ans = match2.group(1) if match2 else None
            if output_ans is None:
                for opt_letter in ['A', 'B', 'C', 'D']:
                    if str(output).lower().strip() == str(test_list[i][opt_letter]).lower().strip():
                        output_ans = opt_letter
                        break

            if output_ans is None:
                count_not_found += 1

        gold_ans = gold['Answer']
        if output_ans == gold_ans:
            correct += 1
            by_subject_correct[subject] += 1

    if print_per_category_acc:
        for subject in by_subject_len:
            print(f"{subject}: {by_subject_len[subject]}, {by_subject_correct[subject] / by_subject_len[subject]}")

    print("count_not_found:", count_not_found)

    acc = correct / len(output_list)
    print("score:", acc)
    return acc


def evaluate_mmlu_redux_boxed(input_datapath, test_datapath, print_per_category_acc=False):
    """Evaluate MMLU-Redux with boxed answer format.
    
    MMLU-Redux is a refined version of MMLU with improved question quality.
    
    Args:
        input_datapath: Path to model output JSONL file (or base path for splits)
        test_datapath: Path to MMLU-Redux test JSON file
        print_per_category_acc: Whether to print per-category accuracy (default: False)
    
    Returns:
        float: Accuracy score
    """
    if os.path.exists(input_datapath):
        output_list = read_jsonl_data(input_datapath)
        test_list = json.load(open(test_datapath, "r"))
    else:
        test_list_full = json.load(open(test_datapath, "r"))
        test_list = []
        output_list = []
        for i in range(8):
            input_datapath_i = os.path.join('/'.join(input_datapath.split("/")[:-1]), f"mmlu_redux_{i*375}to{((i+1)*375)}_thinking.jsonl")
            if os.path.exists(input_datapath_i):
                output_list.extend(read_jsonl_data(input_datapath_i))
                test_list += test_list_full[i*375:(i+1)*375]

    print("#Test_samples:", len(test_list), "#Output_samples:", len(output_list))
    assert len(output_list) == len(test_list)

    correct = 0
    count_not_found = 0

    by_subject_len = {}
    by_subject_correct = {}

    for i, (output, gold) in enumerate(zip(output_list, test_list)):
        subject = gold['category']
        if subject not in by_subject_len:
            by_subject_len[subject] = 0
            by_subject_correct[subject] = 0
        by_subject_len[subject] += 1

        output_ans = get_option_char(output)
        if output_ans is None:
            pattern2 = r'\((A|B|C|D)\)'
            match2 = re.search(pattern2, output)
            output_ans = match2.group(1) if match2 else None
            if output_ans is None:
                for j, opt_letter in enumerate(['A', 'B', 'C', 'D']):
                    if str(output).lower().strip() == str(test_list[i]['options'][j]).lower().strip():
                        output_ans = opt_letter
                        break

            if output_ans is None:
                count_not_found += 1

        gold_ans = gold['answer']
        if output_ans == gold_ans:
            correct += 1
            by_subject_correct[subject] += 1

    if print_per_category_acc:
        for subject in by_subject_len:
            print(f"{subject}: {by_subject_len[subject]}, {by_subject_correct[subject] / by_subject_len[subject]}")

    print("count_not_found:", count_not_found)

    acc = correct / len(output_list)
    print("score:", acc)
    return acc


def evaluate_mmlu_pro_boxed(input_datapath, test_datapath, print_per_category_acc=False):
    """Evaluate MMLU-Pro with boxed answer format.
    
    MMLU-Pro extends MMLU with more challenging questions and up to 16 options (A-P).
    
    Args:
        input_datapath: Path to model output JSONL file (or base path for splits)
        test_datapath: Path to MMLU-Pro test JSON file
        print_per_category_acc: Whether to print per-category accuracy (default: False)
    
    Returns:
        float: Accuracy score
    """
    if os.path.exists(input_datapath):
        output_list = read_jsonl_data(input_datapath)
        test_list = json.load(open(test_datapath, "r"))
    else:
        test_list_full = json.load(open(test_datapath, "r"))
        test_list = []
        output_list = []
        for i in range(32):
            input_datapath_i = os.path.join('/'.join(input_datapath.split("/")[:-1]), f"mmlu_pro_{i*376}to{((i+1)*376)}_thinking.jsonl")
            if os.path.exists(input_datapath_i):
                output_list.extend(read_jsonl_data(input_datapath_i))
                test_list += test_list_full[i*376:(i+1)*376]
        if len(output_list) == 0:
            for i in range(64):
                input_datapath_i = os.path.join('/'.join(input_datapath.split("/")[:-1]), f"mmlu_pro_{i*188}to{((i+1)*188)}_thinking.jsonl")
                if os.path.exists(input_datapath_i):
                    output_list.extend(read_jsonl_data(input_datapath_i))
                    test_list += test_list_full[i*188:(i+1)*188]
                else:
                    print(f"mmlu_pro_{i*188}to{((i+1)*188)}_thinking.jsonl not found")

    print("#Test_samples:", len(test_list), "#Output_samples:", len(output_list))
    assert len(output_list) == len(test_list)

    correct = 0
    count_not_found = 0

    by_subject_len = {}
    by_subject_correct = {}

    for output, gold in zip(output_list, test_list):
        subject = gold['category']
        if subject not in by_subject_len:
            by_subject_len[subject] = 0
            by_subject_correct[subject] = 0
        by_subject_len[subject] += 1

        output_ans = get_option_char(output)
        if output_ans is None:
            pattern2 = r'\((A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P)\)'
            pattern2_re = re.compile(pattern2, re.IGNORECASE)
            matches2 = pattern2_re.findall(output)
            if len(matches2) >= 1:
                output_ans = matches2[-1]
            else:
                output_ans = None

        if output_ans is None:
            count_not_found += 1

        gold_ans = gold['answer']
        if output_ans and output_ans.lower() == gold_ans.lower():
            correct += 1
            by_subject_correct[subject] += 1

    if print_per_category_acc:
        for subject in by_subject_len:
            print(f"{subject}: {by_subject_len[subject]}, {by_subject_correct[subject] / by_subject_len[subject]}")

    acc = correct / len(output_list)
    print("count_not_found:", count_not_found)
    print("score:", acc)

    return acc


def get_args():
    """Parse command-line arguments for MMLU evaluation script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MMLU Benchmark Evaluation")
    parser.add_argument("--modelfolder", type=str, required=True, 
                       help="Path to model output folder")
    parser.add_argument("--testfolder", type=str, required=True,
                       help="Path to test data folder")
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print per-category accuracy')
    args = parser.parse_args()
    return args


def main():
    """Main evaluation function for MMLU benchmark variants."""
    args = get_args()
    model_folder = args.modelfolder
    test_datafolder = args.testfolder
    print_per_category_acc = args.verbose

    print("Evaluating MMLU Pass@1 results...")
    import glob
    input_datapaths = glob.glob(f"{model_folder}/outputs_*/mmlu_r1_0to15000.jsonl")
    test_datapath = os.path.join(test_datafolder, "mmlu/mmlu_test.csv")

    accs = []
    for input_datapath in input_datapaths:
        acc = evaluate_mmlu_boxed(input_datapath, test_datapath, print_per_category_acc)
        accs.append(acc)
    mmlu_acc = np.mean(accs)
    mmlu_std = np.std(accs) if len(accs) > 1 else 0
    print("avg acc for MMLU:", mmlu_acc, "std:", mmlu_std)


    input_datapaths = os.path.join(f"{model_folder}/outputs_vllm073_topp0.95_seed111", "mmlu_pro_thinking.jsonl")
    test_datapath = os.path.join(test_datafolder, "mmlu_pro/test.json")
    mmlu_pro_accs = []
    acc = evaluate_mmlu_pro_boxed(input_datapaths, test_datapath, print_per_category_acc)
    mmlu_pro_accs.append(acc)
    mmlu_pro_acc = np.mean(mmlu_pro_accs)
    mmlu_pro_std = np.std(mmlu_pro_accs) if len(mmlu_pro_accs) > 1 else 0
    print("avg acc for MMLU-Pro:", mmlu_pro_acc, "std:", mmlu_pro_std)

if __name__ == "__main__":
    main()