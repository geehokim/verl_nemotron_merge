"""Code generation benchmark evaluation utilities.
This module provides evaluation functions for coding benchmarks including:
- LiveCodeBench (v5 and v6)
It includes code execution, test case verification, and correctness checking.
"""

import argparse
import copy
import glob
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
from tqdm import tqdm

from tools.code_verifier_utils import run_test


def check_coding_correctness(problem_to_check: Optional[dict], timeout, debug=False):
    """Check correctness of code generation with a global timeout.
    
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside run_test.
    
    Args:
        problem_to_check: Dictionary containing code and test cases
        timeout: Timeout in seconds for each test case
        debug: Whether to enable debug mode (default: False)
    
    Returns:
        bool: True if all test cases pass, False otherwise
    """
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(problem_to_check, debug, result, metadata_list, timeout):
        try:
            res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception as e:
            result.append([-1 for i in range(len(problem_to_check['input_output']))])
            metadata_list.append(e)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    total_timeout = (timeout + 1) * len(problem_to_check['input_output']) + 10
    p = multiprocessing.Process(target=_temp_run, args=(problem_to_check, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()

    judge_value = bool(result and np.all(np.array(result[0]) > 0))
    return judge_value


def update_results(result, timeout=10):
    """Update results with correctness checking.
    
    Args:
        result: Dictionary containing generated code and test cases
        timeout: Timeout in seconds for code execution (default: 6)
    
    Returns:
        dict: Response entry with correctness status and reason
    """
    response_entry = {
        "content": result['generation'],
        "correctness": None,
        "reason": None,
    }

    problem_to_check = copy.deepcopy(result)
    curr_res = check_coding_correctness(problem_to_check, timeout=timeout)

    if curr_res:
        response_entry["correctness"] = True
        response_entry["reason"] = ""
    else:
        response_entry["correctness"] = False
        response_entry["reason"] = "Code is incorrect."

    return response_entry


def evaluate_livecodebench(input_datapath, test_datapath):
    """Evaluate LiveCodeBench code generation performance.
    
    Args:
        input_datapath: Path to model output JSONL file
        test_datapath: Path to LiveCodeBench test JSON file
    
    Returns:
        float: Accuracy score (proportion of correctly solved problems)
    """
    print("reading from %s" % input_datapath)
    id2generation = {}
    with open(input_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            id2generation[item['task_id']] = item['output']
    print("length of id2generation:", len(id2generation))

    print("reading from %s" % test_datapath)
    with open(test_datapath, "r") as f:
        test_list = json.load(f)
    print("length of test_list:", len(test_list))

    combined_results = {}
    for data_item in test_list:
        id_ = data_item['question_id']
        output = id2generation[id_]

        all_testcases = data_item['private_test_cases'] + json.loads(data_item['public_test_cases'])
        
        metadata = json.loads(data_item['metadata'])
        if "func_name" in metadata:
            func_name = metadata['func_name']
        else:
            func_name = ""
        
        combined_results[id_] = {
            'input_output': all_testcases,
            'starter_code': func_name,
            'question_id': id_,
            'generation': output
        }

    total_questions = len(combined_results)
    print("length of combined_results:", total_questions)

    total_correct = 0
    total_finish = 0
    records = []

    with ProcessPoolExecutor(max_workers=32) as executor:

        future_to_task = {}
        token_usages = {}
        for idx, (q_id, result) in enumerate(combined_results.items()):
            future_to_task[
                executor.submit(
                    update_results, result
                )
            ] = idx

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Processing Generations",
        ):
            idx = future_to_task[future]
            response_entry = future.result()
            total_correct += response_entry["correctness"]
            total_finish += 1
            records.append(response_entry)
    
    acc = total_correct / total_questions
    print("accuracy:", acc)

    return acc


def get_args():
    """Parse command-line arguments for code evaluation script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Code Benchmark Evaluation")
    parser.add_argument("--modelfolder", type=str, required=True, 
                       help="Path to model output folder")
    parser.add_argument("--testfolder", type=str, required=True,
                       help="Path to test data folder")
    args = parser.parse_args()
    return args

PATTERN = re.compile(
    r"(?ms)^```python(?:\w+)?\n"          # opener at start of a line
    r"(?P<code>(?:(?!^```python).)*?)"    # content that never starts a new ###python
    r"^\s*```\s*$"                        # closer '###' on its own line
)

def has_code(response):
    """Check if response contains Python code blocks.
    
    Args:
        response: Model output string
    
    Returns:
        str: Last code blocks found in the response
    """
    matches = list(PATTERN.finditer(response))
    return matches[-1].group("code") if matches else None

def check_finish(input_datapath):
    finish_rates = []
    with open(input_datapath, "r") as f:
        for line in f:
            item = json.loads(line)
            if not item['reason']:
                finish_rates.append(0)
            output = item['output']
            finish_rates.append(1 if has_code(output) else 0)

    return np.mean(finish_rates)


def main():
    """Main evaluation function for code generation benchmarks."""
    args = get_args()

    model_folder = args.modelfolder
    test_datafolder = args.testfolder

    # Evaluate LiveCodeBench v5
    tmp_list = []
    finish_list = []
    input_datapaths = glob.glob(model_folder+"/outputs_*/lcb5_2408_2502.jsonl")
    for input_datapath in input_datapaths:
        test_datapath = os.path.join(test_datafolder, "livecodebench/test_aug2024tojan2025.json")
        print("="*80)
        lines = open(input_datapath).readlines()
        if len(lines) != 279:
            print(f"skipping {input_datapath} due to incorrect number of lines {len(lines)}")
            continue
        tmp_acc = evaluate_livecodebench(input_datapath, test_datapath)
        finish_rate = check_finish(input_datapath)
        tmp_list.append(tmp_acc)
        finish_list.append(finish_rate)

    acc = np.mean(tmp_list)
    finish = np.mean(finish_list)
    finish_std = np.std(finish_list)/(len(finish_list)**0.5)

    lcb5_acc = acc
    lcb5_finish = finish
    lcb5_std = np.std(tmp_list)/(len(tmp_list)**0.5)

    print("="*80)
    print("avg acc for livecodebench v5 (2408-2502): %.4f (std of mean: %.4f) (runs: %d)" % (lcb5_acc, lcb5_std, len(tmp_list)))
    print("avg finish rate for livecodebench v5 (2408-2502): %.4f (std of mean: %.4f) (runs: %d)" % (finish, finish_std, len(finish_list)))

    # Evaluate LiveCodeBench v6
    tmp_list = []
    finish_list = []
    input_datapaths = glob.glob(model_folder+"/outputs_*/lcb6_2502_2505.jsonl")
    for input_datapath in input_datapaths:
        test_datapath = os.path.join(test_datafolder, "livecodebench/test_feb2025toApr2025.json")
        print("="*80)
        lines = open(input_datapath).readlines()
        if len(lines) != 175:
            print(f"skipping {input_datapath} due to incorrect number of lines {len(lines)}")
            continue
        tmp_acc = evaluate_livecodebench(input_datapath, test_datapath)
        finish_rate = check_finish(input_datapath)
        tmp_list.append(tmp_acc)
        finish_list.append(finish_rate)

    acc = np.mean(tmp_list)
    finish = np.mean(finish_list)
    finish_std = np.std(finish_list) / (len(finish_list) ** 0.5)

    lcb6_acc = acc
    lcb6_finish = finish
    lcb6_std = np.std(tmp_list)/(len(tmp_list)**0.5)

    print("="*80)
    print("avg acc for livecodebench v6 (2502-2505): %.4f (std of mean: %.4f) (runs: %d)" % (lcb6_acc, lcb6_std, len(tmp_list)))
    print("avg finish rate for livecodebench v6 (2502-2505): %.4f (std of mean: %.4f) (runs: %d)" % (finish, finish_std, len(finish_list)))

    print("Final Accuracy for LiveCodeBench v5 (2408-2502): %.4f" % (lcb5_acc)) 
    print("Final Accuracy for LiveCodeBench v6 (2408-2505): %.4f" % ((lcb5_acc * 279 + lcb6_acc * 175)/454))

if __name__ == "__main__":
    main()
