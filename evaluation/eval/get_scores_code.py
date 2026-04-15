"""Code generation benchmark evaluation utilities.
This module provides evaluation functions for coding benchmarks including:
- LiveCodeBench (v5 and v6)
It includes code execution, test case verification, and correctness checking.
"""

import argparse
import copy
import ctypes
import glob
import json
import multiprocessing
import os
import platform
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

from coding_ground_truth import normalize_ground_truth

# Use the AceReason Evaluation Toolkit verifier as the single source of truth
# for code correctness judgement. The local fork in
# ``evaluation/eval/tools/code_verifier_utils.py`` had drifted from the
# upstream reference in ways that materially changed reward semantics
# (notably: no ``</think>`` completeness check, single-string ``has_code``
# instead of a list, and a less robust ``replace_newlines``). To stay
# faithful to the Nemotron paper's described reward function we route
# directly to ``eval_release/tools/code_verifier.py`` and keep our only
# delta there to the explicit ``reliability_guard`` memory cap.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ACEREASON_TOOLS = _Path(__file__).resolve().parents[2] / "eval_release" / "tools"
if str(_ACEREASON_TOOLS) not in _sys.path:
    _sys.path.insert(0, str(_ACEREASON_TOOLS))

from code_verifier import run_test  # noqa: E402  (path-dependent import)


# glibc malloc_trim: forces free heap chunks back to the OS. Useful as a
# safety net when the parent process accumulates fragmented arenas, but
# each call walks the heap free list (~10-100 ms on a heap of a few GB),
# which adds up across hundreds of reward calls per training step. The
# AceReason verifier does the heavy decode in the child, so the parent
# barely allocates anything per call — the original "monotonic RSS leak"
# concern that motivated this hook turned out to be a non-issue
# (validated by a flat ~175 GB plateau over 100 steps without the hook).
# Leave the helper available for opt-in via env (set
# ``VERL_CODING_MALLOC_TRIM=1``) but skip it by default for throughput.
_LIBC: Optional[ctypes.CDLL] = None
try:
    if platform.system() == "Linux":
        _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
        _LIBC.malloc_trim.restype = ctypes.c_int
except OSError:
    _LIBC = None

_MALLOC_TRIM_ENABLED = os.getenv("VERL_CODING_MALLOC_TRIM", "0") == "1"


def _release_parent_memory() -> None:
    if not _MALLOC_TRIM_ENABLED or _LIBC is None:
        return
    try:
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def _get_verifier_mp_context():
    """Return multiprocessing context for verifier subprocesses.

    Default is ``fork``: cheapest spawn (~ms instead of ~50-100 ms for
    forkserver), and the original COW-bloat concern is now mitigated by
    (a) AceReason's ``</think>`` completeness check rejecting infinite
    generations before they execute, and (b) the explicit ``RLIMIT_AS``
    cap inside the child. Override with ``VERL_CODING_MP_START_METHOD``
    if a future deployment ever needs the cleaner forkserver/spawn
    isolation back.
    """
    preferred = os.getenv("VERL_CODING_MP_START_METHOD", "fork").strip().lower()
    supported = multiprocessing.get_all_start_methods()
    if preferred not in supported:
        if "fork" in supported:
            preferred = "fork"
        elif "forkserver" in supported:
            preferred = "forkserver"
        elif supported:
            preferred = supported[0]
        else:
            preferred = "fork"
    return multiprocessing.get_context(preferred)


def _run_test_in_subprocess(problem_to_check, debug, timeout, child_conn):
    """Run verifier in an isolated child process and return the result through a pipe.

    Using Pipe keeps the existing process isolation/timeout behavior without
    spawning an extra multiprocessing.Manager server process for every sample.
    """
    try:
        res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
        child_conn.send((res, metadata))
    except Exception as e:
        fallback = [-1 for _ in range(len(problem_to_check["input_output"]))]
        try:
            child_conn.send((fallback, repr(e)))
        except Exception:
            pass
    finally:
        child_conn.close()


def _decode_and_run_test_in_subprocess(
    raw_ground_truth: Any,
    solution_str: str,
    debug: bool,
    timeout: int,
    child_conn,
) -> None:
    """Child entrypoint that decodes the ground-truth payload inline.

    The base64+zlib+pickle decode of the (potentially MB-sized) ground truth
    happens HERE, inside the short-lived child process. The expanded Python
    dict/list/string objects therefore never live in the parent's address
    space, and the OS reclaims them in full when the child exits — no
    pymalloc/glibc fragmentation accumulates in the long-lived parent.
    """
    num_tests = 0
    try:
        gt = normalize_ground_truth(raw_ground_truth)
        num_tests = int(gt.get("num_tests") or len(gt.get("input_output", [])) or 0)
        problem_to_check = {
            "input_output": gt["input_output"],
            "starter_code": gt["fn_name"] if gt["fn_name"] else "",
            "generation": solution_str,
        }
        res, metadata = run_test(problem_to_check, debug=debug, timeout=timeout)
        child_conn.send((res, metadata, num_tests))
    except Exception as e:
        fallback = [-1] * max(num_tests, 1)
        try:
            child_conn.send((fallback, repr(e), num_tests))
        except Exception:
            pass
    finally:
        child_conn.close()


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

    total_timeout = (timeout + 1) * len(problem_to_check['input_output']) + 10
    ctx = _get_verifier_mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_run_test_in_subprocess,
        args=(problem_to_check, debug, timeout, child_conn),
    )
    p.start()
    child_conn.close()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()
        p.join()

    result = None
    try:
        if parent_conn.poll():
            result, _metadata = parent_conn.recv()
    except EOFError:
        result = None
    finally:
        parent_conn.close()

    judge_value = bool(result and np.all(np.array(result) > 0))
    _release_parent_memory()
    return judge_value


def check_coding_correctness_from_raw(
    raw_ground_truth: Any,
    solution_str: str,
    timeout: int = 6,
    debug: bool = False,
    estimated_num_tests: int = 50,
) -> Tuple[bool, int]:
    """Verify generated code without materializing the decoded ground truth in the parent.

    The raw (still encoded) ground-truth payload is passed straight through to
    a freshly-spawned child process, where ``normalize_ground_truth`` does the
    heavy decode. The parent therefore never grows its pymalloc/glibc heap
    with the big intermediate dicts/strings — the dominant source of the
    monotonically-increasing parent RSS observed across long training runs.

    Args:
        raw_ground_truth: Raw (encoded or pre-parsed) ground-truth payload.
        solution_str: Generated solution to verify.
        timeout: Per-testcase timeout in seconds (passed through to ``run_test``).
        debug: Whether to enable verifier debug printing.
        estimated_num_tests: Conservative upper bound used to derive the
            global wall-clock cap for the child. Real ``num_tests`` is reported
            back from the child for the caller's bookkeeping.

    Returns:
        Tuple of ``(passed, num_tests)`` where ``num_tests`` is the actual
        number of test cases discovered after decoding (0 on decode failure).
    """
    total_timeout = (timeout + 1) * max(int(estimated_num_tests), 1) + 10
    ctx = _get_verifier_mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_decode_and_run_test_in_subprocess,
        args=(raw_ground_truth, solution_str, debug, timeout, child_conn),
    )
    p.start()
    child_conn.close()
    p.join(timeout=total_timeout + 1)
    if p.is_alive():
        p.kill()
        p.join()

    result = None
    num_tests = 0
    try:
        if parent_conn.poll():
            recv = parent_conn.recv()
            result = recv[0]
            num_tests = int(recv[2]) if len(recv) >= 3 else 0
    except EOFError:
        result = None
    finally:
        parent_conn.close()

    judge_value = bool(result and np.all(np.array(result) > 0))
    _release_parent_memory()
    return judge_value, num_tests


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
