#!/usr/bin/env python3
"""Quick debug harness: feed sample coding problems to compute_score and print
the resulting reward dict. Used to verify the reward pipeline end-to-end.

Run:
    python evaluation/eval/debug_reward_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.eval.verl_custom_reward import compute_score  # noqa: E402


# --- sample problem: read an integer N, print N*N ------------------------------
GROUND_TRUTH = {
    "inputs": ["2\n", "5\n", "10\n"],
    "outputs": ["4\n", "25\n", "100\n"],
    "fn_name": "",
}

EXTRA_INFO_EN = {
    "num_tests": 3,
    "fn_name": "",
    "test_method": "stdio",
    "test_time_limit": 6,
    "prompt_language": "en",
}

CORRECT_SOLUTION = """<think>
We need to read an integer and print its square.
</think>

```python
n = int(input())
print(n * n)
```
"""

WRONG_SOLUTION = """<think>
Print N+N instead.
</think>

```python
n = int(input())
print(n + n)
```
"""

CODE_SWITCH_SOLUTION = """<think>
이 문제는 입력받은 정수의 제곱을 출력하는 문제입니다.
정수를 읽고 그 값을 제곱해서 출력하면 됩니다. 아주 간단한 문제이고,
파이썬으로 풀면 int(input())으로 읽고 n*n을 출력하면 됩니다. 한국어로 계속 추론을 진행합니다.
</think>

```python
n = int(input())
print(n * n)
```
"""

CASES = [
    ("correct (en)", CORRECT_SOLUTION, EXTRA_INFO_EN),
    ("wrong (en)", WRONG_SOLUTION, EXTRA_INFO_EN),
    ("correct + Korean reasoning (en prompt)", CODE_SWITCH_SOLUTION, EXTRA_INFO_EN),
]


def run():
    for label, solution, extra_info in CASES:
        print(f"\n===== {label} =====")
        result = compute_score(
            data_source="nemotron_cascade_rl_coding",
            solution_str=solution,
            ground_truth=GROUND_TRUTH,
            extra_info=extra_info,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
