"""Convert evaluation datasets into VERL-style JSONL format.

Caches converted results as ``_verl_ready.parquet`` next to the source data.
On subsequent runs the parquet is loaded directly, skipping conversion.
"""

import argparse
import json
import os
from typing import Iterable, List

import pandas as pd


_COMPLEX_COLS = {"prompt", "reward_model", "extra_info"}


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parquet_path(benchmark_folder: str, task: str) -> str:
    return os.path.join(benchmark_folder, task, "_verl_ready.parquet")


def _save_parquet(rows: List[dict], path: str):
    df = pd.DataFrame(rows)
    for col in _COMPLEX_COLS & set(df.columns):
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _load_parquet(path: str) -> List[dict]:
    df = pd.read_parquet(path)
    for col in _COMPLEX_COLS & set(df.columns):
        df[col] = df[col].apply(json.loads)
    return df.to_dict(orient="records")


def _aime_prompt(question: str) -> str:
    return question.strip() + "\nPlease reason step by step, and put your final answer within \\boxed{}."


def convert_aime24(benchmark_folder: str) -> List[dict]:
    input_path = os.path.join(benchmark_folder, "aime24/test.jsonl")
    rows = []
    for idx, item in enumerate(_iter_jsonl(input_path)):
        question = item.get("question", item.get("problem", "")).strip()
        rows.append(
            {
                "data_source": "local/aime24",
                "prompt": [{"role": "user", "content": _aime_prompt(question)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "id": item.get("id", idx),
                    "question": question,
                    "url": item.get("url"),
                },
            }
        )
    return rows


def convert_aime25(benchmark_folder: str) -> List[dict]:
    input_path = os.path.join(benchmark_folder, "aime25/test.jsonl")
    rows = []
    for idx, item in enumerate(_iter_jsonl(input_path)):
        question = item["problem"].strip()
        rows.append(
            {
                "data_source": "local/aime25",
                "prompt": [{"role": "user", "content": _aime_prompt(question)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "id": item.get("id", idx),
                    "question": question,
                    "url": item.get("url"),
                    "year": item.get("year"),
                },
            }
        )
    return rows


def convert_aime26(benchmark_folder: str) -> List[dict]:
    input_path = os.path.join(benchmark_folder, "aime26/test.jsonl")
    rows = []
    for idx, item in enumerate(_iter_jsonl(input_path)):
        question = item["problem"].strip()
        rows.append(
            {
                "data_source": "local/aime26",
                "prompt": [{"role": "user", "content": _aime_prompt(question)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "id": item.get("id", idx),
                    "question": question,
                    "url": item.get("url"),
                    "year": item.get("year"),
                },
            }
        )
    return rows


def convert_ifeval(benchmark_folder: str) -> List[dict]:
    input_path = os.path.join(benchmark_folder, "ifeval/input_data.jsonl")
    rows = []
    for idx, item in enumerate(_iter_jsonl(input_path)):
        prompt = item["prompt"].strip()
        rows.append(
            {
                "data_source": "local/ifeval",
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "instruction_following",
                "reward_model": {
                    "style": "ifeval",
                    "ground_truth": {
                        "instruction_id_list": item["instruction_id_list"],
                        "kwargs": item["kwargs"],
                    },
                },
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "key": item["key"],
                    "prompt": prompt,
                    "instruction_id_list": item["instruction_id_list"],
                    "kwargs": item["kwargs"],
                },
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Convert eval datasets to VERL format")
    parser.add_argument(
        "--benchmark-folder",
        type=str,
        required=True,
        help="Root folder containing aime24/aime25/ifeval datasets",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="ifeval,aime24,aime25,aime26",
        help="Comma-separated tasks to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for converted JSONL files (default: <benchmark-folder>/verl_format)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    output_dir = args.output_dir or os.path.join(args.benchmark_folder, "verl_format")
    os.makedirs(output_dir, exist_ok=True)

    converters = {
        "aime24": convert_aime24,
        "aime25": convert_aime25,
        "aime26": convert_aime26,
        "ifeval": convert_ifeval,
    }

    for task in tasks:
        if task not in converters:
            raise ValueError(f"Unsupported task: {task}")

        cache_path = _parquet_path(args.benchmark_folder, task)
        if os.path.exists(cache_path):
            rows = _load_parquet(cache_path)
            print(f"[{task}] LOADED {len(rows)} samples from cache: {cache_path}")
        else:
            rows = converters[task](args.benchmark_folder)
            _save_parquet(rows, cache_path)
            print(f"[{task}] CONVERTED {len(rows)} samples & SAVED to: {cache_path}")

        output_path = os.path.join(output_dir, f"{task}.jsonl")
        _write_jsonl(output_path, rows)
        print(f"  -> wrote JSONL: {output_path}")


if __name__ == "__main__":
    main()
