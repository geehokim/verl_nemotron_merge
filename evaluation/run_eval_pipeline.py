"""Unified eval pipeline for ifeval/aime24/aime25.

Pipeline stages:
1) Convert raw datasets to VERL-format JSONL (optional)
2) Run inference.py for each task
3) Run task-specific scorers
4) Write merged summary JSON
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


SUPPORTED_TASKS = {"ifeval", "aime24", "aime25"}


def _run_cmd(cmd: List[str], cwd: str):
    print("=" * 80)
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_output_folder_name(topp: float, seed: int, think: bool) -> str:
    if topp < 1:
        folder = f"outputs_vllm073_topp{topp}_seed{seed}"
    else:
        folder = "outputs_vllm073"
    if not think:
        folder = "nothink_" + folder
    return folder


def _build_output_name(task: str, start_idx: int, end_idx: int) -> str:
    if start_idx != -1 and end_idx != -1:
        return f"{task}_{start_idx}to{end_idx}.jsonl"
    return f"{task}.jsonl"


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Unified evaluation pipeline")

    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tokenizer-folder", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, required=True)
    parser.add_argument("--benchmark-folder", type=str, required=True)

    parser.add_argument(
        "--tasks",
        type=str,
        default="ifeval,aime24,aime25",
        help="Comma-separated tasks (subset of ifeval,aime24,aime25)",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=str, default=None)
    parser.add_argument("--yarn-factor", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--max-output-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--start-idx", type=int, default=-1)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--use-r1", action="store_true", default=False)
    parser.add_argument("--no-think", action="store_true", default=False)

    parser.add_argument("--skip-inference", action="store_true", default=False)
    parser.add_argument("--skip-verl-convert", action="store_true", default=False)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Full path to save results (takes priority over --results-base-dir)",
    )
    parser.add_argument(
        "--results-base-dir",
        type=str,
        default=None,
        help="Base directory for results. Saves to <results-base-dir>/<save-name>/",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Name used for results subdirectory under --results-base-dir (default: --model-name)",
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Supported: {sorted(SUPPORTED_TASKS)}")

    eval_root = os.path.dirname(os.path.abspath(__file__))
    save_name = args.save_name or args.model_name
    if args.results_dir:
        results_dir = args.results_dir
    elif args.results_base_dir:
        results_dir = os.path.join(args.results_base_dir, save_name)
    else:
        results_dir = os.path.join(args.model_folder, args.model_name, "analysis_results")
    os.makedirs(results_dir, exist_ok=True)

    if not args.skip_verl_convert:
        convert_cmd = [
            args.python_bin,
            "prepare_verl_eval_data.py",
            "--benchmark-folder",
            args.benchmark_folder,
            "--tasks",
            ",".join(tasks),
            "--output-dir",
            os.path.join(results_dir, "verl_format"),
        ]
        _run_cmd(convert_cmd, cwd=eval_root)

    if not args.skip_inference:
        for task in tasks:
            infer_cmd = [
                args.python_bin,
                "inference.py",
                "--model-folder",
                args.model_folder,
                "--model-name",
                args.model_name,
                "--tokenizer-folder",
                args.tokenizer_folder,
                "--tokenizer-name",
                args.tokenizer_name,
                "--benchmark-folder",
                args.benchmark_folder,
                "--eval-dataset",
                task,
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--yarn-factor",
                str(args.yarn_factor),
                "--temperature",
                str(args.temperature),
                "--topk",
                str(args.topk),
                "--topp",
                str(args.topp),
                "--max-output-len",
                str(args.max_output_len),
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--start-idx",
                str(args.start_idx),
                "--end-idx",
                str(args.end_idx),
            ]
            if args.device_id:
                infer_cmd.extend(["--device-id", args.device_id])
            if args.use_r1:
                infer_cmd.append("--use_r1")
            if args.no_think:
                infer_cmd.append("--no-think")

            _run_cmd(infer_cmd, cwd=eval_root)

    think = not args.no_think
    output_folder = _build_output_folder_name(args.topp, args.seed, think)
    output_root = os.path.join(args.model_folder, args.model_name, output_folder)

    task_summaries: Dict[str, dict] = {}
    merged_rows: List[dict] = []

    for task in tasks:
        input_datapath = os.path.join(
            output_root, _build_output_name(task, args.start_idx, args.end_idx)
        )
        judged_output = os.path.join(results_dir, f"{task}.jsonl")
        summary_output = os.path.join(results_dir, f"{task}.summary.json")

        if task in ("aime24", "aime25"):
            test_datapath = os.path.join(args.benchmark_folder, task, "test.jsonl")
            score_cmd = [
                args.python_bin,
                "eval/get_scores_aime.py",
                "--input-datapath",
                input_datapath,
                "--test-datapath",
                test_datapath,
                "--task",
                task,
                "--output-jsonl",
                judged_output,
                "--summary-json",
                summary_output,
                "--start-idx",
                str(args.start_idx),
                "--end-idx",
                str(args.end_idx),
            ]
        elif task == "ifeval":
            test_datapath = os.path.join(args.benchmark_folder, "ifeval", "input_data.jsonl")
            score_cmd = [
                args.python_bin,
                "eval/get_scores_ifeval.py",
                "--input-datapath",
                input_datapath,
                "--test-datapath",
                test_datapath,
                "--output-jsonl",
                judged_output,
                "--summary-json",
                summary_output,
                "--start-idx",
                str(args.start_idx),
                "--end-idx",
                str(args.end_idx),
            ]
        else:
            raise ValueError(f"Unexpected task: {task}")

        _run_cmd(score_cmd, cwd=eval_root)
        task_summaries[task] = _read_json(summary_output)
        task_rows = _read_jsonl(judged_output)
        for row in task_rows:
            # Make each row self-contained for later filtering (e.g., judge_em==1).
            row["model_name"] = args.model_name
            row["model_folder"] = args.model_folder
        merged_rows.extend(task_rows)

    merged_rows_json = os.path.join(results_dir, "all_tasks_results.json")
    merged_rows_jsonl = os.path.join(results_dir, "all_tasks_results.jsonl")
    with open(merged_rows_json, "w", encoding="utf-8") as f:
        json.dump(merged_rows, f, ensure_ascii=False, indent=2)
    with open(merged_rows_jsonl, "w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    merged_summary = {
        "model_folder": args.model_folder,
        "model_name": args.model_name,
        "tasks": tasks,
        "inference_output_folder": output_root,
        "results_dir": results_dir,
        "task_summaries": task_summaries,
        "all_tasks_results_json": merged_rows_json,
        "all_tasks_results_jsonl": merged_rows_jsonl,
        "num_total_examples": len(merged_rows),
    }
    merged_summary_path = os.path.join(results_dir, "summary.json")
    with open(merged_summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Pipeline completed. Merged summary: {merged_summary_path}")


if __name__ == "__main__":
    main()
