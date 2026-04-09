"""Evaluate IFEval outputs and save per-example judgment results.

This script is standalone and only depends on vendored IFEval code under:
  eval/third_party/instruction_following_eval
"""

import argparse
import importlib
import json
import os
import sys
from typing import Dict, List


def _ensure_vendor_imports():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    third_party_root = os.path.join(this_dir, "third_party")
    if third_party_root not in sys.path:
        sys.path.insert(0, third_party_root)

    from instruction_following_eval import evaluation_lib  # pylint: disable=import-outside-toplevel

    return evaluation_lib


def _ensure_runtime_dependencies():
    required_packages = ["nltk", "langdetect"]
    missing = []
    loaded = {}
    for package in required_packages:
        try:
            loaded[package] = importlib.import_module(package)
        except ModuleNotFoundError:
            missing.append(package)

    if missing:
        missing_str = ", ".join(missing)
        raise ModuleNotFoundError(
            "Missing required packages for IFEval scorer: "
            f"{missing_str}. Install with:\n"
            "pip install nltk langdetect"
        )

    return loaded["nltk"]


def _ensure_nltk_resources(nltk_module):
    # Required by instruction_following_eval/instructions_util.py.
    try:
        nltk_module.data.find("tokenizers/punkt")
    except LookupError:
        nltk_module.download("punkt", quiet=True)

    # Newer NLTK versions may use this split package.
    try:
        nltk_module.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk_module.download("punkt_tab", quiet=True)
        except Exception:
            pass


def _read_input_response_map(input_datapath: str) -> Dict[str, dict]:
    prompt_to_item = {}
    with open(input_datapath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("prompt")
            response = item.get("response")
            if prompt is None or response is None:
                raise ValueError(
                    f"IFEval output line missing 'prompt' or 'response': {item.keys()}"
                )
            prompt_to_item[prompt] = item
    return prompt_to_item


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_ifeval(input_datapath: str, test_datapath: str, start_idx: int = -1, end_idx: int = -1):
    nltk_module = _ensure_runtime_dependencies()
    _ensure_nltk_resources(nltk_module)
    evaluation_lib = _ensure_vendor_imports()

    prompt_to_item = _read_input_response_map(input_datapath)
    prompt_to_response = {k: v["response"] for k, v in prompt_to_item.items()}

    inputs = evaluation_lib.read_prompt_list(test_datapath)
    if start_idx != -1 and end_idx != -1:
        inputs = inputs[start_idx:end_idx]

    strict_outputs = []
    loose_outputs = []
    for inp in inputs:
        if inp.prompt not in prompt_to_response:
            raise KeyError(f"Prompt not found in model outputs for key={inp.key}")
        strict_outputs.append(
            evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        )
        loose_outputs.append(
            evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        )

    strict_prompt_acc = _safe_mean(
        [1.0 if o.follow_all_instructions else 0.0 for o in strict_outputs]
    )
    loose_prompt_acc = _safe_mean(
        [1.0 if o.follow_all_instructions else 0.0 for o in loose_outputs]
    )

    strict_inst_acc = _safe_mean(
        [
            1.0 if followed else 0.0
            for o in strict_outputs
            for followed in o.follow_instruction_list
        ]
    )
    loose_inst_acc = _safe_mean(
        [
            1.0 if followed else 0.0
            for o in loose_outputs
            for followed in o.follow_instruction_list
        ]
    )

    results = []
    for inp, strict_out, loose_out in zip(inputs, strict_outputs, loose_outputs):
        raw_item = prompt_to_item[inp.prompt]
        rollout_reasoning = raw_item.get("reason_text", "")
        rollout_has_reasoning = bool(raw_item.get("reason", False))
        rollout_full = (
            f"{rollout_reasoning}\n{strict_out.response}".strip()
            if rollout_reasoning
            else strict_out.response
        )
        results.append(
            {
                "task": "ifeval",
                "example_id": inp.key,
                "task_id": raw_item.get("task_id", inp.key),
                # Unified fields for downstream analysis.
                "query": inp.prompt,
                "answer": {
                    "instruction_id_list": inp.instruction_id_list,
                    "kwargs": inp.kwargs,
                },
                "rollout": strict_out.response,
                "rollout_has_reasoning": rollout_has_reasoning,
                "rollout_reasoning": rollout_reasoning,
                "rollout_full": rollout_full,
                "prompt": inp.prompt,
                "response": strict_out.response,
                "instruction_id_list": inp.instruction_id_list,
                "kwargs": inp.kwargs,
                "follow_all_instructions_strict": strict_out.follow_all_instructions,
                "follow_instruction_list_strict": strict_out.follow_instruction_list,
                "follow_all_instructions_loose": loose_out.follow_all_instructions,
                "follow_instruction_list_loose": loose_out.follow_instruction_list,
                # Main boolean for cross-model Venn analysis (strict criterion).
                "judge_em": 1 if strict_out.follow_all_instructions else 0,
            }
        )

    summary = {
        "task": "ifeval",
        "num_examples": len(results),
        "strict_prompt_accuracy": strict_prompt_acc,
        "strict_instruction_accuracy": strict_inst_acc,
        "loose_prompt_accuracy": loose_prompt_acc,
        "loose_instruction_accuracy": loose_inst_acc,
    }

    return results, summary


def _default_output_jsonl(input_datapath: str) -> str:
    base, ext = os.path.splitext(input_datapath)
    if ext.lower() == ".jsonl":
        return base + ".judged.jsonl"
    return input_datapath + ".judged.jsonl"


def _default_summary_json(input_datapath: str) -> str:
    base, _ = os.path.splitext(input_datapath)
    return base + ".summary.json"


def parse_args():
    parser = argparse.ArgumentParser(description="IFEval scorer (standalone)")
    parser.add_argument(
        "--input-datapath",
        type=str,
        required=True,
        help="Path to inference output JSONL for ifeval",
    )
    parser.add_argument(
        "--test-datapath",
        type=str,
        required=True,
        help="Path to IFEval input_data.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Per-example judged JSONL output path",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Summary JSON output path",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=-1,
        help="Optional start index for subset evaluation (must pair with --end-idx)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="Optional end index for subset evaluation (must pair with --start-idx)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_jsonl = args.output_jsonl or _default_output_jsonl(args.input_datapath)
    summary_json = args.summary_json or _default_summary_json(args.input_datapath)

    output_dir = os.path.dirname(output_jsonl)
    summary_dir = os.path.dirname(summary_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)

    results, summary = evaluate_ifeval(
        input_datapath=args.input_datapath,
        test_datapath=args.test_datapath,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"IFEval judged examples written to: {output_jsonl}")
    print(f"IFEval summary written to: {summary_json}")
    print(
        "strict_prompt_accuracy:",
        f"{summary['strict_prompt_accuracy']:.4f}",
        "| strict_instruction_accuracy:",
        f"{summary['strict_instruction_accuracy']:.4f}",
    )
    print(
        "loose_prompt_accuracy:",
        f"{summary['loose_prompt_accuracy']:.4f}",
        "| loose_instruction_accuracy:",
        f"{summary['loose_instruction_accuracy']:.4f}",
    )


if __name__ == "__main__":
    main()
