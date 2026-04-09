"""Evaluate AIME outputs and save per-example judgment results."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


_PATTERN_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.DOTALL)
_PATTERN_BOLD = re.compile(r"\*\*(.*?)\*\*", re.DOTALL)
_PATTERN_BRACKET = re.compile(r"\\\[\n(.*?)\n\\\]", re.DOTALL)
_PATTERN_IS = re.compile(r"is \\\((.*?)\\\)", re.DOTALL)
_PATTERN_ESCAPED_BRACKET = re.compile(r"\\\[\\n(.*?)\\n\\\]", re.DOTALL)


def _is_completely_wrapped_by_text(input_string):
    pattern = r"^\\text{(.*)}$"
    match = re.match(pattern, input_string)
    if match:
        extracted_content = match.group(1)
        extracted_content = extracted_content.replace("(", "").replace(")", "").replace(",", "")
        return extracted_content
    return None


def math_answer_cleaning(answer):
    extracted_content = _is_completely_wrapped_by_text(answer)
    answer = extracted_content if extracted_content else answer

    answer = answer.replace(",\\!", "").replace("{,}", "").replace("\\$", "")
    answer = answer.replace("dfrac{", "frac{").replace("tfrac{", "frac{")
    answer = answer.replace("^\\circ", "").replace("^{\\circ}", "")
    answer = answer.replace("\\quad", "")
    answer = re.sub(r"\\,\\text\{.*?\}", "", answer)
    answer = re.sub(r"\\text\{.*?\}", "", answer)
    answer = re.sub(r"(\s\^\{-\d+\})", "", answer)
    answer = answer.replace(" ", "").replace("\n", "").replace("\\n", "")
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^{([+-]?\d+)}", r"\1e\2", answer)
    answer = re.sub(r"([+-]?\d*\.?\d+)[\\]times10\^([+-]?\d+)", r"\1e\2", answer)
    answer = re.sub(r"(\d+)\^{(\d+)}", r"\1^\2", answer)
    answer = re.sub(r"10\^\{(-?\d+)\}", r"1e\1", answer)
    answer = answer.replace(",", "").lower()

    if answer.endswith("\\"):
        answer = answer[:-1]

    func_pattern = r"^[a-zA-Z_]\w*\([a-zA-Z_]\w*\)$"
    if "=" in answer and (re.match(func_pattern, answer.split("=")[0]) or len(answer.split("=")[0]) <= 3):
        answer = answer.split("=", 1)[1]

    return answer


def round_number(answer):
    try:
        if float(answer) < 1:
            return f"{float(answer):.2g}"
    except Exception:
        pass
    return answer


def _calculate_numbers(input_string):
    try:
        return eval(input_string)
    except Exception:
        return None


def is_equal_after_calculation(extracted_answer, gold):
    gold = re.sub(r"\\frac{(.*?)}{(.*?)}", r"(\1/\2)", gold)
    extracted_answer = re.sub(r"\\frac{(.*?)}{(.*?)}", r"(\1/\2)", extracted_answer)
    gold_result = _calculate_numbers(gold)
    extracted_answer_result = _calculate_numbers(extracted_answer)
    return bool(gold_result is not None and gold_result == extracted_answer_result)


def _simple_math_equal(pred, gold):
    if pred is None or gold is None:
        return False
    if str(pred).strip().lower() == str(gold).strip().lower():
        return True
    try:
        if float(pred) == float(gold):
            return True
    except Exception:
        pass
    return False


def _extract_answer(text: str) -> Tuple[str, str]:
    matches = _PATTERN_BOXED.findall(text)
    if matches:
        return matches[-1], "boxed"
    matches = _PATTERN_BOLD.findall(text)
    if matches:
        return matches[-1], "bold"
    matches = _PATTERN_BRACKET.findall(text)
    if matches:
        return matches[-1], "bracket"
    matches = _PATTERN_IS.findall(text)
    if matches:
        return matches[-1], "is_paren"
    matches = _PATTERN_ESCAPED_BRACKET.findall(text)
    if matches:
        return matches[-1], "escaped_bracket"
    return None, "none"


def _normalize_id(example_id):
    return str(example_id)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_aime(
    input_datapath: str,
    test_datapath: str,
    task: str,
    start_idx: int = -1,
    end_idx: int = -1,
):
    outputs = _read_jsonl(input_datapath)
    tests = _read_jsonl(test_datapath)
    if start_idx != -1 and end_idx != -1:
        tests = tests[start_idx:end_idx]

    output_by_id: Dict[str, dict] = {}
    for idx, out in enumerate(outputs):
        if "task_id" in out:
            output_by_id[_normalize_id(out["task_id"])] = out
        else:
            output_by_id[_normalize_id(idx)] = out

    judged = []
    correct = 0
    missing_output = 0
    unparsable_pred = 0

    for idx, item in enumerate(tests):
        example_id = item.get("id", idx)
        key = _normalize_id(example_id)
        out = output_by_id.get(key)
        if out is None:
            out = outputs[idx] if idx < len(outputs) else {"output": ""}
            missing_output += 1

        response = out.get("output", "")
        rollout_reasoning = out.get("reason_text", "")
        rollout_has_reasoning = bool(out.get("reason", False))
        rollout_full = (
            f"{rollout_reasoning}\n{response}".strip()
            if rollout_reasoning
            else response
        )
        gold = str(item.get("answer", ""))
        pred_raw, pred_source = _extract_answer(response)

        is_correct = False
        pred_clean = None
        if pred_raw is None:
            unparsable_pred += 1
        else:
            pred_clean = math_answer_cleaning(pred_raw)
            gold_clean = math_answer_cleaning(gold)
            if _simple_math_equal(pred_clean, gold_clean):
                is_correct = True
            elif round_number(pred_clean) == round_number(gold_clean):
                is_correct = True
            elif is_equal_after_calculation(pred_clean, gold_clean):
                is_correct = True

        if is_correct:
            correct += 1

        question = item.get("question", item.get("problem", ""))
        judged.append(
            {
                "task": task,
                "example_id": example_id,
                "task_id": out.get("task_id", example_id),
                # Unified fields for downstream analysis.
                "query": question,
                "answer": gold,
                "rollout": response,
                "rollout_has_reasoning": rollout_has_reasoning,
                "rollout_reasoning": rollout_reasoning,
                "rollout_full": rollout_full,
                "question": question,
                "response": response,
                "pred": pred_clean,
                "pred_source": pred_source,
                "judge_em": 1 if is_correct else 0,
            }
        )

    num_examples = len(tests)
    summary = {
        "task": task,
        "num_examples": num_examples,
        "accuracy": (correct / num_examples) if num_examples > 0 else 0.0,
        "num_correct": correct,
        "num_unparsable_pred": unparsable_pred,
        "num_missing_output": missing_output,
    }
    return judged, summary


def _default_output_jsonl(input_datapath: str) -> str:
    base, ext = os.path.splitext(input_datapath)
    if ext.lower() == ".jsonl":
        return base + ".judged.jsonl"
    return input_datapath + ".judged.jsonl"


def _default_summary_json(input_datapath: str) -> str:
    base, _ = os.path.splitext(input_datapath)
    return base + ".summary.json"


def parse_args():
    parser = argparse.ArgumentParser(description="AIME scorer with per-example output")
    parser.add_argument("--input-datapath", type=str, required=True, help="Inference output JSONL path")
    parser.add_argument("--test-datapath", type=str, required=True, help="AIME test JSONL path")
    parser.add_argument("--task", type=str, required=True, choices=["aime24", "aime25"])
    parser.add_argument("--output-jsonl", type=str, default=None, help="Per-example judged JSONL output path")
    parser.add_argument("--summary-json", type=str, default=None, help="Summary JSON output path")
    parser.add_argument("--start-idx", type=int, default=-1)
    parser.add_argument("--end-idx", type=int, default=-1)
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

    judged, summary = evaluate_aime(
        input_datapath=args.input_datapath,
        test_datapath=args.test_datapath,
        task=args.task,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"{args.task} judged examples written to: {output_jsonl}")
    print(f"{args.task} summary written to: {summary_json}")
    print(f"{args.task} accuracy: {summary['accuracy']:.4f} ({summary['num_correct']}/{summary['num_examples']})")


if __name__ == "__main__":
    main()
