"""Tests for IF (Instruction Following) reward function.

Tests cover IFEval taxonomy instruction types, None-kwargs handling,
ground_truth parsing, and integration with default_compute_score routing.
"""

import json
import sys
import traceback


def _gt(instruction_id_list, kwargs, prompt="Test prompt."):
    return {
        "key": 0,
        "prompt": prompt,
        "instruction_id_list": instruction_id_list,
        "kwargs": kwargs,
    }


def test_title_pass():
    """detectable_format:title — response has <<title>>"""
    from verl.utils.reward_score import default_compute_score

    gt = _gt(["detectable_format:title"], [{}],
             prompt="Write something with a title in double angular brackets.")
    result = default_compute_score("if", "<<My Great Title>>\nHere is my answer.", gt)
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"
    assert result["acc"] is True


def test_title_fail():
    """detectable_format:title — response missing title"""
    from verl.utils.reward_score import default_compute_score

    gt = _gt(["detectable_format:title"], [{}],
             prompt="Write something with a title in double angular brackets.")
    result = default_compute_score("if", "Here is my answer without any title.", gt)
    assert result["score"] == 0.0, f"Expected 0.0, got {result['score']}"
    assert result["acc"] is False


def test_word_count_pass():
    """length_constraints:number_words — at least 50 words, response has 60+"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write at least 50 words about nature."
    response = " ".join(["word"] * 60) + "."
    gt = _gt(["length_constraints:number_words"],
             [{"num_words": 50, "relation": "at least"}],
             prompt=prompt)
    result = default_compute_score("if", response, gt)
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"


def test_word_count_fail():
    """length_constraints:number_words — less than 20 words, response has 50"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write less than 20 words."
    response = " ".join(["word"] * 50) + "."
    gt = _gt(["length_constraints:number_words"],
             [{"num_words": 20, "relation": "less than"}],
             prompt=prompt)
    result = default_compute_score("if", response, gt)
    assert result["score"] == 0.0, f"Expected 0.0, got {result['score']}"


def test_no_comma_pass():
    """punctuation:no_comma — response without commas"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write without using commas."
    gt = _gt(["punctuation:no_comma"], [{}], prompt=prompt)
    result = default_compute_score("if", "This is a response without any commas at all.", gt)
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"


def test_no_comma_fail():
    """punctuation:no_comma — response with commas"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write without using commas."
    gt = _gt(["punctuation:no_comma"], [{}], prompt=prompt)
    result = default_compute_score("if", "This is a response, with commas, in it.", gt)
    assert result["score"] == 0.0, f"Expected 0.0, got {result['score']}"


def test_multi_instruction_partial_fail():
    """Two instructions (title + no_comma), only one passes → score=0.0"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write with title and no commas."
    gt = _gt(["detectable_format:title", "punctuation:no_comma"], [{}, {}], prompt=prompt)
    # Has title but also has commas
    result = default_compute_score("if", "<<My Title>>\nHello, world, this has commas.", gt)
    assert result["score"] == 0.0, f"Expected 0.0 (partial), got {result['score']}"
    assert result["strict_follow_instruction_list"] == [True, False]


def test_none_kwargs():
    """kwargs with all 23 fields (unused=None) should work without TypeError"""
    from verl.utils.reward_score import default_compute_score

    prompt = "Write with a title."
    all_none_kwargs = {
        'capital_frequency': None, 'capital_relation': None, 'end_phrase': None,
        'first_word': None, 'forbidden_words': None, 'frequency': None,
        'keyword': None, 'keywords': None, 'let_frequency': None,
        'let_relation': None, 'letter': None, 'nth_paragraph': None,
        'num_bullets': None, 'num_highlights': None, 'num_paragraphs': None,
        'num_placeholders': None, 'num_sections': None, 'num_sentences': None,
        'num_words': None, 'postscript_marker': None, 'prompt_to_repeat': None,
        'relation': None, 'section_spliter': None,
    }
    gt = _gt(["detectable_format:title"], [all_none_kwargs], prompt=prompt)
    result = default_compute_score("if", "<<Title>>\nContent here.", gt)
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"


def test_ground_truth_json_string():
    """ground_truth as JSON string should be parsed correctly"""
    from verl.utils.reward_score.if_reward import compute_score

    gt_dict = {
        "key": 0,
        "prompt": "No commas please.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    gt_str = json.dumps(gt_dict)
    result = compute_score("No commas in this response.", gt_str)
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"


def test_real_parquet_smoke():
    """Load 1 row from actual IF parquet and compute reward"""
    import pandas as pd
    from verl.utils.reward_score import default_compute_score

    path = "/131_data/geeho/data/Nemotron-Cascade-RL-Instruction-Following/ifrl_if_verl_ready.parquet"
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print("  SKIP (parquet not found)")
        return

    row = df.iloc[0]
    gt = row["reward_model"]["ground_truth"]

    # Construct a response that should pass the first instruction
    instruction_ids = list(gt["instruction_id_list"])
    print(f"  Instructions: {instruction_ids}")

    # Just verify it runs without error
    result = default_compute_score("if", "<<Title>>\nSome short text.", gt)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "score" in result
    print(f"  Score: {result['score']}, instruction_count: {result['instruction_count']}")


def main():
    tests = [
        ("title_pass", test_title_pass),
        ("title_fail", test_title_fail),
        ("word_count_pass", test_word_count_pass),
        ("word_count_fail", test_word_count_fail),
        ("no_comma_pass", test_no_comma_pass),
        ("no_comma_fail", test_no_comma_fail),
        ("multi_instruction_partial_fail", test_multi_instruction_partial_fail),
        ("none_kwargs", test_none_kwargs),
        ("ground_truth_json_string", test_ground_truth_json_string),
        ("real_parquet_smoke", test_real_parquet_smoke),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            print(f"[RUN]  {name}")
            fn()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
