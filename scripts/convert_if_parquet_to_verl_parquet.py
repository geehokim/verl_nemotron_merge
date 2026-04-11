"""Convert IF (Instruction Following) parquet to verl-compatible format with data_source='if'.

Changes from the source parquet:
  - data_source: "nemotron_cascade_rl_if" -> "if"
  - kwargs: removes None-valued fields (IFEval build_description expects only relevant kwargs)
  - ability: kept as "if" (or set to "instruction_following")

Usage:
    python scripts/convert_if_parquet_to_verl_parquet.py \
        --input_path /path/to/ifrl_final_release_verl_ready.parquet \
        --output_path /path/to/output.parquet
"""

import argparse
import json

import pandas as pd


def _clean_kwargs(kwargs_list):
    """Remove None-valued fields from each kwargs dict."""
    cleaned = []
    for kw in kwargs_list:
        if isinstance(kw, dict):
            cleaned.append({k: v for k, v in kw.items() if v is not None})
        else:
            cleaned.append({})
    return cleaned


def convert_if_parquet(input_path: str, output_path: str):
    df_in = pd.read_parquet(input_path)
    print(f"Input: {len(df_in)} rows, columns={list(df_in.columns)}")
    print(f"Original data_source: {df_in['data_source'].unique()}")

    rows = []
    for idx in range(len(df_in)):
        row = df_in.iloc[idx]
        prompt = list(row["prompt"])
        rm = row["reward_model"]
        ei = row["extra_info"]

        if isinstance(rm, str):
            rm = json.loads(rm)
        if isinstance(ei, str):
            ei = json.loads(ei)

        gt = rm.get("ground_truth", {})
        if isinstance(gt, str):
            gt = json.loads(gt)

        # Clean kwargs
        kwargs_raw = gt.get("kwargs", [])
        if hasattr(kwargs_raw, "tolist"):
            kwargs_raw = kwargs_raw.tolist()
        kwargs_cleaned = _clean_kwargs(list(kwargs_raw))

        instruction_id_list = gt.get("instruction_id_list", [])
        if hasattr(instruction_id_list, "tolist"):
            instruction_id_list = instruction_id_list.tolist()

        new_gt = {
            "key": gt.get("key", idx),
            "prompt": gt.get("prompt", ""),
            "instruction_id_list": list(instruction_id_list),
            "kwargs": kwargs_cleaned,
        }

        extra_info = dict(ei) if isinstance(ei, dict) else {}

        rows.append({
            "data_source": "if",
            "prompt": prompt,
            "ability": "instruction_following",
            "reward_model": {
                "style": "rule",
                "ground_truth": new_gt,
            },
            "extra_info": extra_info,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(output_path, index=False)
    print(f"Converted {len(rows)} samples -> {output_path}")
    print(f"Columns: {list(df_out.columns)}")
    print(f"data_source: {df_out['data_source'].unique()}")

    # Verify sample
    sample_rm = rows[0]["reward_model"]
    print(f"\nSample ground_truth:")
    print(json.dumps(sample_rm["ground_truth"], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    convert_if_parquet(args.input_path, args.output_path)
