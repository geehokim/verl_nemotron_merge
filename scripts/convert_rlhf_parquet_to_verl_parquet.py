"""Convert Nemotron-Cascade RLHF parquet to verl-compatible parquet format.

The input already has prompts in chat format but lacks the verl-required columns:
  ability, reward_model, extra_info.

Since RLHF data has no ground_truth, reward_model.ground_truth is set to ""
and the data_source is unified to "nemotron_cascade_rl_rlhf".

Usage:
    python scripts/convert_rlhf_parquet_to_verl_parquet.py \
        --input_path /path/to/rlhf_train_data.parquet \
        --output_path /path/to/rlhf_train_data_verl_ready.parquet
"""

import argparse
import json

import pandas as pd


def convert_rlhf_to_verl_parquet(input_path: str, output_path: str):
    df_in = pd.read_parquet(input_path)
    print(f"Input: {len(df_in)} rows, columns={list(df_in.columns)}")
    print(f"Original data_source distribution:\n{df_in['data_source'].value_counts().to_string()}\n")

    rows = []
    for idx in range(len(df_in)):
        row_in = df_in.iloc[idx]
        prompt = list(row_in["prompt"])  # numpy array -> list of dicts

        row = {
            "data_source": "nemotron_cascade_rl_rlhf",
            "prompt": prompt,
            "ability": "general",
            "reward_model": {
                "style": "model",
                "ground_truth": "",
            },
            "extra_info": {
                "split": "train",
                "index": int(row_in.get("index", idx)),
                "original_data_source": str(row_in["data_source"]),
                "category": str(row_in.get("category", "")),
                "cat": int(row_in.get("cat", -1)),
            },
        }
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(output_path, index=False)
    print(f"Converted {len(rows)} samples -> {output_path}")
    print(f"Columns: {list(df_out.columns)}")
    print(f"\nSample row:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    convert_rlhf_to_verl_parquet(args.input_path, args.output_path)
