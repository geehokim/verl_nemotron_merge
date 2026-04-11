"""Convert a math JSON dataset to verl-compatible parquet format.

Usage:
    python scripts/convert_math_json_to_verl_parquet.py \
        --input_path /path/to/math.json \
        --output_path /path/to/output.parquet
"""

import argparse
import json

import pandas as pd


def convert_math_json_to_verl_parquet(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data):
        row = {
            "data_source": "nemotron_cascade_rl_math",
            "prompt": [{"role": "user", "content": item["problem"]}],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(item["answer"]),
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "source": item.get("source", "math"),
            },
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"Converted {len(rows)} samples -> {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample row:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    convert_math_json_to_verl_parquet(args.input_path, args.output_path)
