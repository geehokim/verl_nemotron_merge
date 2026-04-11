"""
Convert Nemotron Cascade RL Instruction-Following parquet to verl-compatible format.

Usage:
    python convert_ifrl_to_verl.py \
        --input_path /path/to/ifrl_final_release.parquet \
        --output_path /path/to/ifrl_final_release_verl_ready.parquet
"""

import argparse
import pandas as pd


def convert_to_verl(input_path: str, output_path: str):
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Original columns: {df.columns.tolist()}")

    records = []
    for idx, row in df.iterrows():
        # prompt: convert numpy array of dicts -> list of dicts
        prompt_messages = [dict(m) for m in row["prompt"]]
        prompt_text = prompt_messages[0]["content"] if prompt_messages else ""

        # instruction_id_list & kwargs: convert numpy arrays -> lists
        instruction_id_list = list(row["instruction_id_list"])
        kwargs = [dict(k) for k in row["kwargs"]]

        records.append({
            "data_source": "nemotron_cascade_rl_if",
            "prompt": prompt_messages,
            "ability": "if",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "key": int(row["index"]),
                    "prompt": prompt_text,
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs,
                },
            },
            "extra_info": {
                "split": "train",
                "index": int(row["index"]),
                "key": int(row["index"]),
                "prompt": prompt_text,
                "instruction_count": len(instruction_id_list),
            },
        })

    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(out_df)} rows to {output_path}")
    print(f"Output columns: {out_df.columns.tolist()}")

    # Verify
    verify = pd.read_parquet(output_path)
    print(f"\nVerification - reload shape: {verify.shape}")
    print(f"Sample row[0]:")
    for col in verify.columns:
        val = verify[col].iloc[0]
        print(f"  {col}: {repr(val)[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    convert_to_verl(args.input_path, args.output_path)
