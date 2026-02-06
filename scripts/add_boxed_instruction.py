#!/usr/bin/env python3
"""
Add boxed format instruction to Nemotron-Cascade Math RL dataset.

This script modifies the dataset to include instructions for the model to
output answers in LaTeX \\boxed{} format, which is required by the reward function.

Usage:
    python scripts/add_boxed_instruction.py
"""

import pandas as pd
from pathlib import Path

# System message that instructs the model to use boxed format
# This follows the Nemotron-Cascade paper's requirement for structured reasoning
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a mathematical reasoning assistant. When solving problems:\n"
        "1. Show your step-by-step reasoning inside <think>...</think> tags\n"
        "2. After your reasoning, provide your final answer in LaTeX \\boxed{} format\n"
        "3. Example: <think>Let me calculate... 2+2=4</think> The answer is \\boxed{4}\n"
        "\n"
        "Important: Always use \\boxed{your_answer} for your final answer!"
    )
}

# Alternative: Add instruction directly to user message
BOXED_INSTRUCTION = (
    "\n\nPlease provide your final answer in the format \\boxed{your_answer} "
    "after showing your reasoning."
)


def add_system_message(prompt_list: list) -> list:
    """Add system message to prompt if not already present.

    Args:
        prompt_list: List of message dicts with 'role' and 'content' keys

    Returns:
        Modified prompt list with system message prepended
    """
    # Check if system message already exists
    has_system = any(msg.get("role") == "system" for msg in prompt_list)

    if not has_system:
        return [SYSTEM_MESSAGE] + prompt_list
    return prompt_list


def add_instruction_to_user_message(prompt_list: list) -> list:
    """Add boxed format instruction to user message.

    Args:
        prompt_list: List of message dicts with 'role' and 'content' keys

    Returns:
        Modified prompt list with instruction added to user message
    """
    modified = []
    for msg in prompt_list:
        if msg.get("role") == "user":
            # Add instruction to user message
            new_content = msg["content"] + BOXED_INSTRUCTION
            modified.append({"role": "user", "content": new_content})
        else:
            modified.append(msg)
    return modified


def process_dataset(
    input_path: str,
    output_path: str,
    method: str = "system",
    dry_run: bool = False,
):
    """Process dataset to add boxed format instructions.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        method: 'system' (add system message) or 'user' (modify user message)
        dry_run: If True, only print statistics without saving
    """
    print(f"Loading dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Process prompts
    modified_prompts = []
    for idx, prompt in enumerate(df['prompt']):
        if method == "system":
            modified = add_system_message(prompt)
        elif method == "user":
            modified = add_instruction_to_user_message(prompt)
        else:
            raise ValueError(f"Unknown method: {method}")

        modified_prompts.append(modified)

        # Show examples for first 2 samples
        if idx < 2:
            print(f"\n=== Sample {idx + 1} ===")
            print(f"Original: {prompt}")
            print(f"Modified: {modified}")

    # Update dataframe
    df['prompt'] = modified_prompts

    if dry_run:
        print("\n[DRY RUN] Not saving file. Statistics:")
        print(f"  - Total samples: {len(df)}")
        print(f"  - Output path would be: {output_path}")
        return

    # Save modified dataset
    print(f"\nSaving modified dataset to: {output_path}")
    df.to_parquet(output_path, index=False)
    print("✅ Done!")

    # Verify
    df_verify = pd.read_parquet(output_path)
    print(f"\nVerification: Loaded {len(df_verify)} samples from output file")
    print(f"First prompt after reload: {df_verify['prompt'].iloc[0]}")


def main():
    """Main function to process both train and validation datasets."""
    # Paths
    base_dir = Path("/mnt/ddn/vuvlm/geeho/datasets/Nemotron-Cascade-RL-Math")
    eval_dir = Path("/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/nemotron_evaluation/data/aime25")

    train_input = base_dir / "math_verl_ready.parquet"
    train_output = base_dir / "math_verl_ready_with_instruction.parquet"

    val_input = eval_dir / "test_verl_ready.parquet"
    val_output = eval_dir / "test_verl_ready_with_instruction.parquet"

    # Process train dataset
    print("=" * 80)
    print("Processing TRAIN dataset")
    print("=" * 80)
    process_dataset(
        input_path=str(train_input),
        output_path=str(train_output),
        method="system",  # Use 'system' or 'user'
        dry_run=False,     # Set to True to preview without saving
    )

    # Process validation dataset
    print("\n" + "=" * 80)
    print("Processing VALIDATION dataset")
    print("=" * 80)
    process_dataset(
        input_path=str(val_input),
        output_path=str(val_output),
        method="system",
        dry_run=False,
    )

    print("\n" + "=" * 80)
    print("✅ All datasets processed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Update your training script to use the new dataset paths:")
    print(f"   train_files={train_output}")
    print(f"   val_files={val_output}")
    print("\n2. Run a test to verify the model now outputs \\boxed{{}} format")
    print("\n3. Monitor training logs for 'extraction_failed' rate (should drop to ~0%)")


if __name__ == "__main__":
    main()
