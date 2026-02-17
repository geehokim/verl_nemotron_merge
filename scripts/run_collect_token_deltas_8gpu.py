#!/usr/bin/env python3
"""Distributed 8-GPU precompute for JWCM token-delta records.

This script parallelizes the expensive `collect_token_delta_records` stage across
multiple GPUs/processes (typically 8 via `torchrun --nproc_per_node=8`).

For each task (`if`, `math`), the script:
1. Loads validation prompts from `<validation_root>/<task>_validation.parquet`.
2. Shards prompts by rank using deterministic row striding.
3. Computes token-level `Δlog p = log p_rl - log p_base` on RL-generated tokens.
4. Writes rank-local artifacts.
5. Aggregates rank-local artifacts on rank 0 into notebook-loadable global files.

Global outputs per task are written to:
- `<output_root>/<task>/all_token_deltas.csv`
- `<output_root>/<task>/sequence_cache.pt`
- `<output_root>/<task>/token_summary.json`
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TaskModelSpec:
    """Container for one task model path.

    Args:
        name: Task name key used for file naming.
        model_path: Local checkpoint path for the RL model.
    """

    name: str
    model_path: Path


@dataclass(frozen=True)
class DistributedContext:
    """Distributed runtime context.

    Args:
        rank: Global rank index.
        world_size: Number of processes.
        local_rank: Local rank index on current node.
        device: Device string assigned to this rank.
        backend: Distributed backend string.
        distributed: Whether process group is initialized.
    """

    rank: int
    world_size: int
    local_rank: int
    device: str
    backend: str | None
    distributed: bool


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.

    Returns:
        None.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save dictionary payload to JSON file.

    Args:
        payload: JSON-serializable mapping.
        output_path: Destination path.

    Returns:
        None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def load_json(input_path: Path) -> Dict[str, Any]:
    """Load dictionary payload from JSON file.

    Args:
        input_path: Source path.

    Returns:
        Parsed dictionary.
    """

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Convert CLI dtype string to torch dtype object.

    Args:
        dtype_name: One of `bf16`, `fp16`, `fp32`.

    Returns:
        Corresponding torch dtype object.
    """

    lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in lookup:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return lookup[dtype_name]


def init_distributed(timeout_minutes: int) -> DistributedContext:
    """Initialize distributed context from `torchrun` environment.

    Args:
        timeout_minutes: Process group initialization timeout.

    Returns:
        DistributedContext object.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        # Single-process fallback keeps script debuggable without torchrun.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
            backend=None,
            distributed=False,
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=timeout_minutes),
        )

    rank = int(os.environ.get("RANK", dist.get_rank()))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", dist.get_world_size()))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
        distributed=True,
    )


def barrier_if_needed(context: DistributedContext) -> None:
    """Synchronize all ranks when distributed mode is enabled."""

    if context.distributed:
        dist.barrier()


def finalize_distributed(context: DistributedContext) -> None:
    """Destroy process group if distributed mode is enabled."""

    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero(context: DistributedContext) -> bool:
    """Return whether this process is rank 0."""

    return context.rank == 0


def rank_zero_print(context: DistributedContext, message: str) -> None:
    """Print message only on rank 0 to reduce log duplication."""

    if is_rank_zero(context):
        print(message)


def shard_dataframe_by_rank(df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """Shard dataframe by row striding for rank-parallel processing.

    Args:
        df: Input dataframe.
        rank: Current rank.
        world_size: Number of ranks.

    Returns:
        Rank-local dataframe shard.
    """

    if world_size <= 1:
        return df.copy()
    return df.iloc[rank::world_size].reset_index(drop=True)


def load_causal_lm(
    model_name_or_path: str | Path,
    torch_dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load causal LM and tokenizer on target device.

    Args:
        model_name_or_path: HF model id or local path.
        torch_dtype: Target dtype.
        device: Device string.

    Returns:
        Tuple of `(model, tokenizer)`.
    """

    resolved_path = str(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer_with_mistral_regex_fix(
        model_name_or_path=resolved_path,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sanitize_generation_config_for_greedy(model=model)

    return model, tokenizer


def load_tokenizer_with_mistral_regex_fix(
    model_name_or_path: str,
) -> AutoTokenizer:
    """Load tokenizer with optional `fix_mistral_regex=True` safety flag.

    Some checkpoints inherit tokenizer configs that trigger a warning about
    incorrect Mistral regex handling. Passing `fix_mistral_regex=True` fixes
    tokenization where the underlying tokenizer class supports this argument.

    Args:
        model_name_or_path: HF model id or local path.

    Returns:
        Loaded tokenizer instance.
    """

    try:
        # Preferred path for tokenizers that expose this fix flag.
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Fallback for tokenizer classes that do not accept the flag.
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )


def sanitize_generation_config_for_greedy(model: AutoModelForCausalLM) -> None:
    """Reset sampling-only generation fields to avoid noisy greedy warnings.

    RL checkpoints can store sampling parameters in generation config
    (`temperature`, `top_p`, `top_k`). In this pipeline we run deterministic
    greedy generation (`do_sample=False`), so these fields are unused and may
    trigger repetitive warnings.

    Args:
        model: Loaded causal language model.

    Returns:
        None. Mutates `model.generation_config` in-place when available.
    """

    if getattr(model, "generation_config", None) is None:
        return

    generation_config = model.generation_config
    generation_config.do_sample = False

    # Reset to canonical defaults used for greedy-compatible configs.
    # This suppresses warnings about unused sampling flags.
    if hasattr(generation_config, "temperature"):
        generation_config.temperature = 1.0
    if hasattr(generation_config, "top_p"):
        generation_config.top_p = 1.0
    if hasattr(generation_config, "top_k"):
        generation_config.top_k = 50


@torch.no_grad()
def collect_token_delta_records(
    task_name: str,
    validation_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM,
    rl_model: AutoModelForCausalLM,
    max_prompt_tokens: int,
    max_new_tokens: int,
    device: str,
    rank: int,
) -> tuple[pd.DataFrame, Dict[int, torch.Tensor], Dict[str, Any]]:
    """Collect token-level `Δlog p` records on RL-generated trajectories.

    Args:
        task_name: Task name (`if` or `math`).
        validation_df: Rank-local validation dataframe with `prompt_text` and `sample_id`.
        tokenizer: Tokenizer for generation and decode.
        base_model: Base reference model.
        rl_model: RL fine-tuned model for the task.
        max_prompt_tokens: Prompt truncation length.
        max_new_tokens: Maximum generation length.
        device: Device string for computation.
        rank: Current rank (for progress-bar labeling).

    Returns:
        Tuple of:
            - token-level delta dataframe
            - sequence cache (`sample_id -> full_ids`)
            - summary dictionary
    """

    records: List[Dict[str, Any]] = []
    sequence_cache: Dict[int, torch.Tensor] = {}
    generated_token_count = 0

    iterator = tqdm(
        validation_df.itertuples(index=False),
        total=len(validation_df),
        desc=f"Rank{rank:02d} Collect Δlogp ({task_name})",
    )
    for row in iterator:
        prompt_text = str(row.prompt_text)

        encoded_prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = encoded_prompt["input_ids"].to(device)
        attention_mask = encoded_prompt["attention_mask"].to(device)

        generated_ids = rl_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        full_ids = generated_ids[0]
        prompt_len = int(input_ids.shape[1])
        full_len = int(full_ids.shape[0])

        # Skip degenerate generation that produces no continuation.
        if full_len <= prompt_len:
            continue

        sequence_cache[int(row.sample_id)] = full_ids.detach().cpu()
        generated_token_count += full_len - prompt_len

        full_batch = full_ids.unsqueeze(0)
        full_attention = torch.ones_like(full_batch, device=device)

        rl_logits = rl_model(input_ids=full_batch, attention_mask=full_attention).logits
        base_logits = base_model(input_ids=full_batch, attention_mask=full_attention).logits

        rl_log_probs = torch.log_softmax(rl_logits[:, :-1, :].to(torch.float32), dim=-1)
        base_log_probs = torch.log_softmax(base_logits[:, :-1, :].to(torch.float32), dim=-1)

        positions = torch.arange(prompt_len, full_len, device=device)
        shifted_positions = positions - 1
        target_token_ids = full_ids[positions]

        rl_token_logp = rl_log_probs[0, shifted_positions, target_token_ids]
        base_token_logp = base_log_probs[0, shifted_positions, target_token_ids]
        delta_logp = rl_token_logp - base_token_logp

        for idx in range(int(positions.numel())):
            token_id = int(target_token_ids[idx].item())
            token_text = tokenizer.decode([token_id])
            records.append(
                {
                    "task": task_name,
                    "sample_id": int(row.sample_id),
                    "dataset_index": int(row.dataset_index),
                    "token_position": int(positions[idx].item()),
                    "token_id": token_id,
                    "token_text": token_text,
                    "logp_rl": float(rl_token_logp[idx].item()),
                    "logp_base": float(base_token_logp[idx].item()),
                    "delta_logp": float(delta_logp[idx].item()),
                }
            )

    records_df = pd.DataFrame(records)
    summary = {
        "task": task_name,
        "num_prompts_local": int(len(validation_df)),
        "num_token_records_local": int(len(records_df)),
        "generated_token_count_local": int(generated_token_count),
    }
    return records_df, sequence_cache, summary


def merge_rank_csv_files(rank_csv_paths: Sequence[Path], output_csv_path: Path) -> Dict[str, Any]:
    """Merge rank-local CSV files into a global CSV file.

    Args:
        rank_csv_paths: Ordered rank-local CSV paths.
        output_csv_path: Destination merged CSV path.

    Returns:
        Summary dictionary with row and file counts.
    """

    frames: List[pd.DataFrame] = []
    files_used = 0
    for csv_path in rank_csv_paths:
        if not csv_path.exists():
            continue
        if csv_path.stat().st_size == 0:
            continue
        frame = pd.read_csv(csv_path)
        frames.append(frame)
        files_used += 1

    if frames:
        merged_df = pd.concat(frames, ignore_index=True)
    else:
        merged_df = pd.DataFrame()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_csv_path, index=False)

    return {
        "output_csv_path": str(output_csv_path),
        "num_rows": int(len(merged_df)),
        "files_used": int(files_used),
    }


def merge_rank_sequence_caches(
    rank_sequence_paths: Sequence[Path],
    output_sequence_path: Path,
) -> Dict[str, Any]:
    """Merge rank-local sequence-cache dictionaries into one global dictionary.

    Args:
        rank_sequence_paths: Ordered rank-local `torch.save` paths.
        output_sequence_path: Destination path for global sequence cache.

    Returns:
        Summary dictionary with file and sequence counts.
    """

    merged_cache: Dict[int, torch.Tensor] = {}
    files_used = 0

    for sequence_path in rank_sequence_paths:
        if not sequence_path.exists():
            continue
        local_cache = torch.load(sequence_path, map_location="cpu")

        # Coerce sample IDs to int to avoid key-type issues during notebook load.
        for sample_id, full_ids in local_cache.items():
            merged_cache[int(sample_id)] = full_ids
        files_used += 1

    output_sequence_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_cache, output_sequence_path)

    return {
        "output_sequence_path": str(output_sequence_path),
        "sequence_count": int(len(merged_cache)),
        "files_used": int(files_used),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for distributed precompute."""

    parser = argparse.ArgumentParser(
        description="Distributed 8-GPU precompute for JWCM token-delta records.",
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model ID or local path.",
    )
    parser.add_argument(
        "--if-model-path",
        type=Path,
        default=Path(
            "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/"
            "Qwen3-1.7B-ifrl_ifeval/global_step_50/actor/huggingface"
        ),
        help="IF task model path.",
    )
    parser.add_argument(
        "--math-model-path",
        type=Path,
        default=Path(
            "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/"
            "Qwen3-1.7B-math/stage2/global_step_40/actor/huggingface"
        ),
        help="Math task model path.",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        required=True,
        help="Root directory containing <task>_validation.parquet files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where precomputed token-delta artifacts will be saved.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="if,math",
        help="Comma-separated task names to process (subset of: if,math).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1536,
        help="Maximum prompt tokens before truncation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=120,
        help="Distributed initialization timeout in minutes.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for distributed token-delta precompute."""

    args = parse_args()
    dtype = resolve_torch_dtype(args.dtype)
    dist_ctx = init_distributed(timeout_minutes=args.dist_timeout_minutes)

    # Add rank offset in seed so rank-local random calls stay deterministic
    # while avoiding accidental identical stochastic behavior across ranks.
    set_seed(args.seed + dist_ctx.rank)

    print(
        f"[Rank {dist_ctx.rank}] start | world_size={dist_ctx.world_size} "
        f"| local_rank={dist_ctx.local_rank} | device={dist_ctx.device}"
    )

    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    all_task_specs = {
        "if": TaskModelSpec(name="if", model_path=args.if_model_path),
        "math": TaskModelSpec(name="math", model_path=args.math_model_path),
    }

    task_specs: List[TaskModelSpec] = []
    for task_name in requested_tasks:
        if task_name not in all_task_specs:
            raise ValueError(f"Unknown task name in --tasks: {task_name}")
        task_specs.append(all_task_specs[task_name])

    for task_spec in task_specs:
        if not task_spec.model_path.exists():
            raise FileNotFoundError(f"Task model path does not exist: {task_spec.model_path}")

    if is_rank_zero(dist_ctx):
        args.output_root.mkdir(parents=True, exist_ok=True)
    barrier_if_needed(dist_ctx)

    base_model, _ = load_causal_lm(
        model_name_or_path=args.base_model_id,
        torch_dtype=dtype,
        device=dist_ctx.device,
    )

    for task_spec in task_specs:
        validation_path = args.validation_root / f"{task_spec.name}_validation.parquet"
        if not validation_path.exists():
            raise FileNotFoundError(
                f"Validation file is missing for task '{task_spec.name}': {validation_path}"
            )

        full_validation_df = pd.read_parquet(validation_path)
        local_validation_df = shard_dataframe_by_rank(
            df=full_validation_df,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
        print(
            f"[Rank {dist_ctx.rank}] task={task_spec.name} "
            f"local_validation_rows={len(local_validation_df)}"
        )

        task_model, task_tokenizer = load_causal_lm(
            model_name_or_path=task_spec.model_path,
            torch_dtype=dtype,
            device=dist_ctx.device,
        )

        token_records_df, sequence_cache, local_summary = collect_token_delta_records(
            task_name=task_spec.name,
            validation_df=local_validation_df,
            tokenizer=task_tokenizer,
            base_model=base_model,
            rl_model=task_model,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            device=dist_ctx.device,
            rank=dist_ctx.rank,
        )

        task_output_dir = args.output_root / task_spec.name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        token_csv_rank = task_output_dir / f"all_token_deltas_rank{dist_ctx.rank:02d}.csv"
        sequence_cache_rank = task_output_dir / f"sequence_cache_rank{dist_ctx.rank:02d}.pt"
        summary_rank = task_output_dir / f"token_summary_rank{dist_ctx.rank:02d}.json"

        token_records_df.to_csv(token_csv_rank, index=False)
        torch.save(sequence_cache, sequence_cache_rank)
        save_json(
            {
                "task": task_spec.name,
                "rank": dist_ctx.rank,
                "world_size": dist_ctx.world_size,
                "validation_path": str(validation_path),
                "task_model_path": str(task_spec.model_path),
                "created_at": now_iso(),
                "local_summary": local_summary,
            },
            summary_rank,
        )

        # Release rank-local task model as soon as this task precompute is done.
        del task_model
        del task_tokenizer
        del sequence_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        barrier_if_needed(dist_ctx)

        if is_rank_zero(dist_ctx):
            rank_csv_paths = [
                task_output_dir / f"all_token_deltas_rank{rank:02d}.csv"
                for rank in range(dist_ctx.world_size)
            ]
            rank_sequence_paths = [
                task_output_dir / f"sequence_cache_rank{rank:02d}.pt"
                for rank in range(dist_ctx.world_size)
            ]
            rank_summary_paths = [
                task_output_dir / f"token_summary_rank{rank:02d}.json"
                for rank in range(dist_ctx.world_size)
            ]

            merged_csv_path = task_output_dir / "all_token_deltas.csv"
            merged_sequence_path = task_output_dir / "sequence_cache.pt"
            merged_summary_path = task_output_dir / "token_summary.json"

            csv_merge_summary = merge_rank_csv_files(rank_csv_paths, merged_csv_path)
            sequence_merge_summary = merge_rank_sequence_caches(
                rank_sequence_paths=rank_sequence_paths,
                output_sequence_path=merged_sequence_path,
            )

            rank_summaries: List[Dict[str, Any]] = []
            total_prompts_local = 0
            total_token_records_local = 0
            total_generated_tokens_local = 0

            for summary_path in rank_summary_paths:
                if not summary_path.exists():
                    continue
                payload = load_json(summary_path)
                local = payload.get("local_summary", {})
                total_prompts_local += int(local.get("num_prompts_local", 0))
                total_token_records_local += int(local.get("num_token_records_local", 0))
                total_generated_tokens_local += int(local.get("generated_token_count_local", 0))
                rank_summaries.append(payload)

            aggregated_summary = {
                "task": task_spec.name,
                "created_at": now_iso(),
                "world_size": dist_ctx.world_size,
                "validation_path": str(validation_path),
                "task_model_path": str(task_spec.model_path),
                "csv_merge_summary": csv_merge_summary,
                "sequence_merge_summary": sequence_merge_summary,
                "aggregated_counts": {
                    "num_prompts_total": int(total_prompts_local),
                    "num_token_records_total": int(total_token_records_local),
                    "generated_token_count_total": int(total_generated_tokens_local),
                },
                "rank_summaries": rank_summaries,
            }
            save_json(aggregated_summary, merged_summary_path)

            rank_zero_print(
                dist_ctx,
                (
                    f"[Rank 0] task={task_spec.name} aggregated | "
                    f"token_rows={csv_merge_summary['num_rows']} | "
                    f"sequence_count={sequence_merge_summary['sequence_count']} | "
                    f"summary={merged_summary_path}"
                ),
            )

        barrier_if_needed(dist_ctx)

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rank_zero_print(dist_ctx, "Distributed token-delta precompute finished.")
    finalize_distributed(dist_ctx)


if __name__ == "__main__":
    main()
