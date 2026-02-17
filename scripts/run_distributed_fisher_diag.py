#!/usr/bin/env python3
"""Distributed empirical Fisher diagonal computation for one task model.

This script implements the diagonal Fisher estimate used in Fisher Merging
(Matena & Raffel, 2022):

    F_j = E_{x ~ D}[ E_{y ~ p_theta(y|x)} [ (d log p_theta(y|x) / d theta_j)^2 ] ]

Practical estimator in this script:
1. For each prompt x, sample one continuation y from the task model.
2. Compute sequence log-probability log p_theta(y|x) under teacher forcing.
3. Accumulate squared gradients of that scalar with respect to each parameter.
4. Average across all processed sequences globally (all ranks).

The script is designed for torchrun-based multi-GPU execution with rank sharding.
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
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class DistributedContext:
    """Runtime context for distributed execution.

    Args:
        rank: Global rank id.
        world_size: Number of ranks.
        local_rank: Rank index on the local node.
        device: Device string assigned to this process.
        distributed: Whether process group was initialized.
    """

    rank: int
    world_size: int
    local_rank: int
    device: str
    distributed: bool


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save a JSON payload with pretty formatting.

    Args:
        payload: JSON-serializable mapping object.
        output_path: Destination path.

    Returns:
        None. Payload is written to disk.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed.

    Returns:
        None. Random states are updated in-place.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve dtype string to torch dtype.

    Args:
        dtype_name: One of `bf16`, `fp16`, `fp32`.

    Returns:
        Torch dtype corresponding to the input string.
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
    """Initialize distributed state from torchrun environment variables.

    Args:
        timeout_minutes: Process group initialization timeout.

    Returns:
        DistributedContext with rank/device metadata.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return DistributedContext(
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
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
        distributed=True,
    )


def finalize_distributed(context: DistributedContext) -> None:
    """Destroy distributed process group when enabled.

    Args:
        context: Distributed context.

    Returns:
        None.
    """

    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def load_tokenizer_with_mistral_regex_fix(model_name_or_path: str) -> AutoTokenizer:
    """Load tokenizer with optional Mistral regex fix compatibility flag.

    Args:
        model_name_or_path: Hugging Face model id or local checkpoint path.

    Returns:
        Loaded tokenizer instance.
    """

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )


def load_causal_lm(
    model_name_or_path: str | Path,
    torch_dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load causal LM and tokenizer on the requested device.

    Args:
        model_name_or_path: Hugging Face model id or local checkpoint path.
        torch_dtype: Weight dtype used during model loading.
        device: Runtime device string.

    Returns:
        Tuple `(model, tokenizer)`.
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

    tokenizer = load_tokenizer_with_mistral_regex_fix(resolved_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep generation configuration explicit to avoid checkpoint-specific warnings.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        if hasattr(model.generation_config, "temperature"):
            model.generation_config.temperature = 1.0
        if hasattr(model.generation_config, "top_p"):
            model.generation_config.top_p = 1.0
        if hasattr(model.generation_config, "top_k"):
            model.generation_config.top_k = 50

    return model, tokenizer


def shard_dataframe_by_rank(df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """Shard dataframe rows by rank using deterministic striding.

    Args:
        df: Full dataframe.
        rank: Current rank.
        world_size: Number of ranks.

    Returns:
        Rank-local dataframe shard.
    """

    if world_size <= 1:
        return df.copy()
    return df.iloc[rank::world_size].reset_index(drop=True)


def compute_empirical_fisher_diagonal(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    validation_df: pd.DataFrame,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str,
    rank: int,
    cache_clear_interval: int,
) -> tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Compute local empirical Fisher diagonal sum for one rank.

    Args:
        model: Task model for generation and gradient computation.
        tokenizer: Tokenizer used for prompt encoding and generation.
        validation_df: Rank-local prompt dataframe containing `prompt_text`.
        max_prompt_tokens: Prompt truncation length.
        max_new_tokens: Number of response tokens sampled for each prompt.
        do_sample: Whether to sample (`True`) from model distribution.
        temperature: Sampling temperature used when `do_sample=True`.
        top_p: Nucleus sampling threshold used when `do_sample=True`.
        device: Runtime device.
        rank: Global rank index for progress-bar labeling.
        cache_clear_interval: Number of sequences between CUDA cache cleanup.

    Returns:
        Tuple of:
            - Fisher diagonal dictionary (unnormalized sum over local sequences)
            - Local processing summary dictionary
    """

    # Only include trainable floating-point parameters in Fisher accumulation.
    param_entries: List[tuple[str, torch.nn.Parameter]] = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    ]
    param_names = [name for name, _ in param_entries]
    param_list = [parameter for _, parameter in param_entries]

    fisher_diag: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for name, parameter in param_entries
    }

    processed_sequences = 0
    skipped_empty_generation = 0

    iterator = tqdm(
        validation_df.itertuples(index=False),
        total=len(validation_df),
        desc=f"Rank{rank:02d} Fisher",
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

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)
            generation_kwargs["top_p"] = float(top_p)

        # Sampling continuation y does not require gradients.
        with torch.no_grad():
            generated_ids = model.generate(**generation_kwargs)

        full_ids = generated_ids[0]
        prompt_len = int(input_ids.shape[1])
        full_len = int(full_ids.shape[0])

        # Skip degenerate sequences where generation stops immediately.
        if full_len <= prompt_len:
            skipped_empty_generation += 1
            continue

        # Teacher-force on the sampled sequence to compute log p_theta(y|x).
        full_batch = full_ids.unsqueeze(0)
        full_attention = torch.ones_like(full_batch, device=device)
        logits = model(input_ids=full_batch, attention_mask=full_attention).logits

        shifted_positions = torch.arange(prompt_len, full_len, device=device) - 1
        target_token_ids = full_ids[prompt_len:full_len]

        # Restrict log-softmax to generated token positions only.
        selected_logits = logits[0, shifted_positions, :].to(torch.float32)
        selected_log_probs = torch.log_softmax(selected_logits, dim=-1)
        sequence_log_prob = selected_log_probs.gather(
            dim=1,
            index=target_token_ids.unsqueeze(1),
        ).sum()

        # Fisher diagonal contribution: square of gradient of sequence log-probability.
        grads = torch.autograd.grad(
            sequence_log_prob,
            param_list,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            name = param_names[index]
            grad_fp32 = grad.detach().to(torch.float32)
            fisher_diag[name].addcmul_(grad_fp32, grad_fp32, value=1.0)

        processed_sequences += 1

        # Free intermediate tensors aggressively to keep long runs stable.
        del logits
        del selected_logits
        del selected_log_probs
        del sequence_log_prob
        del grads
        if processed_sequences % cache_clear_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    local_stats = {
        "processed_sequences_local": int(processed_sequences),
        "skipped_empty_generation_local": int(skipped_empty_generation),
        "local_prompt_rows": int(len(validation_df)),
        "num_tracked_parameters": int(len(param_entries)),
    }
    return fisher_diag, local_stats


def reduce_scalar_sum(value: int, device: str, distributed: bool) -> int:
    """Reduce one integer scalar by sum across ranks.

    Args:
        value: Local integer value.
        device: Runtime device for this rank.
        distributed: Whether process group is initialized.

    Returns:
        Global summed integer value.
    """

    tensor = torch.tensor([int(value)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for distributed Fisher computation."""

    parser = argparse.ArgumentParser(
        description="Distributed empirical Fisher diagonal computation for one task model.",
    )
    parser.add_argument("--task-name", type=str, required=True, help="Task key (for metadata).")
    parser.add_argument("--task-model-path", type=Path, required=True, help="Task model path.")
    parser.add_argument("--validation-parquet", type=Path, required=True, help="Validation parquet with prompt_text column.")
    parser.add_argument("--fisher-output-path", type=Path, required=True, help="Output .pt path for Fisher diagonal.")
    parser.add_argument("--summary-output-path", type=Path, required=True, help="Output .json path for run summary.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model loading dtype.")
    parser.add_argument("--max-prompt-tokens", type=int, default=1536, help="Prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum sampled continuation tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Sample continuation tokens from model distribution.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when do_sample is enabled.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus threshold when do_sample is enabled.")
    parser.add_argument("--cache-clear-interval", type=int, default=32, help="CUDA cache clear interval.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--dist-timeout-minutes", type=int, default=180, help="Distributed initialization timeout in minutes.")
    return parser.parse_args()


def main() -> None:
    """Entry point for distributed Fisher diagonal computation."""

    args = parse_args()
    context = init_distributed(timeout_minutes=args.dist_timeout_minutes)

    # Rank-dependent seed keeps stochastic sampling reproducible and independent.
    set_seed(int(args.seed) + int(context.rank))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = resolve_torch_dtype(args.dtype)

    validation_df = pd.read_parquet(args.validation_parquet)
    if "prompt_text" not in validation_df.columns:
        raise ValueError(
            f"Validation parquet must include 'prompt_text'. "
            f"columns={list(validation_df.columns)}"
        )

    local_validation_df = shard_dataframe_by_rank(
        df=validation_df,
        rank=context.rank,
        world_size=context.world_size,
    )

    model, tokenizer = load_causal_lm(
        model_name_or_path=args.task_model_path,
        torch_dtype=dtype,
        device=context.device,
    )

    fisher_diag, local_stats = compute_empirical_fisher_diagonal(
        model=model,
        tokenizer=tokenizer,
        validation_df=local_validation_df,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        device=context.device,
        rank=context.rank,
        cache_clear_interval=int(args.cache_clear_interval),
    )

    processed_sequences_global = reduce_scalar_sum(
        value=local_stats["processed_sequences_local"],
        device=context.device,
        distributed=context.distributed,
    )
    skipped_empty_generation_global = reduce_scalar_sum(
        value=local_stats["skipped_empty_generation_local"],
        device=context.device,
        distributed=context.distributed,
    )
    prompt_rows_global = reduce_scalar_sum(
        value=local_stats["local_prompt_rows"],
        device=context.device,
        distributed=context.distributed,
    )

    if context.distributed:
        for name in fisher_diag.keys():
            dist.all_reduce(fisher_diag[name], op=dist.ReduceOp.SUM)

    if processed_sequences_global > 0:
        normalization = float(processed_sequences_global)
        for name in fisher_diag.keys():
            fisher_diag[name].div_(normalization)

    if context.rank == 0:
        fisher_cpu = {name: tensor.detach().cpu() for name, tensor in fisher_diag.items()}
        args.fisher_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fisher_cpu, args.fisher_output_path)

        summary = {
            "created_at": now_iso(),
            "task_name": args.task_name,
            "task_model_path": str(args.task_model_path),
            "validation_parquet": str(args.validation_parquet),
            "fisher_output_path": str(args.fisher_output_path),
            "dtype": args.dtype,
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "world_size": int(context.world_size),
            "processed_sequences_global": int(processed_sequences_global),
            "skipped_empty_generation_global": int(skipped_empty_generation_global),
            "prompt_rows_global": int(prompt_rows_global),
            "num_tracked_parameters": int(local_stats["num_tracked_parameters"]),
        }
        save_json(summary, args.summary_output_path)
        print(f"Saved Fisher diagonal: {args.fisher_output_path}")
        print(f"Saved Fisher summary: {args.summary_output_path}")

    del model
    del tokenizer
    del fisher_diag
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finalize_distributed(context)


if __name__ == "__main__":
    main()
