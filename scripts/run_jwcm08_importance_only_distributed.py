#!/usr/bin/env python3
"""Distributed JWCM08 importance-only pipeline.

This script is a distributed companion to:
`scripts/run_jwcm08_importance_only.py`.

Goal:
- Keep the same importance output format for downstream merging/analysis.
- Speed up slow attribution by sharding work across multiple GPUs with `torchrun`.

Distributed execution model:
1. Each rank loads the same base/task checkpoints.
2. Rollout rows are filtered deterministically, then sharded by rank (striding).
3. Each rank computes local token-level `delta_logp` records.
4. Global critical-token thresholds are computed from all ranks jointly.
5. Each rank computes local importance partial sums on its shard.
6. Partial importance tensors are reduced to rank 0 and saved.

Notes:
- `max_critical_tokens` is applied per-rank in distributed mode to avoid expensive
  global top-k coordination overhead.
- `max_backprop_tokens` is treated as a global target and converted to an
  approximate per-rank budget (`ceil(global/world_size)`).
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


# Ensure repo root is importable when this script is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_jwcm08_importance_only import (  # noqa: E402
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_FISHER_TASK_ROOT,
    DEFAULT_FISHER_VALIDATION_ROOT,
    DEFAULT_IF_MODEL_PATH,
    DEFAULT_MATH_MODEL_PATH,
    DEFAULT_OUTPUT_ROOT,
    AttributionConfig,
    CriticalTokenConfig,
    TaskSpec,
    TokenSelectionConfig,
    _int_to_optional_limit,
    _resolve_selection_modes,
    build_abs_task_vector_cpu,
    build_critical_sequence_payload,
    build_mode_suffix,
    collect_token_delta_records_from_fisher_rollouts,
    initialize_importance_buffers,
    load_causal_lm,
    now_iso,
    resolve_torch_dtype,
    save_json,
    set_seed,
)


@dataclass(frozen=True)
class DistributedContext:
    """Distributed runtime context.

    Args:
        rank: Global rank id.
        world_size: Number of ranks.
        local_rank: Local rank id on node.
        device: Device string assigned to this rank.
        distributed: Whether process group is initialized.
        backend: Backend name (`nccl` or `gloo`) when distributed.
    """

    rank: int
    world_size: int
    local_rank: int
    device: str
    distributed: bool
    backend: str | None


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for distributed JWCM importance computation.

    Args:
        base_model_id: Base model id/path.
        output_root: Root directory for artifacts.
        seed: Global random seed.
        model_dtype: Torch dtype used for model loading.
        device: Device string for this rank.
    """

    base_model_id: str
    output_root: Path
    seed: int
    model_dtype: torch.dtype
    device: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for distributed JWCM importance pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Distributed JWCM08 importance-only pipeline. "
            "Run with torchrun for multi-GPU execution."
        ),
    )
    parser.add_argument("--task", type=str, choices=["if", "math", "all"], default="all")
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["positive", "negative", "absolute", "all"],
        default="positive",
        help="Critical-token selection mode. `all` runs all three modes sequentially.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.20,
        help="Tail selection ratio p in (0, 1]. Default is 0.20 (20%%).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Minimum absolute threshold stabilizer for token selection.",
    )
    parser.add_argument(
        "--max-critical-tokens",
        type=int,
        default=-1,
        help=(
            "Optional cap on selected tokens per rank and per mode. "
            "Use -1 for no cap."
        ),
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=-1,
        help="Prompt truncation budget. Use -1 for no prompt truncation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max-response-tokens",
        dest="max_new_tokens",
        type=int,
        default=-1,
        help="Response truncation budget. Use -1 for no response truncation.",
    )
    parser.add_argument(
        "--include-incorrect-rows",
        action="store_true",
        help="Include incorrect rollout rows. Default behavior uses only correct rows when available.",
    )
    parser.add_argument(
        "--max-rollouts-per-sample",
        type=int,
        default=-1,
        help="Optional cap on rollout rows per sample. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-total-rollouts",
        "--max-rollout-rows",
        dest="max_total_rollouts",
        type=int,
        default=-1,
        help="Optional global rollout-row cap before sharding. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--validation-samples-per-task",
        type=int,
        default=-1,
        help="Optional validation-row cap. Use -1 to use all rows.",
    )
    parser.add_argument(
        "--attribution-mode",
        type=str,
        choices=["exact_token_abs", "sequence_sum_approx"],
        default="exact_token_abs",
    )
    parser.add_argument(
        "--max-backprop-tokens",
        type=int,
        default=-1,
        help="Approximate global backprop token budget per mode. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--no-normalize-by-token-count",
        action="store_true",
        help="Disable normalization by global processed token count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--base-model-id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--if-model-path", type=Path, default=DEFAULT_IF_MODEL_PATH)
    parser.add_argument(
        "--if-validation-path",
        type=Path,
        default=DEFAULT_FISHER_VALIDATION_ROOT / "if_validation.parquet",
    )
    parser.add_argument(
        "--if-rollout-path",
        type=Path,
        default=DEFAULT_FISHER_TASK_ROOT / "if" / "correct_rollout_trajectories.parquet",
    )
    parser.add_argument("--math-model-path", type=Path, default=DEFAULT_MATH_MODEL_PATH)
    parser.add_argument(
        "--math-validation-path",
        type=Path,
        default=DEFAULT_FISHER_VALIDATION_ROOT / "math_validation.parquet",
    )
    parser.add_argument(
        "--math-rollout-path",
        type=Path,
        default=DEFAULT_FISHER_TASK_ROOT / "math" / "correct_rollout_trajectories.parquet",
    )

    parser.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=180,
        help="Process-group initialization timeout in minutes.",
    )
    return parser.parse_args()


def init_distributed_context(timeout_minutes: int) -> DistributedContext:
    """Initialize distributed context from torchrun environment variables.

    Args:
        timeout_minutes: Process-group initialization timeout.

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
            backend=None,
        )

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=int(timeout_minutes)),
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
        backend=backend,
    )


def finalize_distributed_context(context: DistributedContext) -> None:
    """Finalize distributed process group if initialized.

    Args:
        context: Distributed context.

    Returns:
        None.
    """

    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier_if_needed(context: DistributedContext) -> None:
    """Synchronize all ranks when distributed mode is active.

    Args:
        context: Distributed context.

    Returns:
        None.
    """

    if context.distributed:
        dist.barrier()


def rank_zero_print(context: DistributedContext, message: str) -> None:
    """Print message only on rank 0 to avoid duplicated logs.

    Args:
        context: Distributed context.
        message: Message text.

    Returns:
        None.
    """

    if context.rank == 0:
        print(message, flush=True)


def _validate_top_p(top_p: float) -> None:
    """Validate top-p range.

    Args:
        top_p: Selection ratio.

    Returns:
        None. Raises ValueError on invalid range.
    """

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"`top_p` must be in (0, 1], got {top_p}")


def _build_task_specs(args: argparse.Namespace) -> List[TaskSpec]:
    """Build selected task specs from CLI args.

    Args:
        args: Parsed CLI args.

    Returns:
        List of selected TaskSpec objects.
    """

    all_tasks = {
        "if": TaskSpec(
            name="if",
            model_path=args.if_model_path,
            fisher_validation_path=args.if_validation_path,
            fisher_correct_rollout_path=args.if_rollout_path,
        ),
        "math": TaskSpec(
            name="math",
            model_path=args.math_model_path,
            fisher_validation_path=args.math_validation_path,
            fisher_correct_rollout_path=args.math_rollout_path,
        ),
    }
    if args.task == "all":
        return [all_tasks["if"], all_tasks["math"]]
    return [all_tasks[str(args.task)]]


def _validate_required_paths(task_specs: Sequence[TaskSpec]) -> None:
    """Validate required model/data paths.

    Args:
        task_specs: Selected task specs.

    Returns:
        None. Raises FileNotFoundError on missing path.
    """

    for task_spec in task_specs:
        for required_path in [
            task_spec.model_path,
            task_spec.fisher_validation_path,
            task_spec.fisher_correct_rollout_path,
        ]:
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required path does not exist for task '{task_spec.name}': {required_path}"
                )


def shard_dataframe_by_rank(df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """Shard dataframe rows by deterministic rank striding.

    Args:
        df: Input dataframe.
        rank: Global rank.
        world_size: Number of ranks.

    Returns:
        Rank-local dataframe shard.
    """

    if world_size <= 1:
        return df.copy()
    return df.iloc[rank::world_size].reset_index(drop=True)


def prepare_rollout_rows_for_task(
    correct_rollout_df: pd.DataFrame,
    allowed_sample_ids: set[int] | None,
    cfg: CriticalTokenConfig,
    sampling_seed: int,
) -> pd.DataFrame:
    """Prepare globally filtered rollout rows before rank sharding.

    Filtering steps mirror notebook/standalone logic:
    1. Optional `is_correct` filtering.
    2. Optional sample-id whitelist filtering.
    3. Optional per-sample rollout cap.
    4. Optional global rollout-row sampling cap.

    Args:
        correct_rollout_df: Raw correct-rollout dataframe.
        allowed_sample_ids: Optional sample-id whitelist.
        cfg: Critical-token config with filtering controls.
        sampling_seed: Deterministic seed for global rollout sampling.

    Returns:
        Filtered rollout dataframe ready for rank sharding.
    """

    working_df = correct_rollout_df.copy()

    if cfg.use_only_correct_rows and "is_correct" in working_df.columns:
        working_df = working_df[working_df["is_correct"].astype(bool)].copy()

    if allowed_sample_ids is not None:
        working_df = working_df[working_df["sample_index"].isin(sorted(allowed_sample_ids))].copy()

    if cfg.max_rollouts_per_sample is not None:
        working_df = (
            working_df.groupby("sample_index", sort=False)
            .head(int(cfg.max_rollouts_per_sample))
            .reset_index(drop=True)
        )

    if cfg.max_total_rollouts is not None and len(working_df) > int(cfg.max_total_rollouts):
        working_df = (
            working_df.sample(
                n=int(cfg.max_total_rollouts),
                random_state=int(sampling_seed),
                replace=False,
            )
            .sort_values("sample_index")
            .reset_index(drop=True)
        )

    return working_df.reset_index(drop=True)


def all_reduce_sum_int(value: int, context: DistributedContext, device: str) -> int:
    """All-reduce integer scalar with SUM operation.

    Args:
        value: Local integer value.
        context: Distributed context.
        device: Runtime device string.

    Returns:
        Global summed integer.
    """

    tensor = torch.tensor([int(value)], dtype=torch.int64, device=device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def compute_global_threshold(
    local_deltas: np.ndarray,
    mode: str,
    top_p: float,
    epsilon: float,
    context: DistributedContext,
    device: str,
) -> Dict[str, float]:
    """Compute global selection threshold across all ranks.

    This helper gathers local `delta_logp` arrays from all ranks, computes a
    global quantile threshold on rank 0, and broadcasts the result.

    Args:
        local_deltas: Rank-local `delta_logp` array.
        mode: Selection mode (`positive`, `negative`, `absolute`).
        top_p: Tail ratio.
        epsilon: Minimum absolute threshold stabilizer.
        context: Distributed context.
        device: Runtime device.

    Returns:
        Dictionary with `quantile_threshold`, `final_threshold`, and `global_token_count`.
    """

    local_payload = np.asarray(local_deltas, dtype=np.float64)

    if context.distributed:
        gathered: List[np.ndarray | None] = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered, local_payload)
        if context.rank == 0:
            valid_arrays = [
                np.asarray(array, dtype=np.float64)
                for array in gathered
                if array is not None and int(np.asarray(array).size) > 0
            ]
            if valid_arrays:
                global_deltas = np.concatenate(valid_arrays, axis=0)
            else:
                global_deltas = np.empty((0,), dtype=np.float64)
        else:
            global_deltas = np.empty((0,), dtype=np.float64)
    else:
        global_deltas = local_payload

    if context.rank == 0:
        if global_deltas.size <= 0:
            raise RuntimeError("No token deltas available to compute global threshold.")

        if mode == "positive":
            quantile_threshold = float(np.quantile(global_deltas, 1.0 - top_p))
            final_threshold = float(max(epsilon, quantile_threshold))
        elif mode == "negative":
            quantile_threshold = float(np.quantile(global_deltas, top_p))
            final_threshold = float(min(-epsilon, quantile_threshold))
        elif mode == "absolute":
            abs_deltas = np.abs(global_deltas)
            quantile_threshold = float(np.quantile(abs_deltas, 1.0 - top_p))
            final_threshold = float(max(epsilon, quantile_threshold))
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")

        token_count = float(global_deltas.size)
        payload_tensor = torch.tensor(
            [quantile_threshold, final_threshold, token_count],
            dtype=torch.float64,
            device=device,
        )
    else:
        payload_tensor = torch.zeros((3,), dtype=torch.float64, device=device)

    if context.distributed:
        dist.broadcast(payload_tensor, src=0)

    return {
        "quantile_threshold": float(payload_tensor[0].item()),
        "final_threshold": float(payload_tensor[1].item()),
        "global_token_count": int(payload_tensor[2].item()),
    }


def select_local_critical_tokens_with_global_threshold(
    token_records_df: pd.DataFrame,
    mode: str,
    final_threshold: float,
    max_critical_tokens: int | None,
) -> pd.DataFrame:
    """Select local critical tokens using a globally computed threshold.

    Args:
        token_records_df: Rank-local token-delta dataframe.
        mode: Selection mode.
        final_threshold: Global threshold value for this mode.
        max_critical_tokens: Optional per-rank cap.

    Returns:
        Local critical-token dataframe.
    """

    if token_records_df.empty:
        return token_records_df.copy()

    working_df = token_records_df.copy()
    if mode == "positive":
        critical_df = working_df[working_df["delta_logp"] >= float(final_threshold)].copy()
        critical_df.sort_values("delta_logp", ascending=False, inplace=True)
    elif mode == "negative":
        critical_df = working_df[working_df["delta_logp"] <= float(final_threshold)].copy()
        critical_df.sort_values("delta_logp", ascending=True, inplace=True)
    elif mode == "absolute":
        working_df["abs_delta_logp"] = working_df["delta_logp"].abs()
        critical_df = working_df[working_df["abs_delta_logp"] >= float(final_threshold)].copy()
        critical_df.sort_values(["abs_delta_logp", "delta_logp"], ascending=[False, False], inplace=True)
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    if max_critical_tokens is not None and len(critical_df) > int(max_critical_tokens):
        # In distributed mode this cap is intentionally per-rank to avoid
        # expensive global top-k synchronization and payload exchange.
        critical_df = critical_df.head(int(max_critical_tokens)).copy()

    critical_df.reset_index(drop=True, inplace=True)
    return critical_df


def compute_importance_scores_distributed(
    task_model: AutoModelForCausalLM,
    local_critical_payload: Sequence[Mapping[str, Any]],
    abs_delta: Mapping[str, torch.Tensor],
    cfg: AttributionConfig,
    context: DistributedContext,
    device: str,
) -> tuple[Dict[str, torch.Tensor] | None, Dict[str, Any]]:
    """Compute and reduce JWCM importance scores across distributed ranks.

    Implementation strategy:
    - Each rank accumulates local partial importance on GPU.
    - GPU tensors are reduced (SUM) to rank 0.
    - Rank 0 applies global normalization and returns CPU tensors.

    Args:
        task_model: Task model for gradient computation.
        local_critical_payload: Rank-local critical-sequence payload.
        abs_delta: Absolute task vector (`|Δθ|`) on CPU.
        cfg: Attribution config.
        context: Distributed context.
        device: Runtime device.

    Returns:
        Tuple:
        - Rank 0: global importance dictionary on CPU; other ranks: `None`
        - Metadata dictionary with local/global counters
    """

    abs_delta_gpu: Dict[str, torch.Tensor] = {
        name: tensor.to(device, non_blocking=True)
        for name, tensor in abs_delta.items()
    }
    importance_gpu: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(tensor)
        for name, tensor in abs_delta_gpu.items()
    }

    param_entries = [
        (name, parameter)
        for name, parameter in task_model.named_parameters()
        if parameter.requires_grad
    ]
    param_names = [name for name, _ in param_entries]
    param_list = [parameter for _, parameter in param_entries]

    # Convert global budget to approximate per-rank budget for distributed runs.
    if cfg.max_backprop_tokens is None:
        local_backprop_budget = None
    elif context.world_size > 1:
        local_backprop_budget = int(math.ceil(float(cfg.max_backprop_tokens) / float(context.world_size)))
    else:
        local_backprop_budget = int(cfg.max_backprop_tokens)

    processed_sequences_local = 0
    processed_tokens_local = 0
    hit_local_budget = False

    iterator = tqdm(
        local_critical_payload,
        desc=f"Rank{context.rank:02d} Attribution ({cfg.mode})",
        disable=(context.rank != 0),
    )
    for entry in iterator:
        if hit_local_budget:
            break

        full_ids_cpu = entry["full_ids"]
        critical_positions = [int(pos) for pos in entry["critical_positions"] if int(pos) > 0]
        if not critical_positions:
            continue

        full_ids = full_ids_cpu.to(device).unsqueeze(0)
        outputs = task_model(input_ids=full_ids)
        logits = outputs.logits

        if cfg.mode == "exact_token_abs":
            shifted_positions = [position - 1 for position in critical_positions]
            target_ids = [int(full_ids[0, position].item()) for position in critical_positions]

            critical_logits = logits[0, shifted_positions, :].to(torch.float32)
            critical_log_probs = torch.log_softmax(critical_logits, dim=-1)

            for token_idx in range(len(critical_positions)):
                scalar_log_prob = critical_log_probs[token_idx, target_ids[token_idx]]
                retain_graph = token_idx < (len(critical_positions) - 1)

                grads = torch.autograd.grad(
                    scalar_log_prob,
                    param_list,
                    retain_graph=retain_graph,
                    create_graph=False,
                    allow_unused=True,
                )

                for grad_index, grad in enumerate(grads):
                    if grad is None:
                        continue
                    name = param_names[grad_index]
                    if name in importance_gpu:
                        grad_abs = grad.to(torch.float32).abs_()
                        importance_gpu[name].add_(grad_abs.mul_(abs_delta_gpu[name]))

                processed_tokens_local += 1
                if local_backprop_budget is not None and processed_tokens_local >= local_backprop_budget:
                    hit_local_budget = True
                    break

        elif cfg.mode == "sequence_sum_approx":
            log_probs = torch.log_softmax(logits[:, :-1, :].to(torch.float32), dim=-1)
            scalar_terms: List[torch.Tensor] = []
            for token_position in critical_positions:
                shifted = token_position - 1
                target_id = int(full_ids[0, token_position].item())
                scalar_terms.append(log_probs[0, shifted, target_id])

            summed_scalar = torch.stack(scalar_terms).sum()
            grads = torch.autograd.grad(
                summed_scalar,
                param_list,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

            for grad_index, grad in enumerate(grads):
                if grad is None:
                    continue
                name = param_names[grad_index]
                if name in importance_gpu:
                    grad_abs = grad.to(torch.float32).abs_()
                    importance_gpu[name].add_(grad_abs.mul_(abs_delta_gpu[name]))

            processed_tokens_local += len(critical_positions)
        else:
            raise ValueError(f"Unknown attribution mode: {cfg.mode}")

        processed_sequences_local += 1
        del outputs
        del logits

        if processed_sequences_local % 64 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if local_backprop_budget is not None and processed_tokens_local >= local_backprop_budget:
            hit_local_budget = True

    # Reduce counters globally for final metadata and normalization denominator.
    token_tensor = torch.tensor([processed_tokens_local], dtype=torch.int64, device=device)
    sequence_tensor = torch.tensor([processed_sequences_local], dtype=torch.int64, device=device)
    if context.distributed:
        dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(sequence_tensor, op=dist.ReduceOp.SUM)
    processed_tokens_global = int(token_tensor.item())
    processed_sequences_global = int(sequence_tensor.item())

    # Reduce large importance tensors to rank 0 only.
    for name in param_names:
        if name not in importance_gpu:
            continue
        if context.distributed:
            dist.reduce(importance_gpu[name], dst=0, op=dist.ReduceOp.SUM)

    if context.rank == 0:
        importance_cpu = {name: tensor.cpu() for name, tensor in importance_gpu.items()}
        if cfg.normalize_by_token_count and processed_tokens_global > 0:
            scale = float(processed_tokens_global)
            for name in importance_cpu.keys():
                importance_cpu[name].div_(scale)
    else:
        importance_cpu = None

    del abs_delta_gpu
    del importance_gpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "mode": cfg.mode,
        "processed_sequences_local": int(processed_sequences_local),
        "processed_tokens_local": int(processed_tokens_local),
        "processed_sequences_global": int(processed_sequences_global),
        "processed_tokens_global": int(processed_tokens_global),
        "max_backprop_tokens_global": cfg.max_backprop_tokens,
        "max_backprop_tokens_local_budget": local_backprop_budget,
        "normalize_by_token_count": bool(cfg.normalize_by_token_count),
        "world_size": int(context.world_size),
    }
    return importance_cpu, metadata


def main() -> None:
    """Run distributed JWCM08 importance-only pipeline."""

    args = parse_args()
    context = init_distributed_context(timeout_minutes=int(args.dist_timeout_minutes))

    try:
        _validate_top_p(top_p=float(args.top_p))

        # Keep the same seed on all ranks so rollout filtering/sampling is identical
        # before deterministic rank-based sharding.
        set_seed(int(args.seed))

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        task_specs = _build_task_specs(args=args)
        _validate_required_paths(task_specs=task_specs)

        runtime = RuntimeConfig(
            base_model_id=str(args.base_model_id),
            output_root=Path(args.output_root),
            seed=int(args.seed),
            model_dtype=resolve_torch_dtype(str(args.dtype)),
            device=str(context.device),
        )
        critical_cfg = CriticalTokenConfig(
            max_prompt_tokens=_int_to_optional_limit(int(args.max_prompt_tokens)),
            max_new_tokens=_int_to_optional_limit(int(args.max_new_tokens)),
            use_only_correct_rows=not bool(args.include_incorrect_rows),
            max_rollouts_per_sample=_int_to_optional_limit(int(args.max_rollouts_per_sample)),
            max_total_rollouts=_int_to_optional_limit(int(args.max_total_rollouts)),
        )
        selection_modes = _resolve_selection_modes(mode=str(args.selection_mode))
        selection_template = TokenSelectionConfig(
            mode="positive",
            top_p=float(args.top_p),
            epsilon=float(args.epsilon),
            max_critical_tokens=_int_to_optional_limit(int(args.max_critical_tokens)),
        )
        attribution_cfg = AttributionConfig(
            mode=str(args.attribution_mode),
            max_backprop_tokens=_int_to_optional_limit(int(args.max_backprop_tokens)),
            normalize_by_token_count=not bool(args.no_normalize_by_token_count),
        )
        validation_samples_per_task = _int_to_optional_limit(int(args.validation_samples_per_task))

        # Create output directories on rank 0 only, then synchronize.
        dirs = {
            "validation": runtime.output_root / "validation",
            "critical": runtime.output_root / "critical_tokens",
            "importance": runtime.output_root / "importance",
            "metadata": runtime.output_root / "metadata",
        }
        if context.rank == 0:
            runtime.output_root.mkdir(parents=True, exist_ok=True)
            for path in dirs.values():
                path.mkdir(parents=True, exist_ok=True)
        barrier_if_needed(context)

        rank_zero_print(context, f"Distributed context: rank={context.rank} world_size={context.world_size} device={context.device}")
        rank_zero_print(context, f"Output root: {runtime.output_root}")
        rank_zero_print(context, f"Selection modes: {selection_modes}, top_p={selection_template.top_p}")

        run_summary: Dict[str, Any] = {
            "created_at": now_iso(),
            "runtime": asdict(runtime),
            "distributed": {
                "world_size": int(context.world_size),
                "backend": context.backend,
            },
            "critical_config": asdict(critical_cfg),
            "selection_config_template": asdict(selection_template),
            "selection_modes": list(selection_modes),
            "attribution_config": asdict(attribution_cfg),
            "validation_samples_per_task": validation_samples_per_task,
            "task_specs": [asdict(task_spec) for task_spec in task_specs],
            "task_summaries": {},
        }

        base_model, _ = load_causal_lm(
            model_name_or_path=runtime.base_model_id,
            torch_dtype=runtime.model_dtype,
            device=runtime.device,
        )

        for task_spec in task_specs:
            rank_zero_print(context, f"\n=== Task: {task_spec.name} ===")

            validation_df = pd.read_parquet(task_spec.fisher_validation_path)
            required_validation_columns = {"sample_id", "dataset_index", "prompt_text"}
            missing_columns = sorted(required_validation_columns - set(validation_df.columns))
            if missing_columns:
                raise ValueError(
                    f"Validation parquet missing required columns for task '{task_spec.name}': {missing_columns}"
                )

            if validation_samples_per_task is None:
                selected_validation_df = validation_df.copy()
            else:
                selected_validation_df = validation_df.head(int(validation_samples_per_task)).copy()

            if selected_validation_df.empty:
                raise ValueError(
                    f"Selected validation set is empty for task '{task_spec.name}'. "
                    f"source_rows={len(validation_df)} selected={validation_samples_per_task}"
                )

            selected_validation_df["sample_id"] = selected_validation_df["sample_id"].astype(np.int64)
            selected_validation_df["dataset_index"] = selected_validation_df["dataset_index"].astype(np.int64)

            selected_sample_ids = set(int(value) for value in selected_validation_df["sample_id"].tolist())
            sample_id_to_dataset_index = {
                int(row.sample_id): int(row.dataset_index)
                for row in selected_validation_df.itertuples(index=False)
            }

            if context.rank == 0:
                validation_copy_path = dirs["validation"] / f"{task_spec.name}_validation.parquet"
                selected_validation_df.to_parquet(validation_copy_path, index=False)
            else:
                validation_copy_path = dirs["validation"] / f"{task_spec.name}_validation.parquet"
            barrier_if_needed(context)

            task_model, task_tokenizer = load_causal_lm(
                model_name_or_path=task_spec.model_path,
                torch_dtype=runtime.model_dtype,
                device=runtime.device,
            )

            raw_rollout_df = pd.read_parquet(task_spec.fisher_correct_rollout_path)
            prepared_rollout_df = prepare_rollout_rows_for_task(
                correct_rollout_df=raw_rollout_df,
                allowed_sample_ids=selected_sample_ids,
                cfg=critical_cfg,
                sampling_seed=runtime.seed,
            )
            shard_rollout_df = shard_dataframe_by_rank(
                df=prepared_rollout_df,
                rank=context.rank,
                world_size=context.world_size,
            )
            rank_zero_print(
                context,
                (
                    f"[{task_spec.name}] rollout rows: "
                    f"raw={len(raw_rollout_df)} prepared={len(prepared_rollout_df)} "
                    f"rank0_shard={len(shard_rollout_df) if context.rank == 0 else 'n/a'}"
                ),
            )

            # Disable filtering in the collector because it was already applied globally.
            shard_collect_cfg = CriticalTokenConfig(
                max_prompt_tokens=critical_cfg.max_prompt_tokens,
                max_new_tokens=critical_cfg.max_new_tokens,
                use_only_correct_rows=False,
                max_rollouts_per_sample=None,
                max_total_rollouts=None,
            )
            local_token_df, local_sequence_cache, local_token_summary = collect_token_delta_records_from_fisher_rollouts(
                task_name=task_spec.name,
                correct_rollout_df=shard_rollout_df,
                tokenizer=task_tokenizer,
                base_model=base_model,
                rl_model=task_model,
                cfg=shard_collect_cfg,
                device=runtime.device,
                allowed_sample_ids=None,
                sample_id_to_dataset_index=sample_id_to_dataset_index,
                sampling_seed=runtime.seed,
            )

            if not local_token_df.empty:
                local_token_df["sample_id"] = local_token_df["sample_id"].astype(np.int64)
                local_token_df["sequence_id"] = local_token_df["sequence_id"].astype(np.int64)
                local_token_df["token_position"] = local_token_df["token_position"].astype(np.int64)
                local_token_df["token_id"] = local_token_df["token_id"].astype(np.int64)

            global_token_record_count = all_reduce_sum_int(
                value=int(len(local_token_df)),
                context=context,
                device=runtime.device,
            )
            if global_token_record_count <= 0:
                raise RuntimeError(f"No token records produced for task '{task_spec.name}'.")

            abs_delta = build_abs_task_vector_cpu(
                base_model=base_model,
                task_model=task_model,
            )

            local_deltas = (
                local_token_df["delta_logp"].to_numpy(dtype=np.float64)
                if not local_token_df.empty
                else np.empty((0,), dtype=np.float64)
            )

            mode_summaries: Dict[str, Any] = {}
            for mode in selection_modes:
                threshold_info = compute_global_threshold(
                    local_deltas=local_deltas,
                    mode=mode,
                    top_p=selection_template.top_p,
                    epsilon=selection_template.epsilon,
                    context=context,
                    device=runtime.device,
                )

                local_critical_df = select_local_critical_tokens_with_global_threshold(
                    token_records_df=local_token_df,
                    mode=mode,
                    final_threshold=float(threshold_info["final_threshold"]),
                    max_critical_tokens=selection_template.max_critical_tokens,
                )
                local_critical_count = int(len(local_critical_df))
                global_critical_count = all_reduce_sum_int(
                    value=local_critical_count,
                    context=context,
                    device=runtime.device,
                )

                local_payload = build_critical_sequence_payload(
                    critical_df=local_critical_df,
                    sequence_cache=local_sequence_cache,
                )

                if local_payload:
                    importance_cpu, importance_meta = compute_importance_scores_distributed(
                        task_model=task_model,
                        local_critical_payload=local_payload,
                        abs_delta=abs_delta,
                        cfg=attribution_cfg,
                        context=context,
                        device=runtime.device,
                    )
                else:
                    # All ranks must still participate in distributed reduction path.
                    # We pass empty payload to preserve synchronization behavior.
                    importance_cpu, importance_meta = compute_importance_scores_distributed(
                        task_model=task_model,
                        local_critical_payload=[],
                        abs_delta=abs_delta,
                        cfg=attribution_cfg,
                        context=context,
                        device=runtime.device,
                    )

                mode_suffix = build_mode_suffix(mode=mode, top_p=selection_template.top_p)
                if context.rank == 0:
                    if importance_cpu is None:
                        # Defensive fallback; this branch should not trigger for rank 0.
                        importance_cpu = initialize_importance_buffers(abs_delta=abs_delta)
                    importance_path = dirs["importance"] / f"importance_{task_spec.name}_{mode_suffix}.pt"
                    torch.save(importance_cpu, importance_path)
                    rank_zero_print(context, f"[{task_spec.name}] saved importance ({mode}): {importance_path}")

                    mode_summaries[mode] = {
                        "mode_suffix": mode_suffix,
                        "selection_mode": mode,
                        "selection_top_p": float(selection_template.top_p),
                        "selection_epsilon": float(selection_template.epsilon),
                        "quantile_threshold": float(threshold_info["quantile_threshold"]),
                        "final_threshold": float(threshold_info["final_threshold"]),
                        "global_token_count": int(threshold_info["global_token_count"]),
                        "global_critical_count": int(global_critical_count),
                        "local_critical_count_rank0": int(local_critical_count),
                        "max_critical_tokens_per_rank": selection_template.max_critical_tokens,
                        "importance_path": str(importance_path),
                        "importance_meta": importance_meta,
                    }

                del local_critical_df
                del local_payload
                if importance_cpu is not None:
                    del importance_cpu
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if context.rank == 0:
                run_summary["task_summaries"][task_spec.name] = {
                    "task_model_path": str(task_spec.model_path),
                    "validation_source_path": str(task_spec.fisher_validation_path),
                    "rollout_source_path": str(task_spec.fisher_correct_rollout_path),
                    "validation_copy_path": str(validation_copy_path),
                    "num_selected_validation_samples": int(len(selected_sample_ids)),
                    "local_token_summary_rank0": local_token_summary,
                    "global_token_record_count": int(global_token_record_count),
                    "modes": mode_summaries,
                }

            del abs_delta
            del local_token_df
            del local_sequence_cache
            del task_model
            del task_tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            barrier_if_needed(context)

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if context.rank == 0:
            summary_path = dirs["metadata"] / "jwcm08_importance_only_distributed_run_summary.json"
            save_json(run_summary, summary_path)
            rank_zero_print(context, f"\nSaved distributed run summary: {summary_path}")
            rank_zero_print(context, "Distributed JWCM importance-only pipeline finished.")

    finally:
        finalize_distributed_context(context)


if __name__ == "__main__":
    main()
