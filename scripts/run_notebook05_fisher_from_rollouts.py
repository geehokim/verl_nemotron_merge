#!/usr/bin/env python3
"""Compute Fisher information from rollout parquet files used in notebook 05.

This script extracts and productionizes the Fisher-computation portion from:
`merging_analysis/05_fisher_merging_precision_weighted.ipynb`.

Design goals:
1. Run outside Jupyter so it can be launched in background jobs (`nohup`, `tmux`).
2. Reuse the same rollout artifacts (`correct_rollout_trajectories.parquet`).
3. Save Fisher outputs in notebook-compatible names for direct reuse:
   - `fisher_diag_{task}.pt`
   - `fisher_diag_{task}_summary.json`
4. Default to using all available rollout rows (no sampling cap).
5. Default to no response-token truncation (`--max-response-tokens -1`).

Notes:
- The default estimator is the same one-shot sequence estimator from notebook 05:
  `g = d/dtheta [sum_t log p(y_t | x, y_<t)]`, accumulate `g^2`.
- Optional tokenwise estimator is provided for experimentation:
  accumulate `sum_t (d/dtheta log p_t)^2` with cross-terms removed.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TaskSpec:
    """Per-task model and data specification.

    Args:
        name: Short task key (e.g., `if`, `math`) used in output naming.
        model_path: Local Hugging Face model checkpoint path.
        validation_parquet: Validation parquet containing `prompt` column.
        rollout_parquet: Rollout parquet containing `sample_index` and `output`.
    """

    name: str
    model_path: Path
    validation_parquet: Path
    rollout_parquet: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for model loading, output paths, and reproducibility.

    Args:
        output_root: Root directory where Fisher artifacts will be written.
        output_subdir: Relative directory under `output_root` for Fisher tensors.
        metadata_subdir: Relative directory under `output_root` for metadata.
        seed: Global random seed.
        model_dtype_name: Model loading dtype alias (`bf16`, `fp16`, `fp32`).
        fisher_device: Device used for Fisher gradient computation.
        grad_checkpointing: Whether to enable gradient checkpointing.
    """

    output_root: Path
    output_subdir: str
    metadata_subdir: str
    seed: int
    model_dtype_name: str
    fisher_device: str
    grad_checkpointing: bool


@dataclass(frozen=True)
class FisherConfig:
    """Fisher estimation hyperparameters.

    Args:
        estimator: Estimator type (`sequence` or `tokenwise`).
        max_prompt_tokens: Prompt truncation length; `None` means no truncation.
        max_response_tokens: Response truncation length; `None` means no truncation.
        max_rollout_rows: Optional deterministic row cap. `None` means all rows.
        log_every: Progress-print interval.
    """

    estimator: str
    max_prompt_tokens: int | None
    max_response_tokens: int | None
    max_rollout_rows: int | None
    log_every: int


# Notebook-05-compatible defaults.
DEFAULT_IF_MODEL_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ifrl_ifeval/global_step_50/actor/huggingface"
)
DEFAULT_MATH_MODEL_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-math/stage2/global_step_40/actor/huggingface"
)
DEFAULT_IF_VALIDATION_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/merging_analysis/merging_analysis/artifacts/jwcm_v2/validation_data/val_if_512.parquet"
)
DEFAULT_MATH_VALIDATION_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/verl_nemotron_merge/merging_analysis/merging_analysis/artifacts/jwcm_v2/validation_data/val_math_512.parquet"
)
DEFAULT_IF_ROLLOUT_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/qwen3_1.7b_fisher_pipeline_verbose/tasks/if/correct_rollout_trajectories.parquet"
)
DEFAULT_MATH_ROLLOUT_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/qwen3_1.7b_fisher_pipeline_verbose/tasks/math/correct_rollout_trajectories.parquet"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge")


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.

    Returns:
        None. RNG states are updated in-place.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert runtime objects to JSON-serializable values.

    Args:
        obj: Arbitrary runtime object.

    Returns:
        JSON-compatible representation.
    """

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_json_compatible(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_compatible(value) for value in obj]
    return obj


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save mapping payload to JSON with pretty formatting.

    Args:
        payload: Mapping payload to serialize.
        output_path: Destination file path.

    Returns:
        None. File is written to disk.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = to_json_compatible(dict(payload))
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2, ensure_ascii=False)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve dtype alias string into torch dtype object.

    Args:
        dtype_name: Dtype alias (`bf16`, `fp16`, `fp32`).

    Returns:
        Torch dtype object.
    """

    lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in lookup:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return lookup[dtype_name]


def load_tokenizer_with_mistral_regex_fix(model_name_or_path: str) -> AutoTokenizer:
    """Load tokenizer with optional Mistral regex compatibility flag.

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


def normalize_prompt_messages(prompt_obj: Any) -> List[Dict[str, str]]:
    """Normalize parquet prompt object into chat-template message dictionaries.

    Why this function exists:
    - Validation parquet rows may store `prompt` as list/NumPy array objects.
    - Chat template requires list[dict(role, content)] in canonical format.

    Args:
        prompt_obj: Prompt object loaded from parquet.

    Returns:
        List of normalized chat messages with `role` and `content`.
    """

    if isinstance(prompt_obj, np.ndarray):
        messages = prompt_obj.tolist()
    elif isinstance(prompt_obj, list):
        messages = prompt_obj
    else:
        raise TypeError(f"Unsupported prompt type: {type(prompt_obj)}")

    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"Prompt message must be dict, got {type(message)}")
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message.get("content", "")),
            }
        )
    return normalized


def build_index_to_prompt_map(validation_parquet: Path) -> Dict[int, List[Dict[str, str]]]:
    """Build source-row-position to prompt-message mapping.

    Why row-position indexing is used:
    - Rollout parquet `sample_index` comes from VERL `extra_info["index"]`.
    - In this pipeline, that value maps to row position in validation parquet.

    Args:
        validation_parquet: Source validation parquet path.

    Returns:
        Mapping `{sample_index -> normalized_prompt_messages}`.
    """

    validation_df = pd.read_parquet(validation_parquet)
    if "prompt" not in validation_df.columns:
        raise ValueError(
            f"Validation parquet must include 'prompt'. "
            f"path={validation_parquet} columns={list(validation_df.columns)}"
        )

    index_to_prompt: Dict[int, List[Dict[str, str]]] = {}
    for row_position, prompt_obj in enumerate(validation_df["prompt"].tolist()):
        index_to_prompt[int(row_position)] = normalize_prompt_messages(prompt_obj)
    return index_to_prompt


def load_fisher_model_and_tokenizer(
    model_name_or_path: str | Path,
    torch_dtype: torch.dtype,
    device: str,
    grad_checkpointing: bool,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load causal LM + tokenizer prepared for Fisher backward passes.

    Args:
        model_name_or_path: HF model id or local checkpoint path.
        torch_dtype: Weight dtype for loading.
        device: Runtime device string.
        grad_checkpointing: Whether to enable activation checkpointing.

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

    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # Activation checkpointing lowers peak memory during backward at the
        # cost of extra compute, which is useful for long rollout responses.
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        # Backward + checkpointing require `use_cache=False`.
        model.config.use_cache = False

    tokenizer = load_tokenizer_with_mistral_regex_fix(resolved_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _tokenize_text_without_special_tokens(
    tokenizer: AutoTokenizer,
    text: str,
    max_tokens: int | None,
) -> torch.Tensor:
    """Tokenize free-form text with optional truncation and no special tokens.

    Args:
        tokenizer: Hugging Face tokenizer.
        text: Input text string.
        max_tokens: Optional truncation length. `None` means no truncation.

    Returns:
        1D CPU tensor of token ids.
    """

    tokenize_kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    if max_tokens is None:
        tokenize_kwargs["truncation"] = False
    else:
        tokenize_kwargs["truncation"] = True
        tokenize_kwargs["max_length"] = int(max_tokens)

    tokenized = tokenizer(text, **tokenize_kwargs)
    return tokenized["input_ids"][0].detach().cpu()


def _prepare_log_prob_tensors(
    model: AutoModelForCausalLM,
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare per-token response log-probabilities for one trajectory.

    Args:
        model: Task model in eval mode.
        full_ids_cpu: Concatenated prompt+response ids on CPU.
        prompt_len: Prompt-token length within `full_ids_cpu`.
        device: Runtime device string.

    Returns:
        Tuple:
        - `token_log_probs`: log p for each response token (shape `[T]`).
        - `full_ids`: full ids on device (shape `[1, prompt_plus_response_len]`).
    """

    full_ids = full_ids_cpu.unsqueeze(0).to(device)
    full_attention = torch.ones_like(full_ids, device=device)
    logits = model(input_ids=full_ids, attention_mask=full_attention).logits
    full_len = int(full_ids.shape[1])
    if full_len <= prompt_len:
        raise ValueError("No response tokens in trajectory.")

    shifted_positions = torch.arange(prompt_len, full_len, device=device) - 1
    target_token_ids = full_ids[0, prompt_len:full_len]
    selected_logits = logits[0, shifted_positions, :].to(torch.float32)
    selected_log_probs = torch.log_softmax(selected_logits, dim=-1)
    token_log_probs = selected_log_probs.gather(
        dim=1,
        index=target_token_ids.unsqueeze(1),
    ).squeeze(1)

    # `logits` and derived tensors are intentionally kept alive by autograd
    # through `token_log_probs` until gradient calls finish in caller.
    del logits
    del selected_logits
    del selected_log_probs
    return token_log_probs, full_ids


def _accumulate_sequence_estimator(
    model: AutoModelForCausalLM,
    param_names: Sequence[str],
    param_list: Sequence[torch.nn.Parameter],
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    device: str,
    fisher_sum_cpu: MutableMapping[str, torch.Tensor],
) -> None:
    """Accumulate one-shot sequence estimator contribution into Fisher sum.

    Estimator:
        g = d/dtheta [sum_t log p_t], accumulate g^2.

    Args:
        model: Task model.
        param_names: Parameter names aligned with `param_list`.
        param_list: Trainable floating parameters.
        full_ids_cpu: Prompt+response ids on CPU.
        prompt_len: Prompt length.
        device: Runtime device.
        fisher_sum_cpu: CPU accumulator updated in-place.

    Returns:
        None. Fisher accumulator is updated in-place.
    """

    token_log_probs, _ = _prepare_log_prob_tensors(
        model=model,
        full_ids_cpu=full_ids_cpu,
        prompt_len=prompt_len,
        device=device,
    )
    sequence_log_prob = token_log_probs.sum()
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
        fisher_sum_cpu[name].add_(grad.detach().to(torch.float32).cpu().pow(2))

    del grads
    del token_log_probs
    del sequence_log_prob


def _accumulate_tokenwise_estimator(
    model: AutoModelForCausalLM,
    param_names: Sequence[str],
    param_list: Sequence[torch.nn.Parameter],
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    device: str,
    fisher_sum_cpu: MutableMapping[str, torch.Tensor],
) -> None:
    """Accumulate tokenwise estimator contribution into Fisher sum.

    Estimator:
        sum_t (d/dtheta log p_t)^2

    Why this is separate:
    - It removes cross terms between token gradients.
    - It is significantly more expensive than one-shot sequence estimator.

    Args:
        model: Task model.
        param_names: Parameter names aligned with `param_list`.
        param_list: Trainable floating parameters.
        full_ids_cpu: Prompt+response ids on CPU.
        prompt_len: Prompt length.
        device: Runtime device.
        fisher_sum_cpu: CPU accumulator updated in-place.

    Returns:
        None. Fisher accumulator is updated in-place.
    """

    token_log_probs, _ = _prepare_log_prob_tensors(
        model=model,
        full_ids_cpu=full_ids_cpu,
        prompt_len=prompt_len,
        device=device,
    )
    num_tokens = int(token_log_probs.numel())
    for token_index, token_log_prob in enumerate(token_log_probs):
        retain_graph = token_index < (num_tokens - 1)
        grads = torch.autograd.grad(
            token_log_prob,
            param_list,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        for index, grad in enumerate(grads):
            if grad is None:
                continue
            name = param_names[index]
            fisher_sum_cpu[name].add_(grad.detach().to(torch.float32).cpu().pow(2))
        del grads

    del token_log_probs


def _int_to_optional_limit(value: int) -> int | None:
    """Convert CLI integer with -1 sentinel into optional integer.

    Args:
        value: CLI integer value.

    Returns:
        `None` if value is negative, else the integer itself.
    """

    return None if int(value) < 0 else int(value)


def compute_fisher_from_rollouts(
    task_spec: TaskSpec,
    runtime: RuntimeConfig,
    fisher_cfg: FisherConfig,
    fisher_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Compute Fisher diagonal from rollout responses for one task.

    Args:
        task_spec: Task model/data specification.
        runtime: Runtime settings.
        fisher_cfg: Fisher hyperparameter settings.
        fisher_output_path: Destination `.pt` path.
        summary_output_path: Destination summary `.json` path.

    Returns:
        None. Saves Fisher tensor dictionary and summary JSON.
    """

    preferred_device = str(runtime.fisher_device)
    if preferred_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[Warning][{task_spec.name}] CUDA unavailable; falling back from "
            f"{preferred_device} to cpu.",
            flush=True,
        )
        worker_device = "cpu"
    else:
        worker_device = preferred_device

    fisher_dtype = resolve_torch_dtype(runtime.model_dtype_name)
    model, tokenizer = load_fisher_model_and_tokenizer(
        model_name_or_path=task_spec.model_path,
        torch_dtype=fisher_dtype,
        device=worker_device,
        grad_checkpointing=runtime.grad_checkpointing,
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    rollout_df = pd.read_parquet(task_spec.rollout_parquet)
    if rollout_df.empty:
        raise ValueError(f"Rollout parquet is empty: {task_spec.rollout_parquet}")
    required_columns = {"sample_index", "output"}
    missing_columns = required_columns - set(rollout_df.columns)
    if missing_columns:
        raise ValueError(
            f"Rollout parquet must include {sorted(required_columns)}. "
            f"missing={sorted(missing_columns)} path={task_spec.rollout_parquet}"
        )

    if fisher_cfg.max_rollout_rows is not None and len(rollout_df) > int(fisher_cfg.max_rollout_rows):
        # Deterministic head-crop keeps reproducibility simple for debugging.
        fisher_df = rollout_df.iloc[: int(fisher_cfg.max_rollout_rows)].reset_index(drop=True)
    else:
        fisher_df = rollout_df.reset_index(drop=True)

    index_to_prompt = build_index_to_prompt_map(validation_parquet=task_spec.validation_parquet)
    prompt_token_cache: Dict[int, torch.Tensor] = {}

    # Track every trainable floating parameter, including tied/shared tensors when
    # they appear in `named_parameters()`.
    param_entries: List[tuple[str, torch.nn.Parameter]] = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and torch.is_floating_point(parameter)
    ]
    param_names = [name for name, _ in param_entries]
    param_list = [parameter for _, parameter in param_entries]

    # Keep long-lived Fisher accumulator on CPU to limit persistent GPU memory use.
    fisher_sum_cpu: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        for name, parameter in param_entries
    }

    processed = 0
    skipped_invalid_index = 0
    skipped_missing_prompt = 0
    skipped_empty_response = 0
    skipped_too_short = 0

    iterator = tqdm(
        fisher_df.to_dict(orient="records"),
        total=len(fisher_df),
        desc=f"Fisher[{task_spec.name}]",
    )
    for step_index, row in enumerate(iterator, start=1):
        sample_index_raw = row.get("sample_index", None)
        try:
            sample_index = int(sample_index_raw)
        except Exception:
            skipped_invalid_index += 1
            continue

        prompt_messages = index_to_prompt.get(sample_index, None)
        if prompt_messages is None:
            skipped_missing_prompt += 1
            continue

        if sample_index not in prompt_token_cache:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_token_cache[sample_index] = _tokenize_text_without_special_tokens(
                tokenizer=tokenizer,
                text=prompt_text,
                max_tokens=fisher_cfg.max_prompt_tokens,
            )

        response_text = str(row.get("output", ""))
        response_ids_cpu = _tokenize_text_without_special_tokens(
            tokenizer=tokenizer,
            text=response_text,
            max_tokens=fisher_cfg.max_response_tokens,
        )
        if response_ids_cpu.numel() == 0:
            skipped_empty_response += 1
            continue

        prompt_ids_cpu = prompt_token_cache[sample_index]
        full_ids_cpu = torch.cat([prompt_ids_cpu, response_ids_cpu], dim=0)
        prompt_len = int(prompt_ids_cpu.numel())
        if int(full_ids_cpu.numel()) <= prompt_len:
            skipped_too_short += 1
            continue

        try:
            if fisher_cfg.estimator == "sequence":
                _accumulate_sequence_estimator(
                    model=model,
                    param_names=param_names,
                    param_list=param_list,
                    full_ids_cpu=full_ids_cpu,
                    prompt_len=prompt_len,
                    device=worker_device,
                    fisher_sum_cpu=fisher_sum_cpu,
                )
            elif fisher_cfg.estimator == "tokenwise":
                _accumulate_tokenwise_estimator(
                    model=model,
                    param_names=param_names,
                    param_list=param_list,
                    full_ids_cpu=full_ids_cpu,
                    prompt_len=prompt_len,
                    device=worker_device,
                    fisher_sum_cpu=fisher_sum_cpu,
                )
            else:
                raise ValueError(f"Unsupported estimator: {fisher_cfg.estimator}")

            processed += 1
        except torch.OutOfMemoryError as oom_error:
            raise RuntimeError(
                f"[{task_spec.name}] Fisher OOM at sample_index={sample_index}. "
                f"prompt_tokens={prompt_len} response_tokens={int(response_ids_cpu.numel())}. "
                f"Use a larger GPU or set --max-prompt-tokens/--max-response-tokens."
            ) from oom_error
        finally:
            # Ensure stale gradients are cleared between trajectories.
            model.zero_grad(set_to_none=True)

        if step_index % int(fisher_cfg.log_every) == 0:
            print(
                f"[{task_spec.name}] progress processed={processed}/{len(fisher_df)} "
                f"(step={step_index})",
                flush=True,
            )

        if processed > 0 and processed % 8 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if processed <= 0:
        raise RuntimeError(
            f"Task '{task_spec.name}' produced zero valid trajectories. "
            f"Check sample_index alignment and rollout response text."
        )

    for name in fisher_sum_cpu.keys():
        fisher_sum_cpu[name].div_(float(processed))

    fisher_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fisher_sum_cpu, fisher_output_path)

    summary = {
        "created_at": now_iso(),
        "task_name": task_spec.name,
        "task_model_path": str(task_spec.model_path),
        "validation_parquet": str(task_spec.validation_parquet),
        "rollout_parquet": str(task_spec.rollout_parquet),
        "fisher_output_path": str(fisher_output_path),
        "dtype": runtime.model_dtype_name,
        "device": worker_device,
        "estimator": fisher_cfg.estimator,
        "max_prompt_tokens": fisher_cfg.max_prompt_tokens,
        "max_response_tokens": fisher_cfg.max_response_tokens,
        "max_rollout_rows": fisher_cfg.max_rollout_rows,
        "num_rollout_rows": int(len(rollout_df)),
        "num_used_rollout_rows": int(len(fisher_df)),
        "processed_sequences": int(processed),
        "skipped_invalid_index": int(skipped_invalid_index),
        "skipped_missing_prompt": int(skipped_missing_prompt),
        "skipped_empty_response": int(skipped_empty_response),
        "skipped_too_short": int(skipped_too_short),
        "num_tracked_parameters": int(len(param_entries)),
        "grad_checkpointing": bool(runtime.grad_checkpointing),
    }
    save_json(summary, summary_output_path)

    print(f"[{task_spec.name}] Saved Fisher diagonal: {fisher_output_path}", flush=True)
    print(f"[{task_spec.name}] Saved Fisher summary: {summary_output_path}", flush=True)

    del model
    del tokenizer
    del fisher_sum_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute Fisher diagonals from rollout parquet files (notebook 05 logic) "
            "and save notebook-compatible artifacts."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["if", "math", "all"],
        default="all",
        help="Task selector. `all` computes IF and Math sequentially.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where Fisher artifacts are written.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="fisher_diagonal",
        help="Subdirectory under output root for Fisher tensors.",
    )
    parser.add_argument(
        "--metadata-subdir",
        type=str,
        default="metadata",
        help="Subdirectory under output root for run metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Fisher computation device.",
    )
    parser.add_argument(
        "--disable-grad-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing (enabled by default).",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="sequence",
        choices=["sequence", "tokenwise"],
        help="Fisher estimator: one-shot sequence or tokenwise sum-grad-squared.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=-1,
        help="Prompt truncation length; set -1 for no prompt truncation.",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=-1,
        help="Response truncation length; set -1 for no response truncation.",
    )
    parser.add_argument(
        "--max-rollout-rows",
        type=int,
        default=-1,
        help="Deterministic row cap; set -1 to use all rollout rows.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Progress logging interval.",
    )

    # Per-task path overrides.
    parser.add_argument("--if-model-path", type=Path, default=DEFAULT_IF_MODEL_PATH)
    parser.add_argument("--if-validation-path", type=Path, default=DEFAULT_IF_VALIDATION_PATH)
    parser.add_argument("--if-rollout-path", type=Path, default=DEFAULT_IF_ROLLOUT_PATH)
    parser.add_argument("--math-model-path", type=Path, default=DEFAULT_MATH_MODEL_PATH)
    parser.add_argument("--math-validation-path", type=Path, default=DEFAULT_MATH_VALIDATION_PATH)
    parser.add_argument("--math-rollout-path", type=Path, default=DEFAULT_MATH_ROLLOUT_PATH)
    return parser.parse_args()


def _build_task_specs_from_args(args: argparse.Namespace) -> List[TaskSpec]:
    """Build selected task specs from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        List of task specs selected by `--task`.
    """

    all_tasks = {
        "if": TaskSpec(
            name="if",
            model_path=args.if_model_path,
            validation_parquet=args.if_validation_path,
            rollout_parquet=args.if_rollout_path,
        ),
        "math": TaskSpec(
            name="math",
            model_path=args.math_model_path,
            validation_parquet=args.math_validation_path,
            rollout_parquet=args.math_rollout_path,
        ),
    }

    if args.task == "all":
        return [all_tasks["if"], all_tasks["math"]]
    return [all_tasks[str(args.task)]]


def _validate_required_paths(task_specs: Iterable[TaskSpec]) -> None:
    """Validate that required model/data paths exist.

    Args:
        task_specs: Task specs to validate.

    Returns:
        None. Raises `FileNotFoundError` on missing path.
    """

    for task_spec in task_specs:
        for required_path in [
            task_spec.model_path,
            task_spec.validation_parquet,
            task_spec.rollout_parquet,
        ]:
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Required path does not exist for task '{task_spec.name}': {required_path}"
                )


def main() -> None:
    """Script entrypoint."""

    args = parse_args()
    task_specs = _build_task_specs_from_args(args)
    _validate_required_paths(task_specs)

    runtime = RuntimeConfig(
        output_root=Path(args.output_root),
        output_subdir=str(args.output_subdir),
        metadata_subdir=str(args.metadata_subdir),
        seed=int(args.seed),
        model_dtype_name=str(args.dtype),
        fisher_device=str(args.device),
        grad_checkpointing=not bool(args.disable_grad_checkpointing),
    )
    fisher_cfg = FisherConfig(
        estimator=str(args.estimator),
        max_prompt_tokens=_int_to_optional_limit(int(args.max_prompt_tokens)),
        max_response_tokens=_int_to_optional_limit(int(args.max_response_tokens)),
        max_rollout_rows=_int_to_optional_limit(int(args.max_rollout_rows)),
        log_every=int(args.log_every),
    )

    set_seed(runtime.seed)
    runtime.output_root.mkdir(parents=True, exist_ok=True)
    fisher_dir = runtime.output_root / runtime.output_subdir
    metadata_dir = runtime.output_root / runtime.metadata_subdir
    fisher_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    fisher_output_paths: Dict[str, Path] = {
        task_spec.name: fisher_dir / f"fisher_diag_{task_spec.name}.pt" for task_spec in task_specs
    }
    fisher_summary_paths: Dict[str, Path] = {
        task_spec.name: fisher_dir / f"fisher_diag_{task_spec.name}_summary.json" for task_spec in task_specs
    }

    for task_spec in task_specs:
        rollout_rows = int(pd.read_parquet(task_spec.rollout_parquet).shape[0])
        source_rows = int(pd.read_parquet(task_spec.validation_parquet).shape[0])
        print(
            f"Task={task_spec.name} | source_rows={source_rows} | rollout_rows={rollout_rows}",
            flush=True,
        )

    manifest_payload = {
        "created_at": now_iso(),
        "runtime": asdict(runtime),
        "fisher_config": asdict(fisher_cfg),
        "task_models": {task_spec.name: str(task_spec.model_path) for task_spec in task_specs},
        "validation_parquets": {task_spec.name: str(task_spec.validation_parquet) for task_spec in task_specs},
        "rollout_parquets": {task_spec.name: str(task_spec.rollout_parquet) for task_spec in task_specs},
        "fisher_output_paths": {name: str(path) for name, path in fisher_output_paths.items()},
        "fisher_summary_paths": {name: str(path) for name, path in fisher_summary_paths.items()},
    }
    manifest_path = metadata_dir / "fisher_validation_manifest_from_rollouts.json"
    save_json(manifest_payload, manifest_path)
    print(f"Saved Fisher manifest: {manifest_path}", flush=True)

    for task_spec in task_specs:
        compute_fisher_from_rollouts(
            task_spec=task_spec,
            runtime=runtime,
            fisher_cfg=fisher_cfg,
            fisher_output_path=fisher_output_paths[task_spec.name],
            summary_output_path=fisher_summary_paths[task_spec.name],
        )

    print("Fisher computation finished for all selected tasks.", flush=True)


if __name__ == "__main__":
    main()
