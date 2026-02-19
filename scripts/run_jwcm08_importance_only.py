#!/usr/bin/env python3
"""Compute JWCM v2 importance scores from rollout-reuse token deltas.

This script productionizes the importance-computation stage from:
`merging_analysis/08_jwcm_v2_soft_attribution_merge_fisher_rollout_reuse.ipynb`.

Scope of this script:
1. Reuse Fisher validation parquet and Fisher correct-rollout parquet.
2. Build token-level `delta_logp = logp_rl - logp_base` records.
3. Select critical tokens by configurable criterion.
4. Compute parameter importance tensors (`S_j`) only.
5. Save mode-specific importance files for downstream analysis/merging.

Out of scope:
- No merge checkpoint generation.
- No Fisher computation.

Critical-token selection modes:
- `positive`: positive tail (`delta_logp` high, top p%)
- `negative`: negative tail (`delta_logp` low, bottom p%)
- `absolute`: magnitude tail (`|delta_logp|` high, top p%)
- `all`: run all three modes in one execution
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TaskSpec:
    """Task-specific model/data paths for JWCM importance computation.

    Args:
        name: Task key (`if` or `math`).
        model_path: Local checkpoint path for the RL model.
        fisher_validation_path: Validation parquet reused from Fisher workflow.
        fisher_correct_rollout_path: Correct-rollout parquet reused from Fisher pipeline.
    """

    name: str
    model_path: Path
    fisher_validation_path: Path
    fisher_correct_rollout_path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings for model loading and artifact output.

    Args:
        base_model_id: Base model id/path used as attribution anchor.
        output_root: Output directory for all generated artifacts.
        seed: Global random seed.
        model_dtype: Model loading dtype.
        device: Runtime device string.
    """

    base_model_id: str
    output_root: Path
    seed: int
    model_dtype: torch.dtype
    device: str


@dataclass(frozen=True)
class CriticalTokenConfig:
    """Configuration for token-delta collection and rollout filtering.

    Args:
        max_prompt_tokens: Prompt truncation budget for rollout rows.
            `None` means no prompt truncation.
        max_new_tokens: Response truncation budget for rollout rows.
            `None` means no response truncation.
        use_only_correct_rows: Whether to keep only `is_correct=True` when available.
        max_rollouts_per_sample: Optional per-sample rollout cap.
        max_total_rollouts: Optional global rollout-row cap per task.
    """

    max_prompt_tokens: int | None
    max_new_tokens: int | None
    use_only_correct_rows: bool
    max_rollouts_per_sample: int | None
    max_total_rollouts: int | None


@dataclass(frozen=True)
class TokenSelectionConfig:
    """Configuration for critical-token selection criterion.

    Args:
        mode: One of `positive`, `negative`, `absolute`.
        top_p: Selection ratio in `(0, 1]`; default is 0.20 (top/bottom 20%).
        epsilon: Minimum absolute threshold stabilizer.
        max_critical_tokens: Optional cap on selected critical tokens.
    """

    mode: str
    top_p: float
    epsilon: float
    max_critical_tokens: int | None


@dataclass(frozen=True)
class AttributionConfig:
    """Configuration for JWCM importance backprop computation.

    Args:
        mode: `exact_token_abs` or `sequence_sum_approx`.
        max_backprop_tokens: Optional token budget for attribution runtime control.
        normalize_by_token_count: Whether to divide final importance by processed tokens.
    """

    mode: str
    max_backprop_tokens: int | None
    normalize_by_token_count: bool


DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_IF_MODEL_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-ifrl_ifeval/global_step_50/actor/huggingface"
)
DEFAULT_MATH_MODEL_PATH = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-math/stage2/global_step_40/actor/huggingface"
)
DEFAULT_FISHER_VALIDATION_ROOT = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-fisher-merge/validation"
)
DEFAULT_FISHER_TASK_ROOT = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/qwen3_1.7b_fisher_pipeline_verbose/tasks"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/ddn/vuvlm/geeho/nemotron_cascade_output/Qwen3-1.7B-jwcm-v2-importance-only"
)


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible processing.

    Args:
        seed: Integer random seed.

    Returns:
        None. Global RNG states are updated in-place.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        payload: Mapping payload.
        output_path: Destination file path.

    Returns:
        None. JSON file is written to disk.
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
    """Load tokenizer with optional mistral-regex compatibility argument.

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
    attn_implementation: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load causal LM and tokenizer on target device.

    Args:
        model_name_or_path: Hugging Face model id or local checkpoint path.
        torch_dtype: Weight dtype for loading.
        device: Runtime device string.
        attn_implementation: Optional attention backend override.
            Supported values: ``"flash_attention_2"``, ``"sdpa"``, ``"eager"``.
            When ``None``, the model's default attention implementation is used.
            ``"flash_attention_2"`` requires the ``flash_attn`` package and can
            give 2-4x speedup on both forward and backward passes.

    Returns:
        Tuple `(model, tokenizer)`.
    """

    resolved_path = str(model_name_or_path)
    # Build model kwargs, conditionally adding attn_implementation to stay
    # backwards-compatible with older transformers versions.
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = str(attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(resolved_path, **model_kwargs)
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer_with_mistral_regex_fix(resolved_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _int_to_optional_limit(value: int) -> int | None:
    """Convert `-1` sentinel into `None` for optional integer limits.

    Args:
        value: CLI integer value.

    Returns:
        `None` if value < 0, else the integer itself.
    """

    return None if int(value) < 0 else int(value)


def _resolve_selection_modes(mode: str) -> List[str]:
    """Resolve requested selection mode into explicit executable mode list.

    Args:
        mode: One of `positive`, `negative`, `absolute`, `all`.

    Returns:
        List of concrete modes to execute.
    """

    if mode == "all":
        return ["positive", "negative", "absolute"]
    return [mode]


def _format_percent_label(top_p: float) -> str:
    """Format `top_p` into compact percent label for artifact filenames.

    Args:
        top_p: Ratio in `(0, 1]`.

    Returns:
        Percent string such as `20` or `12_5`.
    """

    percent = float(top_p) * 100.0
    rounded = round(percent)
    if abs(percent - rounded) < 1e-8:
        return str(int(rounded))
    return str(percent).replace(".", "_")


def build_mode_suffix(mode: str, top_p: float) -> str:
    """Build deterministic artifact suffix for one selection mode.

    Args:
        mode: Selection mode (`positive`, `negative`, `absolute`).
        top_p: Selection ratio.

    Returns:
        Filename-safe suffix string.
    """

    percent_label = _format_percent_label(top_p=top_p)
    if mode == "positive":
        return f"positive_top{percent_label}"
    if mode == "negative":
        return f"negative_bottom{percent_label}"
    if mode == "absolute":
        return f"absolute_top{percent_label}"
    raise ValueError(f"Unsupported mode for suffix: {mode}")


@torch.no_grad()
def _gather_target_log_probs_from_logits(
    logits: torch.Tensor,
    shifted_positions: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather target-token log-probabilities without full `log_softmax` materialization.

    This helper preserves the exact token log-probability values while reducing
    peak memory by avoiding an explicit `[T, vocab]` `log_softmax` tensor.

    Args:
        logits: Model logits with shape `[1, seq_len, vocab_size]`.
        shifted_positions: Positions of predictor logits (`token_position - 1`).
        target_token_ids: Target token ids for each selected position.

    Returns:
        1D tensor of target log-probabilities for selected positions.
    """

    selected_logits = logits[0, shifted_positions, :].to(torch.float32)
    target_logits = selected_logits.gather(dim=1, index=target_token_ids.unsqueeze(1)).squeeze(1)
    normalization = torch.logsumexp(selected_logits, dim=-1)
    return target_logits - normalization


@torch.no_grad()
def _compute_token_delta_rows_for_sequence(
    task_name: str,
    sample_id: int,
    sequence_id: int,
    dataset_index: int,
    full_ids: torch.Tensor,
    prompt_len: int,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM,
    rl_model: AutoModelForCausalLM,
    device: str,
) -> List[Dict[str, Any]]:
    """Compute token-level delta-logp rows for one prompt+response sequence.

    Args:
        task_name: Task key.
        sample_id: Validation sample id.
        sequence_id: Unique sequence id (rollout-level).
        dataset_index: Source dataset row index.
        full_ids: Full token ids (`prompt || response`) as 1D tensor on device.
        prompt_len: Number of prompt tokens.
        tokenizer: Tokenizer used for decode diagnostics.
        base_model: Base reference model.
        rl_model: RL target model.
        device: Runtime device string.

    Returns:
        List of per-token dictionaries with `delta_logp` and metadata.
    """

    rows: List[Dict[str, Any]] = []

    full_batch = full_ids.unsqueeze(0)
    full_attention = torch.ones_like(full_batch, device=device)
    full_len = int(full_ids.shape[0])
    positions = torch.arange(prompt_len, full_len, device=device)
    shifted_positions = positions - 1
    target_token_ids = full_ids[positions]

    # Compute RL/Base token log-probs sequentially to avoid holding both large
    # logits tensors in memory at the same time.
    rl_outputs = rl_model(input_ids=full_batch, attention_mask=full_attention, use_cache=False)
    rl_token_logp = _gather_target_log_probs_from_logits(
        logits=rl_outputs.logits,
        shifted_positions=shifted_positions,
        target_token_ids=target_token_ids,
    )
    del rl_outputs

    base_outputs = base_model(input_ids=full_batch, attention_mask=full_attention, use_cache=False)
    base_token_logp = _gather_target_log_probs_from_logits(
        logits=base_outputs.logits,
        shifted_positions=shifted_positions,
        target_token_ids=target_token_ids,
    )
    del base_outputs

    delta_logp = rl_token_logp - base_token_logp

    for idx in range(int(positions.numel())):
        token_id = int(target_token_ids[idx].item())
        rows.append(
            {
                "task": task_name,
                "sample_id": int(sample_id),
                "sequence_id": int(sequence_id),
                "dataset_index": int(dataset_index),
                "token_position": int(positions[idx].item()),
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "logp_rl": float(rl_token_logp[idx].item()),
                "logp_base": float(base_token_logp[idx].item()),
                "delta_logp": float(delta_logp[idx].item()),
            }
        )

    return rows


@torch.no_grad()
def collect_token_delta_records_from_fisher_rollouts(
    task_name: str,
    correct_rollout_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    base_model: AutoModelForCausalLM,
    rl_model: AutoModelForCausalLM,
    cfg: CriticalTokenConfig,
    device: str,
    allowed_sample_ids: set[int] | None,
    sample_id_to_dataset_index: Mapping[int, int] | None,
    sampling_seed: int,
) -> tuple[pd.DataFrame, dict[int, torch.Tensor], dict[str, Any]]:
    """Collect token delta rows from Fisher correct-rollout parquet.

    Args:
        task_name: Task identifier.
        correct_rollout_df: Rollout dataframe with at least `sample_index`, `input`, `output`.
        tokenizer: Shared tokenizer for base and RL models.
        base_model: Base model.
        rl_model: RL model.
        cfg: Token collection and rollout filtering config.
        device: Runtime device.
        allowed_sample_ids: Optional whitelist of validation sample ids.
        sample_id_to_dataset_index: Optional map from sample id to dataset index.
        sampling_seed: Seed for deterministic rollout-row sampling.

    Returns:
        Tuple:
        - token-level dataframe
        - sequence cache (`sequence_id -> full_ids_cpu`)
        - summary dictionary
    """

    required_cols = {"sample_index", "input", "output"}
    missing_cols = sorted(required_cols - set(correct_rollout_df.columns))
    if missing_cols:
        raise ValueError(
            f"Correct rollout parquet for task '{task_name}' is missing required columns: {missing_cols}"
        )

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

    records: List[Dict[str, Any]] = []
    sequence_cache: dict[int, torch.Tensor] = {}
    skipped_empty_output = 0
    skipped_too_short = 0
    used_rollout_rows = 0
    sequence_id_counter = 0

    iterator = tqdm(
        working_df.itertuples(index=False),
        total=len(working_df),
        desc=f"Collect Δlogp (fisher_rollout, {task_name})",
    )
    for row in iterator:
        sample_id = int(getattr(row, "sample_index"))
        prompt_text = str(getattr(row, "input"))
        output_text = str(getattr(row, "output"))

        if output_text.strip() == "":
            skipped_empty_output += 1
            continue

        # Support optional "no truncation" mode for prompt tokens.
        prompt_tokenize_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
        }
        if cfg.max_prompt_tokens is None:
            prompt_tokenize_kwargs["truncation"] = False
        else:
            prompt_tokenize_kwargs["truncation"] = True
            prompt_tokenize_kwargs["max_length"] = int(cfg.max_prompt_tokens)
        encoded_prompt = tokenizer(prompt_text, **prompt_tokenize_kwargs)
        prompt_ids = encoded_prompt["input_ids"][0]

        # Support optional "no truncation" mode for response tokens.
        response_tokenize_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
            "add_special_tokens": False,
        }
        if cfg.max_new_tokens is None:
            response_tokenize_kwargs["truncation"] = False
        else:
            response_tokenize_kwargs["truncation"] = True
            response_tokenize_kwargs["max_length"] = int(cfg.max_new_tokens)
        response_ids = tokenizer(output_text, **response_tokenize_kwargs)["input_ids"][0]

        if int(response_ids.numel()) == 0:
            skipped_too_short += 1
            continue

        full_ids = torch.cat([prompt_ids, response_ids], dim=0).to(device)
        prompt_len = int(prompt_ids.shape[0])
        if int(full_ids.shape[0]) <= prompt_len:
            skipped_too_short += 1
            continue

        sequence_id = int(sequence_id_counter)
        sequence_id_counter += 1
        sequence_cache[sequence_id] = full_ids.detach().cpu()

        dataset_index = (
            int(sample_id_to_dataset_index[sample_id])
            if sample_id_to_dataset_index is not None and sample_id in sample_id_to_dataset_index
            else int(sample_id)
        )

        records.extend(
            _compute_token_delta_rows_for_sequence(
                task_name=task_name,
                sample_id=sample_id,
                sequence_id=sequence_id,
                dataset_index=dataset_index,
                full_ids=full_ids,
                prompt_len=prompt_len,
                tokenizer=tokenizer,
                base_model=base_model,
                rl_model=rl_model,
                device=device,
            )
        )
        used_rollout_rows += 1

    records_df = pd.DataFrame(records)
    summary = {
        "task": task_name,
        "source": "fisher_correct_rollout",
        "input_rollout_rows": int(len(correct_rollout_df)),
        "rows_after_filters": int(len(working_df)),
        "used_rollout_rows": int(used_rollout_rows),
        "num_sequences": int(len(sequence_cache)),
        "num_token_records": int(len(records_df)),
        "skipped_empty_output": int(skipped_empty_output),
        "skipped_too_short": int(skipped_too_short),
        "use_only_correct_rows": bool(cfg.use_only_correct_rows),
        "max_rollouts_per_sample": cfg.max_rollouts_per_sample,
        "max_total_rollouts": cfg.max_total_rollouts,
        "sampling_seed": int(sampling_seed),
    }
    return records_df, sequence_cache, summary


def select_critical_tokens(
    token_records_df: pd.DataFrame,
    selection_cfg: TokenSelectionConfig,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Select critical tokens using a configurable tail-selection criterion.

    Args:
        token_records_df: Token-level dataframe containing `delta_logp`.
        selection_cfg: Selection mode and threshold settings.

    Returns:
        Tuple:
        - selected critical-token dataframe
        - metadata with thresholds and counts
    """

    if token_records_df.empty:
        metadata = {
            "mode": selection_cfg.mode,
            "top_p": float(selection_cfg.top_p),
            "epsilon": float(selection_cfg.epsilon),
            "quantile_threshold": float("nan"),
            "final_threshold": float("nan"),
            "num_selected": 0,
            "max_critical_tokens": selection_cfg.max_critical_tokens,
        }
        return token_records_df.copy(), metadata

    deltas = token_records_df["delta_logp"].to_numpy(dtype=np.float64)

    if selection_cfg.mode == "positive":
        quantile_threshold = float(np.quantile(deltas, 1.0 - selection_cfg.top_p))
        final_threshold = float(max(selection_cfg.epsilon, quantile_threshold))
        mask = deltas >= final_threshold
        sort_columns = ["delta_logp"]
        sort_ascending = [False]
        threshold_rule = "delta_logp >= max(epsilon, q_(1-top_p))"
    elif selection_cfg.mode == "negative":
        quantile_threshold = float(np.quantile(deltas, selection_cfg.top_p))
        # Keep selection on negative side while still respecting configured minimum scale.
        final_threshold = float(min(-selection_cfg.epsilon, quantile_threshold))
        mask = deltas <= final_threshold
        sort_columns = ["delta_logp"]
        sort_ascending = [True]
        threshold_rule = "delta_logp <= min(-epsilon, q_(top_p))"
    elif selection_cfg.mode == "absolute":
        abs_deltas = np.abs(deltas)
        quantile_threshold = float(np.quantile(abs_deltas, 1.0 - selection_cfg.top_p))
        final_threshold = float(max(selection_cfg.epsilon, quantile_threshold))
        mask = abs_deltas >= final_threshold
        sort_columns = ["abs_delta_logp", "delta_logp"]
        sort_ascending = [False, False]
        threshold_rule = "|delta_logp| >= max(epsilon, q_(1-top_p) on |delta|)"
    else:
        raise ValueError(f"Unsupported selection mode: {selection_cfg.mode}")

    critical_df = token_records_df[mask].copy()
    if selection_cfg.mode == "absolute":
        critical_df["abs_delta_logp"] = critical_df["delta_logp"].abs()

    critical_df.sort_values(sort_columns, ascending=sort_ascending, inplace=True)
    critical_df.reset_index(drop=True, inplace=True)

    if selection_cfg.max_critical_tokens is not None and len(critical_df) > int(selection_cfg.max_critical_tokens):
        critical_df = critical_df.head(int(selection_cfg.max_critical_tokens)).copy()
        critical_df.reset_index(drop=True, inplace=True)

    metadata = {
        "mode": selection_cfg.mode,
        "top_p": float(selection_cfg.top_p),
        "epsilon": float(selection_cfg.epsilon),
        "threshold_rule": threshold_rule,
        "quantile_threshold": float(quantile_threshold),
        "final_threshold": float(final_threshold),
        "num_selected": int(len(critical_df)),
        "max_critical_tokens": selection_cfg.max_critical_tokens,
    }
    return critical_df, metadata


def build_critical_sequence_payload(
    critical_df: pd.DataFrame,
    sequence_cache: Mapping[int, torch.Tensor],
) -> List[Dict[str, Any]]:
    """Group critical token positions by sequence for attribution backprop.

    Args:
        critical_df: Critical token dataframe with `sequence_id` and `token_position`.
        sequence_cache: Mapping from sequence id to full token ids.

    Returns:
        Sequence-level payload list for importance backprop.
    """

    payload: List[Dict[str, Any]] = []
    if critical_df.empty:
        return payload

    grouping_column = "sequence_id" if "sequence_id" in critical_df.columns else "sample_id"
    for sequence_id, group_df in critical_df.groupby(grouping_column):
        if int(sequence_id) not in sequence_cache:
            continue
        positions = sorted(int(value) for value in group_df["token_position"].tolist())
        payload.append(
            {
                "sequence_id": int(sequence_id),
                "sample_id": int(group_df["sample_id"].iloc[0]),
                "full_ids": sequence_cache[int(sequence_id)],
                "critical_positions": positions,
            }
        )
    return payload


def validate_parameter_compatibility(
    base_model: AutoModelForCausalLM,
    task_model: AutoModelForCausalLM,
) -> None:
    """Validate named parameter key/shape compatibility between base and task.

    Args:
        base_model: Base model.
        task_model: Task RL model.

    Returns:
        None. Raises `ValueError` on mismatch.
    """

    base_params = dict(base_model.named_parameters())
    task_params = dict(task_model.named_parameters())

    if set(base_params.keys()) != set(task_params.keys()):
        missing_in_task = sorted(set(base_params.keys()) - set(task_params.keys()))
        missing_in_base = sorted(set(task_params.keys()) - set(base_params.keys()))
        raise ValueError(
            "Named parameter keys mismatch. "
            f"missing_in_task={missing_in_task[:5]} missing_in_base={missing_in_base[:5]}"
        )

    for name in base_params.keys():
        if tuple(base_params[name].shape) != tuple(task_params[name].shape):
            raise ValueError(
                f"Shape mismatch for parameter '{name}': "
                f"base={tuple(base_params[name].shape)} task={tuple(task_params[name].shape)}"
            )


def build_abs_task_vector_cpu(
    base_model: AutoModelForCausalLM,
    task_model: AutoModelForCausalLM,
) -> Dict[str, torch.Tensor]:
    """Build absolute task vector `|Δθ|` on CPU float32.

    Args:
        base_model: Base model.
        task_model: Task RL model.

    Returns:
        Mapping from parameter name to `|theta_task - theta_base|`.
    """

    validate_parameter_compatibility(base_model=base_model, task_model=task_model)

    abs_delta: Dict[str, torch.Tensor] = {}
    base_params = dict(base_model.named_parameters())
    task_params = dict(task_model.named_parameters())

    for name in tqdm(base_params.keys(), desc="Build |Δθ| (CPU)"):
        base_tensor = base_params[name].detach().to(torch.float32).cpu()
        task_tensor = task_params[name].detach().to(torch.float32).cpu()
        abs_delta[name] = torch.abs(task_tensor - base_tensor)
    return abs_delta


def initialize_importance_buffers(abs_delta: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Allocate zero importance tensors aligned with `abs_delta`.

    Args:
        abs_delta: Parameter-wise absolute task vector.

    Returns:
        Zero-initialized importance dictionary.
    """

    return {name: torch.zeros_like(tensor, dtype=torch.float32) for name, tensor in abs_delta.items()}


def compute_importance_scores(
    task_model: AutoModelForCausalLM,
    critical_payload: Sequence[Mapping[str, Any]],
    abs_delta: Mapping[str, torch.Tensor],
    cfg: AttributionConfig,
    device: str,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Compute JWCM importance scores from critical tokens.

    Formula:
        S_j = sum_{(x,y_t) in C} | d log p(y_t|x) / d theta_j | * |Delta theta_j|

    Args:
        task_model: Task RL model used for gradient evaluation.
        critical_payload: Sequence-grouped critical token payload.
        abs_delta: Absolute task vector dictionary on CPU.
        cfg: Attribution config.
        device: Runtime device string.

    Returns:
        Tuple:
        - importance dictionary on CPU (`float32`)
        - attribution metadata
    """

    # Keep accumulation on device during backprop to avoid repeated CPU syncs.
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

    processed_sequences = 0
    processed_tokens = 0
    hit_budget = False

    iterator = tqdm(critical_payload, desc=f"Attribution ({cfg.mode})")
    for entry in iterator:
        if hit_budget:
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
                retain = token_idx < (len(critical_positions) - 1)

                grads = torch.autograd.grad(
                    scalar_log_prob,
                    param_list,
                    retain_graph=retain,
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

                processed_tokens += 1
                if cfg.max_backprop_tokens is not None and processed_tokens >= cfg.max_backprop_tokens:
                    hit_budget = True
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

            processed_tokens += len(critical_positions)
        else:
            raise ValueError(f"Unknown attribution mode: {cfg.mode}")

        processed_sequences += 1
        del outputs
        del logits

        if processed_sequences % 64 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if cfg.max_backprop_tokens is not None and processed_tokens >= cfg.max_backprop_tokens:
            hit_budget = True

    importance = {name: tensor.cpu() for name, tensor in importance_gpu.items()}
    if cfg.normalize_by_token_count and processed_tokens > 0:
        scale = float(processed_tokens)
        for name in importance.keys():
            importance[name].div_(scale)

    del abs_delta_gpu
    del importance_gpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "mode": cfg.mode,
        "processed_sequences": int(processed_sequences),
        "processed_tokens": int(processed_tokens),
        "max_backprop_tokens": cfg.max_backprop_tokens,
        "normalize_by_token_count": bool(cfg.normalize_by_token_count),
    }
    return importance, metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for JWCM importance-only pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "JWCM v2 importance-only pipeline from Fisher rollout reuse. "
            "Supports positive/negative/absolute critical-token selection."
        ),
    )
    parser.add_argument("--task", type=str, choices=["if", "math", "all"], default="all")
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["positive", "negative", "absolute", "all"],
        default="positive",
        help=(
            "Critical-token selection mode. Use `all` to compute and save "
            "positive/negative/absolute importance tensors in one run."
        ),
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
        help="Minimum absolute threshold stabilizer for selection.",
    )
    parser.add_argument(
        "--max-critical-tokens",
        type=int,
        default=-1,
        help="Optional cap on selected critical tokens per mode/task. Use -1 for no cap.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=-1,
        help="Prompt truncation budget during rollout-token reconstruction. Use -1 for no prompt truncation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max-response-tokens",
        dest="max_new_tokens",
        type=int,
        default=-1,
        help=(
            "Response truncation budget during rollout-token reconstruction. "
            "Use -1 for no response truncation."
        ),
    )
    parser.add_argument(
        "--include-incorrect-rows",
        action="store_true",
        help=(
            "Include incorrect rollout rows as well. "
            "By default, the script keeps only `is_correct=True` when column exists."
        ),
    )
    parser.add_argument(
        "--max-rollouts-per-sample",
        type=int,
        default=-1,
        help="Optional cap on rollout rows per sample id. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--max-total-rollouts",
        "--max-rollout-rows",
        dest="max_total_rollouts",
        type=int,
        default=-1,
        help=(
            "Optional global rollout-row cap per task. "
            "Use -1 for unlimited."
        ),
    )
    parser.add_argument(
        "--validation-samples-per-task",
        type=int,
        default=-1,
        help="Optional validation-row cap per task. Use -1 to use all rows.",
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
        help="Optional backprop token budget per mode/task. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--no-normalize-by-token-count",
        action="store_true",
        help="Disable final division by processed token count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
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
    return parser.parse_args()


def _validate_top_p(top_p: float) -> None:
    """Validate `top_p` range.

    Args:
        top_p: Selection ratio.

    Returns:
        None. Raises `ValueError` for invalid range.
    """

    if not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"`top_p` must be in (0, 1], got {top_p}")


def _build_task_specs(args: argparse.Namespace) -> List[TaskSpec]:
    """Build selected task specs from CLI arguments.

    Args:
        args: Parsed CLI args.

    Returns:
        List of selected task specs.
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
    """Validate required filesystem paths for all selected tasks.

    Args:
        task_specs: Selected task specs.

    Returns:
        None. Raises `FileNotFoundError` on missing path.
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


def main() -> None:
    """Run JWCM importance-only pipeline."""

    args = parse_args()
    _validate_top_p(top_p=float(args.top_p))

    task_specs = _build_task_specs(args=args)
    _validate_required_paths(task_specs=task_specs)

    preferred_device = str(args.device)
    if preferred_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[Warning] CUDA is unavailable; falling back from {preferred_device} to cpu.",
            flush=True,
        )
        runtime_device = "cpu"
    else:
        runtime_device = preferred_device

    runtime = RuntimeConfig(
        base_model_id=str(args.base_model_id),
        output_root=Path(args.output_root),
        seed=int(args.seed),
        model_dtype=resolve_torch_dtype(str(args.dtype)),
        device=runtime_device,
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
        mode="positive",  # Placeholder per-mode override in task loop.
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

    set_seed(runtime.seed)
    runtime.output_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "validation": runtime.output_root / "validation",
        "critical": runtime.output_root / "critical_tokens",
        "importance": runtime.output_root / "importance",
        "metadata": runtime.output_root / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    run_summary: Dict[str, Any] = {
        "created_at": now_iso(),
        "runtime": asdict(runtime),
        "critical_config": asdict(critical_cfg),
        "selection_config_template": asdict(selection_template),
        "selection_modes": list(selection_modes),
        "attribution_config": asdict(attribution_cfg),
        "validation_samples_per_task": validation_samples_per_task,
        "task_specs": [asdict(task_spec) for task_spec in task_specs],
        "task_summaries": {},
    }

    print(f"Device: {runtime.device}", flush=True)
    print(f"DType: {runtime.model_dtype}", flush=True)
    print(f"Output root: {runtime.output_root}", flush=True)
    print(f"Selection modes: {selection_modes}", flush=True)
    print(f"Selection top_p: {selection_template.top_p}", flush=True)

    base_model, _ = load_causal_lm(
        model_name_or_path=runtime.base_model_id,
        torch_dtype=runtime.model_dtype,
        device=runtime.device,
    )

    for task_spec in task_specs:
        print(f"\n=== Task: {task_spec.name} ===", flush=True)

        validation_df = pd.read_parquet(task_spec.fisher_validation_path)
        required_columns = {"sample_id", "dataset_index", "prompt_text"}
        missing_columns = sorted(required_columns - set(validation_df.columns))
        if missing_columns:
            raise ValueError(
                f"Validation parquet missing required columns for task '{task_spec.name}': {missing_columns}"
            )

        validation_copy_path = dirs["validation"] / f"{task_spec.name}_validation.parquet"
        validation_df.to_parquet(validation_copy_path, index=False)

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

        task_model, task_tokenizer = load_causal_lm(
            model_name_or_path=task_spec.model_path,
            torch_dtype=runtime.model_dtype,
            device=runtime.device,
        )

        correct_rollout_df = pd.read_parquet(task_spec.fisher_correct_rollout_path)
        token_records_df, sequence_cache, token_summary = collect_token_delta_records_from_fisher_rollouts(
            task_name=task_spec.name,
            correct_rollout_df=correct_rollout_df,
            tokenizer=task_tokenizer,
            base_model=base_model,
            rl_model=task_model,
            cfg=critical_cfg,
            device=runtime.device,
            allowed_sample_ids=selected_sample_ids,
            sample_id_to_dataset_index=sample_id_to_dataset_index,
            sampling_seed=runtime.seed,
        )
        if token_records_df.empty:
            raise RuntimeError(
                f"Token records are empty for task '{task_spec.name}'. "
                "Check rollout filtering and tokenization settings."
            )

        token_records_df["sample_id"] = token_records_df["sample_id"].astype(np.int64)
        token_records_df["sequence_id"] = token_records_df["sequence_id"].astype(np.int64)
        token_records_df["token_position"] = token_records_df["token_position"].astype(np.int64)
        token_records_df["token_id"] = token_records_df["token_id"].astype(np.int64)

        token_csv_path = dirs["critical"] / f"{task_spec.name}_all_token_deltas.csv"
        token_records_df.to_csv(token_csv_path, index=False)

        abs_delta = build_abs_task_vector_cpu(base_model=base_model, task_model=task_model)

        task_mode_summary: Dict[str, Any] = {}
        for mode in selection_modes:
            selection_cfg = TokenSelectionConfig(
                mode=mode,
                top_p=selection_template.top_p,
                epsilon=selection_template.epsilon,
                max_critical_tokens=selection_template.max_critical_tokens,
            )
            mode_suffix = build_mode_suffix(mode=mode, top_p=selection_cfg.top_p)

            critical_df, critical_meta = select_critical_tokens(
                token_records_df=token_records_df,
                selection_cfg=selection_cfg,
            )

            critical_csv_path = dirs["critical"] / f"{task_spec.name}_critical_tokens_{mode_suffix}.csv"
            critical_df.to_csv(critical_csv_path, index=False)

            critical_payload = build_critical_sequence_payload(
                critical_df=critical_df,
                sequence_cache=sequence_cache,
            )

            if critical_payload:
                importance, importance_meta = compute_importance_scores(
                    task_model=task_model,
                    critical_payload=critical_payload,
                    abs_delta=abs_delta,
                    cfg=attribution_cfg,
                    device=runtime.device,
                )
            else:
                # Save explicit zero tensors when nothing is selected to keep
                # downstream tooling deterministic and schema-compatible.
                importance = initialize_importance_buffers(abs_delta=abs_delta)
                importance_meta = {
                    "mode": attribution_cfg.mode,
                    "processed_sequences": 0,
                    "processed_tokens": 0,
                    "max_backprop_tokens": attribution_cfg.max_backprop_tokens,
                    "normalize_by_token_count": bool(attribution_cfg.normalize_by_token_count),
                }

            importance_path = dirs["importance"] / f"importance_{task_spec.name}_{mode_suffix}.pt"
            torch.save(importance, importance_path)
            print(
                f"[{task_spec.name}] saved importance ({mode}): {importance_path}",
                flush=True,
            )

            task_mode_summary[mode] = {
                "mode_suffix": mode_suffix,
                "selection": critical_meta,
                "critical_csv_path": str(critical_csv_path),
                "importance_path": str(importance_path),
                "importance_meta": importance_meta,
                "num_critical_tokens": int(len(critical_df)),
                "num_critical_sequences": int(len(critical_payload)),
            }

            del importance
            del critical_df
            del critical_payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        run_summary["task_summaries"][task_spec.name] = {
            "task_model_path": str(task_spec.model_path),
            "validation_source_path": str(task_spec.fisher_validation_path),
            "rollout_source_path": str(task_spec.fisher_correct_rollout_path),
            "validation_copy_path": str(validation_copy_path),
            "token_delta_csv_path": str(token_csv_path),
            "token_summary": token_summary,
            "num_selected_validation_samples": int(len(selected_sample_ids)),
            "num_unique_token_delta_samples": int(token_records_df["sample_id"].nunique()),
            "num_unique_token_delta_sequences": int(token_records_df["sequence_id"].nunique()),
            "modes": task_mode_summary,
        }

        del abs_delta
        del token_records_df
        del sequence_cache
        del task_model
        del task_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary_path = dirs["metadata"] / "jwcm08_importance_only_run_summary.json"
    save_json(run_summary, summary_path)
    print(f"\nSaved run summary: {summary_path}", flush=True)
    print("JWCM importance-only pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
