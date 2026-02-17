#!/usr/bin/env python3
"""Verbose end-to-end Fisher-merging pipeline using VERL rollout outputs.

This script implements the exact workflow requested for Fisher merging:

1) Rollout with existing VERL framework (`verl.trainer.main_ppo` in `val_only` mode),
   using task-specific sampling controls:
   - `top_p`, `top_k`, `n`, `temperature`, `max_new_tokens`, `do_sample`.
2) Keep only correct trajectories from rollout dumps and attach them back to
   source data as `rollout_results`.
3) Compute empirical Fisher diagonal from correct trajectories:
   - forward: sum log-probability over entire generated sequence
   - backward: one-shot gradient for that scalar
   - square: element-wise `g^2`
   - average across trajectories.
4) Merge task models via Fisher precision weighting:
   `theta*_j = sum_i(lambda_i * F_{i,j} * theta_{i,j}) / sum_i(lambda_i * F_{i,j})`

The script is intentionally verbose for traceability and debugging.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class RolloutSettings:
    """Task-specific rollout settings mapped to VERL validation generation config.

    Args:
        n: Number of sampled rollouts per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        do_sample: Whether sampling is enabled.
        max_new_tokens: Maximum generated response tokens.
        max_prompt_length: Prompt token limit for dataset filtering/tokenization.
        val_batch_size: Validation batch size for VERL dataloader.
        train_batch_size: Placeholder train batch size required by VERL even in val-only mode.
        gpu_memory_utilization: vLLM GPU memory utilization target.
        max_num_batched_tokens: vLLM max batched token budget.
        log_prob_micro_batch_size_per_gpu: Micro-batch size for log-prob computations.
        ulysses_sequence_parallel_size: Sequence parallel size for actor/ref workers.
        mode: Rollout mode (`async` recommended for vLLM).
        rollout_name: Rollout engine name (`vllm`).
    """

    n: int = 8
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    do_sample: bool = True
    max_new_tokens: int = 512
    max_prompt_length: int = 2048
    val_batch_size: int = 256
    train_batch_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 34816
    log_prob_micro_batch_size_per_gpu: int = 8
    ulysses_sequence_parallel_size: int = 1
    mode: str = "async"
    rollout_name: str = "vllm"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RolloutSettings":
        """Create rollout settings with defaults from optional dictionary."""

        data = dict(data or {})
        return cls(
            n=int(data.get("n", 8)),
            temperature=float(data.get("temperature", 0.6)),
            top_p=float(data.get("top_p", 0.95)),
            top_k=int(data.get("top_k", 20)),
            do_sample=bool(data.get("do_sample", True)),
            max_new_tokens=int(data.get("max_new_tokens", 512)),
            max_prompt_length=int(data.get("max_prompt_length", 2048)),
            val_batch_size=int(data.get("val_batch_size", 256)),
            train_batch_size=int(data.get("train_batch_size", 1)),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.9)),
            max_num_batched_tokens=int(data.get("max_num_batched_tokens", 34816)),
            log_prob_micro_batch_size_per_gpu=int(data.get("log_prob_micro_batch_size_per_gpu", 8)),
            ulysses_sequence_parallel_size=int(data.get("ulysses_sequence_parallel_size", 1)),
            mode=str(data.get("mode", "async")),
            rollout_name=str(data.get("rollout_name", "vllm")),
        )


@dataclass(frozen=True)
class FisherSettings:
    """Task-specific Fisher estimation settings.

    Args:
        device: Device string for Fisher gradient computation.
        model_dtype: Torch dtype name for loading model weights.
        max_prompt_tokens: Prompt truncation length before response concatenation.
        max_response_tokens: Maximum response tokens used in Fisher computation.
        max_correct_trajectories: Optional cap on number of correct trajectories.
        log_every: Progress logging interval in number of trajectories.
        grad_checkpointing: If True, enable gradient checkpointing to reduce
            activation memory during backward.
        oom_retry_min_response_tokens: Legacy field kept for backward
            config compatibility. The current implementation does not shrink
            response length on OOM; failures are surfaced immediately.
    """

    device: str = "cuda:0"
    model_dtype: str = "bf16"
    max_prompt_tokens: int = 1536
    max_response_tokens: int = 512
    max_correct_trajectories: Optional[int] = None
    log_every: int = 10
    grad_checkpointing: bool = True
    oom_retry_min_response_tokens: int = 512

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "FisherSettings":
        """Create Fisher settings with defaults from optional dictionary."""

        data = dict(data or {})
        max_correct = data.get("max_correct_trajectories", None)
        return cls(
            device=str(data.get("device", "cuda:0")),
            model_dtype=str(data.get("model_dtype", "bf16")),
            max_prompt_tokens=int(data.get("max_prompt_tokens", 1536)),
            max_response_tokens=int(data.get("max_response_tokens", 512)),
            max_correct_trajectories=None if max_correct is None else int(max_correct),
            log_every=int(data.get("log_every", 10)),
            grad_checkpointing=bool(data.get("grad_checkpointing", True)),
            oom_retry_min_response_tokens=int(data.get("oom_retry_min_response_tokens", 512)),
        )


@dataclass(frozen=True)
class TaskSettings:
    """One task entry describing model/data/reward/rollout/fisher settings.

    Args:
        name: Task key.
        model_path: Model path used for rollout and Fisher estimation.
        input_parquet: Input dataset parquet path.
        input_format: Dataset format (`auto`, `if_raw`, `verl_ready`).
        data_source: Optional override for data source key in VERL rows.
        base_reward_function_path: Optional path to task-specific base reward function.
        base_reward_function_name: Reward function name in base reward module.
        correctness_score_threshold: Score threshold used when `acc` is absent.
        gpu_ids: Physical GPU ids used for rollout subprocess.
        rollout: Task-specific rollout settings.
        fisher: Task-specific Fisher settings.
    """

    name: str
    model_path: Path
    input_parquet: Path
    input_format: str
    data_source: Optional[str]
    base_reward_function_path: Optional[Path]
    base_reward_function_name: str
    correctness_score_threshold: float
    gpu_ids: List[int]
    rollout: RolloutSettings
    fisher: FisherSettings

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        default_gpu_ids: Sequence[int],
    ) -> "TaskSettings":
        """Parse task settings from raw dictionary and validate required fields."""

        if "name" not in data:
            raise ValueError("Task config must include 'name'.")
        if "model_path" not in data:
            raise ValueError(f"Task '{data.get('name', '<unknown>')}' missing 'model_path'.")
        if "input_parquet" not in data:
            raise ValueError(f"Task '{data.get('name', '<unknown>')}' missing 'input_parquet'.")

        gpu_ids_raw = data.get("gpu_ids", list(default_gpu_ids))
        gpu_ids = [int(value) for value in gpu_ids_raw]

        base_reward_path_raw = data.get("base_reward_function_path", None)
        base_reward_path = None if base_reward_path_raw in [None, "", "none", "null"] else Path(base_reward_path_raw)

        return cls(
            name=str(data["name"]),
            model_path=Path(str(data["model_path"])),
            input_parquet=Path(str(data["input_parquet"])),
            input_format=str(data.get("input_format", "auto")),
            data_source=None
            if data.get("data_source", None) in [None, "", "none", "null"]
            else str(data.get("data_source")),
            base_reward_function_path=base_reward_path,
            base_reward_function_name=str(data.get("base_reward_function_name", "compute_score")),
            correctness_score_threshold=float(data.get("correctness_score_threshold", 0.5)),
            gpu_ids=gpu_ids,
            rollout=RolloutSettings.from_dict(data.get("rollout", {})),
            fisher=FisherSettings.from_dict(data.get("fisher", {})),
        )


@dataclass(frozen=True)
class RuntimeSettings:
    """Global runtime settings for orchestration and logging.

    Args:
        output_root: Root output directory for all artifacts.
        python_executable: Python executable used for VERL subprocess calls.
        default_gpu_ids: Default GPUs for rollout when task does not override.
        seed: Global random seed.
        project_name: VERL project name for logging.
        use_wandb_logger: Whether to include wandb in VERL logger list.
        reward_wrapper_path: Reward wrapper path used by VERL rollout.
        verbose_console: If True, console logger uses DEBUG level.
        cache_env: Cache environment overrides reused from eval scripts.
    """

    output_root: Path
    python_executable: str
    default_gpu_ids: List[int]
    seed: int
    project_name: str
    use_wandb_logger: bool
    reward_wrapper_path: Path
    verbose_console: bool
    cache_env: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RuntimeSettings":
        """Parse runtime settings from raw dictionary with sane defaults."""

        data = dict(data or {})
        default_python = "/home/nsml/verl/bin/python"
        if not Path(default_python).exists():
            default_python = sys.executable

        cache_env = {
            "PYTHONWARNINGS": "ignore::UserWarning:megatron",
            "TRANSFORMERS_VERBOSITY": "error",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "XDG_CACHE_HOME": "/mnt/tmp/nsml/cache",
            "PIP_CACHE_DIR": "/mnt/tmp/nsml/pip-cache",
            "HF_HOME": "/mnt/tmp/nsml/huggingface",
            "TRANSFORMERS_CACHE": "/mnt/tmp/nsml/huggingface/hub",
            "HF_DATASETS_CACHE": "/mnt/tmp/nsml/huggingface/datasets",
            "TORCH_HOME": "/mnt/tmp/nsml/torch",
            "TRITON_CACHE_DIR": "/mnt/tmp/nsml/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/mnt/tmp/nsml/torchinductor",
            "CUDA_CACHE_PATH": "/mnt/tmp/nsml/cuda-cache",
            "TMPDIR": "/mnt/tmp/nsml/tmp",
        }
        cache_env.update({str(k): str(v) for k, v in dict(data.get("cache_env", {})).items()})

        return cls(
            output_root=Path(str(data.get("output_root", "outputs/fisher_merge_pipeline"))),
            python_executable=str(data.get("python_executable", default_python)),
            default_gpu_ids=[int(value) for value in data.get("default_gpu_ids", [0])],
            seed=int(data.get("seed", 42)),
            project_name=str(data.get("project_name", "fisher-merge-pipeline")),
            use_wandb_logger=bool(data.get("use_wandb_logger", False)),
            reward_wrapper_path=Path(str(data.get("reward_wrapper_path", "scripts/fisher_rollout_reward_wrapper.py"))),
            verbose_console=bool(data.get("verbose_console", True)),
            cache_env=cache_env,
        )


@dataclass(frozen=True)
class MergeSettings:
    """Global Fisher merge settings.

    Args:
        epsilon: Denominator stabilizer.
        lambdas: Task-level lambda coefficients.
        anchor_model_path: Optional anchor model path for architecture/tokenizer.
        output_subdir: Output subdirectory name under runtime output root.
        model_dtype: Dtype for loading models during merge.
    """

    epsilon: float
    lambdas: Dict[str, float]
    anchor_model_path: Optional[Path]
    output_subdir: str
    model_dtype: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MergeSettings":
        """Parse merge settings from raw dictionary with defaults."""

        data = dict(data or {})
        lambdas_raw = dict(data.get("lambdas", {}))
        lambdas = {str(name): float(value) for name, value in lambdas_raw.items()}

        anchor_raw = data.get("anchor_model_path", None)
        anchor_path = None if anchor_raw in [None, "", "none", "null"] else Path(str(anchor_raw))

        return cls(
            epsilon=float(data.get("epsilon", 1e-12)),
            lambdas=lambdas,
            anchor_model_path=anchor_path,
            output_subdir=str(data.get("output_subdir", "fisher_merged_model")),
            model_dtype=str(data.get("model_dtype", "bf16")),
        )


@dataclass(frozen=True)
class PipelineSettings:
    """Top-level pipeline settings combining runtime, tasks, and merge options."""

    runtime: RuntimeSettings
    merge: MergeSettings
    tasks: List[TaskSettings]

    @classmethod
    def from_config_path(cls, config_path: Path) -> "PipelineSettings":
        """Load and parse pipeline settings from YAML/JSON config file."""

        if not config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {config_path}")

        raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Config root must be a mapping, got: {type(raw_cfg)}")

        runtime = RuntimeSettings.from_dict(raw_cfg.get("runtime", {}))
        merge = MergeSettings.from_dict(raw_cfg.get("merge", {}))

        raw_tasks = raw_cfg.get("tasks", None)
        if not isinstance(raw_tasks, list) or len(raw_tasks) == 0:
            raise ValueError("Config must include non-empty list: tasks")

        tasks = [TaskSettings.from_dict(task_cfg, default_gpu_ids=runtime.default_gpu_ids) for task_cfg in raw_tasks]
        return cls(runtime=runtime, merge=merge, tasks=tasks)


# =============================================================================
# Utility Functions
# =============================================================================


def now_iso() -> str:
    """Return UTC ISO-8601 timestamp string."""

    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dir(path: Path) -> None:
    """Create directory recursively if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve dtype string into torch dtype."""

    lookup = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    normalized = str(dtype_name).lower().strip()
    if normalized not in lookup:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")
    return lookup[normalized]


def normalize_prompt_messages(prompt_obj: Any) -> List[Dict[str, str]]:
    """Normalize prompt object to chat-template compatible message list.

    Args:
        prompt_obj: Prompt object from source parquet row.

    Returns:
        List of dictionaries with `role` and `content`.
    """

    if isinstance(prompt_obj, np.ndarray):
        messages = prompt_obj.tolist()
    elif isinstance(prompt_obj, list):
        messages = prompt_obj
    elif isinstance(prompt_obj, str):
        messages = [{"role": "user", "content": prompt_obj}]
    else:
        messages = [{"role": "user", "content": str(prompt_obj)}]

    normalized: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append({"role": "user", "content": str(message)})
            continue
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        normalized.append({"role": role, "content": content})
    return normalized


def normalize_reward_model(reward_model_obj: Any) -> Dict[str, Any]:
    """Normalize reward model field into standard dictionary.

    Args:
        reward_model_obj: Raw reward model field from source row.

    Returns:
        Dictionary with at least `ground_truth`.
    """

    if isinstance(reward_model_obj, dict):
        normalized = dict(reward_model_obj)
    else:
        normalized = {"ground_truth": reward_model_obj}

    if "ground_truth" not in normalized:
        normalized["ground_truth"] = ""

    # Keep explicit style for consistency when style is absent.
    normalized.setdefault("style", "rule")
    return normalized


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable structures."""

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_json_compatible(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_compatible(item) for item in obj]
    return obj


def save_json(payload: Mapping[str, Any], output_path: Path) -> None:
    """Save JSON payload with robust conversion."""

    ensure_dir(output_path.parent)
    serializable = to_json_compatible(dict(payload))
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)


def load_jsonl_records(input_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from file."""

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
    return records


def build_logger(log_file_path: Path, verbose_console: bool = True) -> logging.Logger:
    """Build verbose logger with console + file handlers."""

    ensure_dir(log_file_path.parent)
    logger = logging.getLogger("fisher_merge_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear previous handlers when script is re-run in same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG if verbose_console else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """Log a visually separated section header."""

    logger.info("=" * 100)
    logger.info(title)
    logger.info("=" * 100)


def run_command_streaming(
    command: List[str],
    env: Dict[str, str],
    cwd: Path,
    logger: logging.Logger,
    log_prefix: str,
) -> None:
    """Run subprocess and stream stdout/stderr lines to logger.

    Args:
        command: Command list.
        env: Environment variables.
        cwd: Working directory.
        logger: Logger instance.
        log_prefix: Prefix for streamed subprocess lines.
    """

    logger.info("Executing command:")
    logger.info("  %s", " ".join(command))
    logger.info("Working directory: %s", cwd)
    logger.info("CUDA_VISIBLE_DEVICES=%s", env.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[%s] %s", log_prefix, line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Subprocess failed (exit={return_code}): {' '.join(command)}")


def build_base_env(repo_root: Path, runtime: RuntimeSettings) -> Dict[str, str]:
    """Build subprocess environment aligned with existing eval scripts."""

    env = os.environ.copy()
    env.update(runtime.cache_env)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def load_tokenizer_with_mistral_regex_fix(model_name_or_path: str | Path) -> AutoTokenizer:
    """Load tokenizer with optional mistral regex fix flag.

    Some model/tokenizer classes accept `fix_mistral_regex`; for others this
    argument is unsupported. This helper tries the safe path first, then falls
    back to standard loading.
    """

    resolved = str(model_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved,
            trust_remote_code=True,
        )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name_or_path: str | Path,
    torch_dtype: torch.dtype,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load causal LM + tokenizer with explicit device placement."""

    resolved = str(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    tokenizer = load_tokenizer_with_mistral_regex_fix(resolved)
    return model, tokenizer


# =============================================================================
# Dataset Preparation
# =============================================================================


def infer_input_format(source_df: pd.DataFrame) -> str:
    """Infer input dataset format from column signatures."""

    cols = set(source_df.columns.tolist())
    if {"data_source", "prompt", "reward_model"}.issubset(cols):
        return "verl_ready"
    if {"prompt", "instruction_id_list", "kwargs"}.issubset(cols):
        return "if_raw"
    raise ValueError(
        "Failed to infer input format. Supported signatures: "
        "verl_ready(data_source,prompt,reward_model), "
        "if_raw(prompt,instruction_id_list,kwargs). "
        f"columns={sorted(cols)}"
    )


def _to_py(value: Any) -> Any:
    """Convert numpy/arrow scalars and arrays into Python-native objects."""

    if isinstance(value, np.ndarray):
        return [_to_py(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_py(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_py(val) for key, val in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_if_instruction_payload(
    instruction_id_list: Any,
    kwargs: Any,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Normalize IF raw instruction payload into stable JSON-serializable lists."""

    ids_raw = _to_py(instruction_id_list)
    ids = [str(item) for item in ids_raw] if isinstance(ids_raw, list) else []

    kwargs_raw = _to_py(kwargs)
    if not isinstance(kwargs_raw, list):
        kwargs_raw = []

    normalized_kwargs: List[Dict[str, Any]] = []
    for item in kwargs_raw:
        if isinstance(item, dict):
            cleaned = {str(key): value for key, value in item.items() if value is not None}
            normalized_kwargs.append(cleaned)
        else:
            normalized_kwargs.append({})

    if len(normalized_kwargs) < len(ids):
        normalized_kwargs.extend({} for _ in range(len(ids) - len(normalized_kwargs)))
    elif len(normalized_kwargs) > len(ids):
        normalized_kwargs = normalized_kwargs[: len(ids)]

    return ids, normalized_kwargs


def prepare_verl_rollout_dataset(
    task: TaskSettings,
    task_output_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Prepare task dataset in VERL-ready format with deterministic sample indices.

    Returns:
        Tuple of:
            - source dataframe with helper `_source_row_position`
            - prepared VERL-ready dataframe
            - saved prepared parquet path
    """

    log_section(logger, f"[{task.name}] Prepare rollout dataset")
    logger.info("Loading source parquet: %s", task.input_parquet)
    source_df = pd.read_parquet(task.input_parquet)
    source_df = source_df.reset_index(drop=True)
    source_df["_source_row_position"] = np.arange(len(source_df), dtype=np.int64)
    logger.info("[%s] source shape=%s columns=%s", task.name, source_df.shape, list(source_df.columns))

    input_format = task.input_format
    if input_format == "auto":
        input_format = infer_input_format(source_df=source_df)
    logger.info("[%s] resolved input_format=%s", task.name, input_format)

    prepared_rows: List[Dict[str, Any]] = []
    for row_position, row in enumerate(source_df.to_dict(orient="records")):
        prompt_messages = normalize_prompt_messages(row.get("prompt"))

        if input_format == "if_raw":
            instruction_ids, kwargs = _normalize_if_instruction_payload(
                instruction_id_list=row.get("instruction_id_list"),
                kwargs=row.get("kwargs"),
            )
            ground_truth = {
                "instruction_id_list": instruction_ids,
                "kwargs": kwargs,
            }
            reward_model = {
                "style": "rule",
                "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
            }
            data_source = task.data_source or "nemotron_cascade_rl_if"
            source_index = row.get("index", None)
        elif input_format == "verl_ready":
            reward_model = normalize_reward_model(row.get("reward_model"))
            data_source = str(row.get("data_source", task.data_source or "unknown"))
            source_index = row.get("index", None)
        else:
            raise ValueError(f"[{task.name}] Unsupported input_format: {input_format}")

        extra_info = row.get("extra_info", {}) if isinstance(row.get("extra_info", {}), dict) else {}
        extra_info = dict(extra_info)

        # We enforce unique `extra_info.index` for robust join with rollout dumps.
        if "index" in extra_info:
            extra_info["source_index_old"] = extra_info.get("index")
        extra_info["index"] = int(row_position)
        extra_info["source_row_position"] = int(row_position)

        if source_index is not None:
            try:
                extra_info["source_index"] = int(source_index)
            except Exception:
                extra_info["source_index"] = source_index

        prepared_rows.append(
            {
                "data_source": data_source,
                "prompt": prompt_messages,
                "reward_model": reward_model,
                "extra_info": extra_info,
            }
        )

    prepared_df = pd.DataFrame(prepared_rows)
    prepared_parquet_path = task_output_dir / "rollout_input_verl_ready.parquet"
    ensure_dir(prepared_parquet_path.parent)
    prepared_df.to_parquet(prepared_parquet_path, index=False)

    logger.info("[%s] prepared rows=%d", task.name, len(prepared_df))
    logger.info("[%s] saved prepared VERL parquet: %s", task.name, prepared_parquet_path)
    logger.debug("[%s] prepared sample row=%s", task.name, prepared_df.iloc[0].to_dict() if len(prepared_df) else {})

    return source_df, prepared_df, prepared_parquet_path


def load_or_prepare_rollout_dataset(
    task: TaskSettings,
    task_output_dir: Path,
    logger: logging.Logger,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load existing prepared rollout parquet or regenerate it.

    This helper provides cache-aware behavior for rollout dataset preparation:
    - Default mode (`force=False`): reuse `rollout_input_verl_ready.parquet` when
      it already exists and has the required schema.
    - Force mode (`force=True`): always regenerate the prepared parquet from the
      source input parquet.

    Args:
        task: Task configuration describing source dataset and normalization logic.
        task_output_dir: Task artifact directory under output root.
        logger: Pipeline logger.
        force: If True, bypass cache and always regenerate prepared parquet.

    Returns:
        Tuple of:
            - source dataframe with `_source_row_position`
            - prepared VERL-ready dataframe
            - prepared parquet path
    """

    prepared_parquet_path = task_output_dir / "rollout_input_verl_ready.parquet"
    required_columns = {"data_source", "prompt", "reward_model", "extra_info"}

    if (not force) and prepared_parquet_path.exists():
        log_section(logger, f"[{task.name}] Reuse prepared rollout dataset")
        logger.info(
            "[%s] found existing prepared parquet and will reuse it: %s",
            task.name,
            prepared_parquet_path,
        )

        prepared_df = pd.read_parquet(prepared_parquet_path)
        missing_columns = sorted(required_columns - set(prepared_df.columns))
        if missing_columns:
            logger.warning(
                "[%s] prepared parquet is missing required columns %s; regenerating.",
                task.name,
                missing_columns,
            )
        else:
            source_df = pd.read_parquet(task.input_parquet).reset_index(drop=True)
            source_df["_source_row_position"] = np.arange(len(source_df), dtype=np.int64)
            logger.info(
                "[%s] reused prepared rows=%d columns=%s",
                task.name,
                len(prepared_df),
                list(prepared_df.columns),
            )
            return source_df, prepared_df, prepared_parquet_path

    # Fallback path: force mode or cache miss/incompatible cache.
    return prepare_verl_rollout_dataset(
        task=task,
        task_output_dir=task_output_dir,
        logger=logger,
    )


# =============================================================================
# VERL Rollout Stage
# =============================================================================


def build_rollout_command(
    task: TaskSettings,
    runtime: RuntimeSettings,
    prepared_parquet_path: Path,
    validation_output_dir: Path,
    task_verl_output_dir: Path,
) -> List[str]:
    """Build `python -m verl.trainer.main_ppo ...` command for val-only rollout."""

    rollout = task.rollout
    logger_cfg = '["console","wandb"]' if runtime.use_wandb_logger else '["console"]'

    command = [
        runtime.python_executable,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={prepared_parquet_path}",
        f"data.val_files={prepared_parquet_path}",
        f"data.train_batch_size={rollout.train_batch_size}",
        f"data.val_batch_size={rollout.val_batch_size}",
        f"data.max_prompt_length={rollout.max_prompt_length}",
        f"data.max_response_length={rollout.max_new_tokens}",
        "data.filter_overlong_prompts=True",
        "data.validation_shuffle=False",
        f"actor_rollout_ref.model.path={task.model_path}",
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.betas=[0.9,0.95]",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.kl_loss_coef=0.0",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.dtype=float16",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={rollout.ulysses_sequence_parallel_size}",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
        f"actor_rollout_ref.rollout.name={rollout.rollout_name}",
        f"actor_rollout_ref.rollout.mode={rollout.mode}",
        f"actor_rollout_ref.rollout.n={rollout.n}",
        f"actor_rollout_ref.rollout.temperature={rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={rollout.top_p}",
        f"actor_rollout_ref.rollout.top_k={rollout.top_k}",
        f"actor_rollout_ref.rollout.response_length={rollout.max_new_tokens}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout.max_num_batched_tokens}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.enable_prefix_caching=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.dtype=float16",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout.temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={rollout.top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={rollout.top_k}",
        f"actor_rollout_ref.rollout.val_kwargs.n={rollout.n}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(rollout.do_sample)}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"actor_rollout_ref.ref.ulysses_sequence_parallel_size={rollout.ulysses_sequence_parallel_size}",
        "reward_manager.name=naive",
        "reward_manager.source=register",
        f"custom_reward_function.path={runtime.reward_wrapper_path}",
        "custom_reward_function.name=compute_score",
        # `reward_kwargs` is not present in the base Hydra schema, so these
        # nested keys must be appended with `+` (not plain override).
        "+custom_reward_function.reward_kwargs.fail_on_base_error=False",
        "algorithm.use_kl_in_reward=False",
        f"trainer.logger={logger_cfg}",
        f"trainer.project_name={runtime.project_name}",
        f"trainer.experiment_name=fisher_rollout_{task.name}_{int(time.time())}",
        f"trainer.n_gpus_per_node={len(task.gpu_ids)}",
        "trainer.nnodes=1",
        "trainer.val_only=True",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.validation_data_dir={validation_output_dir}",
        f"trainer.default_local_dir={task_verl_output_dir}",
        "+reward_model.reward_kwargs.overlong_filtering=False",
    ]

    # Task-specific base reward function is optional.
    if task.base_reward_function_path is not None:
        command.append(
            "+custom_reward_function.reward_kwargs.base_reward_function_path="
            f"{task.base_reward_function_path}"
        )
        command.append(
            "+custom_reward_function.reward_kwargs.base_reward_function_name="
            f"{task.base_reward_function_name}"
        )

    return command


def run_rollout_with_verl(
    task: TaskSettings,
    runtime: RuntimeSettings,
    repo_root: Path,
    prepared_parquet_path: Path,
    task_output_dir: Path,
    logger: logging.Logger,
    force: bool,
) -> Path:
    """Execute VERL val-only rollout and return validation JSONL directory.

    Cache behavior:
    - Default mode (`force=False`): if validation JSONL files already exist,
      skip rollout subprocess execution and reuse those files.
    - Force mode (`force=True`): remove rollout directories and run rollout
      again from scratch.
    """

    log_section(logger, f"[{task.name}] Run VERL rollout")
    validation_output_dir = task_output_dir / "validation_data_dir"
    task_verl_output_dir = task_output_dir / "verl_run"

    # Skip rollout when valid cached artifacts are already present.
    if (not force) and validation_output_dir.exists():
        existing_jsonl = sorted(validation_output_dir.glob("*.jsonl"), key=lambda path: path.name)
        if len(existing_jsonl) > 0:
            logger.info(
                "[%s] skipping rollout because existing validation JSONL files were found (%d files).",
                task.name,
                len(existing_jsonl),
            )
            logger.info("[%s] cached validation dir: %s", task.name, validation_output_dir)
            return validation_output_dir

    # Force mode explicitly clears prior rollout outputs to avoid mixed artifacts.
    if force:
        if validation_output_dir.exists():
            logger.info("[%s] --force enabled, removing: %s", task.name, validation_output_dir)
            shutil.rmtree(validation_output_dir)
        if task_verl_output_dir.exists():
            logger.info("[%s] --force enabled, removing: %s", task.name, task_verl_output_dir)
            shutil.rmtree(task_verl_output_dir)

    ensure_dir(validation_output_dir)
    ensure_dir(task_verl_output_dir)

    env = build_base_env(repo_root=repo_root, runtime=runtime)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in task.gpu_ids)
    env["HYDRA_FULL_ERROR"] = "1"

    command = build_rollout_command(
        task=task,
        runtime=runtime,
        prepared_parquet_path=prepared_parquet_path,
        validation_output_dir=validation_output_dir,
        task_verl_output_dir=task_verl_output_dir,
    )
    run_command_streaming(
        command=command,
        env=env,
        cwd=repo_root,
        logger=logger,
        log_prefix=f"rollout-{task.name}",
    )

    logger.info("[%s] rollout completed. validation jsonl dir: %s", task.name, validation_output_dir)
    return validation_output_dir


def collect_rollout_records(validation_output_dir: Path, logger: logging.Logger, task_name: str) -> pd.DataFrame:
    """Collect all JSONL rollout records from validation output directory."""

    if not validation_output_dir.exists():
        raise FileNotFoundError(f"[{task_name}] validation output dir missing: {validation_output_dir}")

    jsonl_files = sorted(validation_output_dir.glob("*.jsonl"), key=lambda path: path.name)
    if not jsonl_files:
        raise FileNotFoundError(f"[{task_name}] no jsonl files found under: {validation_output_dir}")

    logger.info("[%s] found %d rollout jsonl files", task_name, len(jsonl_files))
    all_records: List[Dict[str, Any]] = []
    for jsonl_path in jsonl_files:
        records = load_jsonl_records(jsonl_path)
        logger.info("[%s] loaded %d rows from %s", task_name, len(records), jsonl_path)
        all_records.extend(records)

    rollout_df = pd.DataFrame(all_records)
    logger.info("[%s] combined rollout rows=%d columns=%s", task_name, len(rollout_df), list(rollout_df.columns))
    return rollout_df


def parse_optional_float(value: Any) -> Optional[float]:
    """Parse value to float when possible; otherwise return None."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    try:
        return float(text)
    except Exception:
        return None


def parse_optional_int(value: Any) -> Optional[int]:
    """Parse value to integer when possible; otherwise return None."""

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def extract_correct_rollouts(
    rollout_df: pd.DataFrame,
    correctness_score_threshold: float,
    task_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Filter rollout records to keep correct trajectories only."""

    log_section(logger, f"[{task_name}] Filter correct trajectories")

    if rollout_df.empty:
        raise ValueError(f"[{task_name}] rollout dataframe is empty.")

    records: List[Dict[str, Any]] = []
    for row in rollout_df.to_dict(orient="records"):
        score = parse_optional_float(row.get("score"))
        acc = parse_optional_float(row.get("acc"))
        sample_index = parse_optional_int(row.get("sample_index"))

        # Priority: explicit acc -> fallback to score threshold.
        if acc is not None:
            is_correct = bool(acc >= 0.5)
        elif score is not None:
            is_correct = bool(score >= correctness_score_threshold)
        else:
            is_correct = False

        records.append(
            {
                "sample_index": sample_index,
                "input": str(row.get("input", "")),
                "output": str(row.get("output", "")),
                "score": score,
                "acc": acc,
                "step": parse_optional_int(row.get("step")),
                "is_correct": bool(is_correct),
                # Keep original record for debugging/traceability.
                "raw_record_json": json.dumps(row, ensure_ascii=False),
            }
        )

    parsed_df = pd.DataFrame(records)
    correct_df = parsed_df[parsed_df["is_correct"]].copy()

    logger.info("[%s] total rollout rows=%d", task_name, len(parsed_df))
    logger.info("[%s] correct rollout rows=%d", task_name, len(correct_df))
    logger.info("[%s] correct ratio=%.4f", task_name, len(correct_df) / max(len(parsed_df), 1))

    if correct_df.empty:
        raise ValueError(f"[{task_name}] no correct trajectories were found.")

    if correct_df["sample_index"].isna().any():
        missing = int(correct_df["sample_index"].isna().sum())
        logger.warning(
            "[%s] %d correct rows missing sample_index. "
            "These rows cannot be aligned to source dataset and will be dropped.",
            task_name,
            missing,
        )
        correct_df = correct_df.dropna(subset=["sample_index"]).copy()

    correct_df["sample_index"] = correct_df["sample_index"].astype(np.int64)
    return correct_df


def attach_rollout_results_to_source(
    source_df: pd.DataFrame,
    correct_rollout_df: pd.DataFrame,
    task_name: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Attach `rollout_results` column (JSON list) to source dataframe.

    The column contains only correct trajectories for each source row.
    """

    log_section(logger, f"[{task_name}] Attach rollout_results column")

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in correct_rollout_df.to_dict(orient="records"):
        source_row_position = int(row["sample_index"])
        grouped.setdefault(source_row_position, []).append(
            {
                "output": row["output"],
                "score": row["score"],
                "acc": row["acc"],
                "step": row["step"],
            }
        )

    augmented_df = source_df.copy()
    if "_source_row_position" not in augmented_df.columns:
        augmented_df["_source_row_position"] = np.arange(len(augmented_df), dtype=np.int64)

    # Store rollout results as JSON strings for parquet compatibility.
    rollout_results_json: List[str] = []
    rollout_correct_count: List[int] = []
    for row_position in augmented_df["_source_row_position"].tolist():
        results = grouped.get(int(row_position), [])
        rollout_results_json.append(json.dumps(results, ensure_ascii=False))
        rollout_correct_count.append(len(results))

    augmented_df["rollout_results"] = rollout_results_json
    augmented_df["rollout_correct_count"] = np.asarray(rollout_correct_count, dtype=np.int64)
    logger.info("[%s] attached rollout_results for %d source rows", task_name, len(augmented_df))
    logger.info(
        "[%s] rows with >=1 correct rollout: %d",
        task_name,
        int((augmented_df["rollout_correct_count"] > 0).sum()),
    )
    return augmented_df


# =============================================================================
# Fisher Computation Stage
# =============================================================================


def build_index_to_prompt_map(prepared_df: pd.DataFrame) -> Dict[int, List[Dict[str, str]]]:
    """Build mapping from sample index to normalized prompt messages."""

    mapping: Dict[int, List[Dict[str, str]]] = {}
    for row in prepared_df.to_dict(orient="records"):
        extra_info = row.get("extra_info", {})
        if not isinstance(extra_info, dict):
            continue
        sample_index = parse_optional_int(extra_info.get("index"))
        if sample_index is None:
            continue
        mapping[int(sample_index)] = normalize_prompt_messages(row.get("prompt"))
    return mapping


def _compute_sequence_log_prob_with_labels(
    model: AutoModelForCausalLM,
    param_list: List[torch.nn.Parameter],
    full_ids_cpu: torch.Tensor,
    prompt_len: int,
    device: torch.device,
) -> List[Optional[torch.Tensor]]:
    """Compute per-parameter gradients for one full sequence.

    Why this formulation:
    - The previous implementation built a large `log_softmax` tensor over all
      response positions at once, which caused severe memory spikes.
    - Using `labels` delegates token-level NLL computation to the model's
      internal loss path and avoids explicit giant intermediate tensors.

    Args:
        model: Loaded causal LM.
        param_list: Trainable floating parameters tracked for Fisher.
        full_ids_cpu: Concatenated prompt+response token ids on CPU.
        prompt_len: Number of prompt tokens in `full_ids_cpu`.
        device: CUDA/CPU device for forward+backward.

    Returns:
        Gradient list aligned with `param_list`.
    """

    full_ids = full_ids_cpu.to(device).unsqueeze(0)
    full_attention = torch.ones_like(full_ids, device=device)
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    valid_response_tokens = int((labels != -100).sum().item())
    if valid_response_tokens <= 0:
        raise ValueError("No valid response tokens after prompt masking.")

    outputs = model(
        input_ids=full_ids,
        attention_mask=full_attention,
        labels=labels,
        use_cache=False,
    )
    # `outputs.loss` is mean negative log-likelihood across valid tokens.
    # Convert it back to the sum of log-probabilities used in Fisher formula.
    sequence_log_prob = -outputs.loss.to(torch.float32) * float(valid_response_tokens)
    grads = torch.autograd.grad(
        sequence_log_prob,
        param_list,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    del outputs
    del sequence_log_prob
    return grads


def run_fisher_worker_from_args(args: argparse.Namespace) -> None:
    """Distributed Fisher worker entrypoint.

    This worker is launched via `torch.distributed.run` and computes one shard
    of Fisher statistics on each rank. Rank-0 aggregates shard files into the
    final `fisher_diagonal.pt` and `fisher_summary.json`.
    """

    import torch.distributed as dist

    task_name = str(args.fisher_task_name)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def rank_log(message: str) -> None:
        """Print rank-tagged worker logs with immediate flush."""

        print(f"[fisher-worker:{task_name}:rank{rank}] {message}", flush=True)

    # Initialize distributed process group for synchronization only.
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        worker_device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        worker_device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(hours=12),
        )

    try:
        seed = int(args.fisher_seed)
        set_seed(seed + rank)

        fisher_cfg = FisherSettings(
            device=str(worker_device),
            model_dtype=str(args.fisher_model_dtype),
            max_prompt_tokens=int(args.fisher_max_prompt_tokens),
            max_response_tokens=int(args.fisher_max_response_tokens),
            max_correct_trajectories=None
            if int(args.fisher_max_correct_trajectories) < 0
            else int(args.fisher_max_correct_trajectories),
            log_every=int(args.fisher_log_every),
            grad_checkpointing=bool(int(args.fisher_grad_checkpointing)),
            # Legacy compatibility field. OOM fallback truncation is disabled.
            oom_retry_min_response_tokens=0,
        )

        correct_rollout_path = Path(str(args.fisher_correct_rollout_parquet))
        prepared_parquet_path = Path(str(args.fisher_prepared_parquet))
        fisher_output_path = Path(str(args.fisher_output_path))
        fisher_summary_path = Path(str(args.fisher_summary_path))
        rank_output_dir = Path(str(args.fisher_rank_output_dir))
        ensure_dir(rank_output_dir)

        correct_rollout_df = pd.read_parquet(correct_rollout_path)
        prepared_df = pd.read_parquet(prepared_parquet_path)

        if (
            fisher_cfg.max_correct_trajectories is not None
            and len(correct_rollout_df) > fisher_cfg.max_correct_trajectories
        ):
            correct_rollout_df = correct_rollout_df.sample(
                n=fisher_cfg.max_correct_trajectories,
                random_state=seed,
                replace=False,
            ).reset_index(drop=True)

        # Deterministic rank sharding: each rank processes every `world_size`-th sample.
        shard_df = correct_rollout_df.iloc[rank::world_size].reset_index(drop=True)
        rank_log(
            f"loaded rows: all={len(correct_rollout_df)} shard={len(shard_df)} "
            f"world_size={world_size} device={worker_device}"
        )

        dtype = resolve_torch_dtype(fisher_cfg.model_dtype)
        model, tokenizer = load_causal_lm(
            model_name_or_path=str(args.fisher_model_path),
            torch_dtype=dtype,
            device=str(worker_device),
        )
        model.eval()
        if fisher_cfg.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            # Enable activation recomputation to lower memory during backward.
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            # `use_cache` must be disabled for gradient checkpointing/backward.
            model.config.use_cache = False

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        param_entries: List[Tuple[str, torch.nn.Parameter]] = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and torch.is_floating_point(parameter)
        ]
        param_names = [name for name, _ in param_entries]
        param_list = [parameter for _, parameter in param_entries]

        # Keep Fisher accumulator on CPU to reduce persistent GPU memory usage.
        fisher_sum_cpu: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
            for name, parameter in param_entries
        }

        index_to_prompt = build_index_to_prompt_map(prepared_df=prepared_df)
        prompt_token_cache: Dict[int, torch.Tensor] = {}

        processed = 0
        skipped_missing_index = 0
        skipped_empty_response = 0
        skipped_too_short = 0
        start_time = time.time()

        iterator = tqdm(
            shard_df.itertuples(index=False),
            total=len(shard_df),
            desc=f"Fisher[{task_name}][rank{rank}]",
            disable=(rank != 0),
        )
        for step_index, row in enumerate(iterator, start=1):
            sample_index = int(row.sample_index)
            response_text = str(row.output)

            prompt_messages = index_to_prompt.get(sample_index, None)
            if prompt_messages is None:
                skipped_missing_index += 1
                continue

            if sample_index not in prompt_token_cache:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_encoded = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=fisher_cfg.max_prompt_tokens,
                    add_special_tokens=False,
                )
                prompt_token_cache[sample_index] = prompt_encoded["input_ids"][0].detach().cpu()

            prompt_ids_cpu = prompt_token_cache[sample_index]
            response_ids_full_cpu = tokenizer(
                response_text,
                return_tensors="pt",
                truncation=True,
                max_length=fisher_cfg.max_response_tokens,
                add_special_tokens=False,
            )["input_ids"][0].detach().cpu()

            if response_ids_full_cpu.numel() == 0:
                skipped_empty_response += 1
                continue

            response_len = int(response_ids_full_cpu.numel())
            try:
                response_ids_cpu = response_ids_full_cpu
                full_ids_cpu = torch.cat([prompt_ids_cpu, response_ids_cpu], dim=0)
                prompt_len = int(prompt_ids_cpu.numel())
                full_len = int(full_ids_cpu.numel())
                if full_len <= prompt_len:
                    skipped_too_short += 1
                    continue

                grads = _compute_sequence_log_prob_with_labels(
                    model=model,
                    param_list=param_list,
                    full_ids_cpu=full_ids_cpu,
                    prompt_len=prompt_len,
                    device=worker_device,
                )
                for param_index, grad in enumerate(grads):
                    if grad is None:
                        continue
                    name = param_names[param_index]
                    fisher_sum_cpu[name].add_(grad.detach().to(torch.float32).cpu().pow(2))

                processed += 1
                del grads
            except torch.OutOfMemoryError as oom_error:
                # Surface OOM directly so users can tune `max_response_tokens`
                # without hidden approximation behavior.
                raise RuntimeError(
                    f"[{task_name}] Fisher OOM at sample_index={sample_index} "
                    f"with response_tokens={response_len}. "
                    f"Reduce fisher.max_response_tokens / max_prompt_tokens or "
                    f"increase parallelism."
                ) from oom_error
            finally:
                model.zero_grad(set_to_none=True)

            if step_index % fisher_cfg.log_every == 0 and rank == 0:
                elapsed = time.time() - start_time
                rank_log(f"progress shard_processed={processed}/{len(shard_df)} elapsed={elapsed:.1f}s")

            if processed % 16 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        rank_fisher_path = rank_output_dir / f"fisher_rank_{rank}.pt"
        rank_summary_path = rank_output_dir / f"fisher_rank_{rank}.json"
        torch.save(
            {
                "fisher_sum": fisher_sum_cpu,
                "processed": int(processed),
                "num_tracked_parameters": int(len(param_entries)),
            },
            rank_fisher_path,
        )
        save_json(
            {
                "rank": int(rank),
                "world_size": int(world_size),
                "processed": int(processed),
                "skipped_missing_index": int(skipped_missing_index),
                "skipped_empty_response": int(skipped_empty_response),
                "skipped_too_short": int(skipped_too_short),
                "rank_fisher_path": str(rank_fisher_path),
            },
            rank_summary_path,
        )
        rank_log(f"saved shard fisher: {rank_fisher_path}")

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            global_fisher_sum: Optional[Dict[str, torch.Tensor]] = None
            total_processed = 0
            merged_skipped_missing_index = 0
            merged_skipped_empty_response = 0
            merged_skipped_too_short = 0
            merged_num_tracked_parameters = 0

            for current_rank in range(world_size):
                shard_tensor_path = rank_output_dir / f"fisher_rank_{current_rank}.pt"
                shard_meta_path = rank_output_dir / f"fisher_rank_{current_rank}.json"
                shard_state = torch.load(shard_tensor_path, map_location="cpu")
                shard_meta = json.loads(shard_meta_path.read_text(encoding="utf-8"))

                shard_fisher_sum = shard_state["fisher_sum"]
                shard_processed = int(shard_state["processed"])
                merged_num_tracked_parameters = int(shard_state["num_tracked_parameters"])
                total_processed += shard_processed

                merged_skipped_missing_index += int(shard_meta.get("skipped_missing_index", 0))
                merged_skipped_empty_response += int(shard_meta.get("skipped_empty_response", 0))
                merged_skipped_too_short += int(shard_meta.get("skipped_too_short", 0))

                if global_fisher_sum is None:
                    global_fisher_sum = {name: tensor.clone() for name, tensor in shard_fisher_sum.items()}
                else:
                    for name in global_fisher_sum.keys():
                        global_fisher_sum[name].add_(shard_fisher_sum[name])

            if global_fisher_sum is None or total_processed <= 0:
                raise RuntimeError(f"[{task_name}] no trajectories processed for distributed Fisher.")

            for name in global_fisher_sum.keys():
                global_fisher_sum[name].div_(float(total_processed))

            ensure_dir(fisher_output_path.parent)
            ensure_dir(fisher_summary_path.parent)
            torch.save(global_fisher_sum, fisher_output_path)
            fisher_summary = {
                "task_name": task_name,
                "created_at": now_iso(),
                "model_path": str(args.fisher_model_path),
                "model_dtype": fisher_cfg.model_dtype,
                "world_size": int(world_size),
                "processed_trajectories": int(total_processed),
                "skipped_missing_index": int(merged_skipped_missing_index),
                "skipped_empty_response": int(merged_skipped_empty_response),
                "skipped_too_short": int(merged_skipped_too_short),
                "num_tracked_parameters": int(merged_num_tracked_parameters),
                "elapsed_seconds": float(time.time() - start_time),
                "fisher_path": str(fisher_output_path),
            }
            save_json(fisher_summary, fisher_summary_path)
            rank_log(f"saved merged fisher: {fisher_output_path}")
            rank_log(f"saved fisher summary: {fisher_summary_path}")

        if world_size > 1:
            dist.barrier()

    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def compute_fisher_for_task(
    task: TaskSettings,
    correct_rollout_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    task_output_dir: Path,
    logger: logging.Logger,
    runtime: RuntimeSettings,
    repo_root: Path,
) -> tuple[Path, Dict[str, Any]]:
    """Compute Fisher diagonal using distributed workers across task GPUs.

    This function acts as an orchestrator:
    1. Persist latest rollout/prepared dataframes for worker consumption.
    2. Launch distributed worker subprocess (`torch.distributed.run`).
    3. Load worker-produced Fisher summary + output path.
    """

    log_section(logger, f"[{task.name}] Compute Fisher diagonal")
    fisher_cfg = task.fisher
    fisher_path = task_output_dir / "fisher_diagonal.pt"
    fisher_summary_path = task_output_dir / "fisher_summary.json"
    fisher_rank_output_dir = task_output_dir / "fisher_rank_shards"
    ensure_dir(fisher_rank_output_dir)

    # Persist dataframes so each rank can load exactly the same source artifacts.
    correct_rollout_path = task_output_dir / "correct_rollout_trajectories.parquet"
    prepared_parquet_path = task_output_dir / "rollout_input_verl_ready.parquet"
    correct_rollout_df.to_parquet(correct_rollout_path, index=False)
    prepared_df.to_parquet(prepared_parquet_path, index=False)

    world_size = max(1, len(task.gpu_ids))
    logger.info(
        "[%s] launching distributed Fisher on %d GPUs: %s",
        task.name,
        world_size,
        task.gpu_ids,
    )
    logger.info("[%s] Fisher trajectories candidates: %d", task.name, len(correct_rollout_df))

    command = [
        runtime.python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        str(Path(__file__).resolve()),
        "--fisher-worker",
        "--fisher-task-name",
        task.name,
        "--fisher-model-path",
        str(task.model_path),
        "--fisher-prepared-parquet",
        str(prepared_parquet_path),
        "--fisher-correct-rollout-parquet",
        str(correct_rollout_path),
        "--fisher-output-path",
        str(fisher_path),
        "--fisher-summary-path",
        str(fisher_summary_path),
        "--fisher-rank-output-dir",
        str(fisher_rank_output_dir),
        "--fisher-model-dtype",
        fisher_cfg.model_dtype,
        "--fisher-max-prompt-tokens",
        str(fisher_cfg.max_prompt_tokens),
        "--fisher-max-response-tokens",
        str(fisher_cfg.max_response_tokens),
        "--fisher-max-correct-trajectories",
        str(-1 if fisher_cfg.max_correct_trajectories is None else fisher_cfg.max_correct_trajectories),
        "--fisher-log-every",
        str(fisher_cfg.log_every),
        "--fisher-grad-checkpointing",
        str(1 if fisher_cfg.grad_checkpointing else 0),
        "--fisher-seed",
        str(runtime.seed),
    ]

    env = build_base_env(repo_root=repo_root, runtime=runtime)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in task.gpu_ids)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_command_streaming(
        command=command,
        env=env,
        cwd=repo_root,
        logger=logger,
        log_prefix=f"fisher-{task.name}",
    )

    if not fisher_path.exists():
        raise FileNotFoundError(f"[{task.name}] distributed Fisher output missing: {fisher_path}")
    if not fisher_summary_path.exists():
        raise FileNotFoundError(f"[{task.name}] distributed Fisher summary missing: {fisher_summary_path}")

    summary = json.loads(fisher_summary_path.read_text(encoding="utf-8"))
    logger.info("[%s] saved Fisher diagonal: %s", task.name, fisher_path)
    logger.info("[%s] Fisher summary: %s", task.name, fisher_summary_path)
    return fisher_path, summary


# =============================================================================
# Fisher Merge Stage
# =============================================================================


def validate_parameter_compatibility(models: Mapping[str, AutoModelForCausalLM]) -> None:
    """Validate that all models have identical parameter keys and shapes."""

    items = list(models.items())
    if len(items) < 2:
        return

    ref_name, ref_model = items[0]
    ref_params = dict(ref_model.named_parameters())

    for name, model in items[1:]:
        current_params = dict(model.named_parameters())
        if set(ref_params.keys()) != set(current_params.keys()):
            missing = sorted(set(ref_params.keys()) - set(current_params.keys()))
            extra = sorted(set(current_params.keys()) - set(ref_params.keys()))
            raise ValueError(
                f"Parameter key mismatch between '{ref_name}' and '{name}'. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        for param_name in ref_params.keys():
            if tuple(ref_params[param_name].shape) != tuple(current_params[param_name].shape):
                raise ValueError(
                    f"Shape mismatch for '{param_name}' between '{ref_name}' and '{name}': "
                    f"{tuple(ref_params[param_name].shape)} vs {tuple(current_params[param_name].shape)}"
                )


def validate_fisher_compatibility(
    task_models: Mapping[str, AutoModelForCausalLM],
    fisher_by_task: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    """Validate that Fisher dictionaries match model parameter keys and shapes."""

    for task_name, model in task_models.items():
        if task_name not in fisher_by_task:
            raise ValueError(f"Missing Fisher tensors for task: {task_name}")
        fisher_dict = fisher_by_task[task_name]
        for param_name, param_tensor in model.named_parameters():
            if not torch.is_floating_point(param_tensor.data):
                continue
            if param_name not in fisher_dict:
                raise ValueError(f"Missing Fisher tensor: task={task_name} param={param_name}")
            if tuple(fisher_dict[param_name].shape) != tuple(param_tensor.shape):
                raise ValueError(
                    f"Fisher shape mismatch task={task_name} param={param_name}: "
                    f"fisher={tuple(fisher_dict[param_name].shape)} model={tuple(param_tensor.shape)}"
                )


def merge_with_fisher_precision_inplace(
    merge_model: AutoModelForCausalLM,
    task_models: Mapping[str, AutoModelForCausalLM],
    fisher_by_task: Mapping[str, Mapping[str, torch.Tensor]],
    task_lambdas: Mapping[str, float],
    epsilon: float,
) -> Dict[str, Any]:
    """Apply Fisher precision merge to `merge_model` parameters in-place.

    Formula:
        theta*_j = sum_i(lambda_i * F_{i,j} * theta_{i,j}) / sum_i(lambda_i * F_{i,j})

    Practical detail:
    - For coordinates with near-zero denominator, fallback to anchor parameter
      to avoid undefined 0/0-like behavior.
    """

    validate_parameter_compatibility(task_models)
    validate_fisher_compatibility(task_models=task_models, fisher_by_task=fisher_by_task)

    merge_named = dict(merge_model.named_parameters())
    task_named = {task_name: dict(model.named_parameters()) for task_name, model in task_models.items()}
    task_names = list(task_models.keys())

    zero_denom_count = 0
    total_count = 0

    with torch.no_grad():
        for param_name, merge_param in tqdm(merge_named.items(), desc="Fisher merge"):
            if not torch.is_floating_point(merge_param.data):
                continue

            anchor_fp32 = merge_param.data.detach().to(torch.float32)
            numerator = torch.zeros_like(anchor_fp32)
            denominator = torch.zeros_like(anchor_fp32)

            for task_name in task_names:
                lambda_i = float(task_lambdas.get(task_name, 1.0))
                theta_i = task_named[task_name][param_name].data.detach().to(torch.float32)
                fisher_i = fisher_by_task[task_name][param_name].detach().to(torch.float32)

                weighted_precision = lambda_i * fisher_i
                numerator.add_(weighted_precision * theta_i)
                denominator.add_(weighted_precision)

            merged_fp32 = numerator / (denominator + float(epsilon))
            valid_mask = denominator > float(epsilon)
            merged_fp32 = torch.where(valid_mask, merged_fp32, anchor_fp32)
            merge_param.data.copy_(merged_fp32.to(merge_param.dtype))

            zero_denom_count += int((~valid_mask).sum().item())
            total_count += int(valid_mask.numel())

    return {
        "epsilon": float(epsilon),
        "task_lambdas": {name: float(value) for name, value in task_lambdas.items()},
        "zero_denom_count": int(zero_denom_count),
        "total_param_elements": int(total_count),
        "zero_denom_ratio": float(zero_denom_count / max(total_count, 1)),
    }


def run_fisher_merge(
    pipeline_cfg: PipelineSettings,
    fisher_paths_by_task: Mapping[str, Path],
    output_root: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run Fisher merge across all tasks and save merged checkpoint."""

    log_section(logger, "Run Fisher precision merge")

    if len(pipeline_cfg.tasks) == 0:
        raise ValueError("No tasks configured for merge.")

    merge_cfg = pipeline_cfg.merge
    merge_dtype = resolve_torch_dtype(merge_cfg.model_dtype)

    anchor_model_path = merge_cfg.anchor_model_path or pipeline_cfg.tasks[0].model_path
    logger.info("Anchor model path: %s", anchor_model_path)

    merge_model, merge_tokenizer = load_causal_lm(
        model_name_or_path=anchor_model_path,
        torch_dtype=merge_dtype,
        device="cpu",
    )

    task_models: MutableMapping[str, AutoModelForCausalLM] = {}
    for task in pipeline_cfg.tasks:
        model, _ = load_causal_lm(
            model_name_or_path=task.model_path,
            torch_dtype=merge_dtype,
            device="cpu",
        )
        task_models[task.name] = model
        logger.info("Loaded task model for merge: task=%s path=%s", task.name, task.model_path)

    fisher_by_task = {
        task_name: torch.load(fisher_path, map_location="cpu")
        for task_name, fisher_path in fisher_paths_by_task.items()
    }
    logger.info("Loaded Fisher tensors for tasks: %s", sorted(fisher_by_task.keys()))

    task_lambdas = {}
    for task in pipeline_cfg.tasks:
        task_lambdas[task.name] = float(merge_cfg.lambdas.get(task.name, 1.0))
        logger.info("Lambda[%s]=%.6f", task.name, task_lambdas[task.name])

    merge_summary = merge_with_fisher_precision_inplace(
        merge_model=merge_model,
        task_models=task_models,
        fisher_by_task=fisher_by_task,
        task_lambdas=task_lambdas,
        epsilon=merge_cfg.epsilon,
    )

    merged_output_dir = output_root / merge_cfg.output_subdir
    ensure_dir(merged_output_dir)
    merge_model.save_pretrained(merged_output_dir, safe_serialization=True)
    merge_tokenizer.save_pretrained(merged_output_dir)

    metadata = {
        "created_at": now_iso(),
        "method": "fisher_precision_weighted_merge",
        "formula": "theta*=sum_i(lambda_i*F_i*theta_i)/sum_i(lambda_i*F_i)",
        "anchor_model_path": str(anchor_model_path),
        "task_model_paths": {task.name: str(task.model_path) for task in pipeline_cfg.tasks},
        "fisher_paths": {name: str(path) for name, path in fisher_paths_by_task.items()},
        "merge_config": asdict(merge_cfg),
        "merge_summary": merge_summary,
        "output_dir": str(merged_output_dir),
    }
    save_json(metadata, merged_output_dir / "merge_metadata.json")
    logger.info("Saved merged model: %s", merged_output_dir)
    logger.info("Saved merge metadata: %s", merged_output_dir / "merge_metadata.json")

    del merge_model
    del merge_tokenizer
    for task_name in list(task_models.keys()):
        del task_models[task_name]
    del fisher_by_task
    gc.collect()

    return {
        "merged_output_dir": merged_output_dir,
        "merge_summary": merge_summary,
    }


# =============================================================================
# Main Orchestration
# =============================================================================


def process_single_task(
    task: TaskSettings,
    pipeline_cfg: PipelineSettings,
    repo_root: Path,
    logger: logging.Logger,
    force: bool,
) -> Dict[str, Any]:
    """Process one task: dataset prep -> rollout -> correct filtering -> Fisher.

    Args:
        task: Task-level settings.
        pipeline_cfg: Full pipeline settings object.
        repo_root: Repository root used as subprocess working directory.
        logger: Pipeline logger.
        force: If True, regenerate rollout artifacts even when cached files exist.
    """

    task_output_dir = pipeline_cfg.runtime.output_root / "tasks" / task.name
    ensure_dir(task_output_dir)

    source_df, prepared_df, prepared_parquet_path = load_or_prepare_rollout_dataset(
        task=task,
        task_output_dir=task_output_dir,
        logger=logger,
        force=force,
    )

    rollout_records_path = task_output_dir / "rollout_records.parquet"
    validation_output_dir = task_output_dir / "validation_data_dir"

    # Prefer cached parsed rollout records when available.
    rollout_df: Optional[pd.DataFrame] = None
    if (not force) and rollout_records_path.exists():
        log_section(logger, f"[{task.name}] Reuse rollout records parquet")
        try:
            rollout_df = pd.read_parquet(rollout_records_path)
            if len(rollout_df) == 0:
                logger.warning(
                    "[%s] cached rollout records parquet is empty; rerunning/recapturing rollout records.",
                    task.name,
                )
                rollout_df = None
            else:
                logger.info(
                    "[%s] reused rollout records parquet rows=%d path=%s",
                    task.name,
                    len(rollout_df),
                    rollout_records_path,
                )
        except Exception as exc:
            logger.warning(
                "[%s] failed to load cached rollout records (%s); rerunning/recapturing rollout records.",
                task.name,
                exc,
            )
            rollout_df = None

    # If parsed records are not cached, ensure rollout JSONL exists (or run rollout),
    # then collect JSONL outputs into parquet.
    if rollout_df is None:
        validation_output_dir = run_rollout_with_verl(
            task=task,
            runtime=pipeline_cfg.runtime,
            repo_root=repo_root,
            prepared_parquet_path=prepared_parquet_path,
            task_output_dir=task_output_dir,
            logger=logger,
            force=force,
        )

        rollout_df = collect_rollout_records(
            validation_output_dir=validation_output_dir,
            logger=logger,
            task_name=task.name,
        )
        rollout_df.to_parquet(rollout_records_path, index=False)
        logger.info("[%s] saved rollout records parquet: %s", task.name, rollout_records_path)

    correct_rollout_df = extract_correct_rollouts(
        rollout_df=rollout_df,
        correctness_score_threshold=task.correctness_score_threshold,
        task_name=task.name,
        logger=logger,
    )
    correct_rollout_df.to_parquet(task_output_dir / "correct_rollout_trajectories.parquet", index=False)
    logger.info(
        "[%s] saved correct trajectories parquet: %s",
        task.name,
        task_output_dir / "correct_rollout_trajectories.parquet",
    )

    source_augmented_df = attach_rollout_results_to_source(
        source_df=source_df,
        correct_rollout_df=correct_rollout_df,
        task_name=task.name,
        logger=logger,
    )
    source_augmented_path = task_output_dir / "source_with_rollout_results.parquet"
    source_augmented_df.to_parquet(source_augmented_path, index=False)
    logger.info("[%s] saved source+rollout parquet: %s", task.name, source_augmented_path)

    fisher_path, fisher_summary = compute_fisher_for_task(
        task=task,
        correct_rollout_df=correct_rollout_df,
        prepared_df=prepared_df,
        task_output_dir=task_output_dir,
        logger=logger,
        runtime=pipeline_cfg.runtime,
        repo_root=repo_root,
    )

    task_summary = {
        "task_name": task.name,
        "model_path": str(task.model_path),
        "input_parquet": str(task.input_parquet),
        "prepared_parquet_path": str(prepared_parquet_path),
        "validation_output_dir": str(validation_output_dir),
        "rollout_records_parquet": str(task_output_dir / "rollout_records.parquet"),
        "correct_rollout_parquet": str(task_output_dir / "correct_rollout_trajectories.parquet"),
        "source_with_rollout_results_parquet": str(source_augmented_path),
        "fisher_path": str(fisher_path),
        "fisher_summary": fisher_summary,
    }
    save_json(task_summary, task_output_dir / "task_summary.json")
    logger.info("[%s] saved task summary: %s", task.name, task_output_dir / "task_summary.json")
    return task_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Verbose Fisher merge pipeline with VERL rollout.")
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="YAML/JSON config path for pipeline settings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and output paths without running rollout/Fisher/merge steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force regeneration of rollout artifacts. When omitted, existing rollout artifacts "
            "(prepared parquet, validation JSONL, rollout_records.parquet) are reused when available."
        ),
    )
    # Internal distributed Fisher worker mode arguments.
    parser.add_argument("--fisher-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-task-name", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-model-path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-prepared-parquet", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-correct-rollout-parquet", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-output-path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-summary-path", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-rank-output-dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-model-dtype", type=str, default="bf16", help=argparse.SUPPRESS)
    parser.add_argument("--fisher-max-prompt-tokens", type=int, default=1536, help=argparse.SUPPRESS)
    parser.add_argument("--fisher-max-response-tokens", type=int, default=512, help=argparse.SUPPRESS)
    parser.add_argument("--fisher-max-correct-trajectories", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--fisher-log-every", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--fisher-grad-checkpointing", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--fisher-seed", type=int, default=42, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Main entrypoint for Fisher merge pipeline."""

    args = parse_args()
    if bool(args.fisher_worker):
        run_fisher_worker_from_args(args)
        return

    if args.config is None:
        raise ValueError("--config is required unless --fisher-worker is used.")

    repo_root = Path(__file__).resolve().parent.parent
    pipeline_cfg = PipelineSettings.from_config_path(args.config)

    ensure_dir(pipeline_cfg.runtime.output_root)
    log_file_path = pipeline_cfg.runtime.output_root / "pipeline.log"
    logger = build_logger(log_file_path=log_file_path, verbose_console=pipeline_cfg.runtime.verbose_console)

    log_section(logger, "Initialize pipeline")
    logger.info("Config path: %s", args.config)
    logger.info("Repo root: %s", repo_root)
    logger.info("Output root: %s", pipeline_cfg.runtime.output_root)
    logger.info("Log file: %s", log_file_path)
    logger.info("Tasks: %s", [task.name for task in pipeline_cfg.tasks])
    logger.info("Force mode: %s", bool(args.force))

    # Save resolved configuration for reproducibility.
    save_json(
        {
            "created_at": now_iso(),
            "config_path": str(args.config),
            "pipeline_config": asdict(pipeline_cfg),
        },
        pipeline_cfg.runtime.output_root / "resolved_config.json",
    )
    logger.info("Saved resolved config: %s", pipeline_cfg.runtime.output_root / "resolved_config.json")

    set_seed(pipeline_cfg.runtime.seed)
    logger.info("Global seed set to: %d", pipeline_cfg.runtime.seed)

    if args.dry_run:
        log_section(logger, "Dry-run completed")
        logger.info("No rollout/Fisher/merge job was executed because --dry-run was provided.")
        return

    task_summaries: List[Dict[str, Any]] = []
    fisher_paths_by_task: Dict[str, Path] = {}

    for task in pipeline_cfg.tasks:
        summary = process_single_task(
            task=task,
            pipeline_cfg=pipeline_cfg,
            repo_root=repo_root,
            logger=logger,
            force=args.force,
        )
        task_summaries.append(summary)
        fisher_paths_by_task[task.name] = Path(summary["fisher_path"])

    merge_result = run_fisher_merge(
        pipeline_cfg=pipeline_cfg,
        fisher_paths_by_task=fisher_paths_by_task,
        output_root=pipeline_cfg.runtime.output_root,
        logger=logger,
    )

    final_summary = {
        "created_at": now_iso(),
        "config_path": str(args.config),
        "output_root": str(pipeline_cfg.runtime.output_root),
        "task_summaries": task_summaries,
        "fisher_paths_by_task": {task: str(path) for task, path in fisher_paths_by_task.items()},
        "merge_result": {
            "merged_output_dir": str(merge_result["merged_output_dir"]),
            "merge_summary": merge_result["merge_summary"],
        },
    }
    save_json(final_summary, pipeline_cfg.runtime.output_root / "pipeline_summary.json")

    log_section(logger, "Pipeline finished")
    logger.info("Final summary: %s", pipeline_cfg.runtime.output_root / "pipeline_summary.json")


if __name__ == "__main__":
    main()
