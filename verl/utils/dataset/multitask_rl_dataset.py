# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
MultiTaskRLHFDataset
====================

Wraps multiple ``RLHFDataset`` instances — one per parquet file — and presents
them as a single flat ``torch.utils.data.Dataset``. Use this whenever you want
to train on parquets that have **incompatible nested schemas** (different
``reward_model`` / ``extra_info`` struct shapes), which is the common case for
joint multi-task RL (e.g. IF + Coding + Math).

Why a wrapper instead of letting ``RLHFDataset`` concat internally
------------------------------------------------------------------
``RLHFDataset._read_files_and_tokenize`` calls
``datasets.concatenate_datasets``, which requires aligned features across all
parquets. With nested struct columns whose fields differ per task this fails:
e.g. IF's ``reward_model.ground_truth`` is itself a struct (instruction lists)
while Math/Coding store it as a flat string. The HF datasets concatenator does
not unify those, and ``pyarrow.concat_tables(promote_options="permissive")``
also rejects struct-vs-string mismatches.

By loading each parquet into its own ``RLHFDataset`` we keep each schema
independent — there is no concatenation step at the arrow level.

Compatibility with ``UniformMultiTaskSampler``
----------------------------------------------
The sampler reads per-example task labels via ``dataset.dataframe[task_field]``.
This wrapper exposes a tiny adapter object as ``self.dataframe`` that supports
exactly that key access pattern (returning a flat list of length ``len(self)``).
"""

from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class _ColumnAdapter:
    """Duck-typed shim that mimics ``datasets.Dataset[col_name] -> list``."""

    def __init__(self, columns: dict[str, list]):
        self._columns = columns

    def __getitem__(self, key: str) -> list:
        if key not in self._columns:
            raise KeyError(
                f"_ColumnAdapter has no column '{key}'. "
                f"Available: {list(self._columns)}"
            )
        return self._columns[key]

    def __contains__(self, key: str) -> bool:
        return key in self._columns


class MultiTaskRLHFDataset(Dataset):
    """A flat ``Dataset`` that delegates to one ``RLHFDataset`` per parquet."""

    def __init__(
        self,
        data_files,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        if len(data_files) == 0:
            raise ValueError("MultiTaskRLHFDataset requires at least one parquet file")

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.data_files = list(data_files)

        # Per-parquet max_samples policy:
        #   max_samples == -1 -> unlimited per sub-dataset (use the full file)
        #   max_samples >  0  -> apply the same cap to EACH sub-dataset, so the
        #                        total is at most max_samples * num_files. This
        #                        keeps every task represented in smoke runs.
        per_sub_max = max_samples if max_samples is not None else -1

        self.sub_datasets: list[RLHFDataset] = []
        for f in self.data_files:
            logger.info("[MultiTaskRLHFDataset] loading sub-parquet: %s", f)
            sub = RLHFDataset(
                data_files=[f],
                tokenizer=tokenizer,
                processor=processor,
                config=config,
                max_samples=per_sub_max,
            )
            self.sub_datasets.append(sub)

        # Build the flat index map: global_idx -> (sub_idx, local_idx).
        # Stored as two parallel lists for cheap O(1) lookup.
        self._sub_idx: list[int] = []
        self._local_idx: list[int] = []
        self._task_labels: list[str] = []
        task_field = config.get("multitask_sampler", {}).get("task_field", "data_source") \
            if hasattr(config, "get") else "data_source"

        for sub_i, sub in enumerate(self.sub_datasets):
            n = len(sub)
            # Pull task labels from the underlying HF dataframe in one shot.
            try:
                labels = list(sub.dataframe[task_field])
            except Exception as exc:
                raise RuntimeError(
                    f"Sub-dataset {self.data_files[sub_i]} is missing task field "
                    f"'{task_field}': {exc}"
                ) from exc
            if len(labels) != n:
                raise RuntimeError(
                    f"Sub-dataset {self.data_files[sub_i]} reported len={n} but "
                    f"task_field column has {len(labels)} values"
                )
            self._sub_idx.extend([sub_i] * n)
            self._local_idx.extend(range(n))
            self._task_labels.extend(str(x) for x in labels)

        # Adapter so callers (e.g. UniformMultiTaskSampler) can read
        # ``dataset.dataframe[task_field]`` like an HF Dataset.
        self.dataframe = _ColumnAdapter({task_field: self._task_labels})

        from collections import Counter
        per_task = Counter(self._task_labels)
        logger.info(
            "[MultiTaskRLHFDataset] %d sub-datasets, total=%d, per-task counts=%s",
            len(self.sub_datasets), len(self), dict(per_task),
        )

    def __len__(self) -> int:
        return len(self._sub_idx)

    def __getitem__(self, idx: int):
        sub_i = self._sub_idx[idx]
        local_i = self._local_idx[idx]
        return self.sub_datasets[sub_i][local_i]

    # ----- checkpointing hooks (called by RayPPOTrainer for resume) -----

    def resume_dataset_state(self):
        """Forward to each sub-dataset's resume hook so caches are rehydrated."""
        for sub in self.sub_datasets:
            if hasattr(sub, "resume_dataset_state"):
                sub.resume_dataset_state()
        # Rebuild flat indices in case the sub-datasets re-tokenized to a
        # different length (e.g. filter_overlong_prompts changed).
        self._sub_idx.clear()
        self._local_idx.clear()
        self._task_labels.clear()
        task_field = "data_source"
        if hasattr(self.config, "get"):
            ms = self.config.get("multitask_sampler", None)
            if ms is not None and hasattr(ms, "get"):
                task_field = ms.get("task_field", "data_source")
        for sub_i, sub in enumerate(self.sub_datasets):
            labels = list(sub.dataframe[task_field])
            n = len(sub)
            self._sub_idx.extend([sub_i] * n)
            self._local_idx.extend(range(n))
            self._task_labels.extend(str(x) for x in labels)
        self.dataframe = _ColumnAdapter({task_field: self._task_labels})
