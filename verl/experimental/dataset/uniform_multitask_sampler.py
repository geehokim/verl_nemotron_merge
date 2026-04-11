# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
UniformMultiTaskSampler
=======================

A torch ``Sampler[int]`` that yields indices in **batch-balanced** order so that
every contiguous chunk of ``train_batch_size`` indices contains a roughly
uniform distribution across the distinct values of a per-sample task field
(default: ``data_source``).

Motivation
----------
When several tasks of very different sizes are concatenated (e.g. IF=109K,
Coding=15K, Math=15K), a plain ``RandomSampler`` produces batches whose task
mix follows the dataset proportions — IF dominates, Coding/Math are
under-represented. For joint multi-task RL we want every optimizer step to see
a balanced gradient signal across tasks.

How it works
------------
At ``__init__`` the sampler:
  1. Reads the task label for every example from ``dataset.dataframe[task_field]``
     and groups indices into per-task pools.
  2. Computes a per-task quota that sums to ``train_batch_size``. Default is
     floor-uniform with the remainder going to the largest task(s).
  3. Computes the per-epoch number of batches according to ``epoch_policy``:
       - ``"largest"`` (default): one full pass over the largest task — smaller
         tasks reshuffle and cycle. Most coverage of all tasks at the cost of
         oversampling small tasks within an epoch.
       - ``"smallest"``: one full pass over the smallest task — large tasks are
         only partially seen each epoch. Use with high ``total_epochs``.
       - ``"sum"``: ``sum(len)/B`` — same total work as concat-and-shuffle.

At ``__iter__`` it:
  1. Seeds an RNG from ``seed + epoch``.
  2. Shuffles each per-task pool.
  3. For ``epoch_batches`` iterations: pulls ``per_task_counts[t]`` indices
     from each task in order (refilling+reshuffling pools that run out
     mid-epoch), shuffles within the batch so the task layout inside the batch
     is also randomized, then yields the batch's indices flat.
  4. Increments ``self.epoch`` so the next epoch is reshuffled differently.

Checkpoint resume
-----------------
``state_dict`` / ``load_state_dict`` persist the epoch counter so that resuming
training continues with the next epoch's shuffle. Mid-epoch resume rolls back
to the start of the current epoch — same limitation as the existing
``NemotronCascadeCurriculumSampler`` in this repo.

Hydra config
------------
::

    +data.sampler.class_path=pkg://verl.experimental.dataset.uniform_multitask_sampler
    +data.sampler.class_name=UniformMultiTaskSampler
    +data.multitask_sampler.task_field=data_source
    +data.multitask_sampler.epoch_policy=largest
    data.dataloader_num_workers=0   # required by create_rl_sampler

Per ``verl/trainer/main_ppo.py::create_rl_sampler``, custom samplers must
operate with ``dataloader_num_workers=0`` so the sampler is the single source
of truth for ordering.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator, Sized

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler

logger = logging.getLogger(__name__)


class UniformMultiTaskSampler(AbstractSampler):
    """Yields indices grouped into batch-balanced chunks across tasks."""

    def __init__(self, data_source: Sized, data_config: DictConfig):
        super().__init__(data_source=data_source, data_config=data_config)
        self.dataset = data_source
        self.batch_size = int(data_config.train_batch_size)
        self.seed = int(data_config.get("seed", 1) or 1)

        ms_cfg = data_config.get("multitask_sampler", None)
        if ms_cfg is None:
            ms_cfg = {}
        self.task_field: str = str(ms_cfg.get("task_field", "data_source"))
        self.epoch_policy: str = str(ms_cfg.get("epoch_policy", "largest"))
        explicit_counts = ms_cfg.get("per_task_counts", None)

        labels = self._extract_task_labels(self.dataset, self.task_field)
        if len(labels) != len(self.dataset):
            raise RuntimeError(
                f"UniformMultiTaskSampler: extracted {len(labels)} labels but "
                f"dataset has {len(self.dataset)} examples"
            )

        # Build per-task index pools.
        task_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            task_to_indices[str(lbl)].append(idx)

        if not task_to_indices:
            raise ValueError("UniformMultiTaskSampler: no tasks discovered")

        # Deterministic task ordering (alphabetical) so quotas + epoch length
        # are reproducible across runs.
        self.tasks: list[str] = sorted(task_to_indices.keys())
        self.task_to_indices: dict[str, list[int]] = {
            t: task_to_indices[t] for t in self.tasks
        }
        self.num_tasks = len(self.tasks)

        # Per-task quota that sums to batch_size.
        if explicit_counts is not None:
            quota = {t: int(explicit_counts[t]) for t in self.tasks}
            if sum(quota.values()) != self.batch_size:
                raise ValueError(
                    f"explicit per_task_counts {quota} sum to {sum(quota.values())}, "
                    f"expected {self.batch_size}"
                )
            self.per_task_counts = quota
        else:
            base = self.batch_size // self.num_tasks
            rem = self.batch_size - base * self.num_tasks
            self.per_task_counts = {t: base for t in self.tasks}
            # Distribute the remainder deterministically: largest tasks first.
            # Tie-break alphabetically (already in self.tasks order).
            order = sorted(
                self.tasks,
                key=lambda t: (-len(self.task_to_indices[t]), t),
            )
            for t in order[:rem]:
                self.per_task_counts[t] += 1
        for t in self.tasks:
            if self.per_task_counts[t] <= 0:
                raise ValueError(
                    f"per_task_count for '{t}' is {self.per_task_counts[t]}; "
                    f"batch_size={self.batch_size} is too small for {self.num_tasks} tasks"
                )

        # Number of batches in one "epoch".
        self.epoch_batches = self._compute_epoch_batches(self.epoch_policy)
        if self.epoch_batches <= 0:
            raise ValueError(
                f"epoch_batches={self.epoch_batches}; check dataset sizes vs per_task_counts"
            )

        self.epoch = 0  # bumped at the end of each __iter__

        sizes = {t: len(self.task_to_indices[t]) for t in self.tasks}
        logger.info(
            "[UniformMultiTaskSampler] tasks=%s sizes=%s per_task_counts=%s "
            "batch_size=%d epoch_batches=%d epoch_policy=%s",
            self.tasks, sizes, self.per_task_counts, self.batch_size,
            self.epoch_batches, self.epoch_policy,
        )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _extract_task_labels(dataset, field: str) -> list:
        """Best-effort extraction of the per-example task field.

        ``RLHFDataset`` exposes the underlying ``datasets.Dataset`` as
        ``self.dataframe`` and supports column access by name, which is the
        fast path. Falls back to per-item indexing for any dataset that doesn't.
        """
        df = getattr(dataset, "dataframe", None)
        if df is not None:
            try:
                return list(df[field])
            except Exception:  # pragma: no cover - unusual datasets
                pass
        return [dataset[i].get(field, "unknown") for i in range(len(dataset))]

    def _compute_epoch_batches(self, policy: str) -> int:
        if policy == "largest":
            return max(
                (len(self.task_to_indices[t]) + self.per_task_counts[t] - 1)
                // self.per_task_counts[t]
                for t in self.tasks
            )
        if policy == "smallest":
            return min(
                len(self.task_to_indices[t]) // self.per_task_counts[t]
                for t in self.tasks
            )
        if policy == "sum":
            total = sum(len(self.task_to_indices[t]) for t in self.tasks)
            return total // self.batch_size
        raise ValueError(f"unknown epoch_policy: {policy}")

    # ------------------------------------------------------------------ Sampler API

    def __len__(self) -> int:
        return self.epoch_batches * self.batch_size

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)

        # Per-task pools (mutable copies that we shuffle and consume).
        pools: dict[str, list[int]] = {
            t: list(self.task_to_indices[t]) for t in self.tasks
        }
        for t in self.tasks:
            rng.shuffle(pools[t])
        cursors: dict[str, int] = {t: 0 for t in self.tasks}

        def take(t: str, n: int) -> list[int]:
            out: list[int] = []
            while len(out) < n:
                if cursors[t] >= len(pools[t]):
                    rng.shuffle(pools[t])
                    cursors[t] = 0
                end = min(cursors[t] + (n - len(out)), len(pools[t]))
                out.extend(pools[t][cursors[t]:end])
                cursors[t] = end
            return out

        for _ in range(self.epoch_batches):
            batch: list[int] = []
            for t in self.tasks:
                batch.extend(take(t, self.per_task_counts[t]))
            # Shuffle within the batch so task ordering inside a step is also
            # randomized — irrelevant for GRPO normalization (which keys on
            # uid) but avoids surprising downstream consumers that look at
            # ordering.
            rng.shuffle(batch)
            yield from batch

        self.epoch += 1

    # ------------------------------------------------------------------ checkpointing

    def state_dict(self) -> dict:
        return {"epoch": int(self.epoch)}

    def load_state_dict(self, state: dict) -> None:
        self.epoch = int(state.get("epoch", 0))
