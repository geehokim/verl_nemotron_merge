# Copyright 2025 Nemotron-Cascade Implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Nemotron-Cascade Curriculum Sampler

This module implements the dynamic filtering curriculum sampler described in the
Nemotron-Cascade paper for Math RL training.

Dynamic Filtering Logic (from paper):
    1. After each epoch, filter out problems with 100% or 0% accuracy
    2. Re-sample hard problems (0% acc) with 10% probability
    3. Re-sample easy problems (100% acc) with 1% probability

This allows the model to focus on problems of appropriate difficulty while
occasionally revisiting very easy or very hard problems.

Reference:
    Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models
"""

import logging
from collections import defaultdict
from collections.abc import Sized
from typing import Iterator

import numpy as np
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

logger = logging.getLogger(__name__)


class NemotronCascadeCurriculumSampler(AbstractCurriculumSampler):
    """
    Dynamic Filtering Curriculum Sampler for Nemotron-Cascade Math RL Training.
    
    This sampler implements the curriculum learning strategy described in the
    Nemotron-Cascade paper. It tracks per-problem accuracy across rollouts and
    dynamically filters problems based on their difficulty.
    
    Key Features:
        1. Tracks accuracy for each problem across multiple rollouts (n=8 in GRPO)
        2. At epoch end, filters out problems with 100% (too easy) or 0% (too hard) accuracy
        3. Resamples filtered problems with specified probabilities:
           - Hard problems (0% acc): 10% probability of inclusion
           - Easy problems (100% acc): 1% probability of inclusion
    
    Attributes:
        hard_resample_prob (float): Probability to resample hard problems (default: 0.10)
        easy_resample_prob (float): Probability to resample easy problems (default: 0.01)
        problem_accuracy (dict): Maps problem index to list of accuracy values
        active_indices (list): List of currently active problem indices for sampling
        filtered_hard (list): List of filtered hard problem indices
        filtered_easy (list): List of filtered easy problem indices
    """
    
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        """
        Initialize the Nemotron-Cascade Curriculum Sampler.
        
        Args:
            data_source (Sized): The dataset to sample from. Must implement __len__.
            data_config (DictConfig): Configuration containing:
                - hard_resample_prob (float, optional): Probability to resample hard problems.
                    Default: 0.10 (10%)
                - easy_resample_prob (float, optional): Probability to resample easy problems.
                    Default: 0.01 (1%)
                - seed (int, optional): Random seed for reproducibility. Default: 42
        """
        self.data_source = data_source
        self.data_config = data_config
        
        # Resample probabilities from Nemotron-Cascade paper
        self.hard_resample_prob = data_config.get("hard_resample_prob", 0.10)
        self.easy_resample_prob = data_config.get("easy_resample_prob", 0.01)
        
        # Random seed for reproducibility
        self.seed = data_config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        
        # Accuracy tracking: {problem_idx: [acc_value_1, acc_value_2, ...]}
        self.problem_accuracy: dict[int, list[float]] = defaultdict(list)
        
        # Active indices management
        self.active_indices: list[int] = list(range(len(data_source)))
        self.filtered_hard: list[int] = []
        self.filtered_easy: list[int] = []
        
        # Epoch tracking
        self.current_epoch = 0
        self.samples_seen_in_epoch = 0
        self.epoch_size = len(data_source)
        
        self._logged_epoch_start = False
        
        logger.info(
            f"NemotronCascadeCurriculumSampler initialized with {len(data_source)} samples. "
            f"Hard resample prob: {self.hard_resample_prob}, Easy resample prob: {self.easy_resample_prob}"
        )
    
    def __len__(self) -> int:
        """Return the number of active samples available for sampling."""
        return len(self.active_indices)
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over shuffled active indices for one epoch."""
        indices = self.active_indices.copy()
        self.rng.shuffle(indices)
        
        if not self._logged_epoch_start:
            logger.info(
                f"Starting epoch {self.current_epoch} with {len(indices)} active samples "
                f"(filtered_hard: {len(self.filtered_hard)}, filtered_easy: {len(self.filtered_easy)})"
            )
            self._logged_epoch_start = True
        
        for idx in indices:
            yield idx
    
    def update(self, batch: DataProto) -> None:
        """
        Update accuracy tracking with results from the latest batch.
        
        This method is called after each training step. It extracts accuracy
        information from the batch and updates the per-problem accuracy tracking.
        
        Args:
            batch (DataProto): Batch containing:
                - non_tensor_batch['acc']: Accuracy values (bool or float, 0-1)
                - non_tensor_batch['index']: Problem indices
                - non_tensor_batch.get('skip', None): Optional skip flags
        """
        # Extract accuracy and index information
        acc_values = batch.non_tensor_batch.get("acc", None)
        if acc_values is None:
            logger.warning("No 'acc' field in batch.non_tensor_batch, skipping update")
            return
        
        indices = batch.non_tensor_batch.get("index", None)
        if indices is None:
            logger.warning("No 'index' field in batch, accuracy tracking may be inaccurate")
            return
        
        skip_flags = batch.non_tensor_batch.get("skip", None)
        
        # Update accuracy tracking
        batch_size = len(acc_values)
        for i in range(batch_size):
            # Skip samples marked for skipping (overlong in Stage 1)
            if skip_flags is not None and skip_flags[i]:
                continue
            
            problem_idx = int(indices[i])
            acc = float(acc_values[i]) if acc_values[i] is not None else None
            
            if acc is not None:
                self.problem_accuracy[problem_idx].append(acc)
        
        # Track epoch progress
        self.samples_seen_in_epoch += batch_size
        
        # Check if epoch completed
        if self.samples_seen_in_epoch >= self.epoch_size:
            self._on_epoch_end()
            self.samples_seen_in_epoch = 0
            self.current_epoch += 1
            self._logged_epoch_start = False
    
    def _on_epoch_end(self) -> None:
        """
        Perform dynamic filtering at the end of each epoch.
        
        1. Compute mean accuracy per problem
        2. Filter 100% (easy) and 0% (hard) accuracy problems
        3. Resample with specified probabilities
        4. Update active_indices
        """
        if not self.problem_accuracy:
            logger.info("No accuracy data collected, skipping epoch-end filtering")
            return
        
        logger.info(f"Performing epoch-end filtering for epoch {self.current_epoch}")
        
        # Compute mean accuracy for each problem
        problem_mean_acc: dict[int, float] = {}
        for problem_idx, acc_list in self.problem_accuracy.items():
            if acc_list:
                problem_mean_acc[problem_idx] = np.mean(acc_list)
        
        # Categorize problems by accuracy
        new_filtered_hard = []
        new_filtered_easy = []
        new_active = []
        
        all_indices = set(range(len(self.data_source)))
        tracked_indices = set(problem_mean_acc.keys())
        untracked_indices = all_indices - tracked_indices
        
        for problem_idx, mean_acc in problem_mean_acc.items():
            if mean_acc == 0.0:
                new_filtered_hard.append(problem_idx)
            elif mean_acc == 1.0:
                new_filtered_easy.append(problem_idx)
            else:
                new_active.append(problem_idx)
        
        # Resample filtered problems
        hard_resampled = [
            idx for idx in new_filtered_hard
            if self.rng.random() < self.hard_resample_prob
        ]
        
        easy_resampled = [
            idx for idx in new_filtered_easy
            if self.rng.random() < self.easy_resample_prob
        ]
        
        # Update active indices
        self.active_indices = (
            new_active + 
            hard_resampled + 
            easy_resampled + 
            list(untracked_indices)
        )
        
        self.filtered_hard = [idx for idx in new_filtered_hard if idx not in hard_resampled]
        self.filtered_easy = [idx for idx in new_filtered_easy if idx not in easy_resampled]
        
        logger.info(
            f"Epoch {self.current_epoch} filtering complete:\n"
            f"  - Total problems tracked: {len(problem_mean_acc)}\n"
            f"  - Hard (0% acc): {len(new_filtered_hard)} -> filtered {len(self.filtered_hard)}, "
            f"resampled {len(hard_resampled)}\n"
            f"  - Easy (100% acc): {len(new_filtered_easy)} -> filtered {len(self.filtered_easy)}, "
            f"resampled {len(easy_resampled)}\n"
            f"  - Appropriate difficulty: {len(new_active)}\n"
            f"  - Untracked (new): {len(untracked_indices)}\n"
            f"  - New active set size: {len(self.active_indices)}"
        )
        
        # Reset accuracy tracking for next epoch
        self.problem_accuracy.clear()
    
    def get_state_dict(self) -> dict:
        """Get the sampler state for checkpointing."""
        return {
            "active_indices": self.active_indices.copy(),
            "filtered_hard": self.filtered_hard.copy(),
            "filtered_easy": self.filtered_easy.copy(),
            "current_epoch": self.current_epoch,
            "samples_seen_in_epoch": self.samples_seen_in_epoch,
            "problem_accuracy": dict(self.problem_accuracy),
            "rng_state": self.rng.bit_generator.state,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load sampler state from a checkpoint."""
        self.active_indices = state_dict["active_indices"].copy()
        self.filtered_hard = state_dict.get("filtered_hard", []).copy()
        self.filtered_easy = state_dict.get("filtered_easy", []).copy()
        self.current_epoch = state_dict["current_epoch"]
        self.samples_seen_in_epoch = state_dict.get("samples_seen_in_epoch", 0)
        self.problem_accuracy = defaultdict(list, state_dict.get("problem_accuracy", {}))
        
        if "rng_state" in state_dict:
            self.rng.bit_generator.state = state_dict["rng_state"]
        
        logger.info(
            f"Loaded sampler state: epoch {self.current_epoch}, "
            f"{len(self.active_indices)} active samples"
        )

