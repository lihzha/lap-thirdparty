"""Multi-dataset orchestration for CoT RLDS datasets."""

import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.coco_caption_dataset import CocoCaption
from openpi_cot.dataloader.dataset_utils import prepare_batched_dataset
from openpi_cot.dataloader.droid_dataset import DroidCoTDataset
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import state_encoding_to_type
from openpi_cot.dataloader.oxe_datasets import DobbeCoTDataset
from openpi_cot.dataloader.oxe_datasets import LiberoCoTDataset
from openpi_cot.dataloader.oxe_datasets import PlanningDataset
from openpi_cot.dataloader.oxe_datasets import SampleR1LiteCoTDataset
from openpi_cot.dataloader.oxe_datasets import _SingleOXECoTDataset
from openpi_cot.dataloader.oxe_utils.data_utils import allocate_threads
from openpi_cot.dataloader.oxe_utils.data_utils import pprint_data_mixture
from openpi_cot.dataloader.oxe_utils.mixtures import OXE_NAMED_MIXTURES
from openpi_cot.dataloader.specs import CoTRldsDatasetSpec
from openpi_cot.dataloader.vqa_base import VQA_DATASET_NAMES
from openpi_cot.dataloader.vqav2_dataset import Vqav2
from openpi_cot.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class OXECoTDatasets:
    spec: ClassVar[CoTRldsDatasetSpec] = CoTRldsDatasetSpec()
    # Use centralized VQA dataset registry
    VQA_DATASETS: ClassVar[set[str]] = VQA_DATASET_NAMES

    def __init__(
        self,
        config: "CoTDataConfig",
        data_dir: str,
        action_dim: int = 32,
        action_horizon: int = 16,
        seed: int = 0,
        split: str = "train",
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        balance_weights: bool = True,  # noqa: FBT001, FBT002
        hash_tables: dict = None,
        standalone=True,
        use_global_normalization: bool = True,
        enable_prediction_training: bool = False,
    ):
        self.hash_tables = hash_tables
        self.batch_size = batch_size

        # Configure RLDS Dataset(s)
        assert config.data_mix in OXE_NAMED_MIXTURES
        mixture_spec = OXE_NAMED_MIXTURES[config.data_mix]
        self.config = config

        dataset_names = [l[0] for l in mixture_spec]
        sample_weights = [l[1] for l in mixture_spec]

        want_val = split == "val"

        # When using global normalization, assert normalization type is NORMAL
        if use_global_normalization:
            assert action_proprio_normalization_type == NormalizationType.NORMAL, (
                "Global normalization only supports NORMAL normalization type"
            )

        total_threads = len(os.sched_getaffinity(0))
        total_read_threads = int(total_threads * 0.4)
        total_transform_threads = int(total_threads * 0.4)
        logging.info(f"Total read threads, {total_read_threads}")
        logging.info(f"Total transform threads, {total_transform_threads}")
        logging.info(f"Length of sample weights: {len(sample_weights)}")

        # Allocate Threads based on Weights
        threads_per_dataset = allocate_threads(total_transform_threads, np.array(sample_weights))
        reads_per_dataset = allocate_threads(total_read_threads, np.array(sample_weights))

        logging.info("Threads per Dataset: %s", threads_per_dataset)
        logging.info("Reads per Dataset: %s", reads_per_dataset)

        datasets, dataset_sizes, all_dataset_statistics = [], [], {}
        dataset_state_encodings = {}  # Track state encoding for each dataset
        logging.info("Constructing datasets...")
        for dataset_name, threads, reads in zip(  # noqa: B905
            dataset_names,
            threads_per_dataset,
            reads_per_dataset,
        ):
            assert threads != tf.data.AUTOTUNE, "threads should not be AUTOTUNE"
            assert reads != tf.data.AUTOTUNE, "reads should not be AUTOTUNE"
            kwargs = dict(
                data_dir=data_dir,
                config=config,
                action_horizon=action_horizon,
                action_dim=action_dim,
                seed=seed,
                split=split,
                action_proprio_normalization_type=action_proprio_normalization_type,
                num_parallel_reads=threads,
                num_parallel_calls=threads,
                standalone=False,
                skip_normalization=use_global_normalization,
                enable_prediction_training=enable_prediction_training,
                pred_prob=config.pred_prob,
                primary_pred_prob=config.primary_pred_prob,
            )
            if dataset_name == "droid":
                ds = DroidCoTDataset(
                    **kwargs,
                    hash_tables=self.hash_tables,
                )
                self.hash_tables = {
                    "cam_table": ds.cam_table,
                    "lang_table": ds.lang_table,
                    "ep_table": ds.ep_table,
                    "instr_table": ds.instr_table,
                    "filter_table": ds.filter_table,
                }
            elif dataset_name == "sample_r1_lite":
                ds = SampleR1LiteCoTDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            elif dataset_name == "dobbe":
                ds = DobbeCoTDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            elif dataset_name.startswith("libero"):
                ds = LiberoCoTDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            elif dataset_name.startswith("planning"):
                ds = PlanningDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            elif dataset_name == "coco_captions":
                ds = CocoCaption(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            elif dataset_name == "vqa":
                ds = Vqav2(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            else:
                ds = _SingleOXECoTDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            # Don't restrict RAM budget on individual datasets
            # Let the final mixed dataset handle memory management
            datasets.append(ds.dataset)
            dataset_statistics = ds.dataset_statistics
            dataset_sizes.append(dataset_statistics["state"].num_transitions)
            all_dataset_statistics[dataset_name] = dataset_statistics
            dataset_state_encodings[dataset_name] = config.state_encoding  # Track state encoding

        # Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        # primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])
        primary_dataset_indices = np.array(list(range(len(sample_weights))))

        # Balance and Normalize Weights
        if balance_weights:
            sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
        unnormalized_sample_weights = sample_weights.copy()
        sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
        #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight == 1.0)
        dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())

        self.sample_weights = sample_weights
        self.unnormalized_sample_weights = unnormalized_sample_weights
        self.dataset_statistics = all_dataset_statistics
        self.dataset_length = dataset_len

        pprint_data_mixture(dataset_names, sample_weights)

        # Apply global normalization if requested
        if use_global_normalization and not config.vis_dataset:
            global_stats_dir = data_dir
            global_stats = self._compute_or_load_global_stats(
                datasets=datasets,
                dataset_names=dataset_names,
                all_dataset_statistics=all_dataset_statistics,
                dataset_state_encodings=dataset_state_encodings,
                save_dir=global_stats_dir,
                action_dim=action_dim,
            )
            logging.info("Applying global normalization with stats: %s", global_stats)

            # Apply state-type-specific normalization to each dataset BEFORE interleaving
            # This avoids tf.case/tf.cond issues entirely
            normalized_datasets = []
            for ds_name, ds in zip(dataset_names, datasets):
                if ds_name in self.VQA_DATASETS:
                    # Skip normalization for VQA datasets
                    normalized_datasets.append(ds)
                    continue
                state_enc = dataset_state_encodings[ds_name]
                state_type = state_encoding_to_type(state_enc)

                # Create normalizer for this state type
                stats = {"actions": global_stats["actions"]}
                state_key_name = f"state_{state_type}"
                if state_key_name in global_stats:
                    stats["state"] = global_stats[state_key_name]

                normalizer = NormalizeActionAndProprio(
                    norm_stats=stats,
                    normalization_type=action_proprio_normalization_type,
                    action_key="actions",
                    state_key="state",
                )

                # Apply normalizer to this dataset
                normalized_datasets.append(ds.map(normalizer, num_parallel_calls=tf.data.AUTOTUNE))

            # Interleave the normalized datasets
            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
                normalized_datasets, self.sample_weights, rerandomize_each_iteration=True, seed=seed
            )
            self.global_statistics = global_stats
        else:
            # No global normalization - just interleave the datasets
            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
                datasets, self.sample_weights, rerandomize_each_iteration=True, seed=seed
            )
            self.global_statistics = None

        # Store parameters needed for creating checkpointable dataset
        self._prepare_batched_params = {
            "want_val": want_val,
            "shuffle": shuffle,
            "shuffle_buffer_size": config.shuffle_buffer_size,
            "seed": seed,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "resize_resolution": config.resize_resolution,
            "primary_image_key": self.spec.primary_image_key,
            "wrist_image_key": self.spec.wrist_image_key,
            "wrist_image_right_key": self.spec.wrist_image_right_key,
        }

        # Store the pre-batched dataset for creating checkpointable versions
        self._pre_batched_dataset = self.dataset

        self.dataset = prepare_batched_dataset(
            dataset=self.dataset,
            want_val=want_val,
            shuffle=shuffle,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=seed,
            max_samples=max_samples,
            batch_size=batch_size,
            resize_resolution=config.resize_resolution,
            primary_image_key=self.spec.primary_image_key,
            wrist_image_key=self.spec.wrist_image_key,
            wrist_image_right_key=self.spec.wrist_image_right_key,
        )

    def _compute_or_load_global_stats(
        self,
        datasets: list[dl.DLataset],
        dataset_names: list[str],
        all_dataset_statistics: dict,
        dataset_state_encodings: dict,
        save_dir: str,
        action_dim: int,
    ) -> dict:
        """Compute or load global normalization statistics across all datasets.

        When using global normalization, we compute mean/std across all datasets
        weighted by their sample counts. Statistics are computed separately for
        each state type (joint_pos, eef_pose, none).

        Note: The statistics are padded to action_dim to match the padded tensors.
        """
        from openpi_cot.shared.adapters.normalize_adapter import ExtendedNormStats

        # # Try to load cached global stats
        # try:
        #     global_stats = load(save_dir)
        #     logging.info(f"Loaded cached global normalization stats from {save_dir}")
        #     return global_stats
        # except FileNotFoundError:
        #     logging.info("Computing global normalization statistics from scratch...")

        # Group datasets by state type
        datasets_by_state_type = {"joint_pos": [], "eef_pose": [], "none": []}
        for dataset_name in dataset_names:
            if dataset_name in self.VQA_DATASETS:
                continue  # Skip VQA datasets for action stats
            state_encoding = dataset_state_encodings[dataset_name]
            state_type = state_encoding_to_type(state_encoding)
            datasets_by_state_type[state_type].append(dataset_name)

        # Compute weighted global statistics for actions
        # Note: Action stats are shared across ALL datasets regardless of state type
        total_action_n = 0
        action_weighted_sum = np.zeros(action_dim, dtype=np.float32)

        for dataset_name, stats in all_dataset_statistics.items():
            if dataset_name in self.VQA_DATASETS:
                continue  # Skip VQA datasets for action stats
            action_n = stats["actions"].num_transitions
            total_action_n += action_n
            # Pad each dataset's mean to action_dim before accumulating
            action_mean_padded = np.pad(
                stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
            )
            action_weighted_sum += action_mean_padded * action_n

        action_global_mean = action_weighted_sum / total_action_n

        # Compute weighted variance using parallel axis theorem
        action_var_sum = np.zeros_like(action_global_mean)

        for dataset_name, stats in all_dataset_statistics.items():
            if dataset_name in self.VQA_DATASETS:
                continue  # Skip VQA datasets for action stats
            action_n = stats["actions"].num_transitions

            # Pad local stats to action_dim for comparison with global stats
            action_local_mean = np.pad(
                stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
            )
            action_local_std = np.pad(
                stats["actions"].std, (0, action_dim - len(stats["actions"].std)), mode="constant", constant_values=0.0
            )

            # var_i + (mean_i - global_mean)^2
            action_local_var = np.square(action_local_std)
            action_mean_diff_sq = np.square(action_local_mean - action_global_mean)
            action_var_sum += action_n * (action_local_var + action_mean_diff_sq)

        action_global_var = action_var_sum / total_action_n
        action_global_std = np.sqrt(action_global_var)

        # For quantiles, use conservative bounds (global min/max across all datasets)
        # Pad each dataset's quantiles to action_dim first, then compute min/max
        action_q01_padded = [
            np.pad(
                stats["actions"].q01, (0, action_dim - len(stats["actions"].q01)), mode="constant", constant_values=0
            )
            for stats in all_dataset_statistics.values()
        ]
        action_q99_padded = [
            np.pad(
                stats["actions"].q99, (0, action_dim - len(stats["actions"].q99)), mode="constant", constant_values=0
            )
            for stats in all_dataset_statistics.values()
        ]
        action_q01 = np.min(action_q01_padded, axis=0)
        action_q99 = np.max(action_q99_padded, axis=0)

        global_stats = {
            "actions": ExtendedNormStats(
                mean=action_global_mean,
                std=action_global_std,
                q01=action_q01,
                q99=action_q99,
                num_transitions=total_action_n,
                num_trajectories=sum(stats["actions"].num_trajectories for stats in all_dataset_statistics.values()),
            ),
        }

        # Compute separate state statistics for each state type. VQA datasets already skipped above.
        for state_type, ds_names in datasets_by_state_type.items():
            if not ds_names:
                continue  # Skip if no datasets of this type

            # Skip normalization for "none" state type (empty state)
            if state_type == "none":
                continue

            # Compute weighted statistics for this state type
            state_stats_subset = {name: all_dataset_statistics[name] for name in ds_names}
            total_state_n = sum(stats["state"].num_transitions for stats in state_stats_subset.values())

            if total_state_n == 0:
                continue  # Skip if no transitions

            # Initialize with action_dim size (all states will be padded to this size)
            state_weighted_sum = np.zeros(action_dim, dtype=np.float32)

            for dataset_name, stats in state_stats_subset.items():
                state_n = stats["state"].num_transitions
                # Pad state mean to action_dim before accumulating
                state_mean_padded = np.pad(
                    stats["state"].mean, (0, action_dim - len(stats["state"].mean)), mode="constant"
                )
                state_weighted_sum += state_mean_padded * state_n

            state_global_mean = state_weighted_sum / total_state_n

            # Pad global mean to action_dim
            state_global_mean = np.pad(state_global_mean, (0, action_dim - len(state_global_mean)), mode="constant")

            # Compute weighted variance
            state_var_sum = np.zeros_like(state_global_mean)

            for dataset_name, stats in state_stats_subset.items():
                state_n = stats["state"].num_transitions

                # Pad local stats to action_dim
                state_local_mean = np.pad(
                    stats["state"].mean, (0, action_dim - len(stats["state"].mean)), mode="constant"
                )
                state_local_std = np.pad(
                    stats["state"].std, (0, action_dim - len(stats["state"].std)), mode="constant", constant_values=0.0
                )

                # var_i + (mean_i - global_mean)^2
                state_local_var = np.square(state_local_std)
                state_mean_diff_sq = np.square(state_local_mean - state_global_mean)
                state_var_sum += state_n * (state_local_var + state_mean_diff_sq)

            state_global_var = state_var_sum / total_state_n
            state_global_std = np.sqrt(state_global_var)

            # For quantiles, use conservative bounds
            # Pad each dataset's quantiles to action_dim first, then compute min/max
            state_q01_padded = [
                np.pad(
                    stats["state"].q01, (0, action_dim - len(stats["state"].q01)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_q99_padded = [
                np.pad(
                    stats["state"].q99, (0, action_dim - len(stats["state"].q99)), mode="constant", constant_values=0
                )
                for stats in state_stats_subset.values()
            ]
            state_q01 = np.min(state_q01_padded, axis=0)
            state_q99 = np.max(state_q99_padded, axis=0)

            # Store with state type-specific key
            global_stats[f"state_{state_type}"] = ExtendedNormStats(
                mean=state_global_mean,
                std=state_global_std,
                q01=state_q01,
                q99=state_q99,
                num_transitions=total_state_n,
                num_trajectories=sum(stats["state"].num_trajectories for stats in state_stats_subset.values()),
            )

        # # Save global stats
        # if jax.process_index() == 0:
        #     save(save_dir, global_stats)
        #     logging.info(f"Saved global normalization stats to {save_dir}")

        return global_stats

    def __iter__(self):
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def __len__(self):
        return self.dataset_length

    @property
    def num_val_batches_per_epoch(self) -> int:
        """Compute number of batches per epoch based on dataset length and batch size."""
        import jax

        return int(
            self.global_statistics["actions"].num_transitions
            // (self.batch_size * jax.process_count())
            * self.config.val_fraction
            * 0.8  # empirically estimated ratio for filtering
        )
