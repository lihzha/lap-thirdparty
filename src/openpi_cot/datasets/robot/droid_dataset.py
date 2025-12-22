"""DROID-specific dataset implementation."""

import json
import logging
import os
from typing import TYPE_CHECKING

import tensorflow as tf

from openpi_cot.datasets.base_dataset import SingleCoTDataset
from openpi_cot.datasets.utils.dataset_utils import print_memory_usage
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.helpers import state_encoding_to_type

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class DroidCoTDataset(SingleCoTDataset):
    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        from openpi_cot.datasets.utils.helpers import extract_episode_path_from_file_path

        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lookup_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 2. Episode-path ↔ Episode-ID table
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.episode_id_to_path_file}", "r") as fp:
            episode_id_to_path = json.load(fp)
        episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}

        keys = tf.constant(list(episode_path_to_id.keys()), dtype=tf.string)
        values = tf.constant(list(episode_path_to_id.values()), dtype=tf.string)
        ep_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.spec.default_ep_value,
        )
        print_memory_usage("After building ep_table")
        return ep_table

    def build_filter_table(self, metadata_path):
        # Store per-trajectory ranges, not per-step flags
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.keep_ranges_file}", "r") as f:
            filter_dict = json.load(f)
        logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")
        keys_tensor = []
        values_tensor = []

        for episode_key, ranges in filter_dict.items():
            for start, end in ranges:
                for t in range(start, end):
                    frame_key = f"{episode_key}--{t}"
                    keys_tensor.append(frame_key)
                    values_tensor.append(True)
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor),
            default_value=False,
        )
        print_memory_usage("After building filter_table (per-step)")
        logging.info("Filter hash table initialized")

        return filter_table

    def get_traj_identifier(self):
        def _get_traj_identifier(traj):
            episode_id = self._episode_id_from_traj(traj, self.ep_table)
            traj["trajectory_id"] = tf.fill([tf.shape(traj["action"])[0]], episode_id)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_restructure(self):
        def restructure(traj):
            """Reformat observation and action keys, sample language instruction."""
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)

            actions = traj["action"]

            # Align lengths across modalities
            traj_len = tf.shape(actions)[0]
            episode_id = traj["trajectory_id"][0]

            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
            # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
            # Use stateless random with episode-specific seed
            random_val = tf.random.stateless_uniform(
                shape=[], seed=[self.seed, tf.strings.to_hash_bucket_fast(episode_id, 2147483647)]
            )
            exterior_img = tf.cond(
                random_val > 0.5,
                lambda: traj["observation"][self.spec.images_list[0]],
                lambda: traj["observation"][self.spec.images_list[1]],
            )

            # Determine state type from state encoding
            state_type_str = state_encoding_to_type(self.config.state_encoding)

            indices = tf.as_string(tf.range(traj_len))
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            _return_dict = {
                "actions": tf.cast(actions, tf.float32),
                "observation": {
                    self.spec.primary_image_key: exterior_img,
                    "state": tf.cast(traj["state"], tf.float32),
                },
                "prompt": instruction,
                "trajectory_id": traj["trajectory_id"],
                "traj_metadata": traj["traj_metadata"],
                "raw_action": tf.cast(actions, tf.float32),
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
                "is_bimanual": tf.fill([traj_len], tf.constant(False)),  # DROID is single-arm
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
                "raw_state": tf.cast(traj["state"], tf.float32),
                "passes_filter": passes_filter,
                "is_navigation": tf.fill([traj_len], tf.constant(False)),
                "has_wrist_image": tf.fill([traj_len], tf.constant(True)),
            }

            if self.use_wrist_image:
                _return_dict["observation"][self.spec.wrist_image_key] = traj["observation"]["wrist_image_left"]
                # Always add right wrist image for consistency (empty strings for DROID which is single-arm)
                # Empty strings will be decoded to zero images later, matching the decoded image shape
            else:
                _return_dict["observation"][self.spec.wrist_image_key] = tf.repeat("", traj_len)
            # _return_dict["observation"][self.spec.wrist_image_right_key] = tf.repeat("", traj_len)

            return _return_dict

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def apply_frame_filters(self):
        def filter_from_dict(frame):
            return frame["passes_filter"]

        self.dataset = self.dataset.filter(filter_from_dict)

        # Remove "passes_filter" key from output
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        self.dataset = self.dataset.map(remove_passes_filter)

        def _remove_raw_action(frame):
            frame.pop("raw_action")
            return frame

        self.dataset = self.dataset.map(_remove_raw_action)

    def apply_traj_filters(self, action_key):
        # ------------------------------------------------------------------
        # Regex helpers for robust path/id extraction
        # ------------------------------------------------------------------

        # First, filter out empty trajectories to avoid index errors
        def _non_empty(traj):
            return tf.greater(tf.shape(traj[action_key])[0], 0)

        self.dataset = self.dataset.filter(_non_empty)

        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        def _has_instruction(traj):
            instr_bytes = self.instr_table.lookup(traj["trajectory_id"][0])
            # Check both that it's not empty and has reasonable length for a serialized tensor
            return tf.logical_and(
                tf.not_equal(instr_bytes, tf.constant(b"", dtype=tf.string)),
                tf.greater(tf.strings.length(instr_bytes), 10),  # Minimum length for valid serialized tensor
            )

        # Filter out any unsuccessful trajectories -- we use the file name to check this
        # Prefer cheap regex path filter first, then id/lang checks
        self.dataset = self.dataset.filter(_path_ok)
        self.dataset = self.dataset.filter(_has_instruction)

    def apply_traj_transforms(
        self,
        action_horizon: int,
        summation_steps: int = 30,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """Apply trajectory transforms and ensure raw_state is padded to match final state dimension."""
        # Call parent's implementation which handles padding and other transforms
        super().apply_traj_transforms(
            action_horizon=action_horizon,
            summation_steps=summation_steps,
            action_key=action_key,
            state_key=state_key,
        )

        # Ensure raw_state has the same dimension as the final padded state
        def pad_raw_state(traj):
            state_dim = tf.shape(traj["observation"][state_key])[-1]
            raw_state_dim = tf.shape(traj["raw_state"])[-1]
            pad_amount = tf.maximum(0, state_dim - raw_state_dim)
            traj["raw_state"] = tf.pad(traj["raw_state"], [[0, 0], [0, pad_amount]])
            # Preserve static shape
            traj["raw_state"].set_shape([None, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(pad_raw_state, self.num_parallel_calls)

    def apply_repack_transforms(self):
        super().apply_repack_transforms()

        def _pop_keys(traj):
            traj.pop("traj_metadata")
            # traj.pop("trajectory_id")

            return traj

        self.dataset = self.dataset.traj_map(_pop_keys, self.num_parallel_calls)

    def __init__(
        self,
        *,  # Force keyword-only arguments
        data_dir: str,
        config: "CoTDataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # Validation support
        # Global seed for all dataset-related randomness
        seed: int = 0,
        split: str = "train",
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        hash_tables: dict = None,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        if num_parallel_calls == -1 or num_parallel_reads == -1:
            total_threads = len(os.sched_getaffinity(0))
            num_parallel_reads = int(total_threads * 0.3)
            num_parallel_calls = int(total_threads * 0.3)

        if hash_tables is not None:
            self.ep_table = hash_tables.get("ep_table")
            self.filter_table = hash_tables.get("filter_table")

        else:
            if self.spec.lang_action_dir_name in config.language_action_dir:
                metadata_path = config.language_action_dir.replace(
                    self.spec.lang_action_dir_name, self.spec.metadata_path_name
                )
            elif self.spec.lang_action_dir_name_base in config.language_action_dir:
                metadata_path = config.language_action_dir.replace(
                    self.spec.lang_action_dir_name_base, self.spec.metadata_path_name
                )
            else:
                raise ValueError(f"Unknown language action directory: {config.language_action_dir}")

            self.ep_table = self.build_lookup_table(metadata_path)
            self.filter_table = self.build_filter_table(metadata_path)
            if standalone:
                self.hash_tables = {
                    "ep_table": self.ep_table,
                    "filter_table": self.filter_table,
                }

        super().__init__(
            dataset_name=config.droid_dataset_name,
            data_dir=data_dir,
            config=config,
            action_dim=action_dim,
            action_horizon=action_horizon,
            action_proprio_normalization_type=action_proprio_normalization_type,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            seed=seed,
            split=split,
            standalone=standalone,
            shuffle=shuffle,
            batch_size=batch_size,
            max_samples=max_samples,
            enable_prediction_training=enable_prediction_training,
            pred_prob=pred_prob,
            primary_pred_prob=primary_pred_prob,
        )
