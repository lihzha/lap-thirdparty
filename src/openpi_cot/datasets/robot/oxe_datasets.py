"""OXE dataset implementations for CoT RLDS datasets."""

import logging
from typing import TYPE_CHECKING

import tensorflow as tf

from openpi_cot.datasets.base_dataset import SingleCoTDataset
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.helpers import state_encoding_to_type

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class SingleOXECoTDataset(SingleCoTDataset):
    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: "CoTDataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        seed: int = 0,
        split: str = "train",
        standalone: bool = True,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        super().__init__(
            dataset_name=dataset_name,
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

    def apply_restructure(self):
        def restructure(traj):
            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
                    continue
                    img_key = self.spec.wrist_image_right_key
                elif new == "wrist":
                    img_key = self.spec.wrist_image_key
                else:
                    raise ValueError(f"Unknown image key: {new}")
                # Check if key exists in observation dict
                if old is None or old not in old_obs:
                    new_obs[img_key] = tf.repeat("", traj_len)  # padding
                else:
                    new_obs[img_key] = old_obs[old]

            if self.state_obs_keys:
                # Note: instead of padding with zeros, we drop the key if it is None
                new_obs["state"] = tf.concat(
                    [tf.cast(old_obs[key], tf.float32) for key in self.state_obs_keys if key is not None],
                    axis=1,
                )

            else:
                new_obs["state"] = tf.zeros((traj_len, 0), dtype=tf.float32)  # Empty state

            # Determine state type from state encoding
            state_type_str = state_encoding_to_type(self.state_encoding)

            # Build a deterministic per-trajectory identifier using a strong hash
            # of the dataset name and the serialized action tensor. This avoids
            # relying on per-dataset metadata with inconsistent schemas.

            traj = {
                "observation": new_obs,
                "language_instruction": traj["language_instruction"],
                "actions": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],
                "raw_action": tf.cast(traj["action"], tf.float32),
                "is_bimanual": tf.fill([traj_len], tf.constant(self.is_bimanual)),
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
                "raw_state": new_obs["state"],
                "is_navigation": tf.fill([traj_len], tf.constant(False)),
                "has_wrist_image": tf.fill([traj_len], tf.constant(self.has_wrist_image)),
            }

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    def get_traj_identifier(self):
        def _get_traj_identifier(traj):
            # apply a standardization function, if provided
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)
            traj_len = tf.shape(traj["action"])[0]
            max_steps = 128
            action_for_hash = tf.cond(
                max_steps >= traj_len,
                lambda: traj["action"],
                lambda: tf.concat([traj["action"][:64], traj["action"][-64:]], axis=0),
            )
            serialized_action = tf.io.serialize_tensor(action_for_hash)
            name_tensor = tf.constant(self.dataset_name, dtype=tf.string)
            sep1 = tf.constant("::", dtype=tf.string)
            sep2 = tf.constant("-", dtype=tf.string)
            to_hash = tf.strings.join([name_tensor, sep1, serialized_action])
            hashed = tf.strings.to_hash_bucket_strong(to_hash, 2147483647, key=[self.seed, 1337])
            traj_uid = tf.strings.join([name_tensor, sep2, tf.strings.as_string(hashed)])
            traj["trajectory_id"] = tf.repeat(traj_uid, traj_len)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_traj_filters(self, action_key):
        def is_nonzero_length(traj):
            return tf.shape(traj[action_key])[0] > 0

        def has_any_instruction(traj):
            instr = traj["language_instruction"]
            instr = tf.reshape(instr, [-1])
            instr = tf.strings.strip(instr)
            return tf.reduce_any(tf.strings.length(instr) > 0)

        self.dataset = self.dataset.filter(has_any_instruction)

        self.dataset = self.dataset.filter(is_nonzero_length)

    def apply_frame_filters(self):
        """
        Optionally applied *per-dataset* transforms that happen at a frame level.
        """

        # Always drop frames with empty/whitespace-only prompts
        def _non_empty_prompt(frame: dict) -> tf.Tensor:
            p = tf.strings.strip(frame["prompt"])  # scalar tf.string after flatten
            return tf.strings.length(p) > 0

        self.dataset = self.dataset.filter(_non_empty_prompt)

    def apply_repack_transforms(self):
        super().apply_repack_transforms()

        def _pop_and_rename_keys(traj):
            # traj.pop("trajectory_id")
            traj["prompt"] = traj["language_instruction"]
            traj.pop("language_instruction")
            traj.pop("raw_action")
            return traj

        self.dataset = self.dataset.traj_map(_pop_and_rename_keys, self.num_parallel_calls)


class DobbeCoTDataset(SingleOXECoTDataset):
    """Custom dataset for dobbe with action range filtering."""

    def apply_traj_filters(self, action_key):
        """Apply trajectory filters including action range filter.

        Filters out trajectories where any action exceeds [q01, q99] bounds.
        """
        # First apply standard filters
        super().apply_traj_filters(action_key)

        min_allowed = -5
        max_allowed = 5

        def _action_within_bounds(traj):
            """Check if all actions are within [q01, q99] range."""
            actions = traj[action_key]

            # Check if any action is below q01 or above q99 (element-wise)
            below_min = tf.reduce_any(tf.less(actions, min_allowed))
            above_max = tf.reduce_any(tf.greater(actions, max_allowed))

            # Keep trajectory only if all actions are within bounds
            return tf.logical_not(tf.logical_or(below_min, above_max))

        logging.info(f"Applying action range filter for dobbe: min={min_allowed}, max={max_allowed}")
        self.dataset = self.dataset.filter(_action_within_bounds)


class NavigationCoTDataset(SingleOXECoTDataset):
    """Custom dataset for Navigation with 2D position state observations."""

    def apply_restructure(self):
        def restructure(traj):
            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
                    continue
                    img_key = self.spec.wrist_image_right_key
                elif new == "wrist":
                    img_key = self.spec.wrist_image_key
                else:
                    raise ValueError(f"Unknown image key: {new}")
                # Check if key exists in observation dict
                if old is None or old not in old_obs:
                    new_obs[img_key] = tf.repeat("", traj_len)  # padding
                else:
                    new_obs[img_key] = old_obs[old]

            if self.state_obs_keys:
                # Note: instead of padding with zeros, we drop the key if it is None
                new_obs["state"] = tf.concat(
                    [tf.cast(old_obs[key], tf.float32) for key in self.state_obs_keys if key is not None],
                    axis=1,
                )

            else:
                new_obs["state"] = tf.zeros((traj_len, 0), dtype=tf.float32)  # Empty state

            # Determine state type from state encoding
            state_type_str = state_encoding_to_type(self.state_encoding)

            # Build a deterministic per-trajectory identifier using a strong hash
            # of the dataset name and the serialized action tensor. This avoids
            # relying on per-dataset metadata with inconsistent schemas.

            traj = {
                "observation": new_obs,
                "language_instruction": traj["language_instruction"],
                "actions": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],
                "raw_action": tf.cast(traj["action"], tf.float32),
                "is_bimanual": tf.fill([traj_len], tf.constant(self.is_bimanual)),
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
                "raw_state": new_obs["state"],
                "is_navigation": tf.fill([traj_len], tf.constant(True)),
                "has_wrist_image": tf.fill([traj_len], tf.constant(False)),
            }

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)


class LiberoCoTDataset(SingleOXECoTDataset):
    """Custom dataset for LIBERO with EEF state observations."""

    def apply_restructure(self):
        """Restructure LIBERO dataset to match expected format."""

        def restructure(traj):
            # Extract required fields
            # NOTE: standardize_fn is already applied in get_traj_identifier() before trajectory_id is added
            # Calling it again here can cause trajectory_id to be lost due to TF graph optimization/caching
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            # Map image keys from LIBERO dataset format to expected format
            # LIBERO uses: image (256x256x3), wrist_image (256x256x3)
            # Expected format: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
            new_obs[self.spec.primary_image_key] = old_obs.get("image")
            new_obs[self.spec.wrist_image_key] = old_obs.get("wrist_image")
            # LIBERO doesn't have right wrist camera
            new_obs[self.spec.wrist_image_right_key] = tf.repeat("", traj_len)

            # State is already in the correct format (8D: [EEF pose (6D), gripper (2D)])
            new_obs["state"] = tf.cast(old_obs["state"], tf.float32)
            # new_obs["state"] = convert_state_encoding(
            #     new_obs["state"],
            #     from_encoding=self.state_encoding,
            #     to_encoding=self.config.state_encoding,
            # )

            # Actions are 7D in LIBERO dataset (delta EEF actions)
            actions = tf.cast(traj["action"], tf.float32)

            # Get language instruction
            language_instruction = traj.get("language_instruction")

            # Preserve trajectory ID from get_traj_identifier (action-based hash)
            # This ensures proper train/val splitting and avoids metadata dependency issues

            # Determine state type (LIBERO uses EEF pose representation)
            state_type_str = state_encoding_to_type(self.state_encoding)

            return {
                "observation": new_obs,
                "language_instruction": language_instruction,
                "actions": actions,
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],  # Preserve existing trajectory_id
                "raw_action": actions,
                "is_bimanual": tf.fill([traj_len], tf.constant(False)),  # LIBERO is single-arm
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
                "raw_state": new_obs["state"],
                "is_navigation": tf.fill([traj_len], tf.constant(False)),
                "has_wrist_image": tf.fill([traj_len], tf.constant(True)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    # Note: Inheriting get_traj_identifier from parent class (SingleOXECoTDataset)
    # which uses action-based hashing for robust trajectory identification.
    # This avoids issues with missing or malformed trajectory metadata.
