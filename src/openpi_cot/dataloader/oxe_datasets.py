"""OXE dataset implementations for CoT RLDS datasets."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.base_dataset import _SingleCoTDataset
from openpi_cot.dataloader.dataset_utils import gather_with_padding
from openpi_cot.dataloader.dataset_utils import print_memory_usage
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import convert_state_encoding
from openpi_cot.dataloader.helpers import state_encoding_to_type
from openpi_cot.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class _SingleOXECoTDataset(_SingleCoTDataset):
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
        skip_normalization: bool = False,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        self.use_json_actions = False

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
            skip_normalization=skip_normalization,
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

            # traj["action"] = convert_action_encoding(
            #     action=traj["action"],
            #     from_encoding=self.action_encoding,
            #     to_encoding=self.config.action_encoding,
            #     to_delta_cartesian_pose=False,
            # )

            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
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
                # new_obs["state"] = convert_state_encoding(
                #     new_obs["state"],
                #     from_encoding=self.state_encoding,
                #     to_encoding=self.config.state_encoding,
                # )
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
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(self.is_bimanual)),
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
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


class DobbeCoTDataset(_SingleOXECoTDataset):
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


class PlanningDataset(_SingleOXECoTDataset):
    """Dataset for planning tasks loaded from HDF5 via TFDS.

    The planning dataset contains:
    - Images: base_image (84x84x3), wrist_image (84x84x3)
    - State: 10D [arm_pos(3), arm_r6(6), gripper_pos(1)]
    - Actions: 10D action vector
    - Language: Fixed instruction per demo
    """

    def apply_traj_transforms(
        self,
        action_horizon: int,
        summation_steps: int = 30,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """
        Compare to original transforms, we omit the following:
        - skip_unlabeled
        - max_action
        - max_proprio
        - goal_relabeling
        - drop_goal_or_instruction
        - subsample_length
        """
        if not self.skip_normalization and not self.vis_dataset:
            self.dataset = self.dataset.traj_map(
                NormalizeActionAndProprio(
                    norm_stats=self.dataset_statistics,
                    normalization_type=self.action_proprio_normalization_type,
                    action_key=action_key,
                    state_key=state_key,
                ),
                self.num_parallel_calls,
            )

        def pad_action_state(traj):
            # Pad actions to action_dim (only if not already padded)
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(traj[action_key], [[0, 0], [0, pad_amount_action]])
            # Ensure static shape is preserved
            traj[action_key].set_shape([None, self.action_dim])

            # Pad state to action_dim (only if not already padded)
            state_last_dim = tf.shape(traj["observation"][state_key])[-1]
            pad_amount_state = tf.maximum(0, self.action_dim - state_last_dim)
            traj["observation"][state_key] = tf.pad(
                traj["observation"][state_key],
                [[0, 0], [0, pad_amount_state]],
            )
            # Ensure static shape is preserved
            traj["observation"][state_key].set_shape([None, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks with proper zero-padding."""
            traj_len = tf.shape(traj[action_key])[0]

            # Use unified gather function with proper zero-padding
            traj[action_key] = gather_with_padding(
                data=traj[action_key],
                sequence_length=traj_len,
                window_size=action_horizon,
            )
            # Ensure static shape is preserved: [T, action_horizon, action_dim]
            traj[action_key].set_shape([None, action_horizon, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)

        def expand_image_dim(traj):
            """Add prediction frame pairs and corresponding language actions.

            Derives prediction language actions from raw_action (same as language_actions),
            padded to summation_steps for consistency.
            """
            # Backward compatibility: add time dimension with single frame
            traj["observation"][self.spec.primary_image_key] = tf.expand_dims(
                traj["observation"][self.spec.primary_image_key], axis=1
            )  # [T, 1, H, W, C]

            traj["observation"][self.spec.wrist_image_key] = tf.expand_dims(
                traj["observation"][self.spec.wrist_image_key], axis=1
            )  # [T, 1, H, W, C]

            # Handle right wrist (for all datasets - bimanual and non-bimanual)
            if self.spec.wrist_image_right_key in traj["observation"]:
                traj["observation"][self.spec.wrist_image_right_key] = tf.expand_dims(
                    traj["observation"][self.spec.wrist_image_right_key], axis=1
                )  # [T, 1, H, W, C]

            return traj

        self.dataset = self.dataset.traj_map(expand_image_dim, self.num_parallel_calls)


class PlanningCoTDataset(_SingleOXECoTDataset):
    def apply_traj_transforms(
        self,
        action_horizon: int,
        summation_steps: int = 30,
        action_key: str = "actions",
        state_key: str = "state",
    ):
        """
        Compare to original transforms, we omit the following:
        - skip_unlabeled
        - max_action
        - max_proprio
        - goal_relabeling
        - drop_goal_or_instruction
        - subsample_length
        """
        if not self.skip_normalization and not self.vis_dataset:
            self.dataset = self.dataset.traj_map(
                NormalizeActionAndProprio(
                    norm_stats=self.dataset_statistics,
                    normalization_type=self.action_proprio_normalization_type,
                    action_key=action_key,
                    state_key=state_key,
                ),
                self.num_parallel_calls,
            )

        def pad_action_state(traj):
            # Pad actions to action_dim (only if not already padded)
            action_last_dim = tf.shape(traj[action_key])[-1]
            pad_amount_action = tf.maximum(0, self.action_dim - action_last_dim)
            traj[action_key] = tf.pad(traj[action_key], [[0, 0], [0, pad_amount_action]])
            # Ensure static shape is preserved
            traj[action_key].set_shape([None, self.action_dim])

            # Pad state to action_dim (only if not already padded)
            state_last_dim = tf.shape(traj["observation"][state_key])[-1]
            pad_amount_state = tf.maximum(0, self.action_dim - state_last_dim)
            traj["observation"][state_key] = tf.pad(
                traj["observation"][state_key],
                [[0, 0], [0, pad_amount_state]],
            )
            # Ensure static shape is preserved
            traj["observation"][state_key].set_shape([None, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(pad_action_state, self.num_parallel_calls)

        def chunk_actions(traj):
            """Splits episode into action chunks with proper zero-padding."""
            traj_len = tf.shape(traj[action_key])[0]

            # Use unified gather function with proper zero-padding
            traj[action_key] = gather_with_padding(
                data=traj[action_key],
                sequence_length=traj_len,
                window_size=action_horizon,
            )
            # Ensure static shape is preserved: [T, action_horizon, action_dim]
            traj[action_key].set_shape([None, action_horizon, self.action_dim])
            return traj

        self.dataset = self.dataset.traj_map(chunk_actions, self.num_parallel_calls)

        def expand_image_dim(traj):
            """Add prediction frame pairs and corresponding language actions.

            Derives prediction language actions from raw_action (same as language_actions),
            padded to summation_steps for consistency.
            """
            # Backward compatibility: add time dimension with single frame
            traj["observation"][self.spec.primary_image_key] = tf.expand_dims(
                traj["observation"][self.spec.primary_image_key], axis=1
            )  # [T, 1, H, W, C]

            traj["observation"][self.spec.wrist_image_key] = tf.expand_dims(
                traj["observation"][self.spec.wrist_image_key], axis=1
            )  # [T, 1, H, W, C]

            # Handle right wrist (for all datasets - bimanual and non-bimanual)
            if self.spec.wrist_image_right_key in traj["observation"]:
                traj["observation"][self.spec.wrist_image_right_key] = tf.expand_dims(
                    traj["observation"][self.spec.wrist_image_right_key], axis=1
                )  # [T, 1, H, W, C]

            return traj

        self.dataset = self.dataset.traj_map(expand_image_dim, self.num_parallel_calls)

    def apply_restructure(self):
        def restructure(traj):
            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            # traj["action"] = convert_action_encoding(
            #     action=traj["action"],
            #     from_encoding=self.action_encoding,
            #     to_encoding=self.config.action_encoding,
            #     to_delta_cartesian_pose=False,
            # )

            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
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
                # new_obs["state"] = convert_state_encoding(
                #     new_obs["state"],
                #     from_encoding=self.state_encoding,
                #     to_encoding=self.config.state_encoding,
                # )
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
                "language_actions": traj["predicate"],
                "actions": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],
                "raw_action": tf.cast(traj["action"], tf.float32),
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(self.is_bimanual)),
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
            }

            return traj

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)


class LiberoCoTDataset(_SingleOXECoTDataset):
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
            new_obs["state"] = convert_state_encoding(
                new_obs["state"],
                from_encoding=self.state_encoding,
                to_encoding=self.config.state_encoding,
            )

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
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(False)),  # LIBERO is single-arm
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

    # Note: Inheriting get_traj_identifier from parent class (_SingleOXECoTDataset)
    # which uses action-based hashing for robust trajectory identification.
    # This avoids issues with missing or malformed trajectory metadata.


class SampleR1LiteCoTDataset(_SingleOXECoTDataset):
    """Custom dataset for sample_r1_lite with EEF pose lookup table."""

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str = "sample_r1_lite",
        data_dir: str,
        config: "CoTDataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
        eef_npz_path: str = None,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        seed: int = 0,
        split: str = "train",
        standalone: bool = True,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        skip_normalization: bool = False,
        enable_prediction_training: bool = False,
    ):
        # Set default EEF NPZ path if not provided
        if eef_npz_path is None:
            eef_npz_path = f"{data_dir}/{dataset_name}/1.0.0/eef_poses.npz"

        self.eef_npz_path = eef_npz_path
        self.use_json_actions = False

        # Build EEF pose lookup table before parent initialization
        self.eef_pose_table, self.episode_lookup = self.build_eef_pose_table(eef_npz_path)

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
            skip_normalization=skip_normalization,
            enable_prediction_training=enable_prediction_training,
        )

    def build_eef_pose_table(self, npz_path: str):
        """Build EEF pose lookup table from NPZ file.

        Returns:
            tuple: (eef_data dict, episode_lookup StaticHashTable)
        """
        print_memory_usage("Before loading EEF poses")

        # Load EEF poses from NPZ file
        with tf.io.gfile.GFile(npz_path, "rb") as f:
            eef_data = np.load(f, allow_pickle=True)
        episode_ids = eef_data["episode_ids"]
        episode_starts = eef_data["episode_starts"]
        episode_lengths = eef_data["episode_lengths"]
        left_eef_pose = eef_data["left_eef_pose"]
        right_eef_pose = eef_data["right_eef_pose"]

        logging.info(f"Loaded EEF poses for {len(episode_ids)} episodes from {npz_path}")

        # Convert to TensorFlow constants for efficient lookup
        left_eef_pose_tf = tf.constant(left_eef_pose, dtype=tf.float32)
        right_eef_pose_tf = tf.constant(right_eef_pose, dtype=tf.float32)
        episode_starts_tf = tf.constant(episode_starts, dtype=tf.int32)
        episode_lengths_tf = tf.constant(episode_lengths, dtype=tf.int32)

        # Create episode ID to index lookup table
        keys = tf.constant([str(eid) for eid in episode_ids], dtype=tf.string)
        values = tf.constant(list(range(len(episode_ids))), dtype=tf.int32)
        episode_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,
        )

        print_memory_usage("After building EEF pose table")

        return {
            "left_eef_pose": left_eef_pose_tf,
            "right_eef_pose": right_eef_pose_tf,
            "episode_starts": episode_starts_tf,
            "episode_lengths": episode_lengths_tf,
        }, episode_lookup

    def get_traj_identifier(self):
        """Get trajectory identifier from episode metadata."""

        def _get_traj_identifier(traj):
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)
            # Extract episode ID from metadata
            episode_id = traj["traj_metadata"]["episode_metadata"]["file_path"]
            if tf.rank(episode_id) > 0:
                episode_id = episode_id[0]
            episode_id = tf.strings.as_string(episode_id)

            traj_len = tf.shape(traj["action"])[0]
            traj["trajectory_id"] = tf.fill([traj_len], episode_id)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_restructure(self):
        """Restructure trajectory using EEF poses from lookup table."""

        def restructure(traj):
            # Get episode ID from trajectory
            episode_id = traj["trajectory_id"][0]

            # Lookup episode index
            ep_idx = self.episode_lookup.lookup(episode_id)

            # Get start and length for this episode
            start = tf.gather(self.eef_pose_table["episode_starts"], ep_idx)
            length = tf.gather(self.eef_pose_table["episode_lengths"], ep_idx)

            # Create indices for gathering poses (truncated to actual trajectory length)
            indices = tf.range(start, start + length, dtype=tf.int32)

            # Gather left and right EEF poses using indices
            left_poses = tf.gather(self.eef_pose_table["left_eef_pose"], indices)
            right_poses = tf.gather(self.eef_pose_table["right_eef_pose"], indices)

            # Get gripper states from original action (last dimension for each arm)
            # Truncate to match EEF pose length
            left_gripper = traj["observation"]["gripper_state_left"] / 100
            right_gripper = traj["observation"]["gripper_state_right"] / 100

            # Construct full state: [left_eef_pose(6), left_gripper(1), right_eef_pose(6), right_gripper(1)]
            traj["observation"]["state"] = tf.concat([left_poses, left_gripper, right_poses, right_gripper], axis=-1)

            # Apply standardization transform
            if self.standardize_fn is not None:
                traj = self.standardize_fn(traj)

            traj_len = tf.shape(traj["action"])[0]

            # Get images
            new_obs = {}
            for new, old in self.image_obs_keys.items():
                if new == "primary":
                    img_key = self.spec.primary_image_key
                elif new == "wrist_right":
                    img_key = self.spec.wrist_image_right_key
                elif new == "wrist":
                    img_key = self.spec.wrist_image_key
                else:
                    raise ValueError(f"Unknown image key: {new}")

                if old is None or old not in traj["observation"]:
                    new_obs[img_key] = tf.repeat("", traj_len)
                else:
                    new_obs[img_key] = traj["observation"][old]

            new_obs["state"] = traj["observation"]["state"]
            new_obs["state"] = convert_state_encoding(
                new_obs["state"],
                from_encoding=self.state_encoding,
                to_encoding=self.config.state_encoding,
            )

            # Determine state type
            state_type_str = state_encoding_to_type(self.state_encoding)

            return {
                "observation": new_obs,
                "language_instruction": traj["language_instruction"],
                "actions": tf.cast(traj["action"], tf.float32),
                "dataset_name": tf.repeat(self.dataset_name, traj_len),
                "trajectory_id": traj["trajectory_id"],
                "raw_action": tf.cast(traj["action"], tf.float32),
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(True)),  # R1 Lite is bimanual
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)
