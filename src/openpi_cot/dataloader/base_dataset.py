"""Base dataset class for CoT RLDS datasets."""

import contextlib
import logging
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds

from openpi_cot.dataloader.dataset_utils import gather_with_padding
from openpi_cot.dataloader.dataset_utils import prepare_batched_dataset
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import StateEncoding
from openpi_cot.dataloader.oxe_utils.data_utils import load_dataset_kwargs
from openpi_cot.dataloader.specs import CoTRldsDatasetSpec
from openpi_cot.shared.adapters.normalize_adapter import check_dataset_statistics
from openpi_cot.shared.adapters.normalize_adapter import get_dataset_statistics
from openpi_cot.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class _SingleCoTDataset:
    spec: ClassVar[CoTRldsDatasetSpec] = CoTRldsDatasetSpec()

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: "CoTDataConfig",
        action_dim: int = 32,
        action_horizon: int = 16,
        action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        seed: int = 0,
        split: str = "train",
        standalone: bool = False,
        shuffle: bool = False,
        batch_size: int = 1,
        max_samples: int | None = None,
        skip_normalization: bool = False,
        enable_prediction_training: bool = False,
        pred_prob: float = 0.2,
        primary_pred_prob: float = 0.5,
    ):
        self.config = config
        self.seed = seed
        self.want_val = split == "val"
        self.dataset_name = dataset_name
        self.action_dim = action_dim
        self.vis_dataset = bool(config.vis_dataset)
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.use_wrist_image = bool(config.use_wrist_image)
        self.standalone = standalone
        self.skip_normalization = skip_normalization
        self.enable_prediction_training = enable_prediction_training
        self.pred_prob = pred_prob
        self.primary_pred_prob = primary_pred_prob
        dataset_kwargs = load_dataset_kwargs(
            dataset_name, data_dir, load_camera_views=("primary", "wrist", "wrist_right")
        )

        logging.info(f"Dataset kwargs: {dataset_kwargs}")
        self.control_frequency: int = int(dataset_kwargs["control_frequency"])  # constant for this dataset
        self.standardize_fn = dataset_kwargs["standardize_fn"]
        self.image_obs_keys = dataset_kwargs["image_obs_keys"]
        self.state_obs_keys = dataset_kwargs["state_obs_keys"]
        self.state_encoding = dataset_kwargs["state_encoding"]
        self.action_encoding = dataset_kwargs["action_encoding"]
        self.is_bimanual = dataset_kwargs.get("is_bimanual", False)

        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)

        # ------------------------------------------------------------------
        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        # ------------------------------------------------------------------
        tf.config.set_visible_devices([], "GPU")
        with contextlib.suppress(Exception):
            tf.config.set_visible_devices([], "TPU")

        # Set global seed for file-level operations (shuffle, interleave)
        # Data-level randomness uses stateless ops with explicit seeds
        tf.random.set_seed(self.seed)

        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Check if we have cached statistics
        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)

        # If no cached stats, compute them first
        if cached_stats is None or self.config.force_recompute_stats:
            logging.info(f"No cached statistics found for {dataset_name} or force recompute. Computing statistics...")
            # Build temporary dataset for stats computation
            self.dataset = self.build_dataset(self.builder)
            self.get_traj_identifier()
            self.apply_restructure()

            # Compute and save statistics
            cached_stats = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="actions",
                state_key="state",
            )
            logging.info(f"Statistics computed and saved for {dataset_name}")

        # Now rebuild dataset using cached stats path for consistent ordering
        self.dataset = self.build_dataset(self.builder)
        self.get_traj_identifier()

        # Set statistics before filtering (needed for dataset-specific filters)
        self.dataset_statistics = cached_stats

        # Apply operations in consistent order: filter -> split -> restructure
        self.apply_traj_filters(action_key="action")
        self.split_val(split_seed=seed)
        self.apply_restructure()

        # If state encoding is NONE, ensure state stats are properly padded
        if self.state_encoding == StateEncoding.NONE:
            from openpi_cot.shared.adapters.normalize_adapter import ExtendedNormStats

            # If cached stats have empty state arrays, pad them to action_dim
            if len(self.dataset_statistics["state"].mean) == 0:
                self.dataset_statistics["state"] = ExtendedNormStats(
                    mean=np.zeros(self.action_dim, dtype=np.float32),
                    std=np.ones(self.action_dim, dtype=np.float32),
                    q01=np.zeros(self.action_dim, dtype=np.float32),
                    q99=np.zeros(self.action_dim, dtype=np.float32),
                    num_transitions=self.dataset_statistics["state"].num_transitions,
                    num_trajectories=self.dataset_statistics["state"].num_trajectories,
                )

        self.apply_traj_transforms(
            action_horizon=action_horizon,
        )

        self.apply_repack_transforms()

        # self.dataset = self.dataset.shuffle(60_000, seed=self.seed)

        self.apply_flatten()

        self.apply_prediction_frame_transform()

        self.apply_frame_filters()

        if standalone:
            # Store parameters needed for creating checkpointable dataset
            self._prepare_batched_params = {
                "want_val": self.want_val,
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

            # Apply common shuffling/take/cache behavior
            self.dataset = prepare_batched_dataset(
                dataset=self.dataset,
                want_val=self.want_val,
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

    def build_dataset_builder(self, ds_name, data_dir):
        if ds_name == "fmb":
            ds_name = "fmb:1.0.0"
        if ds_name == "dobbe":
            ds_name = "dobbe:0.0.1"
        return tfds.builder(ds_name, data_dir=data_dir)

    def build_dataset(self, builder):
        opts = tf.data.Options()
        # Always use deterministic operations for reproducibility
        # File interleaving will be deterministic but still provide good mixing
        opts.experimental_deterministic = bool(self.want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True
        cpu_count = psutil.cpu_count(logical=True) or 16
        opts.experimental_threading.private_threadpool_size = int(max(16, cpu_count))
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=bool(not self.want_val),  # shuffle at file/shard level for deterministic interleaving
            num_parallel_reads=self.num_parallel_reads,
        )
        dataset = dataset.shard(jax.process_count(), jax.process_index())
        # Repeat early to increase interleaving across files/episodes
        dataset = dataset.with_options(opts)
        return dataset

    def split_val(self, split_seed):
        def _split_filter(traj):
            salt = tf.strings.as_string(split_seed)
            anchor = traj["trajectory_id"][0]
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

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

        def group_language_actions(traj):
            """Compute per-timestep summed language actions over future steps.

            For each timestep t, we sum the language actions from t to
            t + summation_steps - 1 (capped at trajectory end). We DO NOT
            chunk the language actions; after flattening, each sample will
            have a single language string aligned to its action chunk.
            """
            traj_len = tf.shape(traj[action_key])[0]

            # Trim to dataset control frequency and pad to fixed window length (summation_steps)
            # Note: self.control_frequency is a Python int constant per dataset instance
            trimmed_len = tf.minimum(tf.cast(self.control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))

            # Use unified gather function with proper zero-padding
            actions_window_trim = gather_with_padding(
                data=traj["raw_action"],
                sequence_length=traj_len,
                window_size=trimmed_len,
            )  # [T, trimmed_len, A]

            # Pad to full summation_steps if needed
            pad_len = int(summation_steps) - trimmed_len

            def _pad_numeric():
                zeros_pad = tf.zeros(
                    [tf.shape(actions_window_trim)[0], pad_len, tf.shape(actions_window_trim)[-1]],
                    dtype=actions_window_trim.dtype,
                )
                return tf.concat([actions_window_trim, zeros_pad], axis=1)

            actions_window = tf.cond(pad_len > 0, _pad_numeric, lambda: actions_window_trim)

            # Unify spec with DROID by converting per-step numeric rows to tf.string via serialization.
            # Result shape: [T, summation_steps] tf.string (each element is a serialized [A] float32 tensor)
            flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
            serialized_flat = tf.map_fn(
                lambda v: tf.io.serialize_tensor(v),
                flat_rows,
                fn_output_signature=tf.string,
            )
            traj["language_actions"] = tf.reshape(
                serialized_flat,
                [tf.shape(actions_window)[0], int(summation_steps)],
            )
            # Set static shape for TensorFlow's shape inference
            traj["language_actions"].set_shape([None, summation_steps])
            return traj

        self.dataset = self.dataset.traj_map(group_language_actions, self.num_parallel_calls)

        def add_prediction_pairs(traj):
            """Add prediction frame pairs and corresponding language actions.

            Derives prediction language actions from raw_action (same as language_actions),
            padded to summation_steps for consistency.
            """
            traj_len = tf.shape(traj[action_key])[0]

            if not self.enable_prediction_training:
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

            # Prediction mode: sample future frame deltas uniformly from [1, max_prediction_horizon]
            max_horizon = getattr(self.config, "max_prediction_horizon", summation_steps)
            max_horizon_clamped = tf.minimum(max_horizon, traj_len - 1)
            max_horizon_clamped = tf.maximum(max_horizon_clamped, 1)  # Ensure at least 1

            # Generate deterministic seed from trajectory_id for stateless random ops
            traj_id_hash = tf.strings.to_hash_bucket_fast(traj["trajectory_id"][0], 2147483647)
            seed_pair = [self.seed, traj_id_hash]

            deltas = tf.random.stateless_uniform(
                [traj_len],
                seed=seed_pair,
                minval=1,  # At least 1 step into future
                maxval=max_horizon_clamped + 1,
                dtype=tf.int32,
            )
            future_indices = tf.minimum(tf.range(traj_len, dtype=tf.int32) + deltas, traj_len - 1)

            # Stack current and future images (primary only)
            current_imgs = traj["observation"][self.spec.primary_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.primary_image_key] = tf.stack(
                [current_imgs, future_imgs], axis=1
            )  # [T, 2, H, W, C]

            current_imgs = traj["observation"][self.spec.wrist_image_key]
            future_imgs = tf.gather(current_imgs, future_indices)
            traj["observation"][self.spec.wrist_image_key] = tf.stack(
                [current_imgs, future_imgs], axis=1
            )  # [T, 2, H, W, C]

            # # Wrist image: single frame only
            # traj["observation"][self.spec.wrist_image_key] = tf.expand_dims(
            #     traj["observation"][self.spec.wrist_image_key], axis=1
            # )  # [T, 1, H, W, C]

            # Right wrist image: single frame only (for all datasets - bimanual and non-bimanual)
            if self.spec.wrist_image_right_key in traj["observation"]:
                # traj["observation"][self.spec.wrist_image_right_key] = tf.expand_dims(
                #     traj["observation"][self.spec.wrist_image_right_key], axis=1
                # )  # [T, 1, H, W, C]

                traj["observation"][self.spec.wrist_image_right_key] = tf.stack(
                    [current_imgs, future_imgs], axis=1
                )  # [T, 2, H, W, C]

            # Numeric case: Use 2D gather with variable-length windows (more efficient than tf.map_fn)
            # Use realized_deltas (not original deltas) to ensure we only gather valid actions
            # that correspond to the actual visual gap between current and future images
            deltas_clamped = tf.minimum(deltas, summation_steps)

            # Use unified gather function with per-timestep windows
            # This handles variable deltas efficiently in a batched 2D operation
            actions_window = gather_with_padding(
                data=traj["raw_action"],
                sequence_length=traj_len,
                window_size=summation_steps,  # Maximum window size
                per_timestep_windows=deltas_clamped,  # Variable window per timestep
            )  # [T, summation_steps, A]

            # Serialize each action in the 2D array
            # Reshape to [T * summation_steps, A] for efficient serialization
            flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
            serialized_flat = tf.map_fn(
                lambda v: tf.io.serialize_tensor(v),
                flat_rows,
                fn_output_signature=tf.string,
            )
            # Reshape back to [T, summation_steps]
            prediction_lang_actions = tf.reshape(
                serialized_flat,
                [traj_len, summation_steps],
            )

            traj["prediction_language_action"] = prediction_lang_actions  # [T, summation_steps]
            traj["prediction_delta"] = deltas

            return traj

        self.dataset = self.dataset.traj_map(add_prediction_pairs, self.num_parallel_calls)

    def apply_repack_transforms(self):
        raise NotImplementedError

    def get_traj_identifier(self):
        raise NotImplementedError

    def apply_restructure(self):
        raise NotImplementedError

    def apply_traj_filters(self):
        raise NotImplementedError

    def apply_frame_filters(self):
        raise NotImplementedError

    def apply_flatten(self):
        # Flatten: map from trajectory dataset to dataset of individual action chunks
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_prediction_frame_transform(self):
        """Apply prediction frame transformation after flattening.

        This method randomly samples frames based on pred_prob and converts them to prediction samples by:
        - Replacing prompt with prediction_prompt
        - Swapping image frames based on primary_pred_prob
        """
        if not self.enable_prediction_training:
            # When prediction is disabled, images are already [1, H, W, C] after add_prediction_pairs
            # No transformation needed
            def add_prediction_mask(sample):
                """Add prediction mask to the sample."""
                sample["is_prediction_sample"] = tf.constant(False, dtype=tf.bool)
                return sample
            self.dataset = self.dataset.frame_map(add_prediction_mask, num_parallel_calls=self.num_parallel_calls)

        # When prediction is enabled, randomly convert samples to prediction samples
        def convert_to_prediction_sample(sample):
            """Randomly convert samples to prediction samples based on pred_prob."""
            # Generate deterministic seed from trajectory_id for reproducibility
            traj_id_str = sample.get("trajectory_id", tf.constant("default", dtype=tf.string))
            if tf.rank(traj_id_str) > 0:
                traj_id_str = traj_id_str[0]
            traj_id_hash = tf.strings.to_hash_bucket_fast(traj_id_str, 2147483647)
            # Also include frame index if available for per-frame randomness
            frame_hash = tf.cast(tf.random.uniform([], 0, 2147483647, dtype=tf.int32), tf.int64)
            seed_pair = [self.seed + traj_id_hash, frame_hash]

            # Decide if this sample should be a prediction sample
            is_pred_sample = tf.random.stateless_uniform([], seed=seed_pair) < self.pred_prob

            # Decide which camera to use for prediction (primary vs wrist)
            use_primary = tf.random.stateless_uniform([], seed=[seed_pair[0] + 1, seed_pair[1]]) < self.primary_pred_prob

            # Get frame 0 and frame 1 for both cameras
            # After flattening, images have shape [t, H, W, C] where t=2 for prediction-enabled
            primary_frame0 = sample["observation"][self.spec.primary_image_key][0:1]  # [1, H, W, C]
            primary_frame1 = sample["observation"][self.spec.primary_image_key][1:2]  # [1, H, W, C]
            wrist_frame0 = sample["observation"][self.spec.wrist_image_key][0:1]  # [1, H, W, C]
            wrist_frame1 = sample["observation"][self.spec.wrist_image_key][1:2]  # [1, H, W, C]

            # Swap frames based on camera choice
            def use_primary_camera():
                # Use primary camera frames: primary_image = frame 0, wrist_image = frame 1
                return primary_frame0, primary_frame1

            def use_wrist_camera():
                # Use wrist camera frames: primary_image = frame 0, wrist_image = frame 1
                return wrist_frame0, wrist_frame1

            pred_primary_img, pred_wrist_img = tf.cond(use_primary, use_primary_camera, use_wrist_camera)

            # For non-prediction samples, use first frame only
            normal_primary_img = primary_frame0
            normal_wrist_img = wrist_frame0

            # Select images based on whether this is a prediction sample
            final_primary_img = tf.cond(is_pred_sample, lambda: pred_primary_img, lambda: normal_primary_img)
            final_wrist_img = tf.cond(is_pred_sample, lambda: pred_wrist_img, lambda: normal_wrist_img)

            sample["observation"][self.spec.primary_image_key] = final_primary_img
            sample["observation"][self.spec.wrist_image_key] = final_wrist_img

            # Handle right wrist image if present
            if self.spec.wrist_image_right_key in sample["observation"]:
                # For now, just use first frame for right wrist
                sample["observation"][self.spec.wrist_image_right_key] = sample["observation"][
                    self.spec.wrist_image_right_key
                ][0:1]

            # Replace prompt with prediction_prompt for prediction samples
            if "prompt" in sample:
                prediction_prompt = tf.constant(
                    getattr(self.config, "prediction_prompt", "What is the robot's movement between two frames?"),
                    dtype=tf.string,
                )
                sample["prompt"] = tf.cond(
                    is_pred_sample, lambda: prediction_prompt, lambda: sample.get("prompt", tf.constant("", dtype=tf.string))
                )

            # For prediction samples, use prediction_language_action if available
            if "prediction_language_action" in sample and "language_action" in sample:
                sample["language_action"] = tf.cond(
                    is_pred_sample,
                    lambda: sample["prediction_language_action"],
                    lambda: sample["language_action"],
                )

            # Replace control_frequency with prediction_delta for prediction samples
            if "control_frequency" in sample and "prediction_delta" in sample:
                sample["control_frequency"] = tf.cond(
                    is_pred_sample,
                    lambda: sample["prediction_delta"],
                    lambda: sample["control_frequency"],
                )

            # Add is_prediction_sample mask
            sample["is_prediction_sample"] = is_pred_sample

            return sample

        self.dataset = self.dataset.frame_map(convert_to_prediction_sample, self.num_parallel_calls)

    def __iter__(self):
        assert self.standalone, "This dataset is not standalone"
        it = self.dataset.as_numpy_iterator()
        while True:
            try:
                batch = next(it)
            except StopIteration:
                logging.info("StopIteration")
                return
            yield batch

    def create_checkpointable_iterator(self):
        """Create an iterator that can be checkpointed.

        This creates a version of the dataset without device-specific operations
        (like prefetch_to_device and with_ram_budget) that cannot be serialized.

        Returns:
            A TensorFlow iterator that can be saved/restored using tf.train.Checkpoint.

        Note:
            This iterator will be slightly slower than the normal iterator because
            it doesn't use device-specific optimizations, but it can be checkpointed.
        """
        # If standalone mode with stored params, create checkpointable version
        if hasattr(self, "_pre_batched_dataset") and hasattr(self, "_prepare_batched_params"):
            checkpointable_dataset = prepare_batched_dataset(
                dataset=self._pre_batched_dataset,
                checkpointable=True,
                **self._prepare_batched_params,
            )
            # Return iterator from underlying TensorFlow dataset, not dlimp wrapper
            # This ensures compatibility with TensorFlow's checkpoint mechanism
            if hasattr(checkpointable_dataset, "dataset"):
                return iter(checkpointable_dataset.dataset)
            return iter(checkpointable_dataset)
        # Fallback to regular dataset (non-standalone mode)
        if hasattr(self.dataset, "dataset"):
            return iter(self.dataset.dataset)
        return iter(self.dataset)

    def __len__(self):
        return self.dataset_statistics["state"].num_transitions
