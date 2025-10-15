import contextlib
from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds

from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import StateEncoding
from openpi_cot.dataloader.helpers import convert_action_encoding
from openpi_cot.dataloader.helpers import extract_episode_path_from_file_path
from openpi_cot.dataloader.helpers import state_encoding_to_type
from openpi_cot.dataloader.oxe_utils.data_utils import allocate_threads
from openpi_cot.dataloader.oxe_utils.data_utils import load_dataset_kwargs
from openpi_cot.dataloader.oxe_utils.data_utils import pprint_data_mixture
from openpi_cot.dataloader.oxe_utils.mixtures import OXE_NAMED_MIXTURES
from openpi_cot.dataloader.oxe_utils.transforms import binarize_gripper_actions
from openpi_cot.shared.adapters.normalize_adapter import check_dataset_statistics
from openpi_cot.shared.adapters.normalize_adapter import get_dataset_statistics
from openpi_cot.transforms import NormalizeActionAndProprio

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    wrist_right_key: str | None = None,
    resize_to: tuple[int, int] | None = (224, 224),
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors.
    Preserves aspect ratio, pads symmetrically, and returns the original dtype semantics
    (uint8 clamped 0-255, float32 clamped to [-1, 1]).
    """

    def _tf_resize_with_pad(image: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        # Compute resized dimensions preserving aspect ratio
        in_h = tf.shape(image)[0]
        in_w = tf.shape(image)[1]
        orig_dtype = image.dtype

        h_f = tf.cast(in_h, tf.float32)
        w_f = tf.cast(in_w, tf.float32)
        th_f = tf.cast(target_h, tf.float32)
        tw_f = tf.cast(target_w, tf.float32)

        ratio = tf.maximum(w_f / tw_f, h_f / th_f)
        resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
        resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)

        # Resize in float32
        img_f32 = tf.cast(image, tf.float32)
        resized_f32 = tf.image.resize(img_f32, [resized_h, resized_w], method=tf.image.ResizeMethod.BILINEAR)

        # Dtype-specific postprocess (python conditional on static dtype)
        if orig_dtype == tf.uint8:
            resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
            const_val = tf.constant(0, dtype=resized.dtype)
        else:
            resized = tf.clip_by_value(resized_f32, -1.0, 1.0)
            const_val = tf.constant(-1.0, dtype=resized.dtype)

        # Compute symmetric padding
        pad_h_total = target_h - resized_h
        pad_w_total = target_w - resized_w
        pad_h0 = pad_h_total // 2
        pad_h1 = pad_h_total - pad_h0
        pad_w0 = pad_w_total // 2
        pad_w1 = pad_w_total - pad_w0

        padded = tf.pad(resized, [[pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], constant_values=const_val)
        return padded

    def _decode_single(img_bytes):
        # If already numeric, cast to uint8 and return
        if img_bytes.dtype != tf.string:
            img = tf.cast(img_bytes, tf.uint8)
        else:
            # Guard against empty placeholders (e.g., padding "")
            has_data = tf.greater(tf.strings.length(img_bytes), 0)
            img = tf.cond(
                has_data,
                lambda: tf.io.decode_image(
                    img_bytes,
                    channels=3,
                    expand_animations=False,
                    dtype=tf.uint8,
                ),
                lambda: tf.zeros([1, 1, 3], dtype=tf.uint8),
            )
        # Optional resize-with-pad to ensure batching shape compatibility
        if resize_to is not None:
            h, w = resize_to
            img = _tf_resize_with_pad(img, h, w)
        return img

    def decode_with_time_dim(img_tensor):
        """Decode images that may have time dimension.

        Handles:
        - Rank 0: scalar encoded string (single image)
        - Rank 1: [T] vector of encoded strings (prediction mode with multiple frames)
        - Rank 3: [H, W, C] single decoded image
        - Rank 4: [T, H, W, C] decoded images with time dimension
        """
        rank = len(img_tensor.shape)

        if rank == 1:  # [T] - multiple encoded strings (prediction mode)
            # Decode each encoded string separately
            # Output: [T, H, W, C] after decoding
            decoded_frames = tf.map_fn(_decode_single, img_tensor, fn_output_signature=tf.uint8)
            # Set explicit shape for downstream processing
            if resize_to is not None:
                h, w = resize_to
                decoded_frames.set_shape([None, h, w, 3])
            return decoded_frames
        if rank == 4:  # [T, H, W, C] - already decoded with time dimension
            # Apply resize if needed (shouldn't normally happen in this path)
            return img_tensor
        # rank == 0 (scalar string) or rank == 3 ([H, W, C])
        # Single frame: decode if string, otherwise return as-is
        return _decode_single(img_tensor)

    def _decode_frame(traj: dict) -> dict:
        traj["observation"][primary_key] = decode_with_time_dim(traj["observation"][primary_key])
        traj["observation"][wrist_key] = decode_with_time_dim(traj["observation"][wrist_key])
        traj["observation"][wrist_right_key] = decode_with_time_dim(traj["observation"][wrist_right_key])

        return traj

    return _decode_frame


def prepare_batched_dataset(
    dataset,
    want_val,
    shuffle,
    shuffle_buffer_size,
    seed,
    max_samples,
    batch_size,
    resize_resolution,
    primary_image_key,
    wrist_image_key,
    wrist_image_right_key=None,
):
    if (not want_val) and shuffle and max_samples is None:
        dataset = dataset.repeat().shuffle(shuffle_buffer_size, seed=seed)
    if max_samples is not None:
        dataset = dataset.take(int(max_samples)).cache().repeat()

    decode_fn = make_decode_images_fn(
        primary_key=primary_image_key,
        wrist_key=wrist_image_key,
        wrist_right_key=wrist_image_right_key,
        resize_to=resize_resolution,
    )
    dataset = dataset.frame_map(decode_fn, tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    try:
        dataset = dataset.prefetch_to_device(2)
    except Exception:
        dataset = dataset.prefetch(2)
    dataset = dataset.with_ram_budget(1)
    return dataset


def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # in MB
    logging.info(f"[{label}] Memory usage: {mem:.2f} MB")


def compute_window_indices(sequence_length: tf.Tensor, window_size: int) -> tf.Tensor:
    """Return [T, window] indices for gathering sliding windows with end padding.

    Builds indices for each timestep t to gather the next `window_size` steps,
    clamped to the final index so the tail windows repeat the last element.
    """
    # Shape: [T, window]
    base = tf.broadcast_to(tf.range(window_size)[None], [sequence_length, window_size])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size])
    indices = base + offsets
    # Cap to the last valid index to repeat the final element
    return tf.minimum(indices, sequence_length - 1)


# Helper: try cardinality; fall back to counting if UNKNOWN/INFINITE
def dataset_size(ds: tf.data.Dataset) -> int:
    c = ds.cardinality().numpy()  # returns int64 or negative sentinel
    if c >= 0:
        return int(c)
    # Count explicitly (works after .filter/.flat_map, etc.)
    return int(ds.reduce(tf.constant(0, tf.int64), lambda x, _: x + 1).numpy())


@dataclass(frozen=True)
class CoTRldsDatasetSpec:
    lang_action_tfrecord_pattern: str = "tfds_language_actions-*.tfrecord.gz"
    lang_action_dir_name: str = "droid-lang-actions"
    lang_action_dir_name_base: str = "droid-base-lang-actions"
    metadata_path_name: str = "metadata"
    episode_id_to_path_file: str = "episode_id_to_path.json"
    cam2base_extrinsics_file: str = "cam2base_extrinsics.json"
    camera_serials_file: str = "camera_serials.json"
    intrinsics_file: str = "intrinsics.json"
    droid_language_annotations_file: str = "droid_language_annotations.json"
    keep_ranges_file: str = "keep_ranges_1_0_1.json"
    images_list: tuple[str, str] = ("exterior_image_1_left", "exterior_image_2_left")
    primary_image_key: str = "base_0_rgb"
    wrist_image_key: str = "left_wrist_0_rgb"
    wrist_image_right_key: str = "right_wrist_0_rgb"
    default_lang_value: bytes = field(
        default_factory=lambda: tf.io.serialize_tensor(tf.constant([], dtype=tf.string)).numpy()
    )
    default_ep_value: tf.Tensor = field(default_factory=lambda: tf.constant("", dtype=tf.string))
    fallback_instructions = tf.constant(
        [
            "Do something useful.",
            "Complete the task.",
            "Perform the task.",
            "Carry out the objective.",
            "Execute the current task.",
            "Accomplish the goal.",
            "Proceed with the task.",
            "Handle the task at hand.",
            "Continue the operation.",
            "Fulfill the task.",
            "Take meaningful steps.",
            "Demonstrate useful behavior.",
            "Act in a useful manner.",
            "Engage in productive actions.",
            "Make useful moves.",
            "Undertake useful actions.",
            "Behave purposefully.",
            "Start the activity.",
        ],
        dtype=tf.string,
    )


class SingleCoTDataset:
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
        # Global seeding for reproducibility across dataset ops
        # ------------------------------------------------------------------
        tf.random.set_seed(self.seed)
        # Configure Tensorflow with no GPU/TPU devices to avoid clobbering JAX/TPU runtime
        tf.config.set_visible_devices([], "GPU")
        with contextlib.suppress(Exception):
            tf.config.set_visible_devices([], "TPU")

        self.builder = self.build_dataset_builder(dataset_name, data_dir)
        self.dataset = self.build_dataset(self.builder)

        self.get_traj_identifier()

        cached_stats, _, _ = check_dataset_statistics(self.builder.data_dir)
        if cached_stats is not None:
            # Prefer early filtering when stats are already available to reduce downstream work.
            self.apply_traj_filters(action_key="action")
            self.split_val(split_seed=seed)
            self.apply_restructure()
            self.dataset_statistics = cached_stats

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
        else:
            # Build required fields first, compute stats on cardinality-preserving pipeline, then filter.
            self.apply_restructure()
            self.dataset_statistics = get_dataset_statistics(
                self.dataset,
                save_dir=self.builder.data_dir,
                action_key="actions",
                state_key="state",
            )
            self.apply_traj_filters(action_key="actions")
            self.split_val(split_seed=seed)

            # If state encoding is NONE, pad the empty state stats to action_dim
            if self.state_encoding == StateEncoding.NONE:
                from openpi_cot.shared.adapters.normalize_adapter import ExtendedNormStats

                # Replace empty state stats with zero-padded stats
                self.dataset_statistics["state"] = ExtendedNormStats(
                    mean=np.zeros(self.action_dim, dtype=np.float32),
                    std=np.ones(self.action_dim, dtype=np.float32),  # Std of 1 to avoid division by zero
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

        self.apply_frame_filters()

        if standalone:
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
        opts.experimental_deterministic = bool(self.want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True
        cpu_count = psutil.cpu_count(logical=True) or 16
        opts.experimental_threading.private_threadpool_size = int(max(16, cpu_count))
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=bool(self.want_val),  # shuffle at file/shard level for first-level randomness
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
            """Splits episode into action chunks using shared indexing utility."""
            traj_len = tf.shape(traj[action_key])[0]

            action_chunk_indices = compute_window_indices(traj_len, action_horizon)
            traj[action_key] = tf.gather(traj[action_key], action_chunk_indices)
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
            # First, create indices for summation (current + future steps)
            summation_indices = compute_window_indices(traj_len, summation_steps)

            # Trim to dataset control frequency and pad to fixed window length (summation_steps)
            # Note: self.control_frequency is a Python int constant per dataset instance
            trimmed_len = tf.minimum(tf.cast(self.control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))
            if self.use_json_actions:
                # Save raw language actions (shape [T]) before windowing for use in prediction
                raw_lang_actions = traj["language_actions"]

                la_window = tf.gather(traj["language_actions"], summation_indices[:, :trimmed_len])
                pad_len = summation_steps - trimmed_len

                def _pad_text():
                    pad = tf.fill([tf.shape(la_window)[0], pad_len], tf.constant("", dtype=tf.string))
                    return tf.concat([la_window, pad], axis=1)

                traj["language_actions"] = tf.cond(pad_len > 0, _pad_text, lambda: la_window)
                # Set static shape for TensorFlow's shape inference
                traj["language_actions"].set_shape([None, summation_steps])

                # Store raw for prediction (shape [T])
                traj["raw_language_actions"] = raw_lang_actions
            else:
                # Gather numeric actions for the future window up to control frequency: [T, trimmed_len, A]
                actions_window_trim = tf.gather(traj["raw_action"], summation_indices[:, :trimmed_len])
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

            deltas = tf.random.uniform(
                [traj_len],
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

            # Wrist image: single frame only
            traj["observation"][self.spec.wrist_image_key] = tf.expand_dims(
                traj["observation"][self.spec.wrist_image_key], axis=1
            )  # [T, 1, H, W, C]

            # Right wrist image: single frame only (for all datasets - bimanual and non-bimanual)
            if self.spec.wrist_image_right_key in traj["observation"]:
                traj["observation"][self.spec.wrist_image_right_key] = tf.expand_dims(
                    traj["observation"][self.spec.wrist_image_right_key], axis=1
                )  # [T, 1, H, W, C]

            # Derive prediction language actions from raw_action, similar to language_actions
            # For each timestep t with delta d, gather actions from t to t+d and pad to summation_steps
            trimmed_len = tf.minimum(tf.cast(self.control_frequency, tf.int32), tf.cast(summation_steps, tf.int32))

            if self.use_json_actions:
                # JSON case: gather from language_actions
                def gather_and_pad_json(t_idx, delta):
                    """Gather language actions from t_idx to t_idx+delta-1, pad to summation_steps."""
                    # Clamp delta to not exceed trimmed_len
                    actual_len = tf.minimum(delta, trimmed_len)
                    # Create indices [t_idx, t_idx+1, ..., t_idx+actual_len-1]
                    indices = tf.range(actual_len) + t_idx
                    indices = tf.minimum(indices, traj_len - 1)

                    # Gather language actions
                    lang_window = tf.gather(traj["raw_language_actions"], indices)

                    # Pad to summation_steps
                    pad_len = summation_steps - actual_len
                    padded = tf.cond(
                        pad_len > 0,
                        lambda: tf.concat([lang_window, tf.fill([pad_len], tf.constant("", dtype=tf.string))], axis=0),
                        lambda: lang_window[:summation_steps],
                    )
                    return padded

                prediction_lang_actions = tf.map_fn(
                    lambda x: gather_and_pad_json(x[0], x[1]),
                    (tf.range(traj_len, dtype=tf.int32), deltas),
                    fn_output_signature=tf.TensorSpec(shape=[summation_steps], dtype=tf.string),
                )
                traj.pop("raw_language_actions")  # No longer needed
            else:
                # Numeric case: gather from raw_action and serialize
                def gather_and_pad_numeric(t_idx, delta):
                    """Gather actions from t_idx to t_idx+delta-1, serialize, pad to summation_steps."""
                    # Clamp delta to not exceed trimmed_len
                    # Create indices [t_idx, t_idx+1, ..., t_idx+actual_len-1]
                    indices = tf.range(delta) + t_idx
                    indices = tf.minimum(indices, traj_len - 1)

                    # Gather raw actions: [actual_len, A]
                    actions_window = tf.gather(traj["raw_action"], indices)

                    # Serialize each action row
                    serialized = tf.map_fn(
                        lambda v: tf.io.serialize_tensor(v),
                        actions_window,
                        fn_output_signature=tf.string,
                    )

                    # Pad to summation_steps with serialized dummy tensors
                    pad_len = summation_steps - delta
                    padded = tf.cond(
                        pad_len > 0,
                        lambda: tf.concat(
                            [
                                serialized,
                                tf.tile(
                                    [
                                        tf.io.serialize_tensor(
                                            tf.zeros(tf.shape(traj["raw_action"])[-1:], dtype=traj["raw_action"].dtype)
                                        )
                                    ],
                                    [pad_len],
                                ),
                            ],
                            axis=0,
                        ),
                        lambda: serialized[:summation_steps],
                    )
                    return padded

                prediction_lang_actions = tf.map_fn(
                    lambda x: gather_and_pad_numeric(x[0], x[1]),
                    (tf.range(traj_len, dtype=tf.int32), deltas),
                    fn_output_signature=tf.TensorSpec(shape=[summation_steps], dtype=tf.string),
                )

            traj["prediction_language_action"] = prediction_lang_actions  # [T, summation_steps]
            traj["prediction_delta"] = deltas  # Store for debugging

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

    def __len__(self):
        return self.dataset_statistics["state"].num_transitions


class DroidCoTDataset(SingleCoTDataset):
    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lang_action_table(self, language_action_dir):
        # ---------------------------------------------------------------------
        # 1. Language-action table (episode_id → serialized tensor)
        # ---------------------------------------------------------------------
        print_memory_usage("Before building lang_action_table")
        features = {
            "episode_id": tf.io.FixedLenFeature([], tf.string),
            "lang_ser": tf.io.FixedLenFeature([], tf.string),
        }

        def _parse(record):
            ex = tf.io.parse_single_example(record, features)
            lang = tf.io.parse_tensor(ex["lang_ser"], out_type=tf.string)  # shape: [T+1]
            return ex["episode_id"], lang

        files = tf.io.gfile.glob(f"{language_action_dir}/{self.spec.lang_action_tfrecord_pattern}")
        ds = (
            tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
            .map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        episodes, lang_serialized = [], []
        for ep_id, lang in ds:
            episodes.append(ep_id.numpy().decode())
            lang_serialized.append(tf.io.serialize_tensor(lang).numpy())

        keys = tf.constant(episodes, dtype=tf.string)
        values = tf.constant(lang_serialized, dtype=tf.string)
        lang_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=self.spec.default_lang_value,
        )
        print_memory_usage("After building lang_table")
        return lang_table

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

    def build_cam_tables(self, metadata_path):
        # ---------------------------------------------------------------------
        # 3. Camera-index table  (episode_id → ext-cam idx)
        # ---------------------------------------------------------------------
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.cam2base_extrinsics_file}", "r") as fp:
            cam2base_extrinsics = json.load(fp)
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.camera_serials_file}", "r") as fp:
            camera_serials = json.load(fp)
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.intrinsics_file}", "r") as fp:
            intrinsics_json = json.load(fp)
        eid_to_cam_dict = {}

        for eid, extr in cam2base_extrinsics.items():
            cams = camera_serials[eid]
            camera_serial = next(k for k in extr if k.isdigit())
            serial_to_name = {v: k for k, v in cams.items()}
            if camera_serial not in serial_to_name:
                continue
            if eid not in intrinsics_json:
                continue

            calib_camera_name = serial_to_name[camera_serial]
            if calib_camera_name == "ext1_cam_serial":
                calib_image_name = self.spec.images_list[0]  # "exterior_image_1_left"
            elif calib_camera_name == "ext2_cam_serial":
                calib_image_name = self.spec.images_list[1]  # "exterior_image_2_left"
            else:
                raise ValueError(f"Unknown camera name: {calib_camera_name}")

            calib_image_idx = self.spec.images_list.index(calib_image_name)
            eid_to_cam_dict[eid] = calib_image_idx

        keys = tf.constant(list(eid_to_cam_dict.keys()), dtype=tf.string)
        values = tf.constant(list(eid_to_cam_dict.values()), dtype=tf.int32)
        cam_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,  # -1 ⇒ fallback camera
        )
        print_memory_usage("After building cam_table")

        return cam_table

    def build_instr_table(self, metadata_path):
        # ---------------------------------------------------------------------
        # 6. Language-instruction table (merged; episode_id → serialized [K])
        # ---------------------------------------------------------------------
        _instr_keys_py = []
        _instr_vals_ser = []
        with tf.io.gfile.GFile(f"{metadata_path}/{self.spec.droid_language_annotations_file}", "r") as fp:
            language_annotations = json.load(fp)
        _instr_keys_py = list(language_annotations.keys())
        for _eid in _instr_keys_py:
            _v = language_annotations[_eid]
            _arr = [
                _v.get("language_instruction1", ""),
                _v.get("language_instruction2", ""),
                _v.get("language_instruction3", ""),
            ]
            _arr = [s for s in _arr if len(s) > 0]
            if len(_arr) == 0:
                _instr_vals_ser.append(b"")
            else:
                _instr_vals_ser.append(tf.io.serialize_tensor(tf.constant(_arr, dtype=tf.string)).numpy())
        _instr_keys = tf.constant(_instr_keys_py, dtype=tf.string)
        _instr_vals = tf.constant(_instr_vals_ser, dtype=tf.string)
        _instr_default = tf.constant(b"", dtype=tf.string)
        instr_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(_instr_keys, _instr_vals),
            default_value=_instr_default,
        )
        print_memory_usage("After building instr_table")
        return instr_table

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
            actions = tf.concat(
                (
                    tf.cast(traj["observation"]["cartesian_position"], tf.float32),
                    binarize_gripper_actions(1 - traj["action_dict"]["gripper_position"], threshold=0.5),
                ),
                axis=-1,
            )
            actions = convert_action_encoding(
                action=actions,
                from_encoding=self.action_encoding,
                to_encoding=self.config.action_encoding,
                to_delta_cartesian_pose=True,
            )
            # # Pad actions to action_dim to ensure fixed shape for dataset interleaving
            # action_pad_amount = self.action_dim - tf.shape(actions)[-1]
            # actions = tf.pad(
            #     actions,
            #     [[0, 0], [0, action_pad_amount]],
            # )
            # # Set static shape for TensorFlow's shape inference
            # actions.set_shape([None, self.action_dim])

            # Align lengths across modalities
            traj_len = tf.shape(actions)[0]
            episode_id = traj["trajectory_id"][0]
            if self.use_json_actions:
                lang_bytes = self.lang_table.lookup(episode_id)
                # Check if lang_bytes is valid before parsing
                lang_tensor = tf.cond(
                    tf.not_equal(lang_bytes, tf.constant(self.spec.default_lang_value, dtype=tf.string)),
                    lambda: tf.io.parse_tensor(lang_bytes, tf.string)[:traj_len],
                    lambda: tf.fill([traj_len], tf.constant("", dtype=tf.string)),
                )

            else:
                lang_tensor = tf.fill([traj_len], tf.constant(""))
            # Sample instruction from merged table
            instr_bytes = self.instr_table.lookup(episode_id)

            def _sample_from_table():
                # Check if instr_bytes is empty (default value)
                # If empty, return empty string - these will be filtered by _has_instruction later
                return tf.cond(
                    tf.logical_and(
                        tf.not_equal(instr_bytes, tf.constant(b"", dtype=tf.string)),
                        tf.greater(tf.strings.length(instr_bytes), 10),
                    ),
                    lambda: tf.random.shuffle(tf.io.parse_tensor(instr_bytes, out_type=tf.string), seed=self.seed)[0],
                    lambda: tf.constant("", dtype=tf.string),
                )

            instruction = _sample_from_table()
            instruction_vec = tf.fill([tf.shape(actions)[0]], instruction)

            if self.use_json_actions:
                cam_idx = self.cam_table.lookup(episode_id)
                cam_images = [
                    traj["observation"][self.spec.images_list[0]],
                    traj["observation"][self.spec.images_list[1]],
                ]
                cam_images = tf.stack(cam_images, axis=0)  # shape (2, H, W, C)
                cam_idx_clamped = tf.clip_by_value(cam_idx, 0, tf.shape(cam_images)[0] - 1)
                exterior_img = tf.gather(cam_images, cam_idx_clamped)
            else:
                # # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[], seed=self.seed) > 0.5,
                    lambda: traj["observation"][self.spec.images_list[0]],
                    lambda: traj["observation"][self.spec.images_list[1]],
                )

            cartesian = tf.cast(traj["observation"]["cartesian_position"], tf.float32)
            gripper = traj["observation"]["gripper_position"]

            gripper = tf.cond(
                tf.equal(tf.rank(cartesian), tf.rank(gripper)),
                lambda: gripper,  # same rank → no change
                lambda: tf.expand_dims(gripper, axis=-1),  # add new axis if rank differs
            )

            state = tf.concat([cartesian, binarize_gripper_actions(1 - gripper, threshold=0.5)], axis=-1)
            # state = convert_state_encoding(
            #     state, from_encoding=self.state_encoding, to_encoding=self.config.state_encoding
            # )

            # Determine state type from state encoding
            state_type_str = state_encoding_to_type(self.config.state_encoding)

            # # Pad state to action_dim to ensure fixed shape for dataset interleaving
            # state_pad_amount = self.action_dim - tf.shape(state)[-1]
            # state = tf.pad(
            #     state,
            #     [[0, 0], [0, state_pad_amount]],
            # )
            # # Set static shape for TensorFlow's shape inference
            # state.set_shape([None, self.action_dim])

            _return_dict = {
                "actions": tf.cast(actions, tf.float32),
                "observation": {
                    self.spec.primary_image_key: exterior_img,
                    "state": tf.cast(state, tf.float32),
                },
                "prompt": instruction_vec,
                "language_actions": lang_tensor,
                "trajectory_id": traj["trajectory_id"],
                "traj_metadata": traj["traj_metadata"],
                "raw_action": tf.cast(actions, tf.float32),
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
                # Attach control_frequency per step for downstream windowing/summarization
                "control_frequency": tf.fill([traj_len], tf.cast(self.control_frequency, tf.int32)),
                "is_bimanual": tf.fill([traj_len], tf.constant(False)),  # DROID is single-arm
                "state_type": tf.fill([traj_len], tf.constant(state_type_str)),
            }

            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + tf.as_string(tf.range(traj_len))
            )
            passes_filter = self.filter_table.lookup(step_id)
            _return_dict["passes_filter"] = passes_filter

            if self.use_wrist_image:
                _return_dict["observation"][self.spec.wrist_image_key] = traj["observation"]["wrist_image_left"]
                # Always add right wrist image for consistency (empty strings for DROID which is single-arm)
                # Empty strings will be decoded to zero images later, matching the decoded image shape
            else:
                _return_dict["observation"][self.spec.wrist_image_key] = tf.repeat("", traj_len)
            _return_dict["observation"][self.spec.wrist_image_right_key] = tf.repeat("", traj_len)

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

        def _id_ok(traj):
            episode_id = traj["trajectory_id"][0]
            if tf.equal(episode_id, self.spec.default_ep_value):
                return tf.constant(value=False, dtype=tf.bool)
            # Look up by episode_id (NOT episode_path). Using episode_path here would filter everything out.
            lang = self.lang_table.lookup(episode_id)
            default_lang_const = tf.constant(self.spec.default_lang_value, dtype=tf.string)
            if tf.equal(lang, default_lang_const):
                return tf.constant(value=False, dtype=tf.bool)
            return tf.logical_and(
                tf.not_equal(episode_id, self.spec.default_ep_value),
                tf.not_equal(lang, default_lang_const),
            )

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
        # if not self.use_json_actions:
        self.dataset = self.dataset.filter(_id_ok)

    def apply_repack_transforms(self):
        def _pop_keys(traj):
            traj.pop("traj_metadata")
            traj.pop("trajectory_id")

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
        skip_normalization: bool = False,
        enable_prediction_training: bool = False,
    ):
        self.use_json_actions = config.use_json_actions

        if num_parallel_calls == -1 or num_parallel_reads == -1:
            total_threads = len(os.sched_getaffinity(0))
            num_parallel_reads = int(total_threads * 0.3)
            num_parallel_calls = int(total_threads * 0.3)

        if hash_tables is not None:
            self.cam_table = hash_tables.get("cam_table")
            self.lang_table = hash_tables.get("lang_table")
            self.ep_table = hash_tables.get("ep_table")
            self.instr_table = hash_tables.get("instr_table")
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

            self.cam_table = self.build_cam_tables(metadata_path)
            self.lang_table = self.build_lang_action_table(config.language_action_dir)
            self.ep_table = self.build_lookup_table(metadata_path)
            self.instr_table = self.build_instr_table(metadata_path)
            self.filter_table = self.build_filter_table(metadata_path)
            if standalone:
                self.hash_tables = {
                    "cam_table": self.cam_table,
                    "lang_table": self.lang_table,
                    "ep_table": self.ep_table,
                    "instr_table": self.instr_table,
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
            skip_normalization=skip_normalization,
            enable_prediction_training=enable_prediction_training,
        )


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
        skip_normalization: bool = False,
        enable_prediction_training: bool = False,
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
        )

    def apply_restructure(self):
        def restructure(traj):
            # extracts images, depth images and proprio from the "observation" dict
            traj_len = tf.shape(traj["action"])[0]
            old_obs = traj["observation"]
            new_obs = {}

            traj["action"] = convert_action_encoding(
                action=traj["action"],
                from_encoding=self.action_encoding,
                to_encoding=self.config.action_encoding,
                to_delta_cartesian_pose=False,
            )

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
        def _pop_and_rename_keys(traj):
            traj.pop("trajectory_id")
            traj["prompt"] = traj["language_instruction"]
            traj.pop("language_instruction")
            traj.pop("raw_action")
            return traj

        self.dataset = self.dataset.traj_map(_pop_and_rename_keys, self.num_parallel_calls)


class SampleR1LiteCoTDataset(SingleOXECoTDataset):
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
            left_gripper = traj["observation"]["gripper_state_left"]
            right_gripper = traj["observation"]["gripper_state_right"]

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


class OXECoTDatasets:
    spec: ClassVar[CoTRldsDatasetSpec] = CoTRldsDatasetSpec()

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

        # Configure RLDS Dataset(s)
        assert config.data_mix in OXE_NAMED_MIXTURES
        mixture_spec = OXE_NAMED_MIXTURES[config.data_mix]

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
            else:
                ds = SingleOXECoTDataset(
                    dataset_name=dataset_name,
                    **kwargs,
                )
            datasets.append(ds.dataset.with_ram_budget(1))
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
            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(normalized_datasets, self.sample_weights)
            self.global_statistics = global_stats
        else:
            # No global normalization - just interleave the datasets
            self.dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, self.sample_weights)
            self.global_statistics = None

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
        from openpi_cot.shared.adapters.normalize_adapter import save

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
            state_encoding = dataset_state_encodings[dataset_name]
            state_type = state_encoding_to_type(state_encoding)
            datasets_by_state_type[state_type].append(dataset_name)

        # Compute weighted global statistics for actions
        # Note: Action stats are shared across ALL datasets regardless of state type
        total_action_n = sum(stats["actions"].num_transitions for stats in all_dataset_statistics.values())
        action_weighted_sum = np.zeros_like(list(all_dataset_statistics.values())[0]["actions"].mean)

        for dataset_name, stats in all_dataset_statistics.items():
            action_n = stats["actions"].num_transitions
            action_weighted_sum += stats["actions"].mean * action_n

        action_global_mean = action_weighted_sum / total_action_n

        # Pad global mean to action_dim (padding with zeros)
        action_global_mean = np.pad(action_global_mean, (0, action_dim - len(action_global_mean)), mode="constant")

        # Compute weighted variance using parallel axis theorem
        action_var_sum = np.zeros_like(action_global_mean)

        for dataset_name, stats in all_dataset_statistics.items():
            action_n = stats["actions"].num_transitions

            # Pad local stats to action_dim for comparison with global stats
            action_local_mean = np.pad(
                stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
            )
            action_local_std = np.pad(
                stats["actions"].std, (0, action_dim - len(stats["actions"].std)), mode="constant"
            )

            # var_i + (mean_i - global_mean)^2
            action_local_var = np.square(action_local_std)
            action_mean_diff_sq = np.square(action_local_mean - action_global_mean)
            action_var_sum += action_n * (action_local_var + action_mean_diff_sq)

        action_global_var = action_var_sum / total_action_n
        action_global_std = np.sqrt(action_global_var)

        # For quantiles, use conservative bounds (global min/max across all datasets)
        action_q01 = np.min([stats["actions"].q01 for stats in all_dataset_statistics.values()], axis=0)
        action_q99 = np.max([stats["actions"].q99 for stats in all_dataset_statistics.values()], axis=0)

        # Pad quantiles to action_dim
        action_q01 = np.pad(action_q01, (0, action_dim - len(action_q01)), mode="constant", constant_values=0)
        action_q99 = np.pad(action_q99, (0, action_dim - len(action_q99)), mode="constant", constant_values=0)

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

        # Compute separate state statistics for each state type
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
                state_local_std = np.pad(stats["state"].std, (0, action_dim - len(stats["state"].std)), mode="constant")

                # var_i + (mean_i - global_mean)^2
                state_local_var = np.square(state_local_std)
                state_mean_diff_sq = np.square(state_local_mean - state_global_mean)
                state_var_sum += state_n * (state_local_var + state_mean_diff_sq)

            state_global_var = state_var_sum / total_state_n
            state_global_std = np.sqrt(state_global_var)

            # For quantiles, use conservative bounds
            state_q01 = np.min([stats["state"].q01 for stats in state_stats_subset.values()], axis=0)
            state_q99 = np.max([stats["state"].q99 for stats in state_stats_subset.values()], axis=0)

            # Pad quantiles to action_dim
            state_q01 = np.pad(state_q01, (0, action_dim - len(state_q01)), mode="constant", constant_values=0)
            state_q99 = np.pad(state_q99, (0, action_dim - len(state_q99)), mode="constant", constant_values=0)

            # Store with state type-specific key
            global_stats[f"state_{state_type}"] = ExtendedNormStats(
                mean=state_global_mean,
                std=state_global_std,
                q01=state_q01,
                q99=state_q99,
                num_transitions=total_state_n,
                num_trajectories=sum(stats["state"].num_trajectories for stats in state_stats_subset.values()),
            )

        # Save global stats
        if jax.process_index() == 0:
            save(save_dir, global_stats)
            logging.info(f"Saved global normalization stats to {save_dir}")

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
