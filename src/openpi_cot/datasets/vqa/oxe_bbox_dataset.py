"""OXE Bounding Box dataset implementation for VQA training.

This module provides bbox dataset classes for OXE datasets (molmoact, bridge)
that loads robot trajectories with object bounding box annotations from JSONL files
and formats them as VQA samples asking "where is the <object>".
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import tensorflow as tf

from openpi_cot.datasets.utils.data_utils import load_dataset_kwargs
from openpi_cot.datasets.utils.helpers import DATASETS_REQUIRING_WRIST_ROTATION
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.specs import CoTRldsDatasetSpec
from openpi_cot.datasets.vqa.bbox_common import (
    ROBOT_BBOX_PROMPT_PARTS_OXE,
    ROBOT_DIRECTION_PROMPT_PARTS_OXE,
    bridge_key_extractor,
    build_annotated_keys_and_frame_table_v2,
    build_annotated_keys_set,
    build_frame_objects_table_v2,
    build_frame_objects_table_v2_direction,
    count_annotated_frames,
    oxe_key_extractor,
    sample_and_format_objects_tf,
    sample_prompt_tf,
)
from openpi_cot.datasets.vqa.vqa_base import VQA_DATASET_ID_MAP

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class OXEBoundingBoxDataset(ABC):
    """Base class for OXE datasets with bounding box annotations for VQA training.

    This class provides common functionality for loading OXE trajectories
    and bbox annotations from JSONL files, creating VQA samples that ask
    "where is the <object>" with bbox answers.

    Subclasses must implement:
    - get_dataset_name(): Return the dataset name string
    - get_bbox_annotations_dir_name(): Return the bbox annotations directory name
    - get_oxe_dataset_name(): Return the OXE dataset name for loading
    - get_primary_image_key(): Return the key for the primary image in observations
    - get_original_image_size(): Return (width, height) of original images for bbox normalization
    - extract_uuid_from_traj(): Extract UUID from trajectory metadata for lookup
    """

    spec: ClassVar[CoTRldsDatasetSpec] = CoTRldsDatasetSpec()

    def __init__(
        self,
        *,  # Force keyword-only arguments
        data_dir: str,
        config: "CoTDataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
        state_dim: int = 10,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
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
        directional: bool = False,
        direction_slope: float = 2.0,
    ):
        if num_parallel_calls == -1 or num_parallel_reads == -1:
            total_threads = len(os.sched_getaffinity(0))
            num_parallel_reads = int(total_threads * 0.3)
            num_parallel_calls = int(total_threads * 0.3)

        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls
        self.config = config
        self.seed = seed
        self.want_val = split == "val"
        self.directional = directional
        self.direction_slope = direction_slope
        self.direction_prob = getattr(config, "direction_prob", 0.5)
        self.dataset_name = self.get_dataset_name()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.vis_dataset = bool(config.vis_dataset)
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.use_wrist_image = False  # VQA has no wrist images
        self.standalone = standalone
        self.action_horizon = action_horizon
        self.want_full_determinism = config.want_full_determinism
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)
        self.data_dir = data_dir

        # VQA-specific settings
        self.control_frequency = 1  # Single frame, no temporal control
        
        # Check if this dataset requires wrist camera rotation
        oxe_dataset_name = self.get_oxe_dataset_name()
        self._needs_wrist_rotation = any(ds in oxe_dataset_name for ds in DATASETS_REQUIRING_WRIST_ROTATION)

        # Get dataset-specific config
        oxe_dataset_name = self.get_oxe_dataset_name()
        self.dataset_kwargs = load_dataset_kwargs(
            oxe_dataset_name, data_dir, load_camera_views=("primary", "wrist", "wrist_right")
        )
        self.standardize_fn = self.dataset_kwargs.get("standardize_fn")

        # Configure TensorFlow with no GPU/TPU devices
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        tf.random.set_seed(self.seed)

        # Build path to bbox annotations directory
        self.bbox_annotations_dir = self._get_bbox_annotations_dir(config)
        logging.info(f"[{self.dataset_name}] Loading bbox annotations from: {self.bbox_annotations_dir}")

        # Build RLDS dataset (lazy operation, should be fast)
        logging.info(f"[{self.dataset_name}] Building RLDS dataset builder...")
        self.builder = self.build_dataset_builder(oxe_dataset_name, data_dir)
        logging.info(f"[{self.dataset_name}] Building RLDS dataset...")
        self.dataset = self.build_dataset(self.builder)

        # Apply trajectory identifier (OXE-style: hash-based)
        # This processes trajectories to add trajectory_id and episode_id
        logging.info(f"[{self.dataset_name}] Computing trajectory identifiers (this may take a while for large datasets)...")
        self.get_traj_identifier()
        logging.info(f"[{self.dataset_name}] Finished computing trajectory identifiers")

        # Split train/val
        logging.info(f"[{self.dataset_name}] Splitting train/val...")
        self.split_val(split_seed=seed)
        logging.info(f"[{self.dataset_name}] Finished train/val split")

        # Apply VQA restructure for bounding box
        # This also builds the frame objects table in a single pass for efficiency
        logging.info(f"[{self.dataset_name}] Restructuring trajectories and building lookup tables...")
        self.apply_restructure()
        logging.info(f"[{self.dataset_name}] Finished restructuring")

        # Apply frame filters (only keep frames with valid bbox annotations)
        self.apply_frame_filters()

        # Create dummy statistics for compatibility
        from openpi_cot.shared.adapters.normalize_adapter import ExtendedNormStats

        num_transitions = self.get_num_transitions()
        self.dataset_statistics = {
            "actions": ExtendedNormStats(
                mean=np.zeros(self.action_dim, dtype=np.float32),
                std=np.ones(self.action_dim, dtype=np.float32),
                q01=np.zeros(self.action_dim, dtype=np.float32),
                q99=np.zeros(self.action_dim, dtype=np.float32),
                num_transitions=num_transitions,
                num_trajectories=0,
            ),
            "state": ExtendedNormStats(
                mean=np.zeros(self.state_dim, dtype=np.float32),
                std=np.ones(self.state_dim, dtype=np.float32),
                q01=np.zeros(self.state_dim, dtype=np.float32),
                q99=np.zeros(self.state_dim, dtype=np.float32),
                num_transitions=num_transitions,
                num_trajectories=0,
            ),
        }

        if standalone:
            from openpi_cot.datasets.utils.dataset_utils import prepare_batched_dataset

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
                aggressive_aug=getattr(config, "aggressive_aug", False),
                aug_wrist_image=getattr(config, "aug_wrist_image", True),
                not_rotate_wrist_prob=getattr(config, "not_rotate_wrist_prob", 0.0),
            )

    # ========== Abstract methods to be implemented by subclasses ==========

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return dataset name for metadata (e.g., 'molmoact_bbox', 'bridge_bbox')."""
        raise NotImplementedError

    @abstractmethod
    def get_bbox_annotations_dir_name(self) -> str:
        """Return the bbox annotations directory name (e.g., 'molmoact-bbox-annotations')."""
        raise NotImplementedError

    @abstractmethod
    def get_oxe_dataset_name(self) -> str:
        """Return the OXE dataset name for loading (e.g., 'molmoact_dataset', 'bridge_v2_oxe')."""
        raise NotImplementedError

    @abstractmethod
    def get_primary_image_key(self) -> str:
        """Return the key for primary image in observations (e.g., 'image', 'image_0')."""
        raise NotImplementedError

    @abstractmethod
    def get_original_image_size(self) -> tuple[int, int]:
        """Return (width, height) of original images for bbox normalization."""
        raise NotImplementedError

    # ========== Common methods ==========
    
    def get_wrist_image_key(self) -> str | None:
        """Return the key for wrist image in observations.
        
        Default implementation looks up from dataset kwargs.
        Subclasses can override to specify a different key or return None if no wrist image.
        """
        image_obs_keys = self.dataset_kwargs.get("image_obs_keys", {})
        return image_obs_keys.get("wrist")
    
    def get_frame_offset(self) -> int:
        """Return the frame offset to account for frames removed by transforms.
        
        Some datasets (like Bridge V2) have transforms that remove the first frame.
        This offset is added to frame indices when looking up bbox annotations.
        For example, if offset=1, processed frame 0 maps to JSONL frame 1.
        """
        return 0

    def use_target_only(self) -> bool:
        """Return whether to use only is_target=True objects from annotations.
        
        If True, only objects marked as target in the JSONL will be included.
        This can filter out potentially noisy annotations.
        """
        return False

    def get_key_extractor(self) -> callable:
        """Return the key extractor function for matching trajectories to JSONL annotations.
        
        Subclasses can override this to use a different key extraction strategy.
        By default, uses file_path as the key (oxe_key_extractor).
        """
        return oxe_key_extractor

    def _get_bbox_annotations_dir(self, config) -> str:
        """Build path to bbox annotations directory."""
        bbox_dir_name = self.get_bbox_annotations_dir_name()
        if self.spec.lang_action_dir_name in config.language_action_dir:
            return config.language_action_dir.replace(
                self.spec.lang_action_dir_name, bbox_dir_name
            )
        elif self.spec.lang_action_dir_name_base in config.language_action_dir:
            return config.language_action_dir.replace(
                self.spec.lang_action_dir_name_base, bbox_dir_name
            )
        else:
            # Fallback: try to construct path from parent directory
            parent_dir = os.path.dirname(config.language_action_dir.rstrip("/"))
            return os.path.join(parent_dir, bbox_dir_name)

    def build_dataset_builder(self, ds_name, data_dir):
        """Build TFDS builder for OXE dataset."""
        import tensorflow_datasets as tfds

        return tfds.builder(ds_name, data_dir=data_dir)

    def build_dataset(self, builder):
        """Build dataset from RLDS."""
        want_full_determinism = self.want_full_determinism or self.want_val
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=bool(not want_full_determinism),
            num_parallel_reads=self.num_parallel_reads,
        )
        dataset = dataset.shard(jax.process_count(), jax.process_index())
        dataset = dataset.with_options(self.get_dataset_ops())
        return dataset

    def get_dataset_ops(self):
        """Get dataset options for performance."""
        import psutil

        opts = tf.data.Options()
        want_full_determinism = self.want_full_determinism or self.want_val
        if want_full_determinism:
            opts.experimental_deterministic = True
        else:
            opts.experimental_deterministic = False
        opts.autotune.enabled = True
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_fusion = True
        opts.experimental_optimization.map_and_filter_fusion = True
        opts.experimental_optimization.inject_prefetch = False
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_warm_start = True
        opts.experimental_threading.private_threadpool_size = int(max(16, psutil.cpu_count(logical=True)))
        return opts

    def build_episode_key_tf(self, traj) -> tf.Tensor:
        """Build the episode key from trajectory metadata for bbox lookup.
        
        Subclasses can override this to use different key formats.
        By default, uses just file_path as the key.
        
        Args:
            traj: Trajectory dict containing traj_metadata
            
        Returns:
            tf.string tensor with the episode key
        """
        # Default: use file_path as the key
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        return file_path

    def get_traj_identifier(self):
        """Add trajectory_id to each trajectory using OXE-style hash-based identifier."""

        def _get_traj_identifier(traj):
            # Apply standardization function if provided
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

            # Build episode key for bbox lookup using the dataset-specific method
            episode_key = self.build_episode_key_tf(traj)
            traj["episode_id"] = tf.repeat(episode_key, traj_len)

            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def split_val(self, split_seed):
        """Split dataset into train/val."""

        def _split_filter(traj):
            salt = tf.strings.as_string(split_seed)
            anchor = traj["trajectory_id"][0]
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def apply_restructure(self):
        """Restructure trajectory data into VQA-style bbox samples."""
        # OPTIMIZATION: Build set of episode_ids with annotations and frame objects table
        # in a single pass to avoid scanning JSONL files multiple times
        key_extractor = self.get_key_extractor()
        orig_w, orig_h = self.get_original_image_size()
        target_h, target_w = self.config.resize_resolution
        
        # Build both annotated keys set and frame objects table in one pass
        # This is much faster than calling them separately
        if not self.directional:
            logging.info(f"[{self.dataset_name}] Building annotated keys set and frame objects table in single pass...")
            annotated_episode_ids, self._frame_objects_table, self._num_transitions = build_annotated_keys_and_frame_table_v2(
                bbox_annotations_dir=self.bbox_annotations_dir,
                key_extractor=key_extractor,
                dataset_name=self.dataset_name,
                orig_size=(orig_w, orig_h),
                target_size=(target_w, target_h),
                target_only=self.use_target_only(),
                direction_slope=self.direction_slope,
            )
            logging.info(f"[{self.dataset_name}] Found {len(annotated_episode_ids)} trajectories with bbox annotations")
            logging.info(f"[{self.dataset_name}] Cached frame objects table (type: {type(self._frame_objects_table).__name__})")
        else:
            # For directional, we still need to use the separate function
            # TODO: Could optimize this further by creating a combined directional version
            annotated_episode_ids = build_annotated_keys_set(
                self.bbox_annotations_dir, key_extractor
            )
            self._frame_objects_table = None  # Will be built in _build_frame_objects_table
            self._num_transitions = None  # Will be computed in get_num_transitions()
        
        # Log sample file_paths from JSONL for debugging
        if annotated_episode_ids:
            sample_paths = list(annotated_episode_ids)[:2]
            logging.info(f"[{self.dataset_name}] Sample JSONL file_paths (keys): {sample_paths}")

        # Enable trajectory-level filtering using episode_id
        # This filters out trajectories that don't have any annotated frames
        if annotated_episode_ids:
            logging.info(f"[{self.dataset_name}] Filtering trajectories to only those with annotations...")
            annotated_ids_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(list(annotated_episode_ids), dtype=tf.string),
                    tf.constant([True] * len(annotated_episode_ids), dtype=tf.bool),
                ),
                default_value=tf.constant(False, dtype=tf.bool),
            )

            def has_any_annotations(traj):
                """Filter trajectories that have at least one annotated frame."""
                episode_id = traj["episode_id"][0]  # episode_id is repeated for all frames
                return annotated_ids_table.lookup(episode_id)

            self.dataset = self.dataset.filter(has_any_annotations)
            logging.info(f"[{self.dataset_name}] Finished filtering trajectories")

        primary_image_key = self.get_primary_image_key()
        frame_offset = self.get_frame_offset()
        wrist_image_key = self.get_wrist_image_key()
        needs_wrist_rotation = self._needs_wrist_rotation

        def restructure(traj):
            """Convert trajectory to VQA bbox format."""
            traj_len = tf.shape(traj["action"])[0]

            # Get primary image from observations
            primary_img = traj["observation"][primary_image_key]

            # Always prepare wrist image if available (will be used for directional samples)
            # For non-directional samples, wrist will be set to empty in finalize_vqa
            if wrist_image_key is not None:
                wrist_img = traj["observation"].get(wrist_image_key)
                if wrist_img is None:
                    wrist_img = tf.repeat("", traj_len)
            else:
                wrist_img = tf.repeat("", traj_len)

            # Create frame indices as strings
            # Apply frame_offset to account for frames removed by transforms
            # e.g., if transform removes first frame, offset=1 so processed frame 0 -> JSONL frame 1
            frame_indices = tf.as_string(tf.range(traj_len) + frame_offset)

            return {
                "observation": {
                    self.spec.primary_image_key: primary_img,
                    self.spec.wrist_image_key: wrist_img,
                    "state": tf.zeros([traj_len, self.state_dim], dtype=tf.float32),
                },
                "trajectory_id": traj["trajectory_id"],
                "episode_id": traj["episode_id"],
                "frame_idx": frame_indices,
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
                # Track if this dataset requires wrist camera rotation
                "needs_wrist_rotation": tf.fill([traj_len], tf.constant(needs_wrist_rotation)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

        # Flatten to individual frames
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_frame_filters(self):
        """Filter and expand frames to create final VQA samples."""
        # Build a lookup table that maps episode_id--frame_idx to pipe-delimited objects
        # This should reuse the cached table from apply_restructure() if available
        logging.info(f"[{self.dataset_name}] Building/retrieving frame objects lookup table...")
        frame_to_objects = self._build_frame_objects_table()
        logging.info(f"[{self.dataset_name}] Frame objects table ready")

        # Filter frames using pure TF lookup (fast) - only keep frames with annotations
        def has_bbox_annotation(frame):
            """Fast filter using pure TF lookup."""
            lookup_key = tf.strings.join([frame["episode_id"], "--", frame["frame_idx"]])
            objects_data = frame_to_objects.lookup(lookup_key)
            return tf.strings.length(objects_data) > 0

        self.dataset = self.dataset.filter(has_bbox_annotation)

        # Sample objects and format caption using pure TensorFlow operations
        # Always sample exactly 1 object for both bbox and directional samples
        max_objects = 1
        direction_prob = self.direction_prob  # Probability of using direction caption instead of bbox

        def lookup_and_sample_objects(frame):
            """Look up objects, sample if needed, and format caption using pure TF."""
            lookup_key = tf.strings.join([frame["episode_id"], "--", frame["frame_idx"]])
            objects_data = frame_to_objects.lookup(lookup_key)

            # Create seed for reproducible sampling based on frame identity
            seed_key = tf.strings.join([frame["trajectory_id"], "_sample_", frame["frame_idx"]])
            seed_hash = tf.strings.to_hash_bucket_fast(seed_key, 2147483647)
            seed_hash_int = tf.cast(seed_hash, tf.int32)
            seed_pair = (self.seed, seed_hash_int)

            # Determine if this sample is directional (same logic as sample_and_format_objects_tf)
            dir_seed = (seed_pair[0] + 7919, seed_pair[1])
            is_directional = tf.random.stateless_uniform([], seed=dir_seed, dtype=tf.float32) < direction_prob

            # Use pure TensorFlow sampling and formatting
            # Always use sample_and_format_objects_tf which handles both bbox and directional
            labels, caption = sample_and_format_objects_tf(
                objects_data, max_objects=max_objects, seed_pair=seed_pair, direction_prob=direction_prob
            )

            frame["object_labels"] = labels
            frame["bbox_caption"] = caption
            frame["is_directional"] = is_directional

            return frame

        self.dataset = self.dataset.frame_map(lookup_and_sample_objects, num_parallel_calls=self.num_parallel_calls)

        # Filter out any invalid entries (e.g., JSON parse errors)
        def has_valid_caption(frame):
            return tf.strings.length(frame["bbox_caption"]) > 0

        self.dataset = self.dataset.filter(has_valid_caption)

        # Convert to final VQA format
        def finalize_vqa(frame):
            """Create final VQA sample with prompt and caption."""
            labels = frame["object_labels"]
            caption = frame["bbox_caption"]
            is_directional = frame["is_directional"]
            wrist_image_key = self.get_wrist_image_key()

            # Sample a prompt template using the shared helper
            seed_key = tf.strings.join([frame["trajectory_id"], "_bbox_", frame["frame_idx"]])
            seed_hash = tf.strings.to_hash_bucket_fast(seed_key, 2147483647)
            seed_hash_int = tf.cast(seed_hash, tf.int32)
            seed_pair = (self.seed, seed_hash_int)

            # Select prompt based on whether sample is directional
            # OXE bbox uses robot base frame (not end-effector) for directional
            # Call sample_prompt_tf for both cases and select the result
            prompt_directional = sample_prompt_tf(ROBOT_DIRECTION_PROMPT_PARTS_OXE, labels, seed_pair)
            prompt_bbox = sample_prompt_tf(ROBOT_BBOX_PROMPT_PARTS_OXE, labels, seed_pair)
            
            prompt = tf.cond(
                is_directional,
                lambda: prompt_directional,
                lambda: prompt_bbox,
            )

            # Create final output
            # Get VQA dataset ID for per-dataset metrics tracking
            dataset_name_str = self.get_dataset_name()
            vqa_dataset_id = VQA_DATASET_ID_MAP.get(dataset_name_str, 0)
            
            # For directional samples: use both primary and wrist images (like robot samples)
            # For bbox samples: use only primary image (keep legacy behavior)
            # has_wrist is True if sample is directional AND wrist image key is available
            has_wrist_cond = tf.logical_and(
                is_directional,
                tf.constant(wrist_image_key is not None, dtype=tf.bool)
            )
            
            # For non-directional samples, set wrist image to empty
            observation = frame["observation"]
            wrist_img = tf.cond(
                has_wrist_cond,
                lambda: observation[self.spec.wrist_image_key],
                lambda: tf.constant("", dtype=tf.string),
            )
            
            final_observation = {
                self.spec.primary_image_key: observation[self.spec.primary_image_key],
                self.spec.wrist_image_key: wrist_img,
                "state": observation["state"],
            }
            
            return {
                "observation": final_observation,
                "prompt": prompt,
                "caption": caption,
                "dataset_name": frame["dataset_name"],
                "time_horizon_seconds": tf.constant(1.0, dtype=tf.float32),
                "is_bimanual": tf.constant(False, dtype=tf.bool),
                "state_type": tf.constant("none", dtype=tf.string),
                "is_vqa_sample": tf.constant(True, dtype=tf.bool),
                "is_prediction_sample": tf.constant(False, dtype=tf.bool),
                "pred_use_primary": tf.constant(False, dtype=tf.bool),
                "raw_state": tf.zeros([self.state_dim], dtype=tf.float32),
                "is_navigation": tf.constant(False, dtype=tf.bool),
                "has_wrist_image": has_wrist_cond,  # Already a tensor, don't wrap in tf.constant
                "needs_wrist_rotation": frame["needs_wrist_rotation"],
                "vqa_dataset_id": tf.constant(vqa_dataset_id, dtype=tf.int32),
                "actions": tf.zeros([self.action_horizon, self.action_dim], dtype=tf.float32),
                "language_actions": tf.zeros([7], dtype=tf.float32),
            }

        self.dataset = self.dataset.frame_map(finalize_vqa, num_parallel_calls=self.num_parallel_calls)

        # Filter out empty prompts/captions
        def has_valid_qa(sample):
            has_prompt = tf.strings.length(sample["prompt"]) > 0
            has_caption = tf.strings.length(sample["caption"]) > 0
            return tf.logical_and(has_prompt, has_caption)

        self.dataset = self.dataset.filter(has_valid_qa)
        
        # Add memory budget constraint to prevent memory explosion during iteration
        # This is critical for large datasets like bridge_bbox with 159k+ frames
        # Apply before batching to limit memory usage during data loading
        self.dataset = self.dataset.with_ram_budget(1)

    def _build_frame_objects_table(self):
        """Build a lookup table from episode_id--frame_idx to pipe-delimited objects.
        
        Uses cached table from apply_restructure() if available (non-directional case),
        otherwise builds it fresh (directional case).
        """
        # If we already built the table in apply_restructure(), reuse it
        if hasattr(self, '_frame_objects_table') and self._frame_objects_table is not None:
            logging.info(f"[{self.dataset_name}] Reusing cached frame objects table")
            return self._frame_objects_table
        
        # Otherwise, build it now (directional case or fallback)
        # This should NOT happen for non-directional case if apply_restructure() ran correctly
        logging.warning(f"[{self.dataset_name}] Building frame objects table (not cached - this may be slow!)")
        orig_w, orig_h = self.get_original_image_size()
        target_h, target_w = self.config.resize_resolution
        
        if self.directional:
            return build_frame_objects_table_v2_direction(
                bbox_annotations_dir=self.bbox_annotations_dir,
                key_extractor=self.get_key_extractor(),
                dataset_name=self.dataset_name,
                orig_size=(orig_w, orig_h),
                target_size=(target_w, target_h),
                direction_slope=self.direction_slope,
            )
        
        return build_frame_objects_table_v2(
            bbox_annotations_dir=self.bbox_annotations_dir,
            key_extractor=self.get_key_extractor(),
            dataset_name=self.dataset_name,
            orig_size=(orig_w, orig_h),
            target_size=(target_w, target_h),
            target_only=self.use_target_only(),
        )

    def get_num_transitions(self) -> int:
        """Return number of transitions computed from JSONL annotation files.
        
        Uses cached count from build_annotated_keys_and_frame_table_v2() if available
        (non-directional case), otherwise scans JSONL files to count (directional case).
        """
        # If we already computed it during table building, reuse it
        if hasattr(self, "_num_transitions") and self._num_transitions is not None:
            return self._num_transitions
        
        # Otherwise, compute it now (directional case or fallback)
        self._num_transitions = count_annotated_frames(
            self.bbox_annotations_dir, self.get_key_extractor()
        )
        return self._num_transitions

    def debug_key_mismatch(self, num_samples: int = 5):
        """Debug helper to compare episode keys between trajectory and JSONL.
        
        Call this method to inspect sample keys from the JSONL annotations.
        Uses the dataset-specific key extractor (file_path for MolmoAct,
        file_path::episode_id for Bridge).
        """
        import json
        
        key_extractor = self.get_key_extractor()
        
        # Get sample keys from JSONL
        jsonl_files = tf.io.gfile.glob(os.path.join(self.bbox_annotations_dir, "*.jsonl"))
        jsonl_keys = []
        jsonl_episode_ids = set()
        for jsonl_file in jsonl_files[:1]:  # Just check first file
            if "merged" in jsonl_file:
                continue
            with tf.io.gfile.GFile(jsonl_file, "r") as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    if not line.strip():
                        continue
                    try:
                        episode_data = json.loads(line)
                        episode_key = key_extractor(episode_data)
                        if episode_key:
                            jsonl_episode_ids.add(episode_key)
                            labels = episode_data.get("labels", [])
                            for label_entry in labels[:2]:  # Just first 2 frames
                                frame_idx = label_entry.get("frame")
                                if frame_idx is not None:
                                    jsonl_keys.append(f"{episode_key}--{frame_idx}")
                        # Also log the raw episode_metadata structure
                        if i == 0:
                            logging.info(f"JSONL episode_metadata structure: {episode_data.get('episode_metadata', {})}")
                    except json.JSONDecodeError:
                        continue
        
        logging.info(f"Sample JSONL episode keys: {list(jsonl_episode_ids)[:3]}")
        logging.info(f"Sample JSONL lookup keys: {jsonl_keys[:5]}")

    def debug_frame_collisions(self):
        """Debug helper to detect frame collisions and key mapping issues in JSONL.
        
        This method analyzes the JSONL annotations to find:
        1. Episodes where the same frame index appears multiple times
        2. Key collisions where different episodes map to the same lookup key
        3. Frame index distribution per episode
        """
        import json
        from collections import defaultdict
        
        key_extractor = self.get_key_extractor()
        
        logging.info(f"=== Analyzing JSONL bbox annotations for {self.dataset_name} ===")
        logging.info(f"Annotations dir: {self.bbox_annotations_dir}")
        logging.info(f"Using key extractor: {key_extractor.__name__}")
        
        jsonl_files = tf.io.gfile.glob(os.path.join(self.bbox_annotations_dir, "*.jsonl"))
        logging.info(f"Found {len(jsonl_files)} JSONL files")
        
        # Track all lookup keys and their sources
        key_to_sources: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        # Track frame indices per episode
        episode_frame_indices: dict[str, list[int]] = defaultdict(list)
        # Track episodes per episode_key
        episode_key_counts: dict[str, int] = defaultdict(int)
        
        total_episodes = 0
        total_frames = 0
        
        for jsonl_file in jsonl_files:
            if "merged" in jsonl_file:
                continue
            file_basename = os.path.basename(jsonl_file)
            with tf.io.gfile.GFile(jsonl_file, "r") as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        episode_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    total_episodes += 1
                    episode_key = key_extractor(episode_data)
                    if not episode_key:
                        continue
                    
                    episode_key_counts[episode_key] += 1
                    
                    labels = episode_data.get("labels", [])
                    for label_entry in labels:
                        frame_idx = label_entry.get("frame")
                        all_objects = label_entry.get("all_objects", [])
                        
                        if frame_idx is None or not all_objects:
                            continue
                        
                        total_frames += 1
                        lookup_key = f"{episode_key}--{frame_idx}"
                        key_to_sources[lookup_key].append((file_basename, episode_key, frame_idx))
                        episode_frame_indices[episode_key].append(frame_idx)
        
        logging.info(f"Total episodes in JSONL: {total_episodes}")
        logging.info(f"Total annotated frames: {total_frames}")
        logging.info(f"Unique episode keys: {len(episode_key_counts)}")
        
        # Check for duplicate episode keys (same key appearing multiple times)
        duplicate_episodes = {k: v for k, v in episode_key_counts.items() if v > 1}
        if duplicate_episodes:
            logging.warning(f"Found {len(duplicate_episodes)} episode keys appearing multiple times:")
            for ep_key, count in list(duplicate_episodes.items())[:5]:
                logging.warning(f"  {ep_key}: appears {count} times")
        else:
            logging.info("No duplicate episode keys found (good)")
        
        # Check for lookup key collisions (same episode_key--frame_idx from different sources)
        collisions = {k: v for k, v in key_to_sources.items() if len(v) > 1}
        if collisions:
            logging.warning(f"Found {len(collisions)} lookup key collisions:")
            for lookup_key, sources in list(collisions.items())[:10]:
                logging.warning(f"  Key '{lookup_key}' has {len(sources)} entries:")
                for src in sources[:3]:
                    logging.warning(f"    from file={src[0]}, episode={src[1][:50]}..., frame={src[2]}")
        else:
            logging.info("No lookup key collisions found (good)")
        
        # Check for duplicate frame indices within same episode
        episodes_with_dup_frames = {}
        for ep_key, frames in episode_frame_indices.items():
            if len(frames) != len(set(frames)):
                from collections import Counter
                frame_counts = Counter(frames)
                dups = {f: c for f, c in frame_counts.items() if c > 1}
                episodes_with_dup_frames[ep_key] = dups
        
        if episodes_with_dup_frames:
            logging.warning(f"Found {len(episodes_with_dup_frames)} episodes with duplicate frame indices:")
            for ep_key, dups in list(episodes_with_dup_frames.items())[:5]:
                logging.warning(f"  Episode {ep_key[:60]}...")
                logging.warning(f"    Duplicate frames: {dups}")
        else:
            logging.info("No duplicate frame indices within episodes (good)")
        
        # Show frame index statistics
        all_frame_indices = []
        for frames in episode_frame_indices.values():
            all_frame_indices.extend(frames)
        
        if all_frame_indices:
            logging.info(f"Frame index range: {min(all_frame_indices)} to {max(all_frame_indices)}")
            logging.info(f"Sample frame indices from first 3 episodes:")
            for i, (ep_key, frames) in enumerate(list(episode_frame_indices.items())[:3]):
                sorted_frames = sorted(frames)
                logging.info(f"  Episode {i+1} ({ep_key[:50]}...): frames {sorted_frames[:10]}{'...' if len(sorted_frames) > 10 else ''}")
        
        return {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "unique_episode_keys": len(episode_key_counts),
            "duplicate_episodes": len(duplicate_episodes),
            "lookup_key_collisions": len(collisions),
            "episodes_with_dup_frames": len(episodes_with_dup_frames),
        }

    def debug_frame_mapping_samples(self, num_samples: int = 5):
        """Debug helper to show actual frame mapping from trajectory to JSONL.
        
        This samples from the dataset pipeline and shows:
        1. The episode_id and frame_idx used for lookup
        2. Whether the lookup key was found in the JSONL table
        3. The objects returned for that frame
        
        Call this BEFORE the dataset is batched/shuffled for clearer output.
        """
        import json
        
        logging.info(f"=== Sampling frame mappings for {self.dataset_name} ===")
        logging.info(f"Frame offset: {self.get_frame_offset()}")
        
        # Build the frame objects table to check lookups
        frame_to_objects = self._build_frame_objects_table()
        
        # Create a simple dataset that shows the mapping before final VQA conversion
        # We need to get samples after restructure but before finalization
        frame_offset = self.get_frame_offset()
        primary_image_key = self.get_primary_image_key()
        
        # Build a fresh dataset to sample from
        builder = self.build_dataset_builder(self.get_oxe_dataset_name(), self.data_dir)
        sample_ds = dl.DLataset.from_rlds(builder, split="all", shuffle=False, num_parallel_reads=1)
        
        # Apply standardization and get episode info
        standardize_fn = self.standardize_fn
        
        def extract_debug_info(traj):
            if standardize_fn is not None:
                traj = standardize_fn(traj)
            
            traj_len = tf.shape(traj["action"])[0]
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            
            # Get first few frame indices (after offset)
            frame_indices = tf.range(tf.minimum(traj_len, 10)) + frame_offset
            
            return {
                "file_path": file_path,
                "traj_len_after_transform": traj_len,
                "frame_indices_for_lookup": frame_indices,
                "raw_frame_0_shape": tf.shape(traj["observation"][primary_image_key][0]),
            }
        
        sample_ds = sample_ds.traj_map(extract_debug_info, num_parallel_calls=1)
        
        # Take a few samples
        samples_seen = 0
        for sample in sample_ds.take(num_samples):
            samples_seen += 1
            file_path = sample["file_path"].numpy().decode("utf-8")
            traj_len = sample["traj_len_after_transform"].numpy()
            frame_indices = sample["frame_indices_for_lookup"].numpy()
            
            logging.info(f"\n--- Sample {samples_seen} ---")
            logging.info(f"  file_path (episode_id): {file_path}")
            logging.info(f"  traj_len after transform: {traj_len}")
            logging.info(f"  First frame indices for lookup (with offset {frame_offset}): {frame_indices.tolist()}")
            
            # Check which lookup keys exist in the table
            found_frames = []
            missing_frames = []
            for frame_idx in frame_indices:
                lookup_key = f"{file_path}--{frame_idx}"
                result = frame_to_objects.lookup(tf.constant(lookup_key, dtype=tf.string))
                result_str = result.numpy().decode("utf-8")
                if result_str:
                    found_frames.append(frame_idx)
                    if len(found_frames) <= 2:  # Show first 2 found
                        # Parse and show object labels
                        objects = result_str.split(";")
                        labels = [obj.split("|")[0] for obj in objects if "|" in obj]
                        logging.info(f"    Frame {frame_idx}: FOUND - objects: {labels[:5]}{'...' if len(labels) > 5 else ''}")
                else:
                    missing_frames.append(frame_idx)
            
            logging.info(f"  Found in JSONL: {len(found_frames)} frames {found_frames[:5]}")
            logging.info(f"  Missing from JSONL: {len(missing_frames)} frames {missing_frames[:5]}")
            
            # Also check what frames ARE in the JSONL for this episode
            # by searching for keys starting with this file_path
            jsonl_frame_indices = []
            for test_frame in range(100):  # Check first 100 possible frames
                test_key = f"{file_path}--{test_frame}"
                test_result = frame_to_objects.lookup(tf.constant(test_key, dtype=tf.string))
                if test_result.numpy().decode("utf-8"):
                    jsonl_frame_indices.append(test_frame)
            
            logging.info(f"  All JSONL frames for this episode: {jsonl_frame_indices[:20]}{'...' if len(jsonl_frame_indices) > 20 else ''}")
        
        logging.info(f"\nTotal samples examined: {samples_seen}")

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


class MolmoActBoundingBoxDataset(OXEBoundingBoxDataset):
    """MolmoAct dataset with bounding box annotations for VQA training.

    Uses the molmoact_dataset from OXE with bbox annotations from JSONL files.
    """

    def get_dataset_name(self) -> str:
        return "molmoact_bbox"

    def get_bbox_annotations_dir_name(self) -> str:
        return "molmoact-bbox-annotations"

    def get_oxe_dataset_name(self) -> str:
        return "molmoact_dataset"

    def get_primary_image_key(self) -> str:
        return "first_view_image"

    def get_original_image_size(self) -> tuple[int, int]:
        # MolmoAct stores images at 224x224 (resized in dataset builder)
        return (224, 224)


class BridgeBoundingBoxDataset(OXEBoundingBoxDataset):
    """Bridge dataset with bounding box annotations for VQA training.

    Uses the bridge_v2_oxe dataset from OXE with bbox annotations from JSONL files.
    
    For Bridge dataset, the episode key is a composite of file_path AND episode_id
    because one file (e.g., out.npy) can contain multiple episodes.
    Key format: "{file_path}::{episode_id}"
    """

    def get_frame_offset(self) -> int:
        # Bridge V2 transform removes the first frame (all-zero action).
        # JSONL annotations were made on raw data, so frame indices need offset by 1.
        # Processed frame 0 -> JSONL frame 1, processed frame 1 -> JSONL frame 2, etc.
        return 0

    def get_dataset_name(self) -> str:
        return "bridge_bbox"

    def get_bbox_annotations_dir_name(self) -> str:
        return "bridge-bbox-annotations"

    def get_oxe_dataset_name(self) -> str:
        return "bridge_v2_oxe"

    def get_primary_image_key(self) -> str:
        return "image_0"

    def get_original_image_size(self) -> tuple[int, int]:
        # Bridge uses 256x256 images
        return (224, 224)

    def use_target_only(self) -> bool:
        # Use only target objects to filter out potentially noisy annotations
        return False

    def get_key_extractor(self) -> callable:
        """Return Bridge-specific key extractor that uses file_path::episode_id."""
        return bridge_key_extractor

    def build_episode_key_tf(self, traj) -> tf.Tensor:
        """Build composite episode key from file_path and episode_id.
        
        For Bridge dataset, one file can contain multiple episodes, so we need
        both file_path AND episode_id to uniquely identify an episode.
        
        Key format: "{file_path}::{episode_id}"
        """
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"][0]
        
        # Convert episode_id to string and join with file_path
        episode_id_str = tf.strings.as_string(episode_id)
        composite_key = tf.strings.join([file_path, "::", episode_id_str])
        
        return composite_key
