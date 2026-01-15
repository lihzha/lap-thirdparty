"""OXE Bounding Box dataset implementation for VQA training.

This module provides bbox dataset classes for OXE datasets (molmoact, bridge)
that loads robot trajectories with object bounding box annotations from JSONL files
and formats them as VQA samples asking "where is the <object>".
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import tensorflow as tf

from openpi_cot.datasets.utils.data_utils import load_dataset_kwargs
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.specs import CoTRldsDatasetSpec
from openpi_cot.datasets.vqa.bbox_common import (
    BBOX_PROMPT_PARTS,
    build_frame_objects_table,
    format_bbox_caption,
    oxe_key_extractor,
    sample_prompt_tf,
)

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
        logging.info(f"Loading bbox annotations from: {self.bbox_annotations_dir}")

        # Build RLDS dataset
        self.builder = self.build_dataset_builder(oxe_dataset_name, data_dir)
        self.dataset = self.build_dataset(self.builder)

        # Apply trajectory identifier (OXE-style: hash-based)
        self.get_traj_identifier()

        # Split train/val
        self.split_val(split_seed=seed)

        # Apply VQA restructure for bounding box
        self.apply_restructure()

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

    @abstractmethod
    def extract_uuid_from_traj(self, traj) -> tf.Tensor:
        """Extract UUID from trajectory metadata for JSONL lookup.

        The UUID should match the format in the JSONL annotations file.
        Returns a tf.string tensor.
        """
        raise NotImplementedError

    # ========== Common methods ==========

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

            # Also extract UUID for bbox lookup
            traj["uuid"] = tf.repeat(self.extract_uuid_from_traj(traj), traj_len)

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
        primary_image_key = self.get_primary_image_key()

        def restructure(traj):
            """Convert trajectory to VQA bbox format."""
            traj_len = tf.shape(traj["action"])[0]

            # Get primary image from observations
            primary_img = traj["observation"][primary_image_key]

            # Create frame indices as strings
            frame_indices = tf.as_string(tf.range(traj_len))

            return {
                "observation": {
                    self.spec.primary_image_key: primary_img,
                    self.spec.wrist_image_key: tf.repeat("", traj_len),
                    "state": tf.zeros([traj_len, self.state_dim], dtype=tf.float32),
                },
                "trajectory_id": traj["trajectory_id"],
                "uuid": traj["uuid"],
                "frame_idx": frame_indices,
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

        # Flatten to individual frames
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_frame_filters(self):
        """Filter and expand frames to create final VQA samples."""
        # Build a lookup table that maps uuid--frame_idx to a list of objects
        frame_to_objects = self._build_frame_objects_table()

        # Target resolution for letterbox transformation
        target_h, target_w = self.config.resize_resolution
        orig_w, orig_h = self.get_original_image_size()

        def lookup_all_objects(frame):
            """Look up ALL objects for this frame and generate caption with all bboxes."""
            # Create lookup key: uuid--frame_idx
            lookup_key = tf.strings.join([frame["uuid"], "--", frame["frame_idx"]])

            # Look up the serialized list of objects
            objects_json = frame_to_objects.lookup(lookup_key)

            # Check if we have annotations
            has_annotations = tf.strings.length(objects_json) > 0

            # Get image bytes for potential dimension check (if needed)
            img_bytes = frame["observation"][self.spec.primary_image_key]

            def parse_all_objects_and_transform(objects_json_bytes, img_bytes_tensor):
                """Parse all objects and transform bbox coordinates for letterbox."""
                if not objects_json_bytes.numpy():
                    return b"", b""

                try:
                    objects = json.loads(objects_json_bytes.numpy().decode("utf-8"))
                    if not objects:
                        return b"", b""

                    # Use shared format_bbox_caption function
                    prompt_labels, caption = format_bbox_caption(
                        objects=objects,
                        orig_w=orig_w,
                        orig_h=orig_h,
                        target_w=target_w,
                        target_h=target_h,
                        apply_letterbox=True,
                    )

                    return prompt_labels.encode("utf-8"), caption.encode("utf-8")

                except Exception as e:
                    logging.warning(f"Error parsing objects JSON: {e}")
                    return b"", b""

            prompt_labels, caption = tf.py_function(
                parse_all_objects_and_transform,
                [objects_json, img_bytes],
                [tf.string, tf.string],
            )
            prompt_labels.set_shape([])
            caption.set_shape([])

            frame["object_labels"] = prompt_labels
            frame["bbox_caption"] = caption
            frame["has_bbox"] = has_annotations

            return frame

        self.dataset = self.dataset.frame_map(lookup_all_objects, num_parallel_calls=self.num_parallel_calls)

        # Filter out frames without annotations
        def has_valid_bbox(frame):
            return tf.logical_and(
                frame["has_bbox"],
                tf.strings.length(frame["bbox_caption"]) > 0,
            )

        self.dataset = self.dataset.filter(has_valid_bbox)

        # Convert to final VQA format
        def finalize_vqa(frame):
            """Create final VQA sample with prompt and caption."""
            labels = frame["object_labels"]
            caption = frame["bbox_caption"]

            # Sample a prompt template using the shared helper
            seed_key = tf.strings.join([frame["trajectory_id"], "_bbox_", frame["frame_idx"]])
            seed_hash = tf.strings.to_hash_bucket_fast(seed_key, 2147483647)
            seed_hash_int = tf.cast(seed_hash, tf.int32)

            prompt = sample_prompt_tf(BBOX_PROMPT_PARTS, labels, (self.seed, seed_hash_int))

            # Create final output
            return {
                "observation": frame["observation"],
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
                "has_wrist_image": tf.constant(False, dtype=tf.bool),
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

    def _build_frame_objects_table(self):
        """Build a lookup table from uuid--frame_idx to list of objects."""
        return build_frame_objects_table(
            bbox_annotations_dir=self.bbox_annotations_dir,
            key_extractor=oxe_key_extractor,
            dataset_name=self.dataset_name,
        )

    def get_num_transitions(self) -> int:
        """Return approximate number of transitions."""
        return 100000

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
        return "image"

    def get_original_image_size(self) -> tuple[int, int]:
        # MolmoAct uses 256x256 images
        return (256, 256)

    def extract_uuid_from_traj(self, traj) -> tf.Tensor:
        """Extract UUID from MolmoAct trajectory metadata.

        MolmoAct UUID format from JSONL:
        gs://pi0-cot/OXE/molmoact_dataset/1.0.0/molmoact_dataset-train.tfrecord-00000-of-00512::episode_0

        The dlimp library stores the source tfrecord file in traj_metadata._source_file
        and episode index in episode_metadata or as _source_episode_idx.
        """
        # Try to extract source file and episode index from traj_metadata
        if "traj_metadata" in traj:
            traj_meta = traj["traj_metadata"]

            # Method 1: Use _source_file and _source_episode_idx if available (dlimp standard)
            if "_source_file" in traj_meta and "_source_episode_idx" in traj_meta:
                source_file = traj_meta["_source_file"]
                episode_idx = traj_meta["_source_episode_idx"]
                if isinstance(source_file, tf.Tensor) and len(source_file.shape) > 0:
                    source_file = source_file[0]
                if isinstance(episode_idx, tf.Tensor) and len(episode_idx.shape) > 0:
                    episode_idx = episode_idx[0]
                return tf.strings.join([
                    source_file,
                    tf.constant("::episode_", dtype=tf.string),
                    tf.strings.as_string(episode_idx)
                ])

            # Method 2: Use episode_metadata if available
            if "episode_metadata" in traj_meta:
                episode_meta = traj_meta["episode_metadata"]
                # Some datasets have episode_id or episode_index
                episode_idx = None
                if "episode_index" in episode_meta:
                    episode_idx = episode_meta["episode_index"]
                elif "episode_id" in episode_meta:
                    episode_idx = episode_meta["episode_id"]

                if episode_idx is not None:
                    if isinstance(episode_idx, tf.Tensor) and len(episode_idx.shape) > 0:
                        episode_idx = episode_idx[0]
                    # Construct UUID using dataset name and episode index
                    # This won't match the exact GCS path but provides a unique key
                    return tf.strings.join([
                        tf.constant(f"gs://pi0-cot/OXE/{self.get_oxe_dataset_name()}/::episode_", dtype=tf.string),
                        tf.strings.as_string(episode_idx)
                    ])

        # Fallback: construct a unique ID from the trajectory actions
        action_sample = traj["action"][0]
        serialized = tf.io.serialize_tensor(action_sample)
        hash_val = tf.strings.to_hash_bucket_fast(serialized, 2147483647)
        return tf.strings.join([
            tf.constant("molmoact-fallback-", dtype=tf.string),
            tf.strings.as_string(hash_val)
        ])


class BridgeBoundingBoxDataset(OXEBoundingBoxDataset):
    """Bridge dataset with bounding box annotations for VQA training.

    Uses the bridge_v2_oxe dataset from OXE with bbox annotations from JSONL files.
    """

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
        return (256, 256)

    def extract_uuid_from_traj(self, traj) -> tf.Tensor:
        """Extract UUID from Bridge trajectory metadata.

        Bridge UUID format from JSONL:
        gs://pi0-cot/OXE/bridge_v2_oxe/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024::episode_0

        The dlimp library stores the source tfrecord file in traj_metadata._source_file
        and episode index in episode_metadata or as _source_episode_idx.
        """
        # Try to extract source file and episode index from traj_metadata
        if "traj_metadata" in traj:
            traj_meta = traj["traj_metadata"]

            # Method 1: Use _source_file and _source_episode_idx if available (dlimp standard)
            if "_source_file" in traj_meta and "_source_episode_idx" in traj_meta:
                source_file = traj_meta["_source_file"]
                episode_idx = traj_meta["_source_episode_idx"]
                if isinstance(source_file, tf.Tensor) and len(source_file.shape) > 0:
                    source_file = source_file[0]
                if isinstance(episode_idx, tf.Tensor) and len(episode_idx.shape) > 0:
                    episode_idx = episode_idx[0]
                return tf.strings.join([
                    source_file,
                    tf.constant("::episode_", dtype=tf.string),
                    tf.strings.as_string(episode_idx)
                ])

            # Method 2: Use episode_metadata if available
            if "episode_metadata" in traj_meta:
                episode_meta = traj_meta["episode_metadata"]
                # Some datasets have episode_id or episode_index
                episode_idx = None
                if "episode_index" in episode_meta:
                    episode_idx = episode_meta["episode_index"]
                elif "episode_id" in episode_meta:
                    episode_idx = episode_meta["episode_id"]

                if episode_idx is not None:
                    if isinstance(episode_idx, tf.Tensor) and len(episode_idx.shape) > 0:
                        episode_idx = episode_idx[0]
                    # Construct UUID using dataset name and episode index
                    return tf.strings.join([
                        tf.constant(f"gs://pi0-cot/OXE/{self.get_oxe_dataset_name()}/::episode_", dtype=tf.string),
                        tf.strings.as_string(episode_idx)
                    ])

        # Fallback: construct a unique ID from the trajectory actions
        action_sample = traj["action"][0]
        serialized = tf.io.serialize_tensor(action_sample)
        hash_val = tf.strings.to_hash_bucket_fast(serialized, 2147483647)
        return tf.strings.join([
            tf.constant("bridge-fallback-", dtype=tf.string),
            tf.strings.as_string(hash_val)
        ])
