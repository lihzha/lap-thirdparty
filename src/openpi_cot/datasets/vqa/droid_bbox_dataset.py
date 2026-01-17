"""DROID Bounding Box dataset implementation for VQA training.

This dataset loads DROID robot trajectories with object bounding box annotations
from JSONL files and formats them as VQA samples asking "where is the <object>".
"""

import json
import logging
import os
from typing import TYPE_CHECKING, ClassVar

import dlimp as dl
import jax
import numpy as np
import tensorflow as tf

from openpi_cot.datasets.base_dataset import SingleCoTDataset
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.helpers import extract_episode_path_from_file_path
from openpi_cot.datasets.utils.specs import CoTRldsDatasetSpec
from openpi_cot.datasets.vqa.bbox_common import (
    ROBOT_BBOX_PROMPT_PARTS_EE,
    ROBOT_DIRECTION_PROMPT_PARTS_EE,
    build_annotated_keys_set,
    build_frame_objects_table_v2,
    build_frame_objects_table_v2_direction,
    count_annotated_frames,
    droid_key_extractor,
    rotate_bbox_loc_tokens_180_tf,
    sample_and_format_objects_direction_tf,
    sample_and_format_objects_tf,
    sample_prompt_tf,
)
from openpi_cot.datasets.vqa.vqa_base import VQA_DATASET_ID_MAP

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


class DroidBoundingBoxDataset(SingleCoTDataset):
    """DROID dataset with bounding box annotations for VQA training.

    This dataset loads DROID trajectories and bbox annotations from JSONL files,
    creating VQA samples that ask "where is the <object>" with bbox answers.
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
        self.dataset_name = "droid_bbox_direction" if directional else "droid_bbox"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.vis_dataset = bool(config.vis_dataset)
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.use_wrist_image = False  # VQA has no wrist images
        self.standalone = standalone
        self.action_horizon = action_horizon
        self.want_full_determinism = config.want_full_determinism
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)

        # VQA-specific settings
        self.control_frequency = 1  # Single frame, no temporal control

        # Configure TensorFlow with no GPU/TPU devices
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        tf.random.set_seed(self.seed)

        # Build path to bbox annotations directory
        # Try both possible directory name patterns
        if self.spec.lang_action_dir_name in config.language_action_dir:
            self.bbox_annotations_dir = config.language_action_dir.replace(
                self.spec.lang_action_dir_name, "droid-bbox-annotations"
            )
        elif self.spec.lang_action_dir_name_base in config.language_action_dir:
            self.bbox_annotations_dir = config.language_action_dir.replace(
                self.spec.lang_action_dir_name_base, "droid-bbox-annotations"
            )
        else:
            # Fallback: try to construct path from parent directory
            parent_dir = os.path.dirname(config.language_action_dir.rstrip("/"))
            self.bbox_annotations_dir = os.path.join(parent_dir, "droid-bbox-annotations")

        logging.info(f"Loading bbox annotations from: {self.bbox_annotations_dir}")

        # Build lookup tables
        if hash_tables is not None:
            self.ep_table = hash_tables.get("ep_table")
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

            if standalone:
                self.hash_tables = {
                    "ep_table": self.ep_table,
                }

        # Build RLDS dataset
        self.builder = self.build_dataset_builder(config.droid_dataset_name, data_dir)
        self.dataset = self.build_dataset(self.builder)

        # Apply trajectory identifier
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
                not_rotate_wrist_prob=getattr(config, "not_rotate_wrist_prob", 0.0),
            )

    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lookup_table(self, metadata_path):
        """Build episode-path to episode-ID lookup table."""
        from openpi_cot.datasets.utils.dataset_utils import print_memory_usage

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

    def build_dataset_builder(self, ds_name, data_dir):
        """Build TFDS builder for DROID."""
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
        """Add trajectory_id to each trajectory."""

        def _get_traj_identifier(traj):
            episode_id = self._episode_id_from_traj(traj, self.ep_table)
            traj["trajectory_id"] = tf.fill([tf.shape(traj["action"])[0]], episode_id)
            return traj

        self.dataset = self.dataset.traj_map(_get_traj_identifier, self.num_parallel_calls)

    def apply_traj_filters(self, action_key):
        """Filter trajectories to only keep those with bbox annotations."""

        # Filter out empty trajectories
        def _non_empty(traj):
            return tf.greater(tf.shape(traj[action_key])[0], 0)

        self.dataset = self.dataset.filter(_non_empty)

        # Filter to only keep successful trajectories
        def _path_ok(traj):
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
            return tf.strings.regex_full_match(file_path, ".*success.*")

        self.dataset = self.dataset.filter(_path_ok)

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
        # OPTIMIZATION: Build set of episode_paths with annotations and filter trajectories first
        # This skips entire trajectories without any bbox annotations
        annotated_episode_paths = build_annotated_keys_set(
            self.bbox_annotations_dir, droid_key_extractor
        )
        logging.info(f"Found {len(annotated_episode_paths)} trajectories with bbox annotations")

        if annotated_episode_paths:
            # Build a lookup table for fast trajectory-level filtering
            annotated_paths_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(list(annotated_episode_paths), dtype=tf.string),
                    tf.constant([True] * len(annotated_episode_paths), dtype=tf.bool),
                ),
                default_value=tf.constant(False, dtype=tf.bool),
            )

            def has_any_annotations(traj):
                """Filter trajectories that have at least one annotated frame."""
                file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
                episode_path = extract_episode_path_from_file_path(file_path)
                return annotated_paths_table.lookup(episode_path)

            self.dataset = self.dataset.filter(has_any_annotations)

        def restructure(traj):
            """Convert trajectory to VQA bbox format."""
            # Get file_path directly from metadata
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"]

            traj_len = tf.shape(traj["action"])[0]

            # Extract episode path from file_path
            # Example file_path: gs://xembodiment_data/r2d2/r2d2-data-full/ILIAD/success/2023-04-21/.../trajectory.h5
            episode_path = extract_episode_path_from_file_path(file_path[0])

            # Always prepare both primary (exterior) and wrist images
            # For directional samples: use both primary and wrist
            # For bbox samples: use only wrist (set as wrist_img, not primary)
            # Randomly sample one of the two exterior images (left cameras from stereo pairs)
            random_val = tf.random.stateless_uniform(
                shape=[], seed=[self.seed, tf.strings.to_hash_bucket_fast(traj["trajectory_id"][0], 2147483647)]
            )
            exterior_img = tf.cond(
                random_val > 0.5,
                lambda: traj["observation"][self.spec.images_list[0]],
                lambda: traj["observation"][self.spec.images_list[1]],
            )
            primary_img = exterior_img
            wrist_img = traj["observation"]["wrist_image_left"]

            # Create frame indices as strings
            frame_indices = tf.as_string(tf.range(traj_len))

            return {
                "observation": {
                    self.spec.primary_image_key: primary_img,
                    self.spec.wrist_image_key: wrist_img,
                    "state": tf.zeros([traj_len, self.state_dim], dtype=tf.float32),
                },
                "trajectory_id": traj["trajectory_id"],
                "episode_path": tf.fill([traj_len], episode_path),
                "frame_idx": frame_indices,
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
                # DROID requires wrist camera rotation by 180 degrees for ALL samples
                "needs_wrist_rotation": tf.fill([traj_len], tf.constant(True)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

        # Flatten to individual frames
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_frame_filters(self):
        """Filter and expand frames to create final VQA samples."""
        # Build a lookup table that maps episode_path--frame_idx to pipe-delimited objects
        frame_to_objects = self._build_frame_objects_table()

        # Filter frames using pure TF lookup (fast) - only keep frames with annotations
        def has_bbox_annotation(frame):
            """Fast filter using pure TF lookup."""
            lookup_key = tf.strings.join([frame["episode_path"], "--", frame["frame_idx"]])
            objects_data = frame_to_objects.lookup(lookup_key)
            return tf.strings.length(objects_data) > 0

        self.dataset = self.dataset.filter(has_bbox_annotation)

        # Sample objects and format caption using pure TensorFlow operations
        max_objects = 2
        direction_prob = 0.5  # Probability of using direction caption instead of bbox

        def lookup_and_sample_objects(frame):
            """Look up objects, sample if needed, and format caption using pure TF."""
            lookup_key = tf.strings.join([frame["episode_path"], "--", frame["frame_idx"]])
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

            # For droid_bbox, rotate bbox coordinates to match rotated wrist image
            # Split caption by " ; " to handle multiple bboxes
            bbox_parts = tf.strings.split(caption, " ; ")
            
            def rotate_bbox_part(bbox_part: tf.Tensor) -> tf.Tensor:
                """Rotate bbox coordinates in a single bbox part."""
                # Split by space to separate loc tokens from label
                parts = tf.strings.split(bbox_part, " ")
                if tf.shape(parts)[0] < 1:
                    return bbox_part
                
                loc_tokens = parts[0]
                label = tf.cond(
                    tf.shape(parts)[0] > 1,
                    lambda: tf.strings.reduce_join(parts[1:], separator=" "),
                    lambda: tf.constant("", dtype=tf.string),
                )
                
                # Rotate loc tokens
                rotated_loc = rotate_bbox_loc_tokens_180_tf(loc_tokens)
                
                # Reconstruct: rotated_loc + " " + label
                if tf.strings.length(label) > 0:
                    return tf.strings.join([rotated_loc, label], separator=" ")
                else:
                    return rotated_loc
            
            # Rotate each bbox part (only for non-directional bbox samples)
            def rotate_caption():
                rotated_parts = tf.map_fn(
                    rotate_bbox_part,
                    bbox_parts,
                    fn_output_signature=tf.TensorSpec([], tf.string),
                )
                return tf.strings.reduce_join(rotated_parts, separator=" ; ")
            
            # Only rotate bbox coordinates for non-directional samples (bbox samples)
            # Directional samples don't need bbox rotation
            final_caption = tf.cond(
                is_directional,
                lambda: caption,  # Keep original for directional
                lambda: rotate_caption(),  # Rotate for bbox samples
            )

            frame["object_labels"] = labels
            frame["bbox_caption"] = final_caption
            frame["is_directional"] = is_directional

            return frame

        self.dataset = self.dataset.frame_map(lookup_and_sample_objects, num_parallel_calls=self.num_parallel_calls)

        # Filter out any invalid entries (e.g., JSON parse errors)
        def has_valid_caption(frame):
            return tf.strings.length(frame["bbox_caption"]) > 0

        self.dataset = self.dataset.filter(has_valid_caption)

        def finalize_vqa(frame):
            """Create final VQA sample with prompt and caption."""
            labels = frame["object_labels"]
            caption = frame["bbox_caption"]
            is_directional = frame["is_directional"]

            # Select prompt parts based on whether sample is directional
            prompt_parts = tf.cond(
                is_directional,
                lambda: ROBOT_DIRECTION_PROMPT_PARTS_EE,
                lambda: ROBOT_BBOX_PROMPT_PARTS_EE,
            )

            # Sample a prompt template using the shared helper
            seed_key = tf.strings.join([frame["trajectory_id"], "_bbox_", frame["frame_idx"]])
            seed_hash = tf.strings.to_hash_bucket_fast(seed_key, 2147483647)
            seed_hash_int = tf.cast(seed_hash, tf.int32)

            prompt = sample_prompt_tf(prompt_parts, labels, (self.seed, seed_hash_int))

            # Create final output
            # Get VQA dataset ID for per-dataset metrics tracking
            dataset_name_str = self.get_dataset_name()
            vqa_dataset_id = VQA_DATASET_ID_MAP.get(dataset_name_str, 0)
            
            # For directional samples: use both primary and wrist images (like robot samples)
            # For bbox samples: use only wrist image (set as wrist_img, primary will be blank)
            has_wrist = is_directional
            
            # For bbox samples, set primary image to empty string (use only wrist)
            observation = frame["observation"]
            primary_img = tf.cond(
                is_directional,
                lambda: observation[self.spec.primary_image_key],
                lambda: tf.constant("", dtype=tf.string),
            )
            wrist_img = observation[self.spec.wrist_image_key]
            
            final_observation = {
                self.spec.primary_image_key: primary_img,
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
                "has_wrist_image": tf.constant(has_wrist, dtype=tf.bool),
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

    def _build_frame_objects_table(self):
        """Build a lookup table from episode_path--frame_idx to pipe-delimited objects."""
        # DROID wrist image size
        orig_w, orig_h = 320, 180
        target_h, target_w = self.config.resize_resolution

        if self.directional:
            return build_frame_objects_table_v2_direction(
                bbox_annotations_dir=self.bbox_annotations_dir,
                key_extractor=droid_key_extractor,
                dataset_name=self.dataset_name,
                orig_size=(orig_w, orig_h),
                target_size=(target_w, target_h),
                direction_slope=self.direction_slope,
            )
        else:
            return build_frame_objects_table_v2(
                bbox_annotations_dir=self.bbox_annotations_dir,
                key_extractor=droid_key_extractor,
                dataset_name=self.dataset_name,
                orig_size=(orig_w, orig_h),
                target_size=(target_w, target_h),
            )

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "droid_bbox_direction" if self.directional else "droid_bbox"

    def get_num_transitions(self) -> int:
        """Return number of transitions computed from JSONL annotation files."""
        if not hasattr(self, "_num_transitions"):
            self._num_transitions = count_annotated_frames(
                self.bbox_annotations_dir, droid_key_extractor
            )
        return self._num_transitions

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
