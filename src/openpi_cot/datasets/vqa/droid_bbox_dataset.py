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
from openpi_cot.datasets.utils.dataset_utils import print_memory_usage
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.helpers import extract_episode_path_from_file_path
from openpi_cot.datasets.utils.specs import CoTRldsDatasetSpec
from openpi_cot.datasets.vqa.vqa_base import ensure_dldataset

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig

# Bounding box prompts - same format as LVIS/PACO
DROID_BBOX_PROMPT_PARTS = [
    ("Show me where the ", " is in the image using a bounding box."),
    ("Draw a bounding box around the ", " in the image."),
    ("Please provide a bounding box for the ", " in this image."),
    ("Locate the ", " in the image by drawing a bounding box."),
    ("mark the ", " with a bounding box."),
    ("Identify the ", " in the image by bounding it."),
    ("Find the ", " and draw a bounding box around it."),
    ("Highlight the ", " with a bounding box."),
    ("Can you draw a bounding box around the ", "?"),
    ("Where is the ", " in the image? Show it with a bounding box."),
    ("Indicate the ", " by marking a bounding box."),
    ("If there is a ", " in the image, draw a bounding box around it."),
    ("If any ", " is present, show one bounding box."),
    ("Show me one ", " in the image by drawing a bounding box."),
    ("Please locate the ", " using a bounding box."),
    ("Detect the ", " and provide its bounding box."),
    ("Find a ", " in the picture and draw a bounding box around it."),
    ("Bounding box task: draw a box around the ", "."),
    ("Object: ", ". Instruction: Draw a bounding box around the object."),
    ("Look for the ", " and mark it with a bounding box."),
    ("Help me find the ", " by drawing a bounding box around it."),
    ("Show me ", " using a bounding box."),
    ("For ", " in the image, draw a bounding box."),
    ("Indicate ", " with a bounding box."),
    ("Please show the region containing the ", " using a bounding box."),
    ("point out the ", " by drawing a bounding box."),
    ("Locate a ", " and provide bounding box."),
    ("Draw a bounding box around all the ", "."),
    ("Find and outline the ", " with a bounding box."),
    ("Mark ", " using bounding box."),
]


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
        self.dataset_name = "droid_bbox"
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
            self.bbox_table = hash_tables.get("bbox_table")
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
            self.bbox_table = self.build_bbox_table()

            if standalone:
                self.hash_tables = {
                    "ep_table": self.ep_table,
                    "bbox_table": self.bbox_table,
                }

        # Build RLDS dataset
        self.builder = self.build_dataset_builder(config.droid_dataset_name, data_dir)
        self.dataset = self.build_dataset(self.builder)

        # Apply trajectory identifier
        self.get_traj_identifier()

        # # Apply trajectory filters (only keep episodes with bbox annotations)
        # self.apply_traj_filters(action_key="action")

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
            )

    def _episode_id_from_traj(self, traj, ep_table):
        """Lookup episode_id from trajectory metadata using regex extraction."""
        file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        episode_path = extract_episode_path_from_file_path(file_path)
        return ep_table.lookup(episode_path)

    def build_lookup_table(self, metadata_path):
        """Build episode-path to episode-ID lookup table."""
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

    def build_bbox_table(self):
        """Build lookup table from (episode_path, frame_index) to bbox annotations.

        The key format is: "episode_path--frame_index"
        The value is a serialized tensor of shape [N, 5] where each row is [label_hash, x_min, y_min, x_max, y_max]
        normalized to [0, 1]. The label string is stored in a separate table.
        """
        logging.info(f"Building bbox lookup table from {self.bbox_annotations_dir}")

        # Find all JSONL files
        jsonl_files = tf.io.gfile.glob(os.path.join(self.bbox_annotations_dir, "*.jsonl"))
        if not jsonl_files:
            logging.warning(f"No JSONL files found in {self.bbox_annotations_dir}")
            # Return table with dummy entry (TF doesn't allow empty tables)
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(["__dummy_key__"], dtype=tf.string),
                    tf.constant([""], dtype=tf.string),
                ),
                default_value=tf.constant(b"", dtype=tf.string),
            )

        logging.info(f"Found {len(jsonl_files)} JSONL files")

        keys_list = []
        values_list = []
        label_keys = []
        label_values = []

        for jsonl_file in jsonl_files:
            logging.info(f"Processing {jsonl_file}")
            with tf.io.gfile.GFile(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        episode_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract episode path from file_path in episode_metadata
                    file_path = episode_data.get("episode_metadata", {}).get("file_path", "")
                    if not file_path:
                        continue

                    # Extract episode path using same logic as DroidCoTDataset
                    # Remove dataset prefix up to r2d2-data or r2d2-data-full
                    import re

                    match = re.search(r"r2d2-data(?:-full)?/(.+?)/trajectory", file_path)
                    if match:
                        episode_path = match.group(1)
                    else:
                        continue

                    # Process each labeled frame
                    labels = episode_data.get("labels", [])
                    for label_entry in labels:
                        frame_idx = label_entry.get("frame")
                        all_objects = label_entry.get("all_objects", [])

                        if frame_idx is None or not all_objects:
                            continue

                        # Create key: episode_path--frame_index
                        key = f"{episode_path}--{frame_idx}"

                        # Process all objects in this frame
                        for obj in all_objects:
                            obj_label = obj.get("label", "")
                            bbox = obj.get("bbox", [])

                            if not obj_label or len(bbox) < 4:
                                continue

                            # Normalize bbox from 0-1000 to 0-1
                            x_min = float(bbox[0]) / 1000.0
                            y_min = float(bbox[1]) / 1000.0
                            x_max = float(bbox[2]) / 1000.0
                            y_max = float(bbox[3]) / 1000.0

                            # Clamp to valid range
                            x_min = max(0.0, min(1.0, x_min))
                            y_min = max(0.0, min(1.0, y_min))
                            x_max = max(0.0, min(1.0, x_max))
                            y_max = max(0.0, min(1.0, y_max))

                            # Create unique key for this object: episode_path--frame_idx--label--bbox_hash
                            bbox_str = f"{x_min:.4f}_{y_min:.4f}_{x_max:.4f}_{y_max:.4f}"
                            obj_key = f"{key}--{obj_label}--{bbox_str}"

                            # Serialize the annotation: label + bbox
                            annotation = {
                                "label": obj_label,
                                "bbox": [x_min, y_min, x_max, y_max],
                            }
                            value = json.dumps(annotation)

                            keys_list.append(obj_key)
                            values_list.append(value)

        logging.info(f"Built bbox table with {len(keys_list)} entries")

        if not keys_list:
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant([], dtype=tf.string),
                    tf.constant([], dtype=tf.string),
                ),
                default_value=tf.constant(b"", dtype=tf.string),
            )

        bbox_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(keys_list, dtype=tf.string),
                tf.constant(values_list, dtype=tf.string),
            ),
            default_value=tf.constant(b"", dtype=tf.string),
        )
        print_memory_usage("After building bbox_table")
        return bbox_table

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

        def restructure(traj):
            """Convert trajectory to VQA bbox format."""
            # Get file_path directly from metadata
            file_path = traj["traj_metadata"]["episode_metadata"]["file_path"]

            traj_len = tf.shape(traj["action"])[0]
            episode_id = traj["trajectory_id"][0]

            # Extract episode path from file_path
            # Example file_path: gs://xembodiment_data/r2d2/r2d2-data-full/ILIAD/success/2023-04-21/.../trajectory.h5
            episode_path = extract_episode_path_from_file_path(file_path[0])

            # Use wrist image for bbox annotations
            primary_img = traj["observation"]["wrist_image_left"]

            # Create frame indices as strings
            frame_indices = tf.as_string(tf.range(traj_len))

            return {
                "observation": {
                    self.spec.primary_image_key: primary_img,
                    self.spec.wrist_image_key: tf.repeat("", traj_len),
                    "state": tf.zeros([traj_len, self.state_dim], dtype=tf.float32),
                },
                "trajectory_id": traj["trajectory_id"],
                "episode_path": tf.fill([traj_len], episode_path),
                "frame_idx": frame_indices,
                "dataset_name": tf.fill([traj_len], tf.constant(self.dataset_name)),
            }

        self.dataset = self.dataset.traj_map(restructure, self.num_parallel_calls)

        # Flatten to individual frames
        self.dataset = self.dataset.flatten(num_parallel_calls=self.num_parallel_calls)

    def apply_frame_filters(self):
        """Filter and expand frames to create final VQA samples."""
        # Build a lookup table that maps episode_path--frame_idx to a list of objects
        frame_to_objects = self._build_frame_objects_table()

        # Target resolution for letterbox transformation
        target_h, target_w = self.config.resize_resolution

        def lookup_all_objects(frame):
            """Look up ALL objects for this frame and generate caption with all bboxes."""
            # Create lookup key: episode_path--frame_idx
            lookup_key = tf.strings.join([frame["episode_path"], "--", frame["frame_idx"]])

            # Look up the serialized list of objects
            objects_json = frame_to_objects.lookup(lookup_key)

            # Check if we have annotations
            has_annotations = tf.strings.length(objects_json) > 0

            # Get image dimensions for letterbox transformation
            # The image is still encoded at this point, so we need to decode to get dimensions
            img_bytes = frame["observation"][self.spec.primary_image_key]

            def parse_all_objects_and_transform(objects_json_bytes, img_bytes_tensor):
                """Parse all objects and transform bbox coordinates for letterbox."""
                if not objects_json_bytes.numpy():
                    return b"", b""

                try:
                    objects = json.loads(objects_json_bytes.numpy().decode("utf-8"))
                    if not objects:
                        return b"", b""

                    # Decode image to get original dimensions
                    img_data = img_bytes_tensor.numpy()
                    if len(img_data) == 0:
                        return b"", b""

                    # Use TensorFlow to decode and get shape
                    import io
                    from PIL import Image
                    try:
                        img = Image.open(io.BytesIO(img_data))
                        orig_w, orig_h = img.size
                    except Exception:
                        # Default to common DROID wrist image size
                        orig_w, orig_h = 640, 480

                    # Compute letterbox transformation parameters
                    # Same logic as _tf_resize_with_pad
                    ratio = max(orig_w / target_w, orig_h / target_h)
                    resized_w = int(orig_w / ratio)
                    resized_h = int(orig_h / ratio)
                    pad_w = (target_w - resized_w) / 2.0
                    pad_h = (target_h - resized_h) / 2.0

                    # Build prompt with all object labels
                    labels = [obj["label"] for obj in objects]
                    unique_labels = list(dict.fromkeys(labels))  # Preserve order, remove duplicates
                    prompt_labels = ", ".join(unique_labels)

                    # Build caption with all bboxes
                    # Format: "<loc...> label1 ; <loc...> label2 ; ..."
                    caption_parts = []
                    for obj in objects:
                        label = obj["label"]
                        bbox = obj["bbox"]  # [x_min, y_min, x_max, y_max] normalized 0-1

                        # Transform bbox coordinates for letterbox
                        # Original coords are normalized (0-1), need to transform to letterboxed space
                        x_min = bbox[0] * (resized_w / target_w) + (pad_w / target_w)
                        y_min = bbox[1] * (resized_h / target_h) + (pad_h / target_h)
                        x_max = bbox[2] * (resized_w / target_w) + (pad_w / target_w)
                        y_max = bbox[3] * (resized_h / target_h) + (pad_h / target_h)

                        # Clamp to valid range
                        x_min = max(0.0, min(1.0, x_min))
                        y_min = max(0.0, min(1.0, y_min))
                        x_max = max(0.0, min(1.0, x_max))
                        y_max = max(0.0, min(1.0, y_max))

                        # Convert to loc tokens (PaLiGemma format: y_min, x_min, y_max, x_max)
                        N = 1024
                        y_min_idx = int(round(y_min * (N - 1)))
                        x_min_idx = int(round(x_min * (N - 1)))
                        y_max_idx = int(round(y_max * (N - 1)))
                        x_max_idx = int(round(x_max * (N - 1)))

                        loc_str = f"<loc{y_min_idx:04d}><loc{x_min_idx:04d}><loc{y_max_idx:04d}><loc{x_max_idx:04d}>"
                        caption_parts.append(f"{loc_str} {label}")

                    caption = " ; ".join(caption_parts)

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
            # Generate prompt asking about all objects
            labels = frame["object_labels"]
            caption = frame["bbox_caption"]

            # Sample a prompt template
            seed_key = tf.strings.join([frame["trajectory_id"], "_bbox_", frame["frame_idx"]])
            seed_hash = tf.strings.to_hash_bucket_fast(seed_key, 2147483647)
            seed_hash_int = tf.cast(seed_hash, tf.int32)

            num_prompts = len(DROID_BBOX_PROMPT_PARTS)
            prompt_idx = tf.random.stateless_uniform(
                [], seed=[self.seed, seed_hash_int], minval=0, maxval=num_prompts, dtype=tf.int32
            )

            # Create branches for each prompt template
            def make_prompt_fn(idx):
                prefix, suffix = DROID_BBOX_PROMPT_PARTS[idx]

                def fn():
                    return tf.strings.join([prefix, labels, suffix])

                return fn

            prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
            prompt = tf.switch_case(prompt_idx, branch_fns=prompt_branches)

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
        """Build a lookup table from episode_path--frame_idx to list of objects."""
        logging.info("Building frame objects lookup table...")
        import re

        frame_to_objects = {}
        sample_keys_logged = 0

        jsonl_files = tf.io.gfile.glob(os.path.join(self.bbox_annotations_dir, "*.jsonl"))

        for jsonl_file in jsonl_files:
            with tf.io.gfile.GFile(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        episode_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    file_path = episode_data.get("episode_metadata", {}).get("file_path", "")
                    if not file_path:
                        continue

                    # Extract episode path using the same logic as extract_episode_path_from_file_path
                    # Remove prefix up to r2d2-data or r2d2-data-full
                    rel = re.sub(r"^.*r2d2-data(?:-full)?/", "", file_path)
                    # Remove /trajectory... suffix
                    episode_path = re.sub(r"/trajectory.*$", "", rel)

                    if not episode_path:
                        continue

                    labels = episode_data.get("labels", [])
                    for label_entry in labels:
                        frame_idx = label_entry.get("frame")
                        all_objects = label_entry.get("all_objects", [])

                        if frame_idx is None or not all_objects:
                            continue

                        key = f"{episode_path}--{frame_idx}"

                        objects_list = []
                        for obj in all_objects:
                            obj_label = obj.get("label", "")
                            bbox = obj.get("bbox", [])

                            if not obj_label or len(bbox) < 4:
                                continue

                            # Normalize bbox
                            x_min = max(0.0, min(1.0, float(bbox[0]) / 1000.0))
                            y_min = max(0.0, min(1.0, float(bbox[1]) / 1000.0))
                            x_max = max(0.0, min(1.0, float(bbox[2]) / 1000.0))
                            y_max = max(0.0, min(1.0, float(bbox[3]) / 1000.0))

                            objects_list.append({
                                "label": obj_label,
                                "bbox": [x_min, y_min, x_max, y_max],
                            })

                        if objects_list:
                            if key in frame_to_objects:
                                frame_to_objects[key].extend(objects_list)
                            else:
                                frame_to_objects[key] = objects_list

        # Convert to lookup table
        keys = []
        values = []
        for k, v in frame_to_objects.items():
            keys.append(k)
            values.append(json.dumps(v))

        logging.info(f"Built frame objects table with {len(keys)} entries")

        if not keys:
            # Return table with dummy entry (TF doesn't allow empty tables)
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    tf.constant(["__dummy_key__"], dtype=tf.string),
                    tf.constant([""], dtype=tf.string),
                ),
                default_value=tf.constant(b"", dtype=tf.string),
            )

        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(keys, dtype=tf.string),
                tf.constant(values, dtype=tf.string),
            ),
            default_value=tf.constant(b"", dtype=tf.string),
        )

    def bbox_to_text(self, bbox: tf.Tensor) -> tf.Tensor:
        """Convert bbox to formatted text representation using paligemma loc tokens.

        Args:
            bbox: Tensor of shape [4] with normalized coordinates [x_min, y_min, x_max, y_max]
                  where coordinates are in range [0, 1].

        Returns:
            Formatted string in PaLiGemma2 bbox format: "<loc_ymin><loc_xmin><loc_ymax><loc_xmax>".
        """
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        # Convert to paligemma loc token indices (0-1023)
        N = 1024
        y_min_idx = tf.cast(tf.round(y_min * (N - 1)), tf.int32)
        x_min_idx = tf.cast(tf.round(x_min * (N - 1)), tf.int32)
        y_max_idx = tf.cast(tf.round(y_max * (N - 1)), tf.int32)
        x_max_idx = tf.cast(tf.round(x_max * (N - 1)), tf.int32)

        # Format as loc tokens in PaLiGemma2 order: y_min, x_min, y_max, x_max
        y_min_token = tf.strings.join(["<loc", tf.strings.as_string(y_min_idx, width=4, fill="0"), ">"])
        x_min_token = tf.strings.join(["<loc", tf.strings.as_string(x_min_idx, width=4, fill="0"), ">"])
        y_max_token = tf.strings.join(["<loc", tf.strings.as_string(y_max_idx, width=4, fill="0"), ">"])
        x_max_token = tf.strings.join(["<loc", tf.strings.as_string(x_max_idx, width=4, fill="0"), ">"])

        # Join in PaLiGemma2 order: y_min, x_min, y_max, x_max
        return tf.strings.join([y_min_token, x_min_token, y_max_token, x_max_token])

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "droid_bbox"

    def get_num_transitions(self) -> int:
        """Return approximate number of transitions."""
        # Estimate based on typical DROID dataset size
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
