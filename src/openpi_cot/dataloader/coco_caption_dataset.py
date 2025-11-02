"""COCO Captions dataset implementation for VQA training."""

from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.base_dataset import _SingleCoTDataset
from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import StateEncoding

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig


# VQA prompts to randomly sample from
VQA_PROMPTS = tf.constant(
    [
        "Describe the image.",
        "Describe what is in the image.",
        "What is in the image?",
        "Tell me what the image is about.",
        "What do you see in this image?",
        "Describe what you see.",
    ],
    dtype=tf.string,
)


class CocoCaption(_SingleCoTDataset):
    """COCO Captions dataset for vision-language training.

    This dataset loads COCO images with captions and formats them to be
    compatible with the robot dataset structure, padding unused fields
    with zeros or empty values.
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        dataset_name: str,
        data_dir: str,
        config: "CoTDataConfig",
        action_horizon: int = 16,
        action_dim: int = 32,
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
        # VQA datasets don't have language actions in the traditional sense
        self.use_json_actions = False

        # Override parent initialization to skip robot-specific setup
        # Store basic config
        self.config = config
        self.seed = seed
        self.want_val = split == "val"
        self.dataset_name = dataset_name
        self.action_dim = action_dim
        self.vis_dataset = bool(config.vis_dataset)
        self.action_proprio_normalization_type = action_proprio_normalization_type
        self.use_wrist_image = False  # VQA has no wrist images
        self.standalone = standalone
        self.skip_normalization = skip_normalization
        self.enable_prediction_training = enable_prediction_training
        self.action_horizon = action_horizon

        # VQA-specific settings
        self.control_frequency = 1  # Single frame, no temporal control
        self.image_obs_keys = {"primary": None}  # Will map COCO image to primary
        self.state_obs_keys = []  # No proprioceptive state
        self.state_encoding = StateEncoding.NONE
        self.action_encoding = ActionEncoding.EEF_POS
        self.is_bimanual = False

        self.num_parallel_reads = tf.data.AUTOTUNE if num_parallel_reads == -1 else num_parallel_reads
        self.num_parallel_calls = tf.data.AUTOTUNE if num_parallel_calls == -1 else num_parallel_calls
        self.val_fraction = getattr(self.config, "val_fraction", 0.02)

        # Configure TensorFlow
        tf.config.set_visible_devices([], "GPU")
        try:
            tf.config.set_visible_devices([], "TPU")
        except Exception:
            pass

        tf.random.set_seed(self.seed)

        # Build TFDS dataset
        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Build dataset
        self.dataset = self.build_dataset(self.builder)

        # No trajectory identifier for VQA (each sample is independent)
        # No trajectory filters

        # Split train/val
        self.split_val(split_seed=seed)

        # Restructure to match robot dataset format
        self.apply_vqa_restructure()

        # Apply minimal transforms (no action chunking for VQA)
        self.apply_vqa_transforms()

        # No repack needed (no trajectory metadata to remove)

        # VQA is already "flat" (no trajectories)
        # But we need to match expected structure

        # Apply frame filters (remove samples with empty captions)
        self.apply_vqa_frame_filters()

        # Create dummy statistics for compatibility
        from openpi_cot.shared.adapters.normalize_adapter import ExtendedNormStats

        self.dataset_statistics = {
            "actions": ExtendedNormStats(
                mean=np.zeros(self.action_dim, dtype=np.float32),
                std=np.ones(self.action_dim, dtype=np.float32),
                q01=np.zeros(self.action_dim, dtype=np.float32),
                q99=np.zeros(self.action_dim, dtype=np.float32),
                num_transitions=82783,
                num_trajectories=0,
            ),
            "state": ExtendedNormStats(
                mean=np.zeros(self.action_dim, dtype=np.float32),
                std=np.ones(self.action_dim, dtype=np.float32),
                q01=np.zeros(self.action_dim, dtype=np.float32),
                q99=np.zeros(self.action_dim, dtype=np.float32),
                num_transitions=82783,
                num_trajectories=0,
            ),
        }

        if standalone:
            from openpi_cot.dataloader.dataset_utils import prepare_batched_dataset

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

            self._pre_batched_dataset = self.dataset

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
        """Build TFDS builder for COCO captions."""
        import tensorflow_datasets as tfds

        return tfds.builder("coco_captions", data_dir=data_dir)

    def build_dataset(self, builder):
        """Build TensorFlow dataset from TFDS builder."""
        import tensorflow_datasets as tfds

        opts = tf.data.Options()
        opts.experimental_deterministic = bool(self.want_val)
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_fusion = True

        read_config = tfds.ReadConfig(
            shuffle_seed=self.seed,
            options=opts,
        )

        ds = builder.as_dataset(
            split="train",  # COCO only has train split, we'll manually split for val
            shuffle_files=not self.want_val,
            read_config=read_config,
        )

        return ds

    def split_val(self, split_seed):
        """Override split_val to work with COCO's native structure."""

        def _split_filter(example):
            # Create trajectory_id from COCO's image/filename and image/id
            salt = tf.strings.as_string(split_seed)
            image_filename = example["image/filename"]
            image_id = tf.strings.as_string(example["image/id"])
            anchor = tf.strings.join(["coco_", image_filename, "_", image_id])
            key = tf.strings.join([salt, anchor])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def apply_vqa_restructure(self):
        """Restructure COCO data to match robot dataset format."""

        def restructure(example):
            """Convert COCO example to robot dataset structure.

            COCO input:
                - image: [H, W, 3] uint8
                - captions: [{'id': int64, 'text': string}, ...]
                - image/id: int64
                - image/filename: string

            Output:
                - observation/base_0_rgb: encoded image bytes
                - observation/left_wrist_0_rgb: "" (empty)
                - observation/right_wrist_0_rgb: "" (empty)
                - observation/state: zeros [action_dim]
                - action: zeros [action_dim]
                - prompt: randomly sampled VQA prompt
                - caption: randomly sampled caption text
                - dataset_name: "coco_caption"
                - control_frequency: 1
                - is_bimanual: False
            """
            # Extract image
            image = example["image"]

            # Encode image to bytes (to match robot dataset format)
            image_encoded = tf.io.encode_jpeg(image, quality=95)

            # Create trajectory_id from image filename and ID
            # Format: "coco_<filename>_<image_id>"
            image_filename = example["image/filename"]
            image_id = tf.strings.as_string(example["image/id"])
            trajectory_id = tf.strings.join(["coco_", image_filename, "_", image_id])

            # Randomly select one caption from the list
            captions = example["captions"]["text"]
            num_captions = tf.shape(captions)[0]
            caption_idx = tf.random.uniform([], 0, num_captions, dtype=tf.int32, seed=self.seed)
            selected_caption = captions[caption_idx]

            # Randomly select a VQA prompt
            num_prompts = tf.shape(VQA_PROMPTS)[0]
            prompt_idx = tf.random.uniform([], 0, num_prompts, dtype=tf.int32, seed=self.seed)
            prompt = VQA_PROMPTS[prompt_idx]

            # Create observation dict
            observation = {
                self.spec.primary_image_key: image_encoded,
                self.spec.wrist_image_key: tf.constant("", dtype=tf.string),
                self.spec.wrist_image_right_key: tf.constant("", dtype=tf.string),
                "state": tf.zeros([self.action_dim], dtype=tf.float32),
            }

            # Create output matching robot dataset structure
            output = {
                "observation": observation,
                "action": tf.zeros([self.action_dim], dtype=tf.float32),
                "prompt": prompt,
                "caption": selected_caption,
                "dataset_name": tf.constant("coco_caption", dtype=tf.string),
                "control_frequency": tf.constant(self.control_frequency, dtype=tf.int32),
                "is_bimanual": tf.constant(False, dtype=tf.bool),
                "enable_prediction_training_mask": tf.constant(False, dtype=tf.bool),
                "trajectory_id": tf.expand_dims(trajectory_id, axis=0),  # [1] for compatibility with split_val
            }

            return output

        self.dataset = self.dataset.map(
            restructure,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_transforms(self):
        """Apply VQA-specific transforms.

        For VQA, we need to:
        1. Create dummy actions (zeros)
        2. Create dummy language_actions (for compatibility)
        3. Add time dimension to images (single frame)
        """

        def transform(example):
            """Add required fields for training."""
            # Create chunked actions (all zeros for VQA)
            example["actions"] = tf.zeros(
                [self.action_horizon, self.action_dim],
                dtype=tf.float32,
            )

            # Create dummy language_actions (empty serialized tensors)
            # Use summation_steps from config or default to 1
            summation_steps = getattr(self.config, "summation_steps", 1)
            dummy_tensor = tf.zeros([self.action_dim], dtype=tf.float32)
            serialized = tf.io.serialize_tensor(dummy_tensor)
            example["language_actions"] = tf.repeat(serialized, summation_steps)

            # Add time dimension to images (single frame for VQA)
            if not self.enable_prediction_training:
                # Single frame: [H, W, 3] -> [1, H, W, 3]
                example["observation"][self.spec.primary_image_key] = tf.expand_dims(
                    example["observation"][self.spec.primary_image_key], axis=0
                )
                example["observation"][self.spec.wrist_image_key] = tf.expand_dims(
                    example["observation"][self.spec.wrist_image_key], axis=0
                )
                example["observation"][self.spec.wrist_image_right_key] = tf.expand_dims(
                    example["observation"][self.spec.wrist_image_right_key], axis=0
                )
            else:
                # Prediction training: duplicate frame
                example["observation"][self.spec.primary_image_key] = tf.stack(
                    [
                        example["observation"][self.spec.primary_image_key],
                        example["observation"][self.spec.primary_image_key],
                    ],
                    axis=0,
                )
                example["observation"][self.spec.wrist_image_key] = tf.stack(
                    [
                        example["observation"][self.spec.wrist_image_key],
                        example["observation"][self.spec.wrist_image_key],
                    ],
                    axis=0,
                )
                example["observation"][self.spec.wrist_image_right_key] = tf.stack(
                    [
                        example["observation"][self.spec.wrist_image_right_key],
                        example["observation"][self.spec.wrist_image_right_key],
                    ],
                    axis=0,
                )

            return example

        self.dataset = self.dataset.map(
            transform,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_frame_filters(self):
        """Filter out samples with empty captions."""

        def has_valid_caption(example):
            """Check if caption is non-empty."""
            return tf.strings.length(example["caption"]) > 0

        self.dataset = self.dataset.filter(has_valid_caption)
