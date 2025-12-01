"""Base class and registry for VQA datasets."""

from typing import TYPE_CHECKING, ClassVar

from dlimp import DLataset
import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.base_dataset import _SingleCoTDataset
from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.helpers import NormalizationType
from openpi_cot.dataloader.helpers import StateEncoding

if TYPE_CHECKING:
    from openpi_cot.training.config import CoTDataConfig

from openpi_cot.dataloader.dataset_utils import dataset_size

# Registry of VQA dataset names
VQA_DATASET_NAMES: set[str] = {"coco_captions", "vqa", "pixmo_cap", "pixmo_point", "lvis"}


def ensure_dldataset(ds, is_flattened=False):
    """Ensure dataset is a DLataset instance."""
    if isinstance(ds, tf.data.Dataset) and not isinstance(ds, DLataset):
        Original = ds.__class__
        MixedDL = type("DLatasetWrapped", (DLataset, Original), {})
        ds.__class__ = MixedDL
        ds.is_flattened = is_flattened
    return ds


class _BaseVQADataset(_SingleCoTDataset):
    """Base class for VQA datasets (COCO Captions, VQAv2, etc.).

    This class contains shared functionality for all VQA datasets:
    - No language actions in the traditional sense
    - Single frame (no temporal structure)
    - No proprioceptive state
    - Dummy actions for compatibility
    - Caption-based outputs

    Subclasses must implement:
    - build_dataset_builder(): Return TFDS builder
    - build_dataset(): Build TensorFlow dataset from builder
    - get_dataset_name(): Return the dataset name string
    - get_num_transitions(): Return approximate number of samples for statistics
    - restructure_example(): Convert dataset-specific example to standard format
    """

    # Dataset-specific name (to be set by subclasses)
    DATASET_NAME: ClassVar[str] = ""

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
        pred_prob: float | None = None,
        primary_pred_prob: float | None = None,
    ):
        # VQA datasets don't have language actions in the traditional sense
        self.use_json_actions = False

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
        self.action_horizon = action_horizon

        # VQA-specific settings (shared across all VQA datasets)
        self.control_frequency = 1  # Single frame, no temporal control
        self.image_obs_keys = {"primary": None}  # Will map VQA image to primary
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

        # Build TFDS dataset (subclass-specific)
        self.builder = self.build_dataset_builder(dataset_name, data_dir)

        # Build dataset (subclass-specific)
        self.dataset = self.build_dataset(self.builder, split)

        # Split train/val (only if using train/val split)
        if split in ["train", "val"]:
            self.split_val(split_seed=seed)

        self.apply_vqa_restructure()

        # Apply minimal transforms
        self.apply_vqa_transforms()

        # Apply frame filters
        self.apply_vqa_frame_filters()

        ds = dataset_size(self.dataset)
        breakpoint()

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
                mean=np.zeros(self.action_dim, dtype=np.float32),
                std=np.ones(self.action_dim, dtype=np.float32),
                q01=np.zeros(self.action_dim, dtype=np.float32),
                q99=np.zeros(self.action_dim, dtype=np.float32),
                num_transitions=num_transitions,
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

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement build_dataset_builder")

    def build_dataset(self, builder, split: str):
        """Build TensorFlow dataset from TFDS builder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement build_dataset")

    def get_dataset_name(self) -> str:
        """Get the dataset name for metadata. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement get_dataset_name")

    def get_num_transitions(self) -> int:
        """Get approximate number of transitions for statistics. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement get_num_transitions")

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from example. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement create_trajectory_id")

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from example. Must be implemented by subclass.

        Returns:
            (prompt, caption) tuple of tensors
        """
        raise NotImplementedError("Subclass must implement extract_prompt_and_caption")

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode image from example. Must be implemented by subclass.

        Returns:
            Encoded image bytes
        """
        raise NotImplementedError("Subclass must implement extract_and_encode_image")

    def split_val(self, split_seed: int):
        """Split dataset into train/val using consistent hashing."""

        def _split_filter(example):
            # Use trajectory_id for consistent splitting
            salt = tf.strings.as_string(split_seed)
            trajectory_id = self.create_trajectory_id(example)
            key = tf.strings.join([salt, trajectory_id])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            thr = tf.cast(int(self.val_fraction * 1000), tf.int64)
            is_val = bucket < thr
            return is_val if self.want_val else tf.logical_not(is_val)

        self.dataset = self.dataset.filter(_split_filter)

    def apply_vqa_restructure(self):
        """Restructure VQA data to match robot dataset format."""

        def restructure(example):
            """Convert VQA example to robot dataset structure."""
            # Extract and encode image (subclass-specific)
            image_encoded = self.extract_and_encode_image(example)

            # Create trajectory_id (subclass-specific)
            trajectory_id = self.create_trajectory_id(example)

            # Extract prompt and caption (subclass-specific)
            prompt, caption = self.extract_prompt_and_caption(example)

            # Create observation dict (shared)
            observation = {
                self.spec.primary_image_key: image_encoded,
                self.spec.wrist_image_key: tf.constant("", dtype=tf.string),
                self.spec.wrist_image_right_key: tf.constant("", dtype=tf.string),
                "state": tf.zeros([self.action_dim], dtype=tf.float32),
            }

            # Create output matching robot dataset structure (shared)
            output = {
                "observation": observation,
                "prompt": prompt,
                "caption": caption,
                "dataset_name": tf.constant(self.get_dataset_name(), dtype=tf.string),
                "control_frequency": tf.constant(self.control_frequency, dtype=tf.int32),
                "is_bimanual": tf.constant(False, dtype=tf.bool),
                "state_type": tf.constant("none", dtype=tf.string),
                "is_vqa_sample": tf.constant(True, dtype=tf.bool),
                "is_prediction_sample": tf.constant(False, dtype=tf.bool),
                "pred_use_primary": tf.constant(False, dtype=tf.bool),
                "raw_state": tf.zeros([self.action_dim], dtype=tf.float32),
            }

            return output

        self.dataset = self.dataset.frame_map(
            restructure,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_transforms(self):
        """Apply VQA-specific transforms (shared across all VQA datasets)."""

        def transform(example):
            """Add required fields for training."""
            # Create chunked actions (all zeros for VQA)
            example["actions"] = tf.zeros(
                [self.action_horizon, self.action_dim],
                dtype=tf.float32,
            )

            # Create dummy language_actions (empty serialized tensors)
            # Use 30 to match robot datasets' default summation_steps
            summation_steps = getattr(self.config, "summation_steps", 30)
            dummy_tensor = tf.zeros([self.action_dim], dtype=tf.float32)
            serialized = tf.io.serialize_tensor(dummy_tensor)
            example["language_actions"] = tf.repeat(serialized, summation_steps)

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

            return example

        self.dataset = self.dataset.frame_map(
            transform,
            num_parallel_calls=self.num_parallel_calls,
        )

    def apply_vqa_frame_filters(self):
        """Filter out samples with empty questions or captions (shared)."""

        def has_valid_qa(example):
            """Check if question/prompt and answer/caption are non-empty."""
            has_question = tf.strings.length(example["prompt"]) > 0
            has_answer = tf.strings.length(example["caption"]) > 0
            return tf.logical_and(has_question, has_answer)

        self.dataset = self.dataset.filter(has_valid_qa)
