"""LVIS dataset implementation for VQA training."""

import tensorflow as tf

from openpi_cot.dataloader.vqa_base import _BaseVQADataset
from openpi_cot.dataloader.vqa_base import ensure_dldataset

# LVIS prompts to randomly sample from
LVIS_PROMPTS = tf.constant(
    [
        "What objects are in this image?",
        "Describe the objects you see.",
        "List the objects in the image.",
        "What can you see in this image?",
        "Identify the objects present.",
        "Describe what is visible in the image.",
    ],
    dtype=tf.string,
)


class Lvis(_BaseVQADataset):
    """LVIS dataset for vision-language training.

    This dataset loads LVIS images with object annotations and formats them to be
    compatible with the robot dataset structure. Each image contains multiple object
    annotations with category labels.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for LVIS."""
        import tensorflow_datasets as tfds

        return tfds.builder("lvis:1.0.0", data_dir=data_dir)

    def build_dataset(self, builder, split: str):
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
            split="train",  # LVIS train split, we'll manually split for val
            shuffle_files=not self.want_val,
            read_config=read_config,
        )

        ds = ensure_dldataset(ds, is_flattened=True)
        return ds

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "lvis"

    def get_num_transitions(self) -> int:
        """Return approximate number of LVIS samples."""
        return 100170  # Approximate number of LVIS train images

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from LVIS image ID."""
        image_id = example["image_id"]
        return tf.strings.join(["lvis_", image_id])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from LVIS example.

        Returns:
            (prompt, caption) where prompt is randomly sampled from LVIS_PROMPTS
            and caption is a description of all objects in the image.
        """
        # Generate deterministic seed from image ID hash
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        # Extract all category names from annotations
        category_names = example["annotations"]["category_name"]

        # Create caption by joining all unique object names
        # Use unique to avoid repetition
        unique_categories = tf.unique(category_names)[0]

        # Sort for consistent ordering
        unique_categories = tf.sort(unique_categories)

        # Join with commas
        caption = tf.strings.reduce_join(unique_categories, separator=", ")

        # Randomly select a prompt
        num_prompts = tf.shape(LVIS_PROMPTS)[0]
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, image_id_hash], minval=0, maxval=num_prompts, dtype=tf.int32
        )
        prompt = LVIS_PROMPTS[prompt_idx]

        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode LVIS image to JPEG bytes."""
        image = example["image"]
        # LVIS images are not pre-encoded, so encode them
        return tf.io.encode_jpeg(image, quality=95)
