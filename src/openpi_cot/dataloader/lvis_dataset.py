"""LVIS dataset implementation for VQA training."""

import tensorflow as tf

from openpi_cot.dataloader.vqa_base import _BaseVQADataset
from openpi_cot.dataloader.vqa_base import ensure_dldataset

# LVIS prompts to randomly sample from - split into prefix and suffix for category insertion
LVIS_PROMPT_PARTS = [
    ("Point out the ", " in the image."),
    ("Where is the ", " in the image?"),
    ("Locate the ", " in this image."),
    ("Identify the ", " in the image."),
    ("Find the ", " in the image."),
    ("Show me where the ", " is."),
]


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

    def bbox_to_text(self, bbox: tf.Tensor) -> tf.Tensor:
        """Convert bbox to formatted text representation using paligemma loc tokens.

        Args:
            bbox: Tensor of shape [2, 2] with normalized coordinates [[x1, y1], [x2, y2]]
                  where coordinates are in range [0, 1].

        Returns:
            Formatted string with bbox corners as paligemma loc tokens like "<locY1><locX1><locY2><locX2>".
        """
        # Extract corner coordinates (already normalized to 0-1)
        top_left = bbox[0]  # [x1, y1]
        bottom_right = bbox[1]  # [x2, y2]

        x1, y1 = top_left[0], top_left[1]
        x2, y2 = bottom_right[0], bottom_right[1]

        # Convert to paligemma loc token indices (0-1023)
        N = 1024
        x1_idx = tf.cast(tf.round(x1 * (N - 1)), tf.int32)
        y1_idx = tf.cast(tf.round(y1 * (N - 1)), tf.int32)
        x2_idx = tf.cast(tf.round(x2 * (N - 1)), tf.int32)
        y2_idx = tf.cast(tf.round(y2 * (N - 1)), tf.int32)

        # Format as loc tokens: <locY1><locX1><locY2><locX2>
        y1_token = tf.strings.join(["<loc", tf.strings.as_string(y1_idx, width=4, fill="0"), ">"])
        x1_token = tf.strings.join(["<loc", tf.strings.as_string(x1_idx, width=4, fill="0"), ">"])
        y2_token = tf.strings.join(["<loc", tf.strings.as_string(y2_idx, width=4, fill="0"), ">"])
        x2_token = tf.strings.join(["<loc", tf.strings.as_string(x2_idx, width=4, fill="0"), ">"])

        # Join all tokens: top-left then bottom-right
        return tf.strings.join([y1_token, x1_token, y2_token, x2_token])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from LVIS example.

        Returns:
            (prompt, caption) where prompt asks about the specific category
            and caption contains the formatted bbox.
        """
        # Generate deterministic seed from image ID hash
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        # Extract the single category name (each example has one annotation)
        category_name = example["annotations"]["category_name"][0]

        # Extract the bbox (shape: [2, 2] with [[x1, y1], [x2, y2]])
        bbox = example["annotations"]["bbox"][0]

        # Randomly select a prompt template using tf.switch_case
        num_prompts = len(LVIS_PROMPT_PARTS)
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, image_id_hash], minval=0, maxval=num_prompts, dtype=tf.int32
        )

        # Create branches for each prompt template
        def make_prompt_fn(idx):
            prefix, suffix = LVIS_PROMPT_PARTS[idx]

            def fn():
                return tf.strings.join([prefix, category_name, suffix])

            return fn

        prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
        prompt = tf.switch_case(prompt_idx, branch_fns=prompt_branches)

        # Format bbox as caption using loc tokens
        caption = self.bbox_to_text(bbox)

        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode LVIS image to JPEG bytes."""
        image = example["image"]
        # LVIS images are not pre-encoded, so encode them
        return tf.io.encode_jpeg(image, quality=95)
