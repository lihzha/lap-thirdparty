"""PixmoPoint dataset implementation for VQA training."""

import tensorflow as tf

from openpi_cot.dataloader.vqa_base import _BaseVQADataset
from openpi_cot.dataloader.vqa_base import ensure_dldataset

# Maximum number of points to include in the answer
MAX_POINTS = 20

# Prompts for PixmoPoint dataset - split into prefix and suffix for label insertion
PIXMO_POINT_PROMPT_PARTS = [
    ("How many ", " are in the image? Point them out."),
    ("Identify all ", " in the image by pointing to them."),
    ("Point out all the ", " in this image."),
    ("Where are the ", " in the image? Point to each one."),
    ("Locate all ", " in the image and point them out."),
]


class PixmoPoint(_BaseVQADataset):
    """PixmoPoint dataset for vision-language training with point annotations.

    This dataset loads images with point annotations and formats them to be
    compatible with the robot dataset structure. Points are converted to text
    format with normalized coordinates.
    """

    def __init__(self, *args, max_points: int = MAX_POINTS, scale: float = 100.0, **kwargs):
        """Initialize PixmoPoint dataset.

        Args:
            max_points: Maximum number of points to include (extras are dropped).
            scale: Scale factor for normalizing coordinates (default 100).
            *args, **kwargs: Passed to base class.
        """
        self.max_points = max_points
        self.scale = scale
        super().__init__(*args, **kwargs)

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for PixmoPoint."""
        import tensorflow_datasets as tfds

        _ = ds_name  # Unused but required by base class signature
        return tfds.builder("pixmo_point", data_dir=data_dir)

    def build_dataset(self, builder, split: str):
        """Build TensorFlow dataset from TFDS builder."""
        import tensorflow_datasets as tfds

        _ = split  # Unused but required by base class signature
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
            split="train",  # PixmoPoint only has train split, we'll manually split for val
            shuffle_files=not self.want_val,
            read_config=read_config,
        )

        ds = ensure_dldataset(ds, is_flattened=True)
        return ds

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "pixmo_point"

    def get_num_transitions(self) -> int:
        """Return approximate number of PixmoPoint samples."""
        # TODO: Update with actual count when known
        return 100000  # Placeholder

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from PixmoPoint image URL and SHA256."""
        image_sha = example["image_sha256"]
        return tf.strings.join(["pixmo_point_", image_sha])

    def points_to_text(
        self,
        points: tf.Tensor,
        # count_text: tf.Tensor,
    ) -> tf.Tensor:
        """Convert points to formatted text representation using paligemma loc tokens.

        Args:
            points: Tensor of shape [N, 2] with x, y coordinates (already normalized to 0-100).

        Returns:
            Formatted string with points as paligemma loc tokens like "<loc0252><loc0233>".
        """
        # Sort points by x*10000 + y for consistent ordering
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        sort_keys = x_coords * 10000.0 + y_coords
        sorted_indices = tf.argsort(sort_keys)
        sorted_points = tf.gather(points, sorted_indices)

        # Extract sorted x and y coordinates
        x_coords = sorted_points[:, 0]
        y_coords = sorted_points[:, 1]

        # Convert to paligemma loc token indices (0-1023)
        N = 1024
        x_indices = tf.cast(tf.round(x_coords / 100.0 * (N - 1)), tf.int32)
        y_indices = tf.cast(tf.round(y_coords / 100.0 * (N - 1)), tf.int32)

        # Format as loc tokens: <locYYYY><locXXXX> for each point
        y_tokens = tf.strings.join(["<loc", tf.strings.as_string(y_indices, width=4, fill="0"), ">"])
        x_tokens = tf.strings.join(["<loc", tf.strings.as_string(x_indices, width=4, fill="0"), ">"])

        # Interleave y and x tokens (y comes first for each point)
        yx_pairs = tf.reshape(tf.stack([y_tokens, x_tokens], axis=1), [-1])

        # Join all tokens together without separator
        return tf.strings.reduce_join(yx_pairs, separator="")

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from PixmoPoint example.

        Returns:
            (prompt, caption) where prompt asks about the label and caption
            contains the formatted points.
        """
        label = example["label"]
        points_data = example["points"]
        count = example["count"]

        # Extract x, y coordinates and stack them
        x_coords = points_data["x"]
        y_coords = points_data["y"]
        points = tf.stack([x_coords, y_coords], axis=1)  # Shape: [N, 2]

        # Normalize to 0-100 scale and round to 1 decimal place
        points = points * (100.0 / self.scale)
        points = tf.round(points * 10.0) / 10.0  # Round to 1 decimal

        # Limit to max_points
        num_points = tf.shape(points)[0]
        num_points_capped = tf.minimum(num_points, self.max_points)
        points = points[:num_points_capped]

        # Generate deterministic seed from image SHA256
        image_sha = example["image_sha256"]
        sha_hash = tf.strings.to_hash_bucket_fast(image_sha, 2147483647)
        sha_hash_int = tf.cast(sha_hash, tf.int32)

        # Randomly select a prompt template using tf.switch_case
        num_prompts = len(PIXMO_POINT_PROMPT_PARTS)
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, sha_hash_int], minval=0, maxval=num_prompts, dtype=tf.int32
        )

        # Create branches for each prompt template
        def make_prompt_fn(idx):
            prefix, suffix = PIXMO_POINT_PROMPT_PARTS[idx]

            def fn():
                return tf.strings.join([prefix, label, suffix])

            return fn

        prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
        prompt = tf.switch_case(prompt_idx, branch_fns=prompt_branches)

        # Format points as answer/caption
        # count_text = tf.strings.join([tf.string(count), " ", label])
        caption = self.points_to_text(points)

        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode PixmoPoint image to JPEG bytes."""
        image = example["image"]
        # Images should already be JPEG encoded in the TFDS dataset
        # If not, encode them
        # Check dtype instead of rank to avoid TensorFlow graph tracing issues
        if image.dtype != tf.string:  # If it's a decoded image tensor
            return tf.io.encode_jpeg(image, quality=95)
        # Already encoded
        return image
