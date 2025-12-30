"""LVIS dataset implementation for VQA training."""

import tensorflow as tf
import tensorflow_datasets as tfds

from openpi_cot.datasets.vqa.vqa_base import BaseVQADataset

# LVIS prompts to randomly sample from - split into prefix and suffix for category insertion
LVIS_PROMPT_PARTS = [
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
    ("If any ", " are present, show one bounding box."),
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

# Prompts for direction classification relative to image center
DIRECTION_PROMPT_PARTS = [
    ("From the image center, which direction is the ", " located?"),
    ("Relative to the center point of the image, where is the ", "?"),
    ("If you stand at the center of the image, which way do you go to reach the ", "?"),
    ("Looking from the center of the frame, in what direction is the ", " situated?"),
    ("Which direction from the center is the ", " in this image?"),
]


class Lvis(BaseVQADataset):
    """LVIS dataset for vision-language training.

    This dataset loads LVIS images with object annotations and formats them to be
    compatible with the robot dataset structure. Each image contains multiple object
    annotations with category labels.
    """

    def __init__(self, *args, directional: bool = False, direction_slope: float = 2.0, **kwargs):
        self.directional = directional
        self.direction_slope = direction_slope
        super().__init__(*args, **kwargs)

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for LVIS."""
        return tfds.builder("lvis:1.0.0", data_dir=data_dir)

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "lvis"

    def get_num_transitions(self) -> int:
        """Return approximate number of LVIS samples."""
        return 1231766  # Approximate number of LVIS train images

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from LVIS image ID and annotation.

        Since the same image can have multiple object annotations (different categories
        or multiple instances of the same category), we need to include annotation-specific
        information to ensure uniqueness.
        """
        image_id = example["image_id"]
        category_name = example["annotations"]["category_name"][0]
        bbox = example["annotations"]["bbox"][0]

        # Create a hash from bbox coordinates to handle multiple instances of same category
        bbox_flat = tf.reshape(bbox, [-1])
        bbox_str = tf.strings.reduce_join(tf.strings.as_string(bbox_flat), separator="_")
        combined = tf.strings.join([image_id, "_", category_name, "_", bbox_str])
        combined_hash = tf.strings.to_hash_bucket_fast(combined, 2147483647)
        combined_hash_str = tf.strings.as_string(combined_hash)
        return tf.strings.join(["lvis_", image_id, "_", combined_hash_str])

    def bbox_to_text(self, bbox: tf.Tensor) -> tf.Tensor:
        """Convert bbox to formatted text representation using paligemma loc tokens.

        Args:
            bbox: Tensor of shape [2, 2] with normalized coordinates [[x_min, y_min], [x_max, y_max]]
                  where coordinates are in range [0, 1].

        Returns:
            Formatted string in PaLiGemma2 bbox format: "<loc_ymin><loc_xmin><loc_ymax><loc_xmax>".
        """
        # Extract corner coordinates (already normalized to 0-1)
        # bbox[0] = [x_min, y_min], bbox[1] = [x_max, y_max]
        top_left = bbox[0]
        bottom_right = bbox[1]

        x_min, y_min = top_left[0], top_left[1]
        x_max, y_max = bottom_right[0], bottom_right[1]

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

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from LVIS example.

        Returns:
            (prompt, caption) where prompt asks about the specific category
            and caption contains the formatted bbox.
        """
        if self.directional:
            return self._extract_direction_prompt_and_caption(example)

        # Generate deterministic seed from image ID hash
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        # Extract the single category name (each example has one annotation)
        category_name = example["annotations"]["category_name"][0]

        # Extract the bbox (shape: [2, 2] with [[x_min, y_min], [x_max, y_max]])
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

    def _direction_from_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """Map bbox center to one of 8 direction strings relative to image center."""
        top_left = bbox[0]
        bottom_right = bbox[1]
        center = (top_left + bottom_right) / 2.0

        x_rel = center[0] - 0.5  # +x is right
        y_rel = 0.5 - center[1]  # invert so +y is up as described

        k = tf.constant(self.direction_slope, dtype=tf.float32)
        inv_k = 1.0 / k

        abs_x = tf.abs(x_rel)
        abs_y = tf.abs(y_rel)

        # Primary axis regions using slopes k and 1/k
        is_forward = y_rel >= k * abs_x
        is_back = y_rel <= -k * abs_x
        is_right = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
        is_right = tf.logical_and(is_right, x_rel >= inv_k * abs_y)
        is_left = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
        is_left = tf.logical_and(is_left, x_rel <= -inv_k * abs_y)

        def forward():
            return tf.constant("forward")

        def back():
            return tf.constant("back")

        def right():
            return tf.constant("right")

        def left():
            return tf.constant("left")

        def diagonal():
            base_dir = tf.where(x_rel < 0.0, tf.constant("left"), tf.constant("right"))
            vert_dir = tf.where(y_rel >= 0.0, tf.constant("forward"), tf.constant("back"))
            return tf.strings.join([base_dir, " and ", vert_dir])

        return tf.case(
            [
                (is_forward, forward),
                (is_back, back),
                (is_right, right),
                (is_left, left),
            ],
            default=diagonal,
            exclusive=True,
        )

    def _extract_direction_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt/caption asking for direction of the object relative to center."""
        image_id = example["image_id"]
        image_id_hash = tf.strings.to_hash_bucket_fast(image_id, 2147483647)
        image_id_hash = tf.cast(image_id_hash, tf.int32)

        category_name = example["annotations"]["category_name"][0]
        bbox = example["annotations"]["bbox"][0]

        num_prompts = len(DIRECTION_PROMPT_PARTS)
        prompt_idx = tf.random.stateless_uniform(
            [], seed=[self.seed, image_id_hash], minval=0, maxval=num_prompts, dtype=tf.int32
        )

        def make_prompt_fn(idx):
            prefix, suffix = DIRECTION_PROMPT_PARTS[idx]

            def fn():
                return tf.strings.join([prefix, category_name, suffix])

            return fn

        prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
        prompt = tf.switch_case(prompt_idx, branch_fns=prompt_branches)

        direction_caption = self._direction_from_bbox(bbox)

        return prompt, direction_caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode LVIS image to JPEG bytes."""
        image = example["image"]
        # LVIS images are not pre-encoded, so encode them
        return tf.io.encode_jpeg(image, quality=95)
