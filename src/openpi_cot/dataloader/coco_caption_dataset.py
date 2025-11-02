"""COCO Captions dataset implementation for VQA training."""

import tensorflow as tf

from openpi_cot.dataloader.vqa_base import _BaseVQADataset
from openpi_cot.dataloader.vqa_base import ensure_dldataset


# COCO caption prompts to randomly sample from
COCO_PROMPTS = tf.constant(
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


class CocoCaption(_BaseVQADataset):
    """COCO Captions dataset for vision-language training.

    This dataset loads COCO images with captions and formats them to be
    compatible with the robot dataset structure.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for COCO captions."""
        import tensorflow_datasets as tfds

        return tfds.builder("coco_captions", data_dir=data_dir)

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
            split="train",  # COCO only has train split, we'll manually split for val
            shuffle_files=not self.want_val,
            read_config=read_config,
        )

        ds = ensure_dldataset(ds, is_flattened=True)
        return ds

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "coco_captions"

    def get_num_transitions(self) -> int:
        """Return approximate number of COCO caption samples."""
        return 82783  # Approximate number of COCO train images

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from COCO image filename and ID."""
        image_filename = example["image/filename"]
        image_id = tf.strings.as_string(example["image/id"])
        return tf.strings.join(["coco_", image_filename, "_", image_id])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from COCO example.

        Returns:
            (prompt, caption) where prompt is randomly sampled from COCO_PROMPTS
            and caption is randomly sampled from the available captions.
        """
        # Randomly select one caption from the list
        captions = example["captions"]["text"]
        num_captions = tf.shape(captions)[0]
        caption_idx = tf.random.uniform([], 0, num_captions, dtype=tf.int32, seed=self.seed)
        selected_caption = captions[caption_idx]

        # Randomly select a prompt
        num_prompts = tf.shape(COCO_PROMPTS)[0]
        prompt_idx = tf.random.uniform([], 0, num_prompts, dtype=tf.int32, seed=self.seed)
        prompt = COCO_PROMPTS[prompt_idx]

        return prompt, selected_caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode COCO image to JPEG bytes."""
        image = example["image"]
        # COCO images are not pre-encoded, so encode them
        return tf.io.encode_jpeg(image, quality=95)
