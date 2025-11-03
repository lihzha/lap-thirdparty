"""VQAv2 dataset implementation for VQA training."""

import tensorflow as tf

from openpi_cot.dataloader.vqa_base import _BaseVQADataset
from openpi_cot.dataloader.vqa_base import ensure_dldataset


class Vqav2(_BaseVQADataset):
    """VQAv2 dataset for vision-language training.

    This dataset loads VQAv2 images with questions and answers and formats them
    to be compatible with the robot dataset structure.
    """

    def build_dataset_builder(self, ds_name: str, data_dir: str):
        """Build TFDS builder for VQAv2.

        Note: This assumes the vqav2.py file has been registered with TFDS
        and the dataset builder is named 'vqa'.
        """
        import tensorflow_datasets as tfds

        return tfds.builder("vqa", data_dir=data_dir)

    def build_dataset(self, builder, split: str):
        """Build TensorFlow dataset from TFDS builder.

        Args:
            builder: TFDS dataset builder
            split: One of "train", "val", "validation", "test", "test-dev"
        """
        import tensorflow_datasets as tfds

        # Map our split names to VQAv2 split names
        # VQAv2 has: train, validation, test, test-dev
        if split in ["train", "val"]:
            # We'll use train split and manually split for val
            tfds_split = "train"
        elif split == "validation":
            tfds_split = "validation"
        elif split == "test":
            tfds_split = "test"
        elif split == "test-dev":
            tfds_split = "test-dev"
        else:
            tfds_split = "train"

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
            split=tfds_split,
            shuffle_files=not self.want_val,
            read_config=read_config,
        )

        ds = ensure_dldataset(ds, is_flattened=True)
        return ds

    def get_dataset_name(self) -> str:
        """Return dataset name for metadata."""
        return "vqa"

    def get_num_transitions(self) -> int:
        """Return approximate number of VQAv2 samples."""
        return 658111  # Approximate number of VQAv2 train samples

    def create_trajectory_id(self, example: dict) -> tf.Tensor:
        """Create trajectory ID from VQAv2 question_id and image_id."""
        question_id = tf.strings.as_string(example["question_id"])
        image_id = tf.strings.as_string(example["image/id"])
        return tf.strings.join(["vqa_", question_id, "_", image_id])

    def extract_prompt_and_caption(self, example: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Extract prompt and caption from VQAv2 example.

        Returns:
            (prompt, caption) where prompt is the question_text
            and caption is the top_answer.
        """
        prompt = example["question_text"]
        caption = example["top_answer"]
        return prompt, caption

    def extract_and_encode_image(self, example: dict) -> tf.Tensor:
        """Extract and encode VQAv2 image to JPEG bytes."""
        image = example["image"]

        # Check if image is already encoded (bytes) or needs encoding
        if image.dtype == tf.string:
            # Already encoded
            return image
        # Encode image to bytes
        return tf.io.encode_jpeg(image, quality=95)
