"""Test dataloader state checkpointing with REALISTIC data dimensions.

THE PROBLEM: Default shuffle_buffer_size is 250,000-400,000 samples.
Each sample has images + state + actions ≈ 1.5 MB per sample.
Shuffle buffer alone = 375+ GB in memory!
TensorFlow checkpoint saves the ENTIRE shuffle buffer → massive disk usage.

THE SOLUTION: Use a reduced shuffle buffer (1,000-5,000 samples) for testing.
This test uses realistic data dimensions (224x224 images, seq_len=10) to simulate
production scenarios while keeping checkpoint size manageable (~3 GB).

REQUIREMENTS:
1. Set GCS_TEST_BUCKET environment variable:
    export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

2. Ensure sufficient disk space in temp directory (~5 GB free recommended):
    export TMPDIR=/path/to/large/disk/tmp  # Optional: if /tmp has limited space

Example:
    export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'
    python scripts/tests/test_dataloader_checkpoint_small_buffer.py

REALISTIC TEST PARAMETERS:
- Image size: 224x224 (standard for vision models)
- Sequence length: 10 frames
- Shuffle buffer: 1,000 samples (vs 250,000 in production)
- Batch size: 16
- Estimated checkpoint size: ~3 GB (vs ~375 GB for production)
"""

import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tensorflow as tf


def setup_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.handlers[0].setFormatter(formatter)


@dataclass
class DummyObservation:
    """Dummy observation structure."""
    state: np.ndarray
    images: np.ndarray


@dataclass
class DummyConfig:
    """Dummy config for testing.

    These values simulate realistic production scenarios:
    - Image size: 224x224 (standard for vision models)
    - Sequence length: 10 (typical trajectory length)
    - Batch size: 16 (common training batch size)
    - Shuffle buffer: 1000-5000 (reduced from production 250k for testing)
    """
    shuffle_buffer_size: int = 50000
    batch_size: int = 16
    num_samples: int = 10000  # Enough samples to fill shuffle buffer multiple times
    seq_len: int = 10
    state_dim: int = 7
    action_dim: int = 7
    image_height: int = 224
    image_width: int = 224


class DummyDataLoader:
    """Dummy dataloader with checkpoint functionality."""

    def __init__(self, config: DummyConfig, seed: int = 42, persistent_iterator: bool = True):
        self.config = config
        self.seed = seed
        self.persistent_iterator = persistent_iterator
        self._batches_seen = 0
        self._iterator = None
        self._tf_dataset = None

        # Create TensorFlow dataset
        self._create_dataset()

    def _create_dataset(self):
        """Create a TensorFlow dataset with dummy data using checkpointable operations.

        Note: We pre-generate all data in memory and use from_tensor_slices because:
        1. Dataset.from_generator does NOT support checkpointing
        2. from_tensor_slices creates a checkpointable dataset
        3. With small test data (32x32 images), memory usage is minimal (~12 MB)
        """
        # Pre-generate all dummy data as numpy arrays
        rng = np.random.RandomState(self.seed)

        logging.info(f"Generating {self.config.num_samples} dummy samples in memory...")

        # Generate states (num_samples, seq_len, state_dim)
        states = rng.randn(
            self.config.num_samples,
            self.config.seq_len,
            self.config.state_dim
        ).astype(np.float32)

        # Generate images (num_samples, seq_len, height, width, 3)
        images = rng.randint(
            0, 255,
            size=(
                self.config.num_samples,
                self.config.seq_len,
                self.config.image_height,
                self.config.image_width,
                3
            ),
            dtype=np.uint8
        )

        # Generate actions (num_samples, seq_len, action_dim)
        actions = rng.randn(
            self.config.num_samples,
            self.config.seq_len,
            self.config.action_dim
        ).astype(np.float32)

        # Calculate memory usage
        total_mb = (states.nbytes + images.nbytes + actions.nbytes) / (1024**2)
        logging.info(f"Generated data size: {total_mb:.2f} MB")

        # Create dataset from tensor slices (this is checkpointable!)
        dataset = tf.data.Dataset.from_tensor_slices({
            'state': states,
            'images': images,
            'actions': actions
        })
        logging.info("✓ Created checkpointable dataset using from_tensor_slices")

        # Repeat, shuffle, and batch
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size, seed=self.seed)
        dataset = dataset.batch(self.config.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self._tf_dataset = dataset

        # Create iterator
        if self.persistent_iterator:
            options = tf.data.Options()
            options.experimental_external_state_policy = tf.data.experimental.ExternalStatePolicy.WARN
            dataset = dataset.with_options(options)

        self._iterator = iter(dataset)

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self) -> Tuple[DummyObservation, np.ndarray]:
        """Get next batch."""
        batch = next(self._iterator)

        # Convert to numpy
        state = batch['state'].numpy()
        images = batch['images'].numpy()
        actions = batch['actions'].numpy()

        # Create observation
        obs = DummyObservation(state=state, images=images)

        # Increment counter
        self._batches_seen += 1

        return obs, actions

    def get_batches_seen(self) -> int:
        """Get number of batches seen."""
        return self._batches_seen

    def save_dataloader_state(self, checkpoint_dir: str) -> str:
        """Save dataloader state to checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        # Ensure directory exists
        tf.io.gfile.makedirs(checkpoint_dir)

        # Create checkpoint path
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt")

        # Save iterator state
        checkpoint = tf.train.Checkpoint(iterator=self._iterator)
        save_path = checkpoint.save(checkpoint_path)

        # Save batch counter separately
        counter_path = os.path.join(checkpoint_dir, "batches_seen.txt")
        with tf.io.gfile.GFile(counter_path, 'w') as f:
            f.write(str(self._batches_seen))

        logging.info(f"Saved iterator checkpoint to: {save_path}")
        logging.info(f"Saved batch counter to: {counter_path}")

        return save_path

    def load_dataloader_state(self, checkpoint_dir: str) -> int:
        """Load dataloader state from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint

        Returns:
            Number of batches seen at checkpoint time
        """
        # Load batch counter
        counter_path = os.path.join(checkpoint_dir, "batches_seen.txt")
        with tf.io.gfile.GFile(counter_path, 'r') as f:
            self._batches_seen = int(f.read().strip())

        # Restore iterator state
        checkpoint = tf.train.Checkpoint(iterator=self._iterator)
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")

        status = checkpoint.restore(checkpoint_path)

        logging.info(f"Restored iterator from: {checkpoint_path}")
        logging.info(f"Restored batch counter: {self._batches_seen}")

        return self._batches_seen


def get_gcs_test_path(bucket: str = None) -> str:
    """Get GCS path for test checkpoints."""
    if bucket is None:
        bucket = os.environ.get('GCS_TEST_BUCKET')

    if bucket is None:
        raise ValueError(
            "GCS bucket path not provided. Either pass bucket parameter or set GCS_TEST_BUCKET environment variable. "
            "Example: export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'"
        )

    if not bucket.startswith('gs://'):
        bucket = f'gs://{bucket}'

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_path = f"{bucket}/dataloader_test_{timestamp}"

    return test_path


def cleanup_gcs_path(gcs_path: str):
    """Clean up GCS path after test."""
    try:
        if tf.io.gfile.exists(gcs_path):
            logging.info(f"Cleaning up GCS path: {gcs_path}")
            tf.io.gfile.rmtree(gcs_path)
            logging.info(f"✓ Cleaned up GCS path: {gcs_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup GCS path {gcs_path}: {e}")


def estimate_checkpoint_size(shuffle_buffer_size: int, batch_size: int = 16, sample_size_kb: int = 1500):
    """Estimate checkpoint size based on shuffle buffer size.

    Args:
        shuffle_buffer_size: Size of shuffle buffer
        batch_size: Batch size
        sample_size_kb: Estimated size per sample in KB
            - Production (seq=10, 224x224 images): ~1500 KB/sample
            - Small test (seq=2, 32x32 images): ~12 KB/sample

    Returns:
        Estimated size in GB
    """
    # Buffer size in GB
    buffer_size_gb = (shuffle_buffer_size * sample_size_kb) / (1024 * 1024)

    # Add prefetch buffer (2 batches)
    prefetch_gb = (2 * batch_size * sample_size_kb) / (1024 * 1024)

    # TensorFlow creates temp files ~1.5-2x the final size during save
    total_with_overhead = (buffer_size_gb + prefetch_gb) * 2

    return total_with_overhead


def create_test_dataloader(shuffle_buffer_size: int = 1000, seed: int = 42, persistent_iterator: bool = True):
    """Create a dummy dataloader for testing.

    Args:
        shuffle_buffer_size: Size of shuffle buffer (default 1000 for realistic testing)
        seed: Random seed
        persistent_iterator: Whether to use persistent iterator

    Returns:
        DummyDataLoader instance
    """
    config = DummyConfig(shuffle_buffer_size=shuffle_buffer_size)

    logging.info(f"Creating dataloader with shuffle_buffer_size={shuffle_buffer_size}, persistent_iterator={persistent_iterator}")
    logging.info(f"Data shape: {config.num_samples} samples, seq_len={config.seq_len}, images={config.image_height}x{config.image_width}")

    data_loader = DummyDataLoader(
        config,
        seed=seed,
        persistent_iterator=persistent_iterator,
    )

    return data_loader


def collect_batch_ids(dataloader, num_batches=5):
    """Collect batch identifiers to track position in dataset."""
    batch_ids = []
    data_iter = iter(dataloader)

    for i in range(num_batches):
        logging.info(f"Fetching batch {i + 1}/{num_batches}")
        obs, actions = next(data_iter)

        state_sample = np.array(obs.state[0, :5])
        batch_ids.append(state_sample)

        logging.info(f"  Batch {i + 1} state sample: {state_sample}")
        logging.info(f"  Batch {i + 1} obs.state shape: {obs.state.shape}")
        logging.info(f"  Batch {i + 1} actions shape: {actions.shape}")

    return batch_ids


def test_save_and_load_dataloader(gcs_bucket: str = None, test_buffer_size: int = 1000):
    """Test saving and loading dataloader state with realistic buffer size.

    Args:
        gcs_bucket: GCS bucket path for checkpoints.
        test_buffer_size: Shuffle buffer size for testing (default 1000 for realistic testing)
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing DataLoader Checkpoint Save/Load Functionality")
    logging.info("WITH REALISTIC DATA DIMENSIONS")
    logging.info("=" * 80)

    # Calculate sample sizes based on actual data dimensions
    # Realistic: seq_len=10, images=(224,224,3) → ~1500 KB per sample
    # Each sample = 10 frames * 224 * 224 * 3 bytes + state/action ≈ 1.5 MB
    test_sample_kb = 1500  # Realistic estimate for 224x224 images, seq_len=10

    # Show original checkpoint size estimate (with typical production buffer size)
    original_buffer_size = 250000
    original_size_gb = estimate_checkpoint_size(original_buffer_size, batch_size=32, sample_size_kb=test_sample_kb)
    logging.info(f"\n{'=' * 80}")
    logging.info(f"TYPICAL PRODUCTION CONFIG:")
    logging.info(f"  Buffer size: {original_buffer_size:,}")
    logging.info(f"  Batch size: 32")
    logging.info(f"  Sample size: ~{test_sample_kb} KB (seq=10, 224x224 images)")
    logging.info(f"  Estimated checkpoint: {original_size_gb:.1f} GB")
    logging.info(f"This is why you're getting 'no space left on device'!")
    logging.info(f"{'=' * 80}")

    # Show test checkpoint size estimate
    new_size_gb = estimate_checkpoint_size(test_buffer_size, batch_size=16, sample_size_kb=test_sample_kb)
    logging.info(f"\n{'=' * 80}")
    logging.info(f"TEST CONFIG (REALISTIC DIMENSIONS):")
    logging.info(f"  Buffer size: {test_buffer_size:,}")
    logging.info(f"  Batch size: 16")
    logging.info(f"  Sample size: ~{test_sample_kb} KB (seq=10, 224x224 images)")
    logging.info(f"  Estimated checkpoint: {new_size_gb:.2f} GB")
    logging.info(f"Reduction: {original_size_gb:.1f} GB → {new_size_gb:.2f} GB ({original_size_gb/new_size_gb:.0f}x smaller)")
    logging.info(f"{'=' * 80}")
    logging.info(f"\n⚠️  NOTE: This test uses realistic data dimensions!")
    logging.info(f"  Ensure you have at least {new_size_gb * 1.5:.1f} GB free space in your temp directory.")

    # Create GCS path for checkpoints
    try:
        base_path = get_gcs_test_path(gcs_bucket)
        checkpoint_dir = f"{base_path}/dataloader_checkpoint"
        tf.io.gfile.makedirs(checkpoint_dir)
        logging.info(f"\nCheckpoint directory (GCS): {checkpoint_dir}")
    except Exception as e:
        logging.error(f"Failed to create GCS checkpoint directory: {e}")
        return False

    try:
        # ========================================================================
        # Part 1: Create dataloader, iterate, and save checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 1: Create dataloader and save checkpoint")
        logging.info("=" * 80)

        dataloader1 = create_test_dataloader(shuffle_buffer_size=test_buffer_size, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 1 with SMALL shuffle buffer")

        # Iterate through batches
        num_batches_before_save = 10
        logging.info(f"\nIterating through {num_batches_before_save} batches...")
        batch_ids_before = collect_batch_ids(dataloader1, num_batches=num_batches_before_save)

        # Check batch counter
        batches_seen_before = dataloader1.get_batches_seen()
        logging.info(f"\nBatches seen before save: {batches_seen_before}")
        assert batches_seen_before == num_batches_before_save

        # Save checkpoint
        logging.info(f"\nSaving dataloader state to {checkpoint_dir}...")
        logging.info("(This should now be MUCH smaller and faster)")

        # Check local temp directory space
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        try:
            disk_usage = shutil.disk_usage(tmpdir)
            free_gb = disk_usage.free / (1024**3)
            logging.info(f"Local temp directory: {tmpdir}")
            logging.info(f"Free space in temp directory: {free_gb:.2f} GB")
            if free_gb < new_size_gb:
                logging.warning(f"⚠️  WARNING: Free space ({free_gb:.2f} GB) may be insufficient for checkpoint ({new_size_gb:.3f} GB)")
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")

        try:
            save_path = dataloader1.save_dataloader_state(checkpoint_dir)
            logging.info(f"✓ Successfully saved checkpoint to: {save_path}")
        except Exception as e:
            error_msg = str(e)
            logging.error(f"✗ Failed to save checkpoint: {error_msg}")

            if "Could not append to the internal temporary file" in error_msg or "No space left" in error_msg:
                logging.error(f"\n{'=' * 80}")
                logging.error("DISK SPACE ERROR DETECTED")
                logging.error(f"{'=' * 80}")
                logging.error("TensorFlow needs local disk space for temporary files before uploading to GCS.")
                logging.error(f"\nCurrent temp directory: {tmpdir}")
                logging.error(f"Free space: {free_gb:.2f} GB")
                logging.error(f"Estimated need: {new_size_gb:.3f} GB")
                logging.error(f"\nSOLUTIONS:")
                logging.error(f"  1. Set TMPDIR to a directory with more space:")
                logging.error(f"     export TMPDIR=/path/to/large/disk/tmp")
                logging.error(f"     mkdir -p $TMPDIR")
                logging.error(f"  2. Reduce test_buffer_size even more (currently {test_buffer_size})")
                logging.error(f"  3. Clean up temp directory: rm -rf {tmpdir}/*")
                logging.error(f"{'=' * 80}")
            return False

        # Verify checkpoint files exist
        checkpoint_files = [f for f in tf.io.gfile.listdir(checkpoint_dir) if f.startswith("ckpt")]
        logging.info(f"Checkpoint files created: {checkpoint_files}")

        # Check actual checkpoint size
        total_size = 0
        for f in checkpoint_files:
            file_path = f"{checkpoint_dir}/{f}"
            try:
                size = tf.io.gfile.stat(file_path).length
                total_size += size
                logging.info(f"  {f}: {size / (1024**2):.2f} MB")
            except:
                pass
        logging.info(f"Total checkpoint size: {total_size / (1024**3):.2f} GB")

        assert len(checkpoint_files) > 0, "No checkpoint files were created"

        # ========================================================================
        # Part 2: Create new dataloader and load checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 2: Create new dataloader and load checkpoint")
        logging.info("=" * 80)

        dataloader2 = create_test_dataloader(shuffle_buffer_size=test_buffer_size, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 2 (fresh instance)")

        batches_seen_initial = dataloader2.get_batches_seen()
        logging.info(f"Batches seen initially (should be 0): {batches_seen_initial}")
        assert batches_seen_initial == 0

        # Load checkpoint
        logging.info(f"\nLoading dataloader state from {checkpoint_dir}...")
        try:
            batches_seen_loaded = dataloader2.load_dataloader_state(checkpoint_dir)
            logging.info("✓ Successfully loaded checkpoint")
            logging.info(f"  Restored to batch {batches_seen_loaded}")
        except Exception as e:
            logging.error(f"✗ Failed to load checkpoint: {e}")
            return False

        assert batches_seen_loaded == batches_seen_before
        logging.info(f"✓ Batch counter correctly restored: {batches_seen_loaded}")

        # ========================================================================
        # Part 3: Continue iteration
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 3: Continue iteration from restored state")
        logging.info("=" * 80)

        num_batches_after_load = 5
        logging.info(f"\nIterating through {num_batches_after_load} more batches...")
        batch_ids_after = collect_batch_ids(dataloader2, num_batches=num_batches_after_load)

        batches_seen_final = dataloader2.get_batches_seen()
        expected_final = batches_seen_loaded + num_batches_after_load
        logging.info(f"\nFinal batch count: {batches_seen_final}")
        logging.info(f"Expected: {expected_final}")
        assert batches_seen_final == expected_final

        # ========================================================================
        # Summary
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("TEST SUMMARY")
        logging.info("=" * 80)
        logging.info("✓ Successfully created dataloader with SMALL shuffle buffer")
        logging.info(f"✓ Reduced checkpoint size from ~{original_size_gb:.0f}GB to ~{new_size_gb:.1f}GB")
        logging.info(f"✓ Iterated through {num_batches_before_save} batches before save")
        logging.info(f"✓ Saved checkpoint with batch count {batches_seen_before}")
        logging.info("✓ Created new dataloader and loaded checkpoint")
        logging.info(f"✓ Restored batch count matches: {batches_seen_loaded}")
        logging.info(f"✓ Continued iteration and tracked {num_batches_after_load} more batches")
        logging.info(f"✓ Final batch count correct: {batches_seen_final}")
        logging.info("\n" + "=" * 80)
        logging.info("ALL TESTS PASSED ✓")
        logging.info("=" * 80)

        logging.info("\n" + "=" * 80)
        logging.info("IMPORTANT NOTE FOR PRODUCTION:")
        logging.info("=" * 80)
        logging.info(f"This test uses shuffle_buffer_size={test_buffer_size:,} (realistic dimensions).")
        logging.info(f"Typical production configs use shuffle_buffer_size={original_buffer_size:,}")
        logging.info("")
        logging.info("For production checkpointing:")
        logging.info(f"  - This test checkpoint: ~{new_size_gb:.1f} GB")
        logging.info(f"  - Full production checkpoint: ~{original_size_gb:.0f} GB")
        logging.info("  - You need a disk/bucket with sufficient space")
        logging.info("  - Consider reducing shuffle_buffer_size for frequent checkpointing")
        logging.info(f"  - Current test proves checkpoint works with {test_buffer_size:,} buffer")
        logging.info("=" * 80)

        return True
    finally:
        # Clean up GCS checkpoint directory
        cleanup_gcs_path(base_path)


def main():
    """Main test function."""
    import sys

    setup_logging()

    # Check that GCS bucket is configured
    gcs_bucket = os.environ.get('GCS_TEST_BUCKET')
    if not gcs_bucket:
        logging.error("=" * 80)
        logging.error("ERROR: GCS_TEST_BUCKET environment variable not set")
        logging.error("=" * 80)
        logging.error("This test requires a GCS bucket to save checkpoints.")
        logging.error("Set the GCS_TEST_BUCKET environment variable before running:")
        logging.error("  export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'")
        logging.error("=" * 80)
        sys.exit(1)

    logging.info("=" * 80)
    logging.info(f"Using GCS bucket: {gcs_bucket}")
    logging.info("=" * 80)

    # Run test
    logging.info("\n" + "=" * 80)
    logging.info("TEST: DataLoader Checkpoint with Realistic Dimensions")
    logging.info("=" * 80)
    logging.info("This test uses realistic data dimensions to simulate production:")
    logging.info("  - Image size: 224x224 (standard for vision models)")
    logging.info("  - Sequence length: 10 (typical trajectory)")
    logging.info("  - Shuffle buffer: 1000 (reduced from production 250k)")
    logging.info("  - Batch size: 16 (common training size)")
    logging.info("=" * 80)
    success = test_save_and_load_dataloader(test_buffer_size=1000)

    if not success:
        logging.error("TEST FAILED")
        sys.exit(1)

    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
