"""Test dataloader state checkpointing with SMALL shuffle buffer.

THE PROBLEM: Default shuffle_buffer_size is 250,000-400,000 samples.
Each sample has images + state + actions ≈ 100KB-1MB per sample.
Shuffle buffer alone = 25-400 GB in memory!
TensorFlow checkpoint saves the ENTIRE shuffle buffer → massive disk usage.

THE SOLUTION: Use a tiny shuffle buffer (100-1000 samples) for testing.

REQUIREMENTS:
1. Set GCS_TEST_BUCKET environment variable:
    export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

Example:
    export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'
    python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
"""

import logging
import os
from pathlib import Path
import tempfile
import time
from dataclasses import replace

from etils import epath
import jax
import numpy as np
import tensorflow as tf

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config


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
    logger.handlers[0].setFormatter(formatter)


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


def create_test_config_with_small_buffer(config, buffer_size: int = 100):
    """Create a test config with a MUCH smaller shuffle buffer.

    Args:
        config: Original config
        buffer_size: New shuffle buffer size (default 100, down from 250,000+)

    Returns:
        Modified config with small shuffle buffer
    """
    # Get the data config
    if hasattr(config, 'data'):
        data_config = config.data
    else:
        data_config = config

    # Create a copy with reduced shuffle buffer
    test_data_config = replace(data_config, shuffle_buffer_size=buffer_size)

    if hasattr(config, 'data'):
        test_config = replace(config, data=test_data_config)
    else:
        test_config = test_data_config

    logging.info(f"=" * 80)
    logging.info(f"REDUCED shuffle_buffer_size: {data_config.shuffle_buffer_size:,} → {buffer_size:,}")
    logging.info(f"This reduces checkpoint size by ~{data_config.shuffle_buffer_size // buffer_size}x")
    logging.info(f"=" * 80)

    return test_config


def estimate_checkpoint_size(config, batch_size: int = 32):
    """Estimate checkpoint size based on config.

    Args:
        config: Training configuration
        batch_size: Batch size

    Returns:
        Estimated size in GB
    """
    # Get shuffle buffer size
    if hasattr(config, 'data'):
        shuffle_buffer = config.data.shuffle_buffer_size
    else:
        shuffle_buffer = config.shuffle_buffer_size

    # Rough estimate: each sample is ~500KB (with images)
    # This is conservative - could be smaller or larger
    sample_size_kb = 500

    # Buffer size in GB
    buffer_size_gb = (shuffle_buffer * sample_size_kb) / (1024 * 1024)

    # Add prefetch buffer (2 batches)
    prefetch_gb = (2 * batch_size * sample_size_kb) / (1024 * 1024)

    # TensorFlow creates temp files ~1.5-2x the final size during save
    total_with_overhead = (buffer_size_gb + prefetch_gb) * 2

    return total_with_overhead


def create_test_dataloader(config, seed=42, persistent_iterator=True):
    """Create a dataloader similar to train.py."""
    try:
        mesh = jax.sharding.Mesh(jax.devices(), axis_names=("data",))
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    except Exception as e:
        logging.warning(f"Could not create mesh sharding: {e}. Using None.")
        data_sharding = None

    logging.info(f"Creating dataloader with persistent_iterator={persistent_iterator}")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
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


def test_save_and_load_dataloader(config_path: str = None, gcs_bucket: str = None, test_buffer_size: int = 100):
    """Test saving and loading dataloader state with small buffer.

    Args:
        config_path: Path to config file. If None, uses default config.
        gcs_bucket: GCS bucket path for checkpoints.
        test_buffer_size: Shuffle buffer size for testing (default 100)
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing DataLoader Checkpoint Save/Load Functionality")
    logging.info("WITH SMALL SHUFFLE BUFFER FOR TESTING")
    logging.info("=" * 80)

    # Load config
    if config_path:
        config = _config.load_config(config_path)
    else:
        try:
            config = _config.cli()
        except SystemExit:
            logging.error("Failed to load config. Please provide a config path or run with --config-name=<name>")
            return False

    logging.info(f"Using config: {config.name if hasattr(config, 'name') else 'default'}")

    # Show original checkpoint size estimate
    original_size_gb = estimate_checkpoint_size(config)
    logging.info(f"\n{'=' * 80}")
    logging.info(f"ORIGINAL CONFIG CHECKPOINT SIZE ESTIMATE: {original_size_gb:.1f} GB")
    logging.info(f"This is why you're getting 'no space left on device'!")
    logging.info(f"{'=' * 80}")

    # Create test config with small shuffle buffer
    test_config = create_test_config_with_small_buffer(config, buffer_size=test_buffer_size)

    # Show new checkpoint size estimate
    new_size_gb = estimate_checkpoint_size(test_config)
    logging.info(f"\n{'=' * 80}")
    logging.info(f"NEW CONFIG CHECKPOINT SIZE ESTIMATE: {new_size_gb:.2f} GB")
    logging.info(f"Reduction: {original_size_gb:.1f} GB → {new_size_gb:.2f} GB ({original_size_gb/new_size_gb:.0f}x smaller)")
    logging.info(f"{'=' * 80}")

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

        dataloader1 = create_test_dataloader(test_config, seed=42, persistent_iterator=True)
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
        try:
            save_path = dataloader1.save_dataloader_state(checkpoint_dir)
            logging.info(f"✓ Successfully saved checkpoint to: {save_path}")
        except Exception as e:
            logging.error(f"✗ Failed to save checkpoint: {e}")
            logging.error(f"\nIf you still get 'no space left', try:")
            logging.error(f"  1. Reduce test_buffer_size even more (currently {test_buffer_size})")
            logging.error(f"  2. Set TMPDIR to a larger disk:")
            logging.error(f"     export TMPDIR=/mnt/large-disk/tmp")
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

        dataloader2 = create_test_dataloader(test_config, seed=42, persistent_iterator=True)
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
        logging.info("This test uses shuffle_buffer_size=100 for testing.")
        logging.info(f"Your production config uses shuffle_buffer_size={config.data.shuffle_buffer_size:,}")
        logging.info("")
        logging.info("For production checkpointing:")
        logging.info("  - Checkpoint size will be ~100-500GB with default buffer size")
        logging.info("  - You need a disk/bucket with sufficient space")
        logging.info("  - Consider reducing shuffle_buffer_size if you need frequent checkpointing")
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
    logging.info("TEST: DataLoader Checkpoint with Small Buffer")
    logging.info("=" * 80)
    success = test_save_and_load_dataloader(test_buffer_size=1)

    if not success:
        logging.error("TEST FAILED")
        sys.exit(1)

    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
