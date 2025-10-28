"""Test real dataloader state checkpointing with OXECoTDatasets.

This test validates the actual production dataloader implementation with checkpointing
functionality. It tests the full pipeline from OXECoTDatasets through CoTRLDSDataLoader.

REQUIREMENTS:
1. Set GCS_TEST_BUCKET environment variable:
    export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

2. Ensure dataset access:
    - For GCS: Ensure you have access to the dataset bucket
    - For local: Ensure dataset is available at specified path

3. Ensure sufficient disk space in temp directory (~5 GB free recommended):
    export TMPDIR=/path/to/large/disk/tmp  # Optional: if /tmp has limited space

Example:
    export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'
    python scripts/tests/test_real_dataloader_checkpoint.py

TEST PARAMETERS:
- Shuffle buffer: 1,000 samples (reduced from production 250,000 for testing)
- Batch size: 4
- Dataset: Single OXE dataset for minimal setup
- Estimated checkpoint size: ~3 GB
"""

import dataclasses
import logging
import os
import pathlib
import shutil
import time
from typing import Any

import jax
import numpy as np
import tensorflow as tf

# Import the actual dataloader
from openpi_cot.dataloader import cot_data_loader
from openpi_cot.dataloader.helpers import ActionEncoding, StateEncoding
from openpi_cot.models.pi_cot_config import PiCoTConfig
from openpi_cot.training.config import CoTDataConfig, RLDSCoTDataConfig, TrainConfig


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
    if logger.handlers:
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
    test_path = f"{bucket}/real_dataloader_test_{timestamp}"

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


def create_test_config(
    shuffle_buffer_size: int = 1000,
    batch_size: int = 4,
    data_dir: str = "gs://pi0-cot/OXE",
) -> TrainConfig:
    """Create a minimal test configuration.

    Args:
        shuffle_buffer_size: Size of shuffle buffer (default 1000 for testing)
        batch_size: Batch size for training
        data_dir: Path to dataset directory

    Returns:
        TrainConfig instance configured for testing
    """
    # Create minimal model config
    model_config = PiCoTConfig(
        action_horizon=10,
        action_dim=32,  # Padded action dimension
        max_token_len=180,
        pi05=True,
        discrete_state_input=True,
        enable_prediction_training=False,
    )

    # Create minimal data config with small buffer
    data_config = RLDSCoTDataConfig(
        repo_id="combined",
        asset_id="combined",
        dataset_type="oxe",
        rlds_data_dir=data_dir,
        data_mix="oxe_magic_soup_plus_minus_aloha_sim",  # Small mix for testing
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples=None,  # Use all available data
        state_encoding=StateEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        resize_resolution=(224, 224),
        use_wrist_image=False,  # Disable to reduce memory
        wrist_image_dropout_prob=0.0,
    )

    # Create train config
    config = TrainConfig(
        name="test_checkpoint",
        model=model_config,
        data=data_config,
        batch_size=batch_size,
        fsdp_devices=1,
        checkpoint_base_dir="/tmp/test_checkpoints",
        assets_base_dir="/tmp/test_assets",
    )

    return config


def collect_batch_ids(dataloader, num_batches: int = 5) -> list[Any]:
    """Collect batch identifiers to track position in dataset.

    Args:
        dataloader: The dataloader to iterate
        num_batches: Number of batches to collect

    Returns:
        List of batch identifiers (state samples)
    """
    batch_ids = []

    for i in range(num_batches):
        logging.info(f"Fetching batch {i + 1}/{num_batches}")
        obs, actions = next(dataloader)

        # Extract a sample from the batch for identification
        # Note: obs is a CoTObservation object
        state_sample = np.array(obs.state[0, :5])  # First sample, first 5 dims
        batch_ids.append(state_sample)

        logging.info(f"  Batch {i + 1} state sample (first 5 dims): {state_sample}")
        logging.info(f"  Batch {i + 1} obs.state shape: {obs.state.shape}")
        logging.info(f"  Batch {i + 1} actions shape: {actions.shape}")

    return batch_ids


def test_save_and_load_real_dataloader(
    gcs_bucket: str = None,
    test_buffer_size: int = 1000,
    batch_size: int = 4,
    data_dir: str = "gs://pi0-cot/OXE",
):
    """Test saving and loading real dataloader state.

    Args:
        gcs_bucket: GCS bucket path for checkpoints
        test_buffer_size: Shuffle buffer size for testing (default 1000)
        batch_size: Batch size for training
        data_dir: Path to dataset directory

    Returns:
        True if test passes, False otherwise
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing Real DataLoader Checkpoint Save/Load Functionality")
    logging.info("=" * 80)

    # Calculate estimated checkpoint size
    sample_size_kb = 1500  # Realistic estimate for 224x224 images
    original_buffer_size = 250000
    original_size_gb = (original_buffer_size * sample_size_kb) / (1024 * 1024) * 2
    new_size_gb = (test_buffer_size * sample_size_kb) / (1024 * 1024) * 2

    logging.info(f"\n{'=' * 80}")
    logging.info(f"TYPICAL PRODUCTION CONFIG:")
    logging.info(f"  Buffer size: {original_buffer_size:,}")
    logging.info(f"  Estimated checkpoint: {original_size_gb:.1f} GB")
    logging.info(f"{'=' * 80}")

    logging.info(f"\n{'=' * 80}")
    logging.info(f"TEST CONFIG:")
    logging.info(f"  Buffer size: {test_buffer_size:,}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Estimated checkpoint: {new_size_gb:.2f} GB")
    logging.info(f"  Reduction: {original_size_gb:.1f} GB → {new_size_gb:.2f} GB ({original_size_gb/new_size_gb:.0f}x smaller)")
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
        # Check disk space
        tmpdir = os.environ.get('TMPDIR', '/tmp')
        try:
            disk_usage = shutil.disk_usage(tmpdir)
            free_gb = disk_usage.free / (1024**3)
            logging.info(f"\nLocal temp directory: {tmpdir}")
            logging.info(f"Free space in temp directory: {free_gb:.2f} GB")
            if free_gb < new_size_gb:
                logging.warning(f"⚠️  WARNING: Free space ({free_gb:.2f} GB) may be insufficient for checkpoint ({new_size_gb:.3f} GB)")
        except Exception as e:
            logging.warning(f"Could not check disk space: {e}")

        # ========================================================================
        # Part 1: Create dataloader, iterate, and save checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 1: Create real dataloader and save checkpoint")
        logging.info("=" * 80)

        # Create config
        logging.info("Creating test configuration...")
        config = create_test_config(
            shuffle_buffer_size=test_buffer_size,
            batch_size=batch_size,
            data_dir=data_dir,
        )

        # Initialize JAX if not already initialized (required for create_data_loader)
        logging.info("Initializing JAX...")
        try:
            # Check if already initialized
            _ = jax.device_count()
        except RuntimeError:
            # Not initialized, initialize now
            try:
                jax.distributed.initialize()
            except Exception as e:
                logging.warning(f"Could not initialize JAX distributed: {e}")
                # Continue anyway - might work in single-process mode

        # Create dataloader with persistent iterator
        logging.info("Creating dataloader with persistent_iterator=True...")
        dataloader1 = cot_data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=True,
            seed=42,
            split="train",
            persistent_iterator=True,
        )
        logging.info("✓ Created dataloader 1")

        # Get iterator
        data_iter1 = iter(dataloader1)

        # Iterate through batches
        num_batches_before_save = 10
        logging.info(f"\nIterating through {num_batches_before_save} batches...")
        batch_ids_before = collect_batch_ids(data_iter1, num_batches=num_batches_before_save)

        # Check batch counter
        batches_seen_before = dataloader1.get_batches_seen()
        logging.info(f"\nBatches seen before save: {batches_seen_before}")
        assert batches_seen_before == num_batches_before_save, f"Expected {num_batches_before_save}, got {batches_seen_before}"

        # Save checkpoint
        logging.info(f"\nSaving dataloader state to {checkpoint_dir}...")
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

        # Create new dataloader with same config
        logging.info("Creating fresh dataloader instance...")
        dataloader2 = cot_data_loader.create_data_loader(
            config,
            sharding=None,
            shuffle=True,
            seed=42,
            split="train",
            persistent_iterator=True,
        )
        logging.info("✓ Created dataloader 2 (fresh instance)")

        batches_seen_initial = dataloader2.get_batches_seen()
        logging.info(f"Batches seen initially (should be 0): {batches_seen_initial}")
        assert batches_seen_initial == 0, f"Expected 0, got {batches_seen_initial}"

        # Load checkpoint
        logging.info(f"\nLoading dataloader state from {checkpoint_dir}...")
        try:
            batches_seen_loaded = dataloader2.load_dataloader_state(checkpoint_dir)
            logging.info("✓ Successfully loaded checkpoint")
            logging.info(f"  Restored to batch {batches_seen_loaded}")
        except Exception as e:
            logging.error(f"✗ Failed to load checkpoint: {e}")
            return False

        assert batches_seen_loaded == batches_seen_before, f"Expected {batches_seen_before}, got {batches_seen_loaded}"
        logging.info(f"✓ Batch counter correctly restored: {batches_seen_loaded}")

        # ========================================================================
        # Part 3: Continue iteration from restored state
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 3: Continue iteration from restored state")
        logging.info("=" * 80)

        # Get iterator for dataloader2
        data_iter2 = iter(dataloader2)

        num_batches_after_load = 5
        logging.info(f"\nIterating through {num_batches_after_load} more batches...")
        batch_ids_after = collect_batch_ids(data_iter2, num_batches=num_batches_after_load)

        batches_seen_final = dataloader2.get_batches_seen()
        expected_final = batches_seen_loaded + num_batches_after_load
        logging.info(f"\nFinal batch count: {batches_seen_final}")
        logging.info(f"Expected: {expected_final}")
        assert batches_seen_final == expected_final, f"Expected {expected_final}, got {batches_seen_final}"

        # ========================================================================
        # Summary
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("TEST SUMMARY")
        logging.info("=" * 80)
        logging.info("✓ Successfully created real dataloader with persistent iterator")
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

        return True

    except Exception as e:
        logging.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
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
    logging.info("TEST: Real DataLoader Checkpoint with OXECoTDatasets")
    logging.info("=" * 80)
    logging.info("This test uses the actual production dataloader:")
    logging.info("  - OXECoTDatasets from dataset_mixer.py")
    logging.info("  - CoTRLDSDataLoader from cot_data_loader.py")
    logging.info("  - Shuffle buffer: 1,000 (reduced from production 250k)")
    logging.info("  - Batch size: 4")
    logging.info("=" * 80)

    success = test_save_and_load_real_dataloader(
        gcs_bucket=gcs_bucket,
        test_buffer_size=1000,
        batch_size=4,
    )

    if not success:
        logging.error("TEST FAILED")
        sys.exit(1)

    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
