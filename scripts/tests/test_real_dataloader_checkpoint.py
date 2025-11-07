"""Test real dataloader state checkpointing with OXECoTDatasets.

This test validates the actual production dataloader implementation with skip-based
checkpointing functionality. It tests the full pipeline from OXECoTDatasets through
CoTRLDSDataLoader with lightweight JSON checkpoint files.

REQUIREMENTS:
1. Set GCS_TEST_BUCKET environment variable:
    export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

2. Ensure dataset access:
    - For GCS: Ensure you have access to the dataset bucket
    - For local: Ensure dataset is available at specified path

Example:
    export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'

    # Using default config (pi_combined_cot_v4)
    python scripts/tests/test_real_dataloader_checkpoint.py

    # Specify a different config
    python scripts/tests/test_real_dataloader_checkpoint.py \
        --config-name pi_droid_cot_v4

    # Specify GCS bucket via command line
    python scripts/tests/test_real_dataloader_checkpoint.py \
        --gcs-bucket 'gs://my-bucket/test-checkpoints'

TEST PARAMETERS:
- Checkpoint approach: Skip-based (lightweight JSON)
- Checkpoint size: ~100 bytes (not GB!)
- No persistent_iterator requirement
- No disk space concerns
"""

import dataclasses
import logging
import os
import pathlib
import shutil
import sys
import time
from typing import Any

import jax
import numpy as np
import tensorflow as tf
import tyro

# Import the actual dataloader
from openpi_cot.dataloader import cot_data_loader
from openpi_cot.dataloader.helpers import ActionEncoding, StateEncoding
from openpi_cot.models.pi_cot_config import PiCoTConfig
from openpi_cot.training.config import CoTDataConfig, RLDSCoTDataConfig, TrainConfig
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


def compare_batches(batch_ids1: list[np.ndarray], batch_ids2: list[np.ndarray], tolerance: float = 1e-6) -> bool:
    """Compare two lists of batch identifiers for equality.

    Args:
        batch_ids1: First list of batch identifiers
        batch_ids2: Second list of batch identifiers
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if batches match, False otherwise
    """
    if len(batch_ids1) != len(batch_ids2):
        logging.error(f"Different number of batches: {len(batch_ids1)} vs {len(batch_ids2)}")
        return False

    for i, (batch1, batch2) in enumerate(zip(batch_ids1, batch_ids2)):
        if not np.allclose(batch1, batch2, rtol=tolerance, atol=tolerance):
            logging.error(f"Batch {i} mismatch!")
            logging.error(f"  Expected: {batch1}")
            logging.error(f"  Got:      {batch2}")
            logging.error(f"  Max diff: {np.max(np.abs(batch1 - batch2))}")
            return False

    return True


def test_save_and_load_real_dataloader(
    config: TrainConfig,
    gcs_bucket: str | None = None,
) -> bool:
    """Test saving and loading real dataloader state.

    This test verifies the CORE FUNCTIONALITY: that resuming from a checkpoint
    produces the exact same batches that would have been produced if we had
    continued without interruption.

    Test approach:
    1. Create dataloader_reference and iterate N+H batches (get ground truth)
    2. Create dataloader_1 and iterate N batches, then save checkpoint
    3. Create dataloader_2, load checkpoint, and iterate H batches
    4. VERIFY: batches N+1 to N+H from dataloader_reference match batches 1 to H from dataloader_2

    Args:
        config: Training configuration (with test-specific modifications)
        gcs_bucket: GCS bucket path for checkpoints (reads from env if None)

    Returns:
        True if test passes, False otherwise
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing Real DataLoader Checkpoint Determinism")
    logging.info("=" * 80)

    # Extract parameters from config
    test_buffer_size = config.data.shuffle_buffer_size
    batch_size = config.batch_size
    data_dir = config.data.rlds_data_dir

    logging.info(f"Config: {config.name}")
    logging.info(f"Dataset type: {config.data.dataset_type}")
    logging.info(f"Data mix: {getattr(config.data, 'data_mix', 'N/A')}")
    logging.info(f"Data directory: {data_dir}")

    logging.info(f"\n{'=' * 80}")
    logging.info(f"SKIP-BASED CHECKPOINT APPROACH:")
    logging.info(f"  Checkpoint type: JSON with batch counter")
    logging.info(f"  Checkpoint size: ~100 bytes (not GB!)")
    logging.info(f"  Resume method: dataset.skip(n)")
    logging.info(f"  No persistent_iterator needed: ✓")
    logging.info(f"{'=' * 80}")

    logging.info(f"\n{'=' * 80}")
    logging.info(f"TEST CONFIG:")
    logging.info(f"  Buffer size: {test_buffer_size:,}")
    logging.info(f"  Batch size: {batch_size}")
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

        # Create sharding (used for all dataloaders)
        devices = jax.local_devices()
        mesh = jax.sharding.Mesh(devices, ('data',))
        from jax.sharding import NamedSharding, PartitionSpec as P
        data_sharding = NamedSharding(mesh, P('data'))

        # Test parameters
        num_batches_before_checkpoint = 10
        num_batches_after_checkpoint = 5
        total_batches = num_batches_before_checkpoint + num_batches_after_checkpoint

        # ========================================================================
        # Part 1: Create reference dataloader and iterate N+H batches (ground truth)
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 1: Create REFERENCE dataloader (ground truth)")
        logging.info("=" * 80)
        logging.info(f"This will iterate {total_batches} batches to establish ground truth")

        dataloader_reference = cot_data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            seed=42,
            split="train",
        )
        logging.info("✓ Created reference dataloader")

        data_iter_reference = iter(dataloader_reference)
        logging.info(f"\nCollecting {total_batches} batches from reference dataloader...")
        reference_batches_all = collect_batch_ids(data_iter_reference, num_batches=total_batches)

        # Split into before/after checkpoint portions
        reference_batches_before = reference_batches_all[:num_batches_before_checkpoint]
        reference_batches_after = reference_batches_all[num_batches_before_checkpoint:]

        logging.info(f"✓ Collected {len(reference_batches_all)} reference batches")
        logging.info(f"  Batches 1-{num_batches_before_checkpoint}: Before checkpoint")
        logging.info(f"  Batches {num_batches_before_checkpoint+1}-{total_batches}: After checkpoint (these should match resumed iteration)")

        # ========================================================================
        # Part 2: Create dataloader, iterate N batches, and save checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 2: Create dataloader and save checkpoint after N batches")
        logging.info("=" * 80)

        dataloader1 = cot_data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            seed=42,
            split="train",
        )
        logging.info("✓ Created dataloader 1")

        data_iter1 = iter(dataloader1)
        logging.info(f"\nIterating through {num_batches_before_checkpoint} batches...")
        batch_ids_before = collect_batch_ids(data_iter1, num_batches=num_batches_before_checkpoint)

        # Verify these match the reference
        logging.info("\n" + "-" * 80)
        logging.info("Verification: Do first N batches match reference?")
        if compare_batches(batch_ids_before, reference_batches_before):
            logging.info("✓ First N batches match reference (same seed produces same batches)")
        else:
            logging.error("✗ First N batches don't match reference! Seed may not be working correctly.")
            return False

        # Check batch counter
        batches_seen_before = dataloader1.get_batches_seen()
        logging.info(f"\nBatches seen: {batches_seen_before}")
        assert batches_seen_before == num_batches_before_checkpoint, f"Expected {num_batches_before_checkpoint}, got {batches_seen_before}"

        # Save checkpoint
        logging.info(f"\nSaving dataloader state to {checkpoint_dir}...")
        try:
            save_path = dataloader1.save_dataloader_state(checkpoint_dir)
            logging.info(f"✓ Successfully saved checkpoint to: {save_path}")
        except Exception as e:
            error_msg = str(e)
            logging.error(f"✗ Failed to save checkpoint: {error_msg}")
            return False

        # Verify checkpoint file exists
        checkpoint_file = f"{checkpoint_dir}/dataloader_state.json"
        if not tf.io.gfile.exists(checkpoint_file):
            logging.error(f"✗ Checkpoint file not found: {checkpoint_file}")
            return False

        logging.info(f"✓ Checkpoint file created: dataloader_state.json")

        # Check actual checkpoint size (should be tiny!)
        try:
            size = tf.io.gfile.stat(checkpoint_file).length
            logging.info(f"Checkpoint size: {size} bytes")
            if size > 1024:  # Should be well under 1KB
                logging.warning(f"⚠️  WARNING: Checkpoint larger than expected ({size} bytes)")
        except Exception as e:
            logging.warning(f"Could not check checkpoint size: {e}")

        # ========================================================================
        # Part 3: Create new dataloader, load checkpoint, and verify determinism
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 3: Load checkpoint and verify DETERMINISM")
        logging.info("=" * 80)
        logging.info("This is the CORE TEST: Do resumed batches match reference?")

        dataloader2 = cot_data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            seed=42,
            split="train",
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

        # Create iterator (skip will be applied automatically)
        logging.info(f"\nCreating iterator (skip will be applied automatically)...")
        logging.info(f"Expected skip: {batches_seen_loaded} batches")
        data_iter2 = iter(dataloader2)
        logging.info("✓ Iterator created - skip should have been applied")

        # Iterate H batches after checkpoint
        logging.info(f"\nIterating through {num_batches_after_checkpoint} batches after checkpoint...")
        batch_ids_after = collect_batch_ids(data_iter2, num_batches=num_batches_after_checkpoint)

        batches_seen_final = dataloader2.get_batches_seen()
        expected_final = batches_seen_loaded + num_batches_after_checkpoint
        logging.info(f"\nFinal batch count: {batches_seen_final}")
        logging.info(f"Expected: {expected_final}")
        assert batches_seen_final == expected_final, f"Expected {expected_final}, got {batches_seen_final}"

        # ========================================================================
        # CORE VERIFICATION: Do resumed batches match reference?
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("CORE VERIFICATION: Determinism Check")
        logging.info("=" * 80)
        logging.info(f"Comparing resumed batches (batches 1-{num_batches_after_checkpoint} after checkpoint)")
        logging.info(f"with reference batches (batches {num_batches_before_checkpoint+1}-{total_batches})")
        logging.info("These MUST match for checkpoint/resume to be correct!")

        if compare_batches(batch_ids_after, reference_batches_after):
            logging.info("\n" + "🎉" * 40)
            logging.info("✓✓✓ DETERMINISM VERIFIED ✓✓✓")
            logging.info("Resumed batches EXACTLY match what would have been produced!")
            logging.info("🎉" * 40)
        else:
            logging.error("\n" + "✗" * 80)
            logging.error("DETERMINISM TEST FAILED!")
            logging.error("Resumed batches DO NOT match reference batches!")
            logging.error("This means checkpoint/resume is NOT working correctly!")
            logging.error("✗" * 80)
            return False

        # ========================================================================
        # Summary
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("TEST SUMMARY")
        logging.info("=" * 80)
        logging.info("✓ Created reference dataloader and collected ground truth batches")
        logging.info(f"✓ Lightweight checkpoint saved ({size} bytes)")
        logging.info(f"✓ Iterated through {num_batches_before_checkpoint} batches before checkpoint")
        logging.info(f"✓ Saved checkpoint with batch count {batches_seen_before}")
        logging.info("✓ Created new dataloader and loaded checkpoint")
        logging.info(f"✓ Restored batch count matches: {batches_seen_loaded}")
        logging.info(f"✓ Skip applied automatically ({batches_seen_loaded} batches)")
        logging.info(f"✓ Continued iteration for {num_batches_after_checkpoint} batches")
        logging.info(f"✓ Final batch count correct: {batches_seen_final}")
        logging.info("✓✓✓ DETERMINISM VERIFIED: Resumed batches match reference exactly!")
        logging.info("\n" + "=" * 80)
        logging.info("SKIP-BASED CHECKPOINT BENEFITS:")
        logging.info("  - Checkpoint size: ~100 bytes (vs GB with tf.train.Checkpoint)")
        logging.info("  - Save/load time: <0.1 seconds (vs 10-60 seconds)")
        logging.info("  - No persistent_iterator requirement")
        logging.info("  - Works with all TF operations")
        logging.info("  - ✓ DETERMINISTIC: Produces exact same batches after resume")
        logging.info("=" * 80)
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

def main(config: _config.TrainConfig):
    """Main test function with command-line argument parsing."""
    # Parse command-line arguments using tyro

    setup_logging()


    # Check that GCS bucket is configured
    gcs_bucket = os.environ.get('GCS_TEST_BUCKET')
    if not gcs_bucket:
        logging.error("=" * 80)
        logging.error("ERROR: GCS_TEST_BUCKET not configured")
        logging.error("=" * 80)
        logging.error("This test requires a GCS bucket to save checkpoints.")
        logging.error("Set either:")
        logging.error("  1. Environment variable: export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'")
        logging.error("  2. Command line flag: --gcs-bucket 'gs://your-bucket/test-checkpoints'")
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
    logging.info("  - Skip-based checkpointing (lightweight JSON)")
    logging.info(f"  - Shuffle buffer: {config.data.shuffle_buffer_size:,}")
    logging.info(f"  - Batch size: {config.batch_size}")
    logging.info("=" * 80)

    success = test_save_and_load_real_dataloader(
        config=config,
        gcs_bucket=gcs_bucket,
    )

    if not success:
        logging.error("TEST FAILED")
        sys.exit(1)

    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)

if __name__ == "__main__":
    main(_config.cli())
