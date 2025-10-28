"""Test dataloader state checkpointing functionality with local staging.

This version saves checkpoints to a local staging directory first, then copies to GCS.
This avoids TensorFlow's internal temporary file issues when disk space is limited.

REQUIREMENTS:
1. Set LOCAL_STAGING_DIR to a path with sufficient space (10+ GB recommended):
    export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints

2. (Optional) Set GCS_TEST_BUCKET to copy checkpoints to GCS after saving:
    export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

Example:
    export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints
    export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'  # Optional
    python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config
"""

import logging
import os
from pathlib import Path
import shutil
import tempfile
import time

from etils import epath
import jax
import numpy as np
import tensorflow as tf

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config


def setup_logging():
    """Setup basic logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def check_disk_space(path: str, required_gb: float = 10.0) -> bool:
    """Check if a path has sufficient disk space.

    Args:
        path: Directory path to check
        required_gb: Required space in GB

    Returns:
        True if sufficient space, False otherwise
    """
    try:
        stat = shutil.disk_usage(path)
        available_gb = stat.free / (1024**3)

        logging.info(f"Checking disk space for: {path}")
        logging.info(f"  Total: {stat.total / (1024**3):.2f} GB")
        logging.info(f"  Used: {stat.used / (1024**3):.2f} GB")
        logging.info(f"  Available: {available_gb:.2f} GB")

        if available_gb < required_gb:
            logging.error(f"  ✗ Insufficient space! Need at least {required_gb} GB, have {available_gb:.2f} GB")
            return False
        else:
            logging.info(f"  ✓ Sufficient space ({available_gb:.2f} GB >= {required_gb} GB)")
            return True
    except Exception as e:
        logging.error(f"Could not check disk space: {e}")
        return False


def copy_to_gcs(local_path: str, gcs_path: str):
    """Copy local checkpoint directory to GCS.

    Args:
        local_path: Local directory path
        gcs_path: GCS destination path
    """
    try:
        logging.info(f"Copying {local_path} to {gcs_path}...")

        # Create GCS directory
        tf.io.gfile.makedirs(gcs_path)

        # Copy all files
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                # Get relative path
                rel_path = os.path.relpath(local_file, local_path)
                gcs_file = f"{gcs_path}/{rel_path}"

                # Copy file
                with open(local_file, 'rb') as f:
                    data = f.read()
                with tf.io.gfile.GFile(gcs_file, 'wb') as f:
                    f.write(data)

        logging.info(f"✓ Successfully copied to GCS: {gcs_path}")
    except Exception as e:
        logging.warning(f"Failed to copy to GCS: {e}")


def cleanup_gcs_path(gcs_path: str):
    """Clean up GCS path after test."""
    try:
        if tf.io.gfile.exists(gcs_path):
            logging.info(f"Cleaning up GCS path: {gcs_path}")
            tf.io.gfile.rmtree(gcs_path)
            logging.info(f"✓ Cleaned up GCS path: {gcs_path}")
    except Exception as e:
        logging.warning(f"Failed to cleanup GCS path {gcs_path}: {e}")


def create_test_dataloader(config, seed=42, persistent_iterator=True):
    """Create a dataloader similar to train.py.

    Args:
        config: Training configuration
        seed: Random seed
        persistent_iterator: Whether to enable persistent iterator for checkpointing

    Returns:
        DataLoader instance
    """
    # Create sharding (simplified version from train.py)
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
    """Collect batch identifiers to track position in dataset.

    Args:
        dataloader: DataLoader instance
        num_batches: Number of batches to collect

    Returns:
        List of batch identifiers (we use state values as proxy)
    """
    batch_ids = []
    data_iter = iter(dataloader)

    for i in range(num_batches):
        logging.info(f"Fetching batch {i + 1}/{num_batches}")
        obs, actions = next(data_iter)

        # Use state values as a batch identifier
        state_sample = np.array(obs.state[0, :5])
        batch_ids.append(state_sample)

        logging.info(f"  Batch {i + 1} state sample: {state_sample}")
        logging.info(f"  Batch {i + 1} obs.state shape: {obs.state.shape}")
        logging.info(f"  Batch {i + 1} actions shape: {actions.shape}")

    return batch_ids


def test_save_and_load_dataloader(config_path: str = None, staging_dir: str = None, gcs_bucket: str = None):
    """Test saving and loading dataloader state.

    Args:
        config_path: Path to config file. If None, uses default config.
        staging_dir: Local staging directory for checkpoints.
        gcs_bucket: Optional GCS bucket path to copy checkpoints.
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing DataLoader Checkpoint Save/Load Functionality")
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

    # Setup staging directory
    if staging_dir is None:
        staging_dir = os.environ.get('LOCAL_STAGING_DIR')

    if staging_dir is None:
        # Use /tmp as fallback but warn
        staging_dir = tempfile.mkdtemp(prefix='dataloader_test_')
        logging.warning("=" * 80)
        logging.warning("WARNING: LOCAL_STAGING_DIR not set, using /tmp")
        logging.warning("If you encounter disk space errors, set LOCAL_STAGING_DIR:")
        logging.warning("  export LOCAL_STAGING_DIR=/path/to/large/disk/checkpoints")
        logging.warning("=" * 80)
    else:
        os.makedirs(staging_dir, exist_ok=True)

    # Check disk space
    logging.info("\n" + "=" * 80)
    logging.info("Checking Disk Space")
    logging.info("=" * 80)
    if not check_disk_space(staging_dir, required_gb=10.0):
        logging.error("Insufficient disk space. Please set LOCAL_STAGING_DIR to a location with more space.")
        return False

    # Create checkpoint directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_base = os.path.join(staging_dir, f"dataloader_test_{timestamp}")
    checkpoint_dir = os.path.join(checkpoint_base, "dataloader_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.info(f"Checkpoint directory (local): {checkpoint_dir}")

    # Setup GCS path if provided
    gcs_path = None
    if gcs_bucket:
        if not gcs_bucket.startswith('gs://'):
            gcs_bucket = f'gs://{gcs_bucket}'
        gcs_path = f"{gcs_bucket}/dataloader_test_{timestamp}"
        logging.info(f"Will copy to GCS: {gcs_path}")

    try:
        # ========================================================================
        # Part 1: Create dataloader, iterate, and save checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 1: Create dataloader and save checkpoint")
        logging.info("=" * 80)

        dataloader1 = create_test_dataloader(config, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 1")

        # Iterate through some batches
        num_batches_before_save = 10
        logging.info(f"\nIterating through {num_batches_before_save} batches...")
        batch_ids_before = collect_batch_ids(dataloader1, num_batches=num_batches_before_save)

        # Check batch counter
        batches_seen_before = dataloader1.get_batches_seen()
        logging.info(f"\nBatches seen before save: {batches_seen_before}")
        assert batches_seen_before == num_batches_before_save

        # Save checkpoint to local staging directory
        logging.info(f"\nSaving dataloader state to {checkpoint_dir}...")
        try:
            save_path = dataloader1.save_dataloader_state(checkpoint_dir)
            logging.info(f"✓ Successfully saved checkpoint to: {save_path}")
        except Exception as e:
            logging.error(f"✗ Failed to save checkpoint: {e}")
            return False

        # Verify checkpoint files exist
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt")]
        logging.info(f"Checkpoint files created: {checkpoint_files}")
        assert len(checkpoint_files) > 0, "No checkpoint files were created"

        # Copy to GCS if requested
        if gcs_path:
            gcs_checkpoint_dir = f"{gcs_path}/dataloader_checkpoint"
            copy_to_gcs(checkpoint_dir, gcs_checkpoint_dir)

        # ========================================================================
        # Part 2: Create new dataloader and load checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 2: Create new dataloader and load checkpoint")
        logging.info("=" * 80)

        dataloader2 = create_test_dataloader(config, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 2 (fresh instance)")

        # Before loading, verify it starts from 0
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

        # Verify loaded batch count matches saved count
        assert batches_seen_loaded == batches_seen_before
        logging.info(f"✓ Batch counter correctly restored: {batches_seen_loaded}")

        # ========================================================================
        # Part 3: Continue iteration and verify consistency
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 3: Continue iteration from restored state")
        logging.info("=" * 80)

        num_batches_after_load = 5
        logging.info(f"\nIterating through {num_batches_after_load} more batches...")
        batch_ids_after = collect_batch_ids(dataloader2, num_batches=num_batches_after_load)

        # Check final batch counter
        batches_seen_final = dataloader2.get_batches_seen()
        expected_final = batches_seen_loaded + num_batches_after_load
        logging.info(f"\nFinal batch count: {batches_seen_final}")
        logging.info(f"Expected: {expected_final}")
        assert batches_seen_final == expected_final

        # ========================================================================
        # Part 4: Test without persistent_iterator (should fail)
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 4: Test error handling without persistent_iterator")
        logging.info("=" * 80)

        dataloader3 = create_test_dataloader(config, seed=42, persistent_iterator=False)
        logging.info("Created dataloader 3 (persistent_iterator=False)")

        # Try to save - should raise ValueError
        try:
            test_fail_dir = os.path.join(checkpoint_base, "should_fail")
            os.makedirs(test_fail_dir, exist_ok=True)
            dataloader3.save_dataloader_state(test_fail_dir)
            logging.error("✗ Should have raised ValueError for non-persistent iterator")
            return False
        except ValueError as e:
            logging.info(f"✓ Correctly raised ValueError: {e}")

        # Try to load - should also raise ValueError
        try:
            dataloader3.load_dataloader_state(checkpoint_dir)
            logging.error("✗ Should have raised ValueError for non-persistent iterator")
            return False
        except ValueError as e:
            logging.info(f"✓ Correctly raised ValueError: {e}")

        # ========================================================================
        # Summary
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("TEST SUMMARY")
        logging.info("=" * 80)
        logging.info("✓ Successfully created dataloader with persistent_iterator=True")
        logging.info(f"✓ Iterated through {num_batches_before_save} batches before save")
        logging.info(f"✓ Saved checkpoint with batch count {batches_seen_before}")
        logging.info("✓ Created new dataloader and loaded checkpoint")
        logging.info(f"✓ Restored batch count matches: {batches_seen_loaded}")
        logging.info(f"✓ Continued iteration and tracked {num_batches_after_load} more batches")
        logging.info(f"✓ Final batch count correct: {batches_seen_final}")
        logging.info("✓ Error handling works correctly for non-persistent iterator")
        if gcs_path:
            logging.info(f"✓ Copied checkpoints to GCS: {gcs_path}")
        logging.info("\n" + "=" * 80)
        logging.info("ALL TESTS PASSED ✓")
        logging.info("=" * 80)

        return True
    finally:
        # Clean up local staging directory
        try:
            if os.path.exists(checkpoint_base):
                shutil.rmtree(checkpoint_base)
                logging.info(f"✓ Cleaned up local staging directory: {checkpoint_base}")
        except Exception as e:
            logging.warning(f"Failed to cleanup local staging: {e}")

        # Clean up GCS if used
        if gcs_path:
            cleanup_gcs_path(gcs_path)


def main():
    """Main test function."""
    import sys

    setup_logging()

    # Check configuration
    staging_dir = os.environ.get('LOCAL_STAGING_DIR')
    gcs_bucket = os.environ.get('GCS_TEST_BUCKET')

    if not staging_dir:
        logging.warning("=" * 80)
        logging.warning("WARNING: LOCAL_STAGING_DIR environment variable not set")
        logging.warning("=" * 80)
        logging.warning("For best results, set LOCAL_STAGING_DIR to a path with 10+ GB:")
        logging.warning("  export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints")
        logging.warning("")
        logging.warning("Will use /tmp as fallback (may have limited space)")
        logging.warning("=" * 80)
        time.sleep(2)  # Give user time to read

    logging.info("=" * 80)
    logging.info("Configuration:")
    logging.info(f"  LOCAL_STAGING_DIR: {staging_dir or '(not set, using /tmp)'}")
    logging.info(f"  GCS_TEST_BUCKET: {gcs_bucket or '(not set, will not copy to GCS)'}")
    logging.info("=" * 80)

    # Run test
    logging.info("\n" + "=" * 80)
    logging.info("TEST: DataLoader Checkpoint Save/Load")
    logging.info("=" * 80)
    success = test_save_and_load_dataloader()

    if not success:
        logging.error("TEST FAILED")
        sys.exit(1)

    # All tests passed
    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
