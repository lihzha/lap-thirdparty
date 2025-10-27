"""Test dataloader state checkpointing functionality.

This test mimics the dataloader creation in train.py and verifies that
save/load functionality works correctly.
"""

import logging
import os
from pathlib import Path
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


def get_gcs_test_path(bucket: str = None) -> str:
    """Get GCS path for test checkpoints.

    Args:
        bucket: GCS bucket path (e.g., 'gs://my-bucket/test-checkpoints').
                If None, reads from GCS_TEST_BUCKET environment variable.
                If neither provided, raises an error.

    Returns:
        GCS path for test checkpoints with unique timestamp
    """
    if bucket is None:
        bucket = os.environ.get('GCS_TEST_BUCKET')

    if bucket is None:
        raise ValueError(
            "GCS bucket path not provided. Either pass bucket parameter or set GCS_TEST_BUCKET environment variable. "
            "Example: export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'"
        )

    # Ensure bucket starts with gs://
    if not bucket.startswith('gs://'):
        bucket = f'gs://{bucket}'

    # Create unique path with timestamp to avoid conflicts
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_path = f"{bucket}/dataloader_test_{timestamp}"

    return test_path


def cleanup_gcs_path(gcs_path: str):
    """Clean up GCS path after test.

    Args:
        gcs_path: GCS path to clean up
    """
    try:
        if tf.io.gfile.exists(gcs_path):
            logging.info(f"Cleaning up GCS path: {gcs_path}")
            # Delete all files in the directory
            files = tf.io.gfile.listdir(gcs_path)
            for file in files:
                file_path = f"{gcs_path}/{file}"
                try:
                    if tf.io.gfile.isdir(file_path):
                        tf.io.gfile.rmtree(file_path)
                    else:
                        tf.io.gfile.remove(file_path)
                except Exception as e:
                    logging.warning(f"Failed to delete {file_path}: {e}")
            # Delete the directory itself
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
        # In a real dataset, each batch should have unique state values
        state_sample = np.array(obs.state[0, :5])  # First sample, first timestep, first 5 dims
        batch_ids.append(state_sample)

        logging.info(f"  Batch {i + 1} state sample: {state_sample}")
        logging.info(f"  Batch {i + 1} obs.state shape: {obs.state.shape}")
        logging.info(f"  Batch {i + 1} actions shape: {actions.shape}")

    return batch_ids


def test_save_and_load_dataloader(config_path: str = None, gcs_bucket: str = None):
    """Test saving and loading dataloader state.

    Args:
        config_path: Path to config file. If None, uses default config.
        gcs_bucket: GCS bucket path for checkpoints. If None, reads from GCS_TEST_BUCKET env var.
    """
    setup_logging()

    logging.info("=" * 80)
    logging.info("Testing DataLoader Checkpoint Save/Load Functionality")
    logging.info("=" * 80)

    # Load config
    if config_path:
        config = _config.load_config(config_path)
    else:
        # Use CLI config (should load from default or command line args)
        try:
            config = _config.cli()
        except SystemExit:
            logging.error("Failed to load config. Please provide a config path or run with --config-name=<name>")
            return False

    logging.info(f"Using config: {config.name if hasattr(config, 'name') else 'default'}")

    # Create GCS path for checkpoints
    try:
        base_path = get_gcs_test_path(gcs_bucket)
        checkpoint_dir = f"{base_path}/dataloader_checkpoint"
        tf.io.gfile.makedirs(checkpoint_dir)
        logging.info(f"Checkpoint directory (GCS): {checkpoint_dir}")
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

        dataloader1 = create_test_dataloader(config, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 1")

        # Iterate through some batches and collect identifiers
        num_batches_before_save = 10
        logging.info(f"\nIterating through {num_batches_before_save} batches...")
        batch_ids_before = collect_batch_ids(dataloader1, num_batches=num_batches_before_save)

        # Check batch counter
        batches_seen_before = dataloader1.get_batches_seen()
        logging.info(f"\nBatches seen before save: {batches_seen_before}")
        assert batches_seen_before == num_batches_before_save, (
            f"Expected {num_batches_before_save} batches, but saw {batches_seen_before}"
        )

        # Save checkpoint
        logging.info(f"\nSaving dataloader state to {checkpoint_dir}...")
        try:
            save_path = dataloader1.save_dataloader_state(str(checkpoint_dir))
            logging.info(f"✓ Successfully saved checkpoint to: {save_path}")
        except Exception as e:
            logging.error(f"✗ Failed to save checkpoint: {e}")
            return False

        # Verify checkpoint files exist
        checkpoint_files = [f for f in tf.io.gfile.listdir(checkpoint_dir) if f.startswith("ckpt")]
        logging.info(f"Checkpoint files created: {checkpoint_files}")
        assert len(checkpoint_files) > 0, "No checkpoint files were created"

        # ========================================================================
        # Part 2: Create new dataloader and load checkpoint
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 2: Create new dataloader and load checkpoint")
        logging.info("=" * 80)

        # Create a new dataloader instance
        dataloader2 = create_test_dataloader(config, seed=42, persistent_iterator=True)
        logging.info("Created dataloader 2 (fresh instance)")

        # Before loading, verify it starts from 0
        batches_seen_initial = dataloader2.get_batches_seen()
        logging.info(f"Batches seen initially (should be 0): {batches_seen_initial}")
        assert batches_seen_initial == 0, f"Expected 0 batches initially, but got {batches_seen_initial}"

        # Load checkpoint
        logging.info(f"\nLoading dataloader state from {checkpoint_dir}...")
        try:
            batches_seen_loaded = dataloader2.load_dataloader_state(str(checkpoint_dir))
            logging.info("✓ Successfully loaded checkpoint")
            logging.info(f"  Restored to batch {batches_seen_loaded}")
        except Exception as e:
            logging.error(f"✗ Failed to load checkpoint: {e}")
            return False

        # Verify loaded batch count matches saved count
        assert batches_seen_loaded == batches_seen_before, (
            f"Loaded batch count {batches_seen_loaded} != saved count {batches_seen_before}"
        )
        logging.info(f"✓ Batch counter correctly restored: {batches_seen_loaded}")

        # ========================================================================
        # Part 3: Continue iteration and verify consistency
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Part 3: Continue iteration from restored state")
        logging.info("=" * 80)

        # Continue iterating from restored dataloader
        num_batches_after_load = 5
        logging.info(f"\nIterating through {num_batches_after_load} more batches...")
        batch_ids_after = collect_batch_ids(dataloader2, num_batches=num_batches_after_load)

        # Check final batch counter
        batches_seen_final = dataloader2.get_batches_seen()
        expected_final = batches_seen_loaded + num_batches_after_load
        logging.info(f"\nFinal batch count: {batches_seen_final}")
        logging.info(f"Expected: {expected_final}")
        assert batches_seen_final == expected_final, (
            f"Expected {expected_final} total batches, but got {batches_seen_final}"
        )

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
            dataloader3.save_dataloader_state(f"{checkpoint_dir}/should_fail")
            logging.error("✗ Should have raised ValueError for non-persistent iterator")
            return False
        except ValueError as e:
            logging.info(f"✓ Correctly raised ValueError: {e}")

        # Try to load - should also raise ValueError
        try:
            dataloader3.load_dataloader_state(str(checkpoint_dir))
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
        logging.info("\n" + "=" * 80)
        logging.info("ALL TESTS PASSED ✓")
        logging.info("=" * 80)

        return True
    finally:
        # Clean up GCS checkpoint directory
        cleanup_gcs_path(base_path)


def test_multiple_save_load_cycles():
    """Test multiple save/load cycles to ensure robustness."""
    setup_logging()

    logging.info("\n" + "=" * 80)
    logging.info("Testing Multiple Save/Load Cycles")
    logging.info("=" * 80)

    # Load config
    try:
        config = _config.cli()
    except SystemExit:
        logging.error("Failed to load config")
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "dataloader_checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)

        dataloader = create_test_dataloader(config, seed=42, persistent_iterator=True)

        # Perform multiple save/load cycles
        num_cycles = 3
        batches_per_cycle = 5

        for cycle in range(num_cycles):
            logging.info(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")

            # Iterate some batches
            logging.info(f"Iterating {batches_per_cycle} batches...")
            batch_ids = collect_batch_ids(dataloader, num_batches=batches_per_cycle)

            # Save checkpoint
            batches_before = dataloader.get_batches_seen()
            logging.info(f"Saving at batch {batches_before}...")
            save_path = dataloader.save_dataloader_state(str(checkpoint_dir))
            logging.info(f"Saved to {save_path}")

            # Load checkpoint (in same dataloader)
            logging.info("Loading checkpoint...")
            batches_after = dataloader.load_dataloader_state(str(checkpoint_dir))

            # Verify
            assert batches_after == batches_before, (
                f"Cycle {cycle}: Batch count mismatch {batches_after} != {batches_before}"
            )
            logging.info(f"✓ Cycle {cycle + 1} passed")

        logging.info("\n✓ All cycles passed!")
        return True


def main():
    """Main test function."""
    import sys

    # Test 1: Basic save/load functionality
    logging.info("\n" + "=" * 80)
    logging.info("TEST 1: Basic Save/Load Functionality")
    logging.info("=" * 80)
    success1 = test_save_and_load_dataloader()

    if not success1:
        logging.error("TEST 1 FAILED")
        sys.exit(1)

    # Test 2: Multiple cycles
    logging.info("\n" + "=" * 80)
    logging.info("TEST 2: Multiple Save/Load Cycles")
    logging.info("=" * 80)
    success2 = test_multiple_save_load_cycles()

    if not success2:
        logging.error("TEST 2 FAILED")
        sys.exit(1)

    # All tests passed
    logging.info("\n" + "=" * 80)
    logging.info("ALL TESTS PASSED SUCCESSFULLY! ✓")
    logging.info("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
