"""Tests for TensorFlow iterator checkpointing functionality."""

import tempfile
from pathlib import Path

import jax
import numpy as np
import pytest
import tensorflow as tf

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config


class TestIteratorCheckpointing:
    """Test iterator save/restore for deterministic resumption."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def minimal_config(self, temp_checkpoint_dir):
        """Create a minimal config for testing with fake data."""
        # Create a minimal config that uses fake data
        config = _config.TrainConfig(
            name="test_iterator_checkpoint",
            checkpoint_dir=temp_checkpoint_dir / "checkpoints",
            batch_size=4,
            seed=42,
            data=_config.DataConfig(
                repo_id="fake",
                rlds_data_dir=None,
                image_size=(224, 224),
            ),
            model=_config.ModelConfig(
                action_horizon=1,
                action_dim=7,
            ),
        )
        return config

    def test_persistent_iterator_parameter(self, minimal_config):
        """Test that persistent_iterator parameter is properly passed through."""
        # Test with persistent_iterator=True
        loader_persistent = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=True,
        )

        # Check if loader has checkpoint methods
        assert hasattr(loader_persistent, "save_iterator_checkpoint"), "Loader should have save method"
        assert hasattr(loader_persistent, "restore_iterator_checkpoint"), "Loader should have restore method"

        # Test with persistent_iterator=False
        loader_non_persistent = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=False,
        )

        # Should still have methods but won't do anything
        assert hasattr(loader_non_persistent, "save_iterator_checkpoint")
        assert hasattr(loader_non_persistent, "restore_iterator_checkpoint")

    def test_iterator_checkpoint_save_and_restore(self, minimal_config, temp_checkpoint_dir):
        """Test that saving and restoring iterator produces consistent batches."""
        # Skip this test if TensorFlow datasets aren't available or if using fake data
        if minimal_config.data.repo_id == "fake":
            pytest.skip("Skipping iterator checkpoint test with fake data (no real TF dataset)")

        checkpoint_dir = str(temp_checkpoint_dir / "tf_iterator")

        # Create first loader and iterate through some batches
        loader1 = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=True,
        )

        iter1 = iter(loader1)
        batches_before_checkpoint = []

        # Get first N batches
        num_batches_before = 5
        for _ in range(num_batches_before):
            batch = next(iter1)
            batches_before_checkpoint.append(batch)

        # Save iterator state
        loader1.save_iterator_checkpoint(checkpoint_dir)

        # Get next M batches from the same iterator (these should match after restore)
        num_batches_after = 3
        expected_batches_after_restore = []
        for _ in range(num_batches_after):
            batch = next(iter1)
            expected_batches_after_restore.append(batch)

        # Create a new loader with the same configuration
        loader2 = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,  # Same seed
            persistent_iterator=True,
        )

        # Restore iterator state
        loader2.restore_iterator_checkpoint(checkpoint_dir)

        # Create iterator after restoration
        iter2 = iter(loader2)

        # Get batches from restored iterator - should match expected_batches_after_restore
        for i in range(num_batches_after):
            restored_batch = next(iter2)
            expected_batch = expected_batches_after_restore[i]

            # Compare observations
            obs_restored, actions_restored = restored_batch
            obs_expected, actions_expected = expected_batch

            # Check that images match (within floating point tolerance)
            for cam_key in obs_restored.images.keys():
                np.testing.assert_allclose(
                    np.array(obs_restored.images[cam_key]),
                    np.array(obs_expected.images[cam_key]),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Restored images for {cam_key} at batch {i} don't match",
                )

            # Check that actions match
            np.testing.assert_allclose(
                np.array(actions_restored),
                np.array(actions_expected),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"Restored actions at batch {i} don't match",
            )

    def test_iterator_checkpoint_nonexistent_restore(self, minimal_config, temp_checkpoint_dir):
        """Test that restoring from non-existent checkpoint doesn't crash."""
        checkpoint_dir = str(temp_checkpoint_dir / "nonexistent")

        loader = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=True,
        )

        # Should not crash, just log a warning
        loader.restore_iterator_checkpoint(checkpoint_dir)

        # Iterator should still work
        iter1 = iter(loader)
        batch = next(iter1)
        assert batch is not None

    def test_non_persistent_iterator_no_save(self, minimal_config, temp_checkpoint_dir):
        """Test that non-persistent iterators don't save/restore."""
        checkpoint_dir = str(temp_checkpoint_dir / "should_not_save")

        loader = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=False,  # Non-persistent
        )

        # Save should be a no-op
        loader.save_iterator_checkpoint(checkpoint_dir)

        # Directory should not be created
        assert not Path(checkpoint_dir).exists(), "Non-persistent loader should not create checkpoint dir"

    def test_multiple_processes_independent_checkpoints(self, minimal_config, temp_checkpoint_dir):
        """Test that each process can save/restore its own iterator state."""
        # This is more of a design verification test
        # In multi-process setting, each process would have its own data_loader instance
        # and save to the same directory structure

        checkpoint_dir = str(temp_checkpoint_dir / "multiprocess")

        # Simulate process 0
        loader_p0 = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=True,
        )

        # Each process should be able to save independently
        loader_p0.save_iterator_checkpoint(checkpoint_dir)

        # Each process should be able to restore independently
        loader_p0_restore = _data_loader.create_data_loader(
            minimal_config,
            shuffle=True,
            seed=42,
            persistent_iterator=True,
        )
        loader_p0_restore.restore_iterator_checkpoint(checkpoint_dir)

        # Should work without errors
        iter_restored = iter(loader_p0_restore)
        batch = next(iter_restored)
        assert batch is not None

    def test_iterable_transformed_dataset_persistent_flag(self):
        """Test that IterableTransformedDataset properly handles persistent_iterator flag."""
        # Create a simple TF dataset
        tf_dataset = tf.data.Dataset.range(100).batch(4)

        # Test with persistent_iterator=True
        dataset_persistent = _data_loader.IterableTransformedDataset(
            batch_size=4,
            dataset=tf_dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        assert dataset_persistent.persistent_iterator is True
        assert hasattr(dataset_persistent, "save_iterator_checkpoint")
        assert hasattr(dataset_persistent, "restore_iterator_checkpoint")
        assert dataset_persistent._tf_iterator is not None
        assert dataset_persistent._tf_checkpoint is not None

        # Test with persistent_iterator=False
        dataset_non_persistent = _data_loader.IterableTransformedDataset(
            batch_size=4,
            dataset=tf_dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=False,
        )

        assert dataset_non_persistent.persistent_iterator is False
        assert dataset_non_persistent._tf_iterator is None
        assert dataset_non_persistent._tf_checkpoint is None

    def test_iterator_checkpoint_preserves_shuffle_state(self, temp_checkpoint_dir):
        """Test that iterator checkpoint preserves shuffle buffer state."""
        # Create a simple dataset with known values
        dataset = tf.data.Dataset.range(1000)
        dataset = dataset.shuffle(buffer_size=100, seed=42, reshuffle_each_iteration=False)
        dataset = dataset.batch(10)

        checkpoint_dir = str(temp_checkpoint_dir / "shuffle_test")

        # Wrap in our checkpointable iterator
        iterable = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        # Iterate and collect some batches
        iter1 = iter(iterable)
        batches_before = []
        for _ in range(5):
            batch = next(iter1)
            batches_before.append(batch)

        # Save checkpoint
        iterable.save_iterator_checkpoint(checkpoint_dir)

        # Continue iterating and collect expected batches
        expected_batches = []
        for _ in range(3):
            batch = next(iter1)
            expected_batches.append(batch)

        # Create new iterator and restore
        iterable2 = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        iterable2.restore_iterator_checkpoint(checkpoint_dir)
        iter2 = iter(iterable2)

        # Get batches from restored iterator
        for i in range(3):
            restored_batch = next(iter2)
            expected_batch = expected_batches[i]

            # Batches should match exactly
            np.testing.assert_array_equal(
                restored_batch,
                expected_batch,
                err_msg=f"Batch {i} after restoration doesn't match expected",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
