"""Tests for TensorFlow iterator checkpointing functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

import openpi_cot.dataloader.cot_data_loader as _data_loader


class TestIteratorCheckpointing:
    """Test iterator save/restore for deterministic resumption."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

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

    def test_iterator_checkpoint_nonexistent_restore(self, temp_checkpoint_dir):
        """Test that restoring from non-existent checkpoint doesn't crash."""
        checkpoint_dir = str(temp_checkpoint_dir / "nonexistent")

        dataset = tf.data.Dataset.range(100).batch(10)
        iterable = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        # Should not crash, just log a warning
        iterable.restore_iterator_checkpoint(checkpoint_dir)

        # Iterator should still work (from the beginning)
        iter1 = iter(iterable)
        batch = next(iter1)
        assert batch is not None

    def test_non_persistent_iterator_no_save(self, temp_checkpoint_dir):
        """Test that non-persistent iterators don't save/restore."""
        checkpoint_dir = str(temp_checkpoint_dir / "should_not_save")

        dataset = tf.data.Dataset.range(100).batch(10)
        iterable = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=False,  # Non-persistent
        )

        # Save should be a no-op (will log a warning)
        iterable.save_iterator_checkpoint(checkpoint_dir)

        # Directory should not be created
        assert not Path(checkpoint_dir).exists(), "Non-persistent iterator should not create checkpoint dir"

    def test_iterator_continues_from_correct_position(self, temp_checkpoint_dir):
        """Test that iterator resumes from the exact position after restore."""
        checkpoint_dir = str(temp_checkpoint_dir / "position_test")

        # Create dataset with known sequential values
        dataset = tf.data.Dataset.range(1000).batch(10)

        # First iterator - consume some batches
        iterable1 = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        iter1 = iter(iterable1)

        # Consume first 5 batches (0-49)
        for _ in range(5):
            _ = next(iter1)

        # Save checkpoint
        iterable1.save_iterator_checkpoint(checkpoint_dir)

        # Get the next batch from original iterator (should be batch 5: [50-59])
        batch_5 = next(iter1)
        expected_values = np.arange(50, 60)
        np.testing.assert_array_equal(batch_5, expected_values)

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

        # First batch from restored iterator should also be batch 5: [50-59]
        restored_batch_5 = next(iter2)
        np.testing.assert_array_equal(restored_batch_5, expected_values)

        # Next batch should be batch 6: [60-69]
        batch_6 = next(iter2)
        np.testing.assert_array_equal(batch_6, np.arange(60, 70))

    def test_cot_rlds_dataloader_checkpoint_methods(self):
        """Test that CoTRLDSDataLoader properly exposes checkpoint methods."""
        dataset = tf.data.Dataset.range(100).batch(10)
        iterable = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=True,
        )

        # Create a mock data config
        class MockDataConfig:
            pass

        data_cfg = MockDataConfig()

        # Test with persistent_iterator=True
        loader_persistent = _data_loader.CoTRLDSDataLoader(
            iterable,
            sharding=None,
            num_batches=None,
            data_cfg=data_cfg,
            persistent_iterator=True,
        )

        assert hasattr(loader_persistent, "save_iterator_checkpoint")
        assert hasattr(loader_persistent, "restore_iterator_checkpoint")

        # Test with persistent_iterator=False
        iterable_non_persistent = _data_loader.IterableTransformedDataset(
            batch_size=10,
            dataset=dataset,
            transforms=[],
            is_batched=True,
            persistent_iterator=False,
        )

        loader_non_persistent = _data_loader.CoTRLDSDataLoader(
            iterable_non_persistent,
            sharding=None,
            num_batches=None,
            data_cfg=data_cfg,
            persistent_iterator=False,
        )

        # Should have methods but they won't do anything
        assert hasattr(loader_non_persistent, "save_iterator_checkpoint")
        assert hasattr(loader_non_persistent, "restore_iterator_checkpoint")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
