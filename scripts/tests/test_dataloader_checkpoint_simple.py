#!/usr/bin/env python3
"""Simple test for dataloader checkpoint functionality.

This is a minimal test that can be run quickly without full training setup.
Usage:
    python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=<your_config>
"""

import logging
import tempfile
from pathlib import Path

import jax

import openpi_cot.dataloader.cot_data_loader as _data_loader
import openpi_cot.training.config as _config


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 80)
    print("DataLoader Checkpoint Test")
    print("=" * 80 + "\n")

    # Load config
    print("Loading configuration...")
    config = _config.cli()
    print(f"✓ Config loaded: {config.name if hasattr(config, 'name') else 'default'}")

    # Create temporary checkpoint directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "test_checkpoint"
        checkpoint_dir.mkdir()
        print(f"✓ Checkpoint directory: {checkpoint_dir}\n")

        # =====================================================================
        # Step 1: Create dataloader with persistent_iterator=True
        # =====================================================================
        print("Step 1: Creating dataloader with persistent_iterator=True...")
        try:
            dataloader1 = _data_loader.create_data_loader(
                config,
                sharding=None,  # Simplified - no sharding needed for test
                shuffle=True,
                seed=42,
                persistent_iterator=True,  # Required for checkpointing
            )
            print("✓ Dataloader created successfully\n")
        except Exception as e:
            print(f"✗ Failed to create dataloader: {e}")
            return 1

        # =====================================================================
        # Step 2: Iterate through some batches
        # =====================================================================
        print("Step 2: Iterating through 5 batches...")
        try:
            data_iter = iter(dataloader1)
            for i in range(5):
                obs, actions = next(data_iter)
                print(f"  Batch {i+1}: obs.state shape={obs.state.shape}, actions shape={actions.shape}")
            print("✓ Successfully processed 5 batches\n")
        except Exception as e:
            print(f"✗ Failed to iterate: {e}")
            return 1

        # Check batch counter
        batches_seen = dataloader1.get_batches_seen()
        print(f"Batches seen: {batches_seen}")
        assert batches_seen == 5, f"Expected 5 batches, got {batches_seen}"
        print("✓ Batch counter is correct\n")

        # =====================================================================
        # Step 3: Save checkpoint
        # =====================================================================
        print("Step 3: Saving checkpoint...")
        try:
            save_path = dataloader1.save_dataloader_state(str(checkpoint_dir))
            print(f"✓ Checkpoint saved to: {save_path}\n")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            return 1

        # Verify checkpoint files exist
        checkpoint_files = list(checkpoint_dir.glob("ckpt*"))
        print(f"Checkpoint files: {[f.name for f in checkpoint_files]}")
        assert len(checkpoint_files) > 0, "No checkpoint files created"
        print("✓ Checkpoint files exist\n")

        # =====================================================================
        # Step 4: Create new dataloader and load checkpoint
        # =====================================================================
        print("Step 4: Creating new dataloader and loading checkpoint...")
        try:
            dataloader2 = _data_loader.create_data_loader(
                config,
                sharding=None,
                shuffle=True,
                seed=42,
                persistent_iterator=True,
            )
            print("✓ New dataloader created")
        except Exception as e:
            print(f"✗ Failed to create second dataloader: {e}")
            return 1

        # Verify starts at 0
        initial_batches = dataloader2.get_batches_seen()
        print(f"Initial batch count: {initial_batches}")
        assert initial_batches == 0, f"Expected 0, got {initial_batches}"
        print("✓ Initial batch count is 0")

        # Load checkpoint
        try:
            restored_batches = dataloader2.load_dataloader_state(str(checkpoint_dir))
            print(f"✓ Checkpoint loaded, restored to batch {restored_batches}\n")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return 1

        # Verify restored count matches
        assert restored_batches == 5, f"Expected 5, got {restored_batches}"
        print("✓ Restored batch count is correct\n")

        # =====================================================================
        # Step 5: Continue iteration
        # =====================================================================
        print("Step 5: Continuing iteration from restored state...")
        try:
            data_iter2 = iter(dataloader2)
            for i in range(3):
                obs, actions = next(data_iter2)
                print(f"  Batch {i+1} (after restore): obs.state shape={obs.state.shape}")
            print("✓ Successfully processed 3 more batches\n")
        except Exception as e:
            print(f"✗ Failed to continue iteration: {e}")
            return 1

        # Check final count
        final_batches = dataloader2.get_batches_seen()
        print(f"Final batch count: {final_batches}")
        assert final_batches == 8, f"Expected 8 (5+3), got {final_batches}"
        print("✓ Final batch count is correct\n")

        # =====================================================================
        # Step 6: Test error handling
        # =====================================================================
        print("Step 6: Testing error handling (persistent_iterator=False)...")
        try:
            dataloader3 = _data_loader.create_data_loader(
                config,
                sharding=None,
                shuffle=True,
                seed=42,
                persistent_iterator=False,  # Checkpointing should fail
            )
            print("✓ Created dataloader with persistent_iterator=False")
        except Exception as e:
            print(f"✗ Failed to create dataloader: {e}")
            return 1

        # Try to save - should fail
        try:
            dataloader3.save_dataloader_state(str(checkpoint_dir / "should_fail"))
            print("✗ ERROR: Should have raised ValueError!")
            return 1
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {str(e)[:80]}...\n")

        # =====================================================================
        # Summary
        # =====================================================================
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print("✓ Step 1: Created dataloader with persistent_iterator=True")
        print("✓ Step 2: Iterated through 5 batches")
        print("✓ Step 3: Saved checkpoint successfully")
        print("✓ Step 4: Loaded checkpoint into new dataloader")
        print("✓ Step 5: Continued iteration from restored state")
        print("✓ Step 6: Error handling works correctly")
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
