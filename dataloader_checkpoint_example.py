#!/usr/bin/env python3
"""Example script demonstrating the skip-based dataloader checkpoint usage.

This example shows how to:
1. Create a dataloader
2. Train for some batches
3. Save the dataloader state (lightweight, just batch counter)
4. Load the state and resume from where you left off

The skip-based approach is much lighter weight than TensorFlow checkpointing:
- Checkpoint file is ~100 bytes (just a JSON with batch counter)
- No persistent_iterator requirement
- Works with all dataset types
- Fast save/load operations
"""

import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def demo_checkpoint_workflow():
    """Demonstrate the checkpoint save/load workflow."""

    # This is a minimal example - in practice you would:
    # 1. Create your actual config
    # 2. Create the dataloader with create_data_loader()
    # 3. Iterate through batches during training

    print("\n" + "="*70)
    print("Dataloader Checkpoint Example")
    print("="*70)

    # Simulate creating a dataloader (replace with your actual dataloader)
    class MockDataLoader:
        """Mock dataloader for demonstration purposes."""
        def __init__(self):
            self._seen_batches = 0
            self._skip_batches = 0

        def __iter__(self):
            """Simulate iteration with skip support."""
            if self._skip_batches > 0:
                logging.info(f"Skipping {self._skip_batches} batches...")
                self._seen_batches = self._skip_batches
                self._skip_batches = 0

            for i in range(self._seen_batches, 100):  # Simulate 100 total batches
                self._seen_batches = i + 1
                yield {"batch_id": i, "data": f"batch_{i}"}

        def save_dataloader_state(self, checkpoint_dir: str) -> str:
            """Save batch counter to JSON."""
            import json
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(checkpoint_dir) / "dataloader_state.json"

            checkpoint_data = {
                "batches_seen": int(self._seen_batches),
                "version": "1.0",
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logging.info(f"✓ Saved checkpoint at batch {self._seen_batches}")
            return str(checkpoint_path)

        def load_dataloader_state(self, checkpoint_dir: str) -> int:
            """Load batch counter from JSON."""
            import json
            checkpoint_path = Path(checkpoint_dir) / "dataloader_state.json"

            if not checkpoint_path.exists():
                raise ValueError(f"No checkpoint found at {checkpoint_path}")

            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            self._seen_batches = checkpoint_data["batches_seen"]
            self._skip_batches = self._seen_batches

            logging.info(f"✓ Loaded checkpoint (will skip to batch {self._seen_batches})")
            return self._seen_batches

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"

        print("\n--- Phase 1: Initial Training ---")
        loader = MockDataLoader()

        # Train for 10 batches
        print("Training for 10 batches...")
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            print(f"  Processing batch {batch['batch_id']}")

        print(f"\nCurrent position: batch {loader._seen_batches}")

        # Save checkpoint
        print("\n--- Phase 2: Save Checkpoint ---")
        checkpoint_path = loader.save_dataloader_state(str(checkpoint_dir))
        checkpoint_size = Path(checkpoint_path).stat().st_size
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Checkpoint size: {checkpoint_size} bytes")

        # Show checkpoint content
        import json
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint content: {json.dumps(checkpoint_data, indent=2)}")

        # Simulate restart: create new loader
        print("\n--- Phase 3: Simulate Restart & Resume ---")
        print("Creating new dataloader (simulating restart)...")
        new_loader = MockDataLoader()

        # Load checkpoint
        batches_seen = new_loader.load_dataloader_state(str(checkpoint_dir))
        print(f"Resuming from batch {batches_seen}")

        # Continue training
        print("\nContinuing training for 5 more batches...")
        for i, batch in enumerate(new_loader):
            if i >= 5:
                break
            print(f"  Processing batch {batch['batch_id']}")

        print(f"\nFinal position: batch {new_loader._seen_batches}")

    print("\n" + "="*70)
    print("✓ Example completed successfully!")
    print("="*70)

    print("\n--- Key Benefits of Skip-Based Approach ---")
    print("✓ Tiny checkpoint files (~100 bytes vs 1-10 GB)")
    print("✓ Fast save/load operations (milliseconds vs seconds)")
    print("✓ No persistent_iterator requirement")
    print("✓ Works with all TensorFlow operations")
    print("✓ GCS-compatible (via tf.io.gfile)")


def usage_notes():
    """Print usage notes for real implementation."""
    print("\n" + "="*70)
    print("Usage in Your Training Code")
    print("="*70)

    print("""
# 1. Create dataloader (no special flags needed!)
from openpi_cot.dataloader.cot_data_loader import create_data_loader

loader = create_data_loader(
    config=your_config,
    shuffle=True,
    seed=42,
)

# 2. Train and save checkpoints periodically
for epoch in range(num_epochs):
    for batch_idx, (obs, actions) in enumerate(loader):
        # ... your training code ...

        # Save checkpoint every N batches
        if batch_idx % checkpoint_interval == 0:
            checkpoint_dir = f"gs://your-bucket/checkpoints/epoch_{epoch}"
            loader.save_dataloader_state(checkpoint_dir)

# 3. Resume from checkpoint after restart
loader = create_data_loader(config=your_config, ...)

# Load the checkpoint
checkpoint_dir = "gs://your-bucket/checkpoints/epoch_5"
batches_seen = loader.load_dataloader_state(checkpoint_dir)
print(f"Resuming from batch {batches_seen}")

# Continue training - iterator will automatically skip to checkpoint position
for obs, actions in loader:
    # ... continues from where you left off ...
    pass
""")

    print("="*70)
    print("Note: The skip happens automatically on the first call to __iter__")
    print("="*70)


if __name__ == "__main__":
    demo_checkpoint_workflow()
    usage_notes()
